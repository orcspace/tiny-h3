"""Execution planner: where every DiT block lives and how the step is partitioned, from measured budgets.

Pure Python (no torch): ``tinyh3 plan --vram-gib 8 --ram-gib 16`` and the UI's pre-flight estimate use it without a
GPU. The runtime calls the same function with live measurements before every request.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .config import DIT, TEXT, SOL, Geometry
from .hardware import Hardware, GiB

MiB = 1 << 20

MIXED_INT8_BLOCKS = (0, 1, 2, 47, 48, 49)
POLICIES = ('int8', 'mixed', 'int4')


# ------------------------------------------------------------------------------------------------ sizes

def linear_bytes(kind: str, n: int, k: int, group: int = 128) -> int:
    if kind == 'int8':
        return n * k + 4 * n
    if kind == 'int4':
        return n * k // 2 + 2 * n * (k // group)
    if kind == 'bf16':
        return 2 * n * k
    raise ValueError(kind)


def block_kinds(policy: str, num_layers: int = DIT.num_layers) -> list[str]:
    if policy == 'int8':
        return ['int8'] * num_layers
    if policy == 'int4':
        return ['int4'] * num_layers
    if policy == 'mixed':
        ends = {i if i >= 0 else num_layers + i for i in (0, 1, 2, -3, -2, -1)}
        return ['int8' if i in ends else 'int4' for i in range(num_layers)]
    raise ValueError(f'unknown policy {policy!r}; expected one of {POLICIES}')


def block_bytes(kind: str) -> int:
    d = DIT.hidden_size
    linears = sum(linear_bytes(kind, n, k) for n, k in DIT.linear_shapes().values())
    adaln = 6 * d * DIT.modality_num * (DIT.adaln_rank * 2 + 4)
    norms = 2 * d * 2 + 2 * DIT.head_dim * 2
    return linears + adaln + norms + 64 * 256  # alignment slack


def text_encoder_bytes(kind: str = 'int8') -> int:
    per_layer = sum(linear_bytes(kind, n, k) for n, k in TEXT.linear_shapes().values())
    return per_layer * TEXT.num_layers_used


def sol_params() -> int:
    d = SOL.dim
    per_layer = d * 3 * d + d * d + d * 2 * d * SOL.ffn_mult + d * SOL.ffn_mult * d
    return per_layer * SOL.num_layers


# ------------------------------------------------------------------------------------------- activations

def activation_bytes(seq: int, head_group: int, ffn_chunk: int) -> int:
    """Peak transient VRAM of one DiT block for ``seq`` tokens (see runtime/dit.py for the matching code)."""
    d, inner, ffn = DIT.hidden_size, DIT.inner_dim, DIT.ffn_dim
    residual = seq * d * 2                       # x, bf16
    x_q = seq * (d + 4)                          # int8 rows + scales
    attn_out = seq * inner * 2 + seq * (inner + 4)  # o (bf16) + its int8 copy for the out-projection
    qkv = int(3 * seq * head_group * DIT.head_dim * 2 * 1.6)  # group q/k/v plus attention-kernel copies
    chunk = min(ffn_chunk, seq)
    ffn_tmp = chunk * (2 * ffn * 2 + ffn + 4 + d * 2)
    rope = seq * DIT.rotary_dim * 4 * 2
    proj_out = seq * d * 2                       # out-projection result before the gated residual add
    return residual + x_q + max(attn_out + qkv, ffn_tmp) + rope + proj_out + 96 * MiB


# --------------------------------------------------------------------------------------------------- plan

@dataclass
class Plan:
    fits: bool
    reason: str = ''
    policy: str = 'mixed'
    vram_budget: int = 0
    ram_budget: int = 0
    seq: int = 0
    head_group: int = 8
    ffn_chunk: int = 4096
    activation: int = 0
    slots: int = 2
    slot_bytes: int = 0
    resident: list = field(default_factory=list)
    pinned: list = field(default_factory=list)
    disk: list = field(default_factory=list)
    block_kinds: list = field(default_factory=list)
    resident_bytes: int = 0
    pinned_bytes: int = 0
    disk_bytes: int = 0
    attention: str = 'sdpa'
    est: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def tier(self, index: int) -> str:
        if index in self.resident:
            return 'vram'
        if index in self.pinned:
            return 'pinned'
        return 'disk'

    def as_dict(self) -> dict:
        d = asdict(self)
        for key in ('vram_budget', 'ram_budget', 'activation', 'slot_bytes', 'resident_bytes', 'pinned_bytes',
                    'disk_bytes'):
            d[key + '_gib'] = round(d[key] / GiB, 2)
        return d

    def summary(self) -> str:
        if not self.fits:
            return f'НЕ ПОМЕЩАЕТСЯ: {self.reason}'
        e = self.est
        return (f'{len(self.resident)} блоков в VRAM, {len(self.pinned)} в pinned-ОЗУ, {len(self.disk)} с диска; '
                f'группа голов {self.head_group}, кусок FFN {self.ffn_chunk}; '
                f'шаг ≈ {e.get("step_s", 0):.1f} с, всего ≈ {e.get("total_s", 0):.0f} с')


def spread(n: int, k: int) -> list[int]:
    """k of n indices spread as evenly as possible (Bresenham)."""
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    return sorted({(i * n) // k + (n // k) // 2 for i in range(k)})


def bit_reversal_order(n: int) -> list[int]:
    bits = max(1, (n - 1).bit_length())
    return sorted(range(n), key=lambda i: int(format(i, f'0{bits}b')[::-1], 2))


def make_plan(hw: Hardware, geometry: Geometry, *, policy: str = 'mixed', steps: int = 4,
              sizes: list[int] | None = None, kinds: list[str] | None = None, attention: str = 'auto',
              text_cached: bool = False, te_bytes: int | None = None, force_head_group: int | None = None,
              force_ffn_chunk: int | None = None, max_resident: int | None = None) -> Plan:
    kinds = kinds or block_kinds(policy)
    sizes = sizes or [block_bytes(k) for k in kinds]
    n = len(sizes)
    seq = geometry.sequence_length
    windows = hw.os == 'Windows'
    plan = Plan(fits=False, policy=policy, seq=seq, block_kinds=kinds)

    if hw.vram_total == 0:
        plan.reason = 'CUDA-видеокарта не найдена'
        return plan

    reserve = int(0.35 * GiB) + (int(0.25 * GiB) if windows else 0)
    plan.vram_budget = vram_budget = max(hw.vram_free - reserve, 0)
    globals_bytes = 64 * MiB

    # 1. Partition the step: the largest head group / FFN chunk whose activations leave at least ~45% of the
    #    budget for weights (resident blocks + streaming slots).
    groups = [force_head_group] if force_head_group else [16, 8, 4, 2, 1]
    chunks = [force_ffn_chunk] if force_ffn_chunk else [8192, 4096, 2048, 1024, 512]
    smallest_slot = min(sizes)
    choice = None
    for g in groups:
        for c in chunks:
            act = activation_bytes(seq, g, c)
            if act + globals_bytes + 2 * smallest_slot <= vram_budget and act <= 0.55 * vram_budget:
                choice = (g, c, act)
                break
        if choice:
            break
    if choice is None:
        g, c = groups[-1], chunks[-1]
        act = activation_bytes(seq, g, c)
        if act + globals_bytes + 2 * smallest_slot > vram_budget:
            plan.reason = (f'активации {act / GiB:.1f} ГБ + слоты не помещаются в бюджет VRAM {vram_budget / GiB:.1f} ГБ '
                           f'при {seq} токенах — уменьшите разрешение или длительность')
            return plan
        choice = (g, c, act)
    plan.head_group, plan.ffn_chunk, plan.activation = choice

    # 2. Residency: keep the biggest blocks first (an int8 block streamed would force int8-sized slots), and spread
    #    the streamed ones across the stack so every copy overlaps the compute of resident neighbours.
    slots = 3 if vram_budget >= 14 * GiB else 2
    plan.slots = slots
    order = sorted(range(n), key=lambda i: (-sizes[i], i))
    room = vram_budget - globals_bytes - plan.activation
    resident: set[int] = set()
    by_kind = {}
    for i in range(n):
        by_kind.setdefault(sizes[i], []).append(i)
    size_classes = sorted(by_kind, reverse=True)
    for size in size_classes:
        members = by_kind[size]
        smaller = [s for s in size_classes if s < size]
        for count in range(len(members), -1, -1):
            streamed_left = len(members) - count
            slot = size if streamed_left else (max(smaller) if smaller else 0)
            used = sum(sizes[i] for i in resident) + count * size + slots * slot
            if used <= room:
                break
        if count < len(members):
            keep = [members[j] for j in range(len(members)) if j not in
                    set(spread(len(members), len(members) - count))]
        else:
            keep = list(members)
        resident.update(keep)
        if count < len(members):
            break
    if max_resident is not None and len(resident) > max_resident:
        drop = sorted(resident, key=lambda i: (sizes[i], i))[:len(resident) - max_resident]
        resident.difference_update(drop)
    streamed = [i for i in range(n) if i not in resident]
    plan.slot_bytes = max((sizes[i] for i in streamed), default=0)
    plan.resident = sorted(resident)
    plan.resident_bytes = sum(sizes[i] for i in resident)
    if plan.resident_bytes + slots * plan.slot_bytes + plan.activation + globals_bytes > vram_budget:
        plan.reason = 'не хватает VRAM даже на слоты стриминга'
        return plan

    # 3. Host tier: pin what RAM allows; the rest is re-read from disk every step, spread by bit reversal.
    ram_reserve = max(int(1.5 * GiB), int(0.08 * hw.ram_total))
    host_working = int(1.6 * GiB) + geometry.video_rows * DIT.hidden_size * 2
    plan.ram_budget = ram_budget = max(hw.ram_available - ram_reserve - host_working, 0)
    pin_cap = int((0.5 if windows else 0.7) * hw.ram_total)
    budget = min(ram_budget, pin_cap)
    pinned, used = [], 0
    for i in sorted(streamed, key=lambda j: bit_reversal_order(n).index(j)):
        if used + sizes[i] <= budget:
            pinned.append(i)
            used += sizes[i]
    plan.pinned = sorted(pinned)
    plan.pinned_bytes = used
    plan.disk = sorted(set(streamed) - set(pinned))
    plan.disk_bytes = sum(sizes[i] for i in plan.disk)
    if plan.disk:
        # disk blocks still pass through a small ring of pinned bounce buffers
        if ram_budget - used < 2 * plan.slot_bytes:
            plan.notes.append('мало ОЗУ под буферы чтения с диска — возможны паузы')

    plan.attention = attention if attention != 'auto' else ('sage' if hw.compute_capability >= (8, 0) else 'sdpa')
    plan.est = estimate(hw, geometry, plan, sizes, steps, text_cached, te_bytes)
    plan.fits = True
    if plan.disk:
        plan.notes.append(f'{len(plan.disk)} блоков читаются с диска каждый шаг '
                          f'({plan.disk_bytes / GiB:.1f} ГБ/шаг при ~{hw.disk_gbps:.1f} ГБ/с)')
    return plan


def estimate(hw: Hardware, geometry: Geometry, plan: Plan, sizes: list[int], steps: int, text_cached: bool,
             te_bytes: int | None) -> dict:
    seq = geometry.sequence_length
    int8_eff = max(hw.int8_tops, 1.0) * 1e12 * 0.55
    attn_rate = max(hw.fp16_tflops, 1.0) * 1e12 * 0.65 * (1.8 if plan.attention == 'sage' else 1.0)
    lin = 2.0 * seq * DIT.block_params() / int8_eff
    attn = 4.0 * seq * seq * DIT.inner_dim / attn_rate
    misc = seq * DIT.hidden_size * 2 * 24 / 300e9
    block_s = lin + attn + misc
    compute = block_s * len(sizes)
    streamed = [i for i in range(len(sizes)) if i not in plan.resident]
    pcie = sum(sizes[i] for i in streamed) / (hw.pcie_gbps * 1e9)
    disk = plan.disk_bytes / (hw.disk_gbps * 1e9)
    step = max(compute, pcie, disk) + 0.05 * len(sizes) * 0.01
    load = (plan.resident_bytes + plan.pinned_bytes) / (hw.disk_gbps * 1e9)
    te = 0.0 if text_cached else (te_bytes or text_encoder_bytes('int8')) / (hw.disk_gbps * 1e9) + 3.0
    sol_tokens = geometry.latent_t * (geometry.latent_h // 2 * 2) * (geometry.latent_w // 2 * 2)
    dwd = 2.0 * sol_tokens * sol_params() / (max(hw.fp16_tflops, 1.0) * 1e12 * 0.5) + 2.0
    total = te + load + steps * step + dwd + 4.0
    return {'block_s': round(block_s, 3), 'step_s': round(step, 2), 'compute_s': round(compute, 2),
            'pcie_s': round(pcie, 2), 'disk_s': round(disk, 2), 'load_s': round(load, 1),
            'text_s': round(te, 1), 'dwd_s': round(dwd, 1), 'total_s': round(total, 1),
            'stall_s': round(max(0.0, max(pcie, disk) - compute) * steps, 2)}


def quick_plan(vram_gib: float, ram_gib: float, height: int, width: int, seconds: float, policy='mixed',
               steps=4, gpu='RTX 3070', disk_gbps=2.0) -> Plan:
    from .hardware import simulated
    from .config import frames_for_seconds
    hw = simulated(vram_gib, ram_gib, gpu, disk_gbps)
    return make_plan(hw, Geometry(height, width, frames_for_seconds(seconds)), policy=policy, steps=steps)
