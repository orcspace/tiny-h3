"""Measured GPU tensor rates and GeForce half-rate detection.

GeForce cards run FP16/BF16 (and TF32) tensor math with FP32 accumulation at half the rate of the same silicon sold
as a professional card; INT8 and FP16-accumulate math are not capped. The Linux kernel module gpu-unlock
(https://github.com/Zagitta/gpu-unlock) removes the cap at boot. TinyH3 never touches the driver itself: it measures
what the card actually does and picks kernels accordingly:

* locked:   attention with FP16 PV accumulation (SageAttention ``pv_accum_dtype='fp16+fp32'``), where available;
* unlocked: BF16 attention kernels (Spark / Flash) at full rate.

The int8 GEMMs (all DiT linears) run at full rate either way.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict

from .config import home

PROFILE_VERSION = 1


@dataclass
class TensorProfile:
    gpu: str
    driver: str
    bf16_tflops: float      # BF16 x BF16 -> FP32 accumulate (cuBLAS)
    fp16acc_tflops: float   # FP16 x FP16 -> FP16 accumulate
    int8_tops: float
    unlocked: bool | None   # None: not a GeForce part / cannot tell
    measured_at: float
    version: int = PROFILE_VERSION

    @property
    def summary(self) -> str:
        state = {True: 'разблокирована (полная скорость BF16)', False: 'BF16 с FP32-накоплением на половинной скорости',
                 None: 'ограничение неприменимо'}[self.unlocked]
        return (f'{self.gpu}: BF16 {self.bf16_tflops:.0f} TFLOPS, FP16-acc {self.fp16acc_tflops:.0f} TFLOPS, '
                f'INT8 {self.int8_tops:.0f} TOPS — {state}')


def _path():
    return home() / 'gpu_profile.json'


def _time_op(fn, warmup=3, iters=10) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000 / iters


def measure(n: int = 4096) -> TensorProfile:
    import torch
    props = torch.cuda.get_device_properties(0)
    flops = 2.0 * n ** 3
    a = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
    bf16 = flops / _time_op(lambda: a @ b) / 1e12
    a16, b16 = a.half(), b.half()
    prev = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    try:
        # cuBLAS may use FP16 accumulation for FP16 GEMMs when reduced-precision reduction is allowed; the
        # comparison below only needs an upper bound of what the tensor cores do without the FP32-accumulate cap.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        fp16acc = flops / _time_op(lambda: a16 @ b16) / 1e12
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = prev
    i8a = torch.randint(-127, 127, (n, n), device='cuda', dtype=torch.int8)
    i8b = torch.randint(-127, 127, (n, n), device='cuda', dtype=torch.int8)
    int8 = flops / _time_op(lambda: torch._int_mm(i8a, i8b)) / 1e12
    del a, b, a16, b16, i8a, i8b
    torch.cuda.empty_cache()
    unlocked = classify(props.name, bf16, int8)
    return TensorProfile(gpu=props.name, driver=torch.version.cuda or '', bf16_tflops=round(bf16, 1),
                         fp16acc_tflops=round(fp16acc, 1), int8_tops=round(int8, 1), unlocked=unlocked,
                         measured_at=time.time())


def classify(name: str, bf16_tflops: float, int8_tops: float) -> bool | None:
    """INT8 runs at 4x the capped BF16-FP32-accumulate rate on a locked GeForce and at ~2x once unlocked."""
    if 'geforce' not in name.lower() and 'rtx 30' not in name.lower() and 'rtx 40' not in name.lower():
        return None
    if bf16_tflops <= 0:
        return None
    ratio = int8_tops / bf16_tflops
    return ratio < 3.0


def load(max_age_days: float = 30) -> TensorProfile | None:
    try:
        raw = json.loads(_path().read_text())
        if raw.get('version') != PROFILE_VERSION or time.time() - raw['measured_at'] > max_age_days * 86400:
            return None
        raw.pop('version', None)
        return TensorProfile(**raw)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def profile(refresh: bool = False) -> TensorProfile | None:
    """The cached profile, measuring once if needed. None without CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except ImportError:
        return None
    cached = None if refresh else load()
    if cached is not None:
        import torch
        if cached.gpu == torch.cuda.get_device_properties(0).name:
            return cached
    prof = measure()
    try:
        _path().parent.mkdir(parents=True, exist_ok=True)
        _path().write_text(json.dumps(asdict(prof), indent=1))
    except OSError:
        pass
    return prof


def advice(prof: TensorProfile | None, os_name: str) -> list[str]:
    """Human-readable recommendations for ``tinyh3 doctor``."""
    if prof is None:
        return ['CUDA-видеокарта не найдена.']
    tips = [prof.summary]
    if prof.unlocked is False:
        if os_name == 'Linux':
            tips.append('Внимание считается в BF16 с FP32-накоплением на половинной скорости. На Linux это снимает '
                        'gpu-unlock (модуль ядра, загружается до драйвера NVIDIA): +50–60% к вниманию, ≈ +35% к шагу '
                        'TinyH3. Установка с подтверждением: scripts/gpu-unlock-linux.sh')
        else:
            tips.append('Карта работает с половинной скоростью BF16 с FP32-накоплением. gpu-unlock официально есть только '
                        'для Linux; сторонние порты под Windows авторы gpu-unlock советуют использовать с большой '
                        'осторожностью. TinyH3 компенсирует это: внимание считается с FP16-накоплением, если '
                        'установлен SageAttention.')
    elif prof.unlocked:
        tips.append('Полная скорость BF16: TinyH3 использует BF16-ядра внимания.')
    return tips


def measure_attention(tokens: int = 16384, heads: int = 8, head_dim: int = 128) -> dict:
    """Milliseconds per dense attention call for every available backend (stored with the profile)."""
    import torch
    from .runtime.attention import attention, available
    q, k, v = (torch.randn(tokens, heads, head_dim, device='cuda', dtype=torch.bfloat16) for _ in range(3))
    out = {}
    for name in available():
        if name == 'spark':
            continue
        try:
            out[name] = round(_time_op(lambda: attention(q, k, v, name), warmup=2, iters=5) * 1000, 2)
        except Exception:  # a backend that fails on this GPU is simply not listed
            out[name] = float('nan')
    try:
        raw = json.loads(_path().read_text())
        raw.setdefault('attention_ms', {})[str(tokens)] = out
        _path().write_text(json.dumps(raw, indent=1))
    except (OSError, ValueError):
        pass
    return out
