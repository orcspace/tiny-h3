"""MiniMax-H3 DiT executor over streamed, quantized blocks.

Functionally equal to diffusers' ``MiniMaxH3Transformer3DModel`` (pruned AdaLN) for one packed sequence, but written
as functions over weight views so that:

* a block's weights are borrowed from the streamer for exactly the time its compute is enqueued;
* attention runs per *head group*: Q/K/V of ``head_group`` heads at a time, so the transient is
  ``L x head_group x 128`` instead of ``L x 3 x 7168``;
* the out-projection runs once on the full attention output, the FFN runs in token chunks;
* norm + AdaLN modulation + Hadamard rotation + int8 quantization is one kernel, as is SwiGLU + rotation + int8.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .. import kernels as K
from ..config import DIT, DiTConfig
from ..kernels import reference as ref
from .attention import attention
from .layout import Layout, rotary

BF16 = torch.bfloat16


def slice_rows(rec: dict, start: int, stop: int) -> dict:
    out = dict(rec)
    out['q'] = rec['q'][start:stop]
    out['scale'] = rec['scale'][start:stop]
    return out


def qk_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.float()).to(x.dtype)


def apply_rope_(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """diffusers ``_apply_rotary_emb`` on (L, H, D) in place: rotate_half over the leading ``rotary_dim`` channels."""
    rd = cos.shape[-1]
    half = rd // 2
    c = cos.to(x.dtype)[:, None, :]
    s = sin.to(x.dtype)[:, None, :]
    r = x[..., :rd]
    x1, x2 = r[..., :half], r[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    x[..., :rd] = r * c + rotated * s
    return x


def gated_add_(x: torch.Tensor, gate: torch.Tensor, rows: torch.Tensor | None, y: torch.Tensor,
               chunk: int = 8192) -> None:
    """``x += gate[rows] * y`` without materializing ``gate[rows]`` for the whole sequence."""
    if rows is None:
        x.add_(y)
        return
    for c0 in range(0, x.shape[0], chunk):
        c1 = min(c0 + chunk, x.shape[0])
        x[c0:c1].add_(gate.index_select(0, rows[c0:c1]) * y[c0:c1])


def adaln_tables(w: dict, temb: torch.Tensor, hidden: int = DIT.hidden_size):
    """Six (num_timesteps * 3, D) modulation tables of one block (pruned AdaLN, folded fp32 bias)."""
    m = F.linear(temb.to(w['adaln_w'].dtype), w['adaln_w'])
    m = (m.float() + w['adaln_b'].float()).to(BF16).view(-1, 6 * hidden)
    return [t.contiguous() for t in m.chunk(6, dim=-1)]


def self_attention(xq, xs, w, L, rope, *, heads, head_dim, head_group, eps, backend, out_dtype=BF16,
                   permutation=None):
    """``backend``: a dense backend name, or a callable ``(q, k, v) -> o`` (e.g. Spark bound to this layer, which
    needs ``head_group == heads``).

    ``permutation`` = (perm, inverse): rows are gathered once on the int8 input, so Q/K/V come out of the GEMM
    already in the backend's order (no copies of the projections), and the attention output is scattered back once.
    """
    if permutation is not None:
        perm, inverse = permutation
        xq, xs = xq.index_select(0, perm), xs.index_select(0, perm)
        if rope is not None:
            rope = tuple(t.index_select(0, perm) for t in rope)
    inner = heads * head_dim
    qkv = w['qkv']
    attend = backend if callable(backend) else (lambda q, k, v: attention(q, k, v, backend))
    o = torch.empty(L, inner, device=xq.device, dtype=out_dtype)
    for h0 in range(0, heads, head_group):
        h1 = min(h0 + head_group, heads)
        a, b = h0 * head_dim, h1 * head_dim
        q = K.qlinear(xq, xs, slice_rows(qkv, a, b)).view(L, h1 - h0, head_dim)
        k = K.qlinear(xq, xs, slice_rows(qkv, inner + a, inner + b)).view(L, h1 - h0, head_dim)
        v = K.qlinear(xq, xs, slice_rows(qkv, 2 * inner + a, 2 * inner + b)).view(L, h1 - h0, head_dim)
        q = qk_norm(q, w['norm_q'], eps)
        k = qk_norm(k, w['norm_k'], eps)
        if rope is not None:
            apply_rope_(q, *rope)
            apply_rope_(k, *rope)
        o[:, a:b] = attend(q, k, v).reshape(L, -1)
        del q, k, v
    if permutation is not None:
        o = o.index_select(0, inverse)
    oq, os_ = K.rotate_quantize(o)
    del o
    return K.qlinear(oq, os_, w['out'])


def dit_block(x, w, rows, mods, rope, *, head_group=8, ffn_chunk=4096, backend='sdpa', cfg=DIT, permutation=None):
    """One modulated H3 block on the packed sequence ``x`` (L, D) in place."""
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
    L = x.shape[0]
    lin = w['linears']
    xq, xs = K.norm_modulate_rotate_quantize(x, w['norm1'], cfg.norm_eps, scale_msa, shift_msa, rows)
    y = self_attention(xq, xs, {**lin, 'norm_q': w['norm_q'], 'norm_k': w['norm_k']}, L, rope,
                       heads=cfg.num_heads, head_dim=cfg.head_dim, head_group=head_group, eps=cfg.qk_norm_eps,
                       backend=backend, permutation=permutation)
    del xq, xs
    gated_add_(x, gate_msa, rows, y)
    del y
    hq, hs = K.norm_modulate_rotate_quantize(x, w['norm2'], cfg.norm_eps, scale_mlp, shift_mlp, rows)
    for c0 in range(0, L, ffn_chunk):
        c1 = min(c0 + ffn_chunk, L)
        h = K.qlinear(hq[c0:c1], hs[c0:c1], lin['ff_in'])
        uq, us = K.swiglu_rotate_quantize(h)
        del h
        y = K.qlinear(uq, us, lin['ff_out'])
        gated_add_(x[c0:c1], gate_mlp, rows[c0:c1], y)
        del y, uq, us
    return x


def plain_block(x, w, *, head_group=56, backend='sdpa', cfg=DIT):
    """Token-refiner block: pre-norm attention and FFN, no modulation, no rotary."""
    L = x.shape[0]
    lin = w['linears']
    xq, xs = K.rotate_quantize(ref.rms_norm(x, w['norm1'], cfg.norm_eps))
    y = self_attention(xq, xs, {**lin, 'norm_q': w['norm_q'], 'norm_k': w['norm_k']}, L, None,
                       heads=cfg.num_heads, head_dim=cfg.head_dim, head_group=head_group, eps=cfg.qk_norm_eps,
                       backend=backend)
    x.add_(y)
    hq, hs = K.rotate_quantize(ref.rms_norm(x, w['norm2'], cfg.norm_eps))
    h = K.qlinear(hq, hs, lin['ff_in'])
    uq, us = K.swiglu_rotate_quantize(h)
    x.add_(K.qlinear(uq, us, lin['ff_out']))
    return x


def time_coords(table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Pruned timestep embedder: linear interpolation in the (table_size, adaln_rank) coordinate table."""
    steps = table.shape[0] - 1
    pos = t.to(table.dtype).flatten().clamp(0.0, 1.0) * steps
    lower = pos.floor().clamp(max=steps - 1).long()
    return torch.lerp(table.index_select(0, lower), table.index_select(0, lower + 1), (pos - lower).unsqueeze(-1))


def config_from_manifest(raw: dict | None) -> DiTConfig:
    if not raw:
        return DIT
    fields = DiTConfig.__dataclass_fields__
    return DiTConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in fields})


class Request:
    """Per-request device state: layout tensors, rotary tables, row ranges."""

    def __init__(self, layout: Layout, device, cfg: DiTConfig = DIT):
        self.layout = layout
        self.device = device
        self.tags = layout.token_tags.to(device)
        self.video_idx = layout.video_idx.to(device)
        self.audio_idx = layout.audio_idx.to(device)
        self.text_idx = layout.text_idx.to(device)
        cos, sin = rotary(layout.position_ids, cfg.rope_freq_dim, cfg.rope_theta)
        self.rope = (cos.to(device), sin.to(device))
        self.ranges = layout.contiguous_ranges()
        self.seq = layout.seq


class DiT:
    def __init__(self, store, *, device='cuda', tiers=None, slots=2, head_group=8, ffn_chunk=4096,
                 attention_backend='sdpa', passes=None, progress=None, spark=None):
        from .streamer import UnitStreamer
        self.store = store
        self.device = torch.device(device)
        self.cfg = config_from_manifest(store.manifest['dit'].get('config'))
        self.g = store.load_safetensors('dit/globals.safetensors', device=self.device)
        n = len(store.units('dit'))
        tiers = tiers or {i: 'vram' for i in range(n)}
        self.streamer = UnitStreamer(store, 'dit', device=self.device, tiers=tiers, slots=slots, passes=passes,
                                     progress=progress)
        self.num_layers = n
        self.head_group = head_group
        self.ffn_chunk = ffn_chunk
        self.backend = attention_backend
        self.spark = spark  # SparkAttention: sparse layers need every head at once
        self.timings: list[float] = []

    def block_weights(self, tensors: dict, unit: dict) -> dict:
        from ..store import linear_records
        w = {k: v for k, v in tensors.items() if not k.startswith('__') and '.' not in k}
        w['linears'] = linear_records(tensors, unit)
        return w

    def forward(self, req: Request, video_rows: torch.Tensor, audio_rows: torch.Tensor, text_states: torch.Tensor,
                t_unique: torch.Tensor, t_index: torch.Tensor, want_features: bool = False, on_block=None):
        g, cfg = self.g, self.cfg
        dev = self.device
        r = req.ranges
        x = torch.empty(req.seq, cfg.hidden_size, device=dev, dtype=BF16)
        x[r['text']] = text_states.to(dev, BF16)
        video = F.linear(video_rows.to(dev, torch.float32), g['proj_in.weight'].float(), g['proj_in.bias'].float())
        n_cond = req.layout.n_cond
        x[r['cond']] = video[:n_cond].to(BF16)
        x[r['video']] = video[n_cond:].to(BF16)
        del video
        x[r['audio']] = F.linear(audio_rows.to(dev, torch.float32), g['audio_proj_in.weight'].float(),
                                 g['audio_proj_in.bias'].float()).to(BF16)

        t_unique = t_unique.to(dev)
        t_index = t_index.to(dev)
        temb = time_coords(g['time_table'].float(), t_unique)
        rows = t_index * cfg.modality_num + req.tags
        self.streamer.prime()
        units = self.store.units('dit')
        if self.spark is not None:
            self.spark.begin_step(req, t_index)
        for i in range(self.num_layers):
            t0 = time.perf_counter()
            tensors = self.streamer.acquire(i)
            w = self.block_weights(tensors, units[i])
            mods = adaln_tables(w, temb, cfg.hidden_size)
            backend, group, permutation = self.backend, self.head_group, None
            if self.spark is not None and not self.spark.is_dense(i, req.seq):
                backend = (lambda q, k, v, layer=i: self.spark(q, k, v, layer, permuted=True))
                group = cfg.num_heads
                permutation = self.spark.permutation()
            dit_block(x, w, rows, mods, req.rope, head_group=group, ffn_chunk=self.ffn_chunk,
                      backend=backend, cfg=cfg, permutation=permutation)
            self.streamer.release(i, tensors)
            del w, mods, tensors
            if on_block:
                on_block(i, time.perf_counter() - t0)

        # Output heads: pruned AdaLN out-norm, shift then scale, indexed per row by timestep index.
        mod = F.linear(temb.to(g['norm_out.linear.weight'].dtype), g['norm_out.linear.weight'])
        shift, scale = (mod.float() + g['norm_out.bias'].float()).to(BF16).chunk(2, dim=-1)
        features = None
        if want_features and 'dwd_norm_out.linear.weight' in g:
            dmod = F.linear(F.silu(temb.float()), g['dwd_norm_out.linear.weight'].float(),
                            g['dwd_norm_out.linear.bias'].float())
            dscale, dshift = dmod.chunk(2, dim=-1)
            vr = req.ranges['video']
            xv = x[vr].float()
            ti = t_index[vr]
            xv = F.layer_norm(xv, (cfg.hidden_size,), eps=cfg.final_norm_eps)
            features = (xv * (1 + dscale.index_select(0, ti)) + dshift.index_select(0, ti)).to(BF16)
            del xv
        out_video, out_audio = [], []
        for name, idx, weight, bias, sink in (
                ('video', req.video_idx, 'proj_out.weight', 'proj_out.bias', out_video),
                ('audio', req.audio_idx, 'audio_proj_out.weight', 'audio_proj_out.bias', out_audio)):
            xs = x.index_select(0, idx)
            ti = t_index.index_select(0, idx)
            h = ref.rms_norm(xs, g['norm_out.norm.weight'], cfg.final_norm_eps).to(BF16)
            h = h * (1.0 + scale.index_select(0, ti)) + shift.index_select(0, ti)
            sink.append(F.linear(h.float(), g[weight].float(), g[bias].float()))
            del xs, h
        del x
        return out_video[0], out_audio[0], features

    def close(self):
        self.streamer.close()
