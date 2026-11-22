"""Pure-PyTorch reference of every TinyH3 numeric kernel.

These functions define the semantics. The Triton kernels in ``triton_ops`` must match them (tests enforce it), and
they are the fallback on machines without Triton and the path used on CPU.

Conventions
-----------
* ConvRot: inputs of every quantized linear are rotated by ``blockdiag(H256)`` along the feature axis, where
  ``H256`` is the normalized *regular* Hadamard matrix built from the order-4 seed (symmetric and orthogonal). The
  weights carry the same rotation, baked in at conversion: ``(x H)(W H)^T = x W^T``.
* Activations: symmetric int8, one fp32 scale per row, ``q = rint(x / s)``, ``s = max|x| / 127``.
* int8 weights: symmetric, one fp32 scale per output channel.
* int4 weights: symmetric in ``[-7, 7]``, groups of 128 along K, one fp16 scale per (row, group). Packed two per
  byte: within each 128-wide group, byte ``j`` holds element ``j`` in its low nibble and element ``j + 64`` in its
  high nibble, so a kernel unpacks one tile into two contiguous K halves without any interleave.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

ROT_GROUP = 256
INT4_GROUP = 128

_HADAMARD: dict = {}


def hadamard(size: int = ROT_GROUP, device='cpu', dtype=torch.float32) -> torch.Tensor:
    key = (size, str(device), dtype)
    if key not in _HADAMARD:
        if size < 4 or size & (size - 1) or math.log(size, 4) % 1:
            raise ValueError(f'Hadamard size must be a power of 4, got {size}')
        seed = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
        m, width = seed, 4
        while width < size:
            m, width = torch.kron(m, seed), width * 4
        _HADAMARD[key] = (m / math.sqrt(size)).to(device=device, dtype=dtype)
    return _HADAMARD[key]


def rotate(x: torch.Tensor, group: int = ROT_GROUP) -> torch.Tensor:
    """``x @ blockdiag(H, ..., H)`` over the last axis, computed in fp32 (fp64 stays fp64)."""
    k = x.shape[-1]
    if k % group:
        raise ValueError(f'{k} features is not a multiple of the rotation group {group}')
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    h = hadamard(group, x.device, dtype)
    return (x.to(dtype).reshape(*x.shape[:-1], k // group, group) @ h).reshape(x.shape)


# ---------------------------------------------------------------------------------------------------- activations

def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp -> (int8 [M, K], fp32 scale [M]); x is 2-D."""
    x = x.float()
    scale = x.abs().amax(dim=-1).clamp_min(1e-12) / 127.0
    q = torch.round(x / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


def rotate_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_rows(rotate(x.reshape(-1, x.shape[-1])))


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y


def norm_modulate(x, norm_weight, eps, scale_table, shift_table, rows):
    """``rmsnorm(x) * (1 + scale[rows]) + shift[rows]`` in fp32; ``rows`` indexes the modulation tables."""
    y = rms_norm(x, norm_weight, eps)
    return y * (1.0 + scale_table.float().index_select(0, rows)) + shift_table.float().index_select(0, rows)


def norm_modulate_rotate_quantize(x, norm_weight, eps, scale_table, shift_table, rows):
    return rotate_quantize(norm_modulate(x, norm_weight, eps, scale_table, shift_table, rows))


def swiglu(h: torch.Tensor, gate_first: bool = False) -> torch.Tensor:
    """H3's FeedForward packs ``[up, gate]`` (diffusers SwiGLU); Sol packs ``[gate, up]``."""
    a, b = h.float().chunk(2, dim=-1)
    gate, up = (a, b) if gate_first else (b, a)
    return up * F.silu(gate)


def swiglu_rotate_quantize(h: torch.Tensor, gate_first: bool = False):
    return rotate_quantize(swiglu(h, gate_first))


# -------------------------------------------------------------------------------------------------------- weights

def quantize_weight_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] -> (int8 [N, K], fp32 scale [N])."""
    w = w.float()
    scale = w.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


CLIP_CANDIDATES = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)


def quantize_weight_int4(w: torch.Tensor, group: int = INT4_GROUP, clip_search: bool = True,
                         row_chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] -> (uint8 [N, K // 2] packed, fp16 scale [N, K // group]).

    With ``clip_search`` every group picks, among a few clipping ratios of its abs-max, the scale with the lowest
    squared reconstruction error: outliers are clipped where that pays off, which RTN at abs-max never does.
    """
    n, k = w.shape
    if k % group or group % 2:
        raise ValueError(f'K={k} must be a multiple of the int4 group {group}')
    packed = torch.empty(n, k // 2, dtype=torch.uint8)
    scales = torch.empty(n, k // group, dtype=torch.float16)
    for r0 in range(0, n, row_chunk):
        r1 = min(r0 + row_chunk, n)
        g = w[r0:r1].float().reshape(r1 - r0, k // group, group)
        amax = g.abs().amax(dim=-1).clamp_min(1e-12)
        best_scale = (amax / 7.0).to(torch.float16)
        if clip_search:
            best_err = None
            for ratio in CLIP_CANDIDATES:
                s = (amax * ratio / 7.0).to(torch.float16).float()
                q = torch.round(g / s[..., None]).clamp_(-7, 7)
                err = (q * s[..., None] - g).pow_(2).sum(-1)
                if best_err is None:
                    best_err, best_scale = err, s.to(torch.float16)
                else:
                    better = err < best_err
                    best_err = torch.where(better, err, best_err)
                    best_scale = torch.where(better, s.to(torch.float16), best_scale)
        q = torch.round(g / best_scale.float()[..., None]).clamp_(-7, 7).to(torch.int8)
        packed[r0:r1] = pack_int4(q.reshape(r1 - r0, k), group).cpu()
        scales[r0:r1] = best_scale.cpu()
    return packed, scales


def pack_int4(q: torch.Tensor, group: int = INT4_GROUP) -> torch.Tensor:
    n, k = q.shape
    half = group // 2
    g = q.reshape(n, k // group, 2, half).to(torch.int16)
    lo, hi = g[:, :, 0] & 0xF, g[:, :, 1] & 0xF
    return (lo | (hi << 4)).to(torch.uint8).reshape(n, k // 2)


def unpack_int4(packed: torch.Tensor, group: int = INT4_GROUP) -> torch.Tensor:
    """uint8 [N, K // 2] -> int8 [N, K]."""
    n, kh = packed.shape
    half = group // 2
    b = packed.reshape(n, kh // half, half).to(torch.int16)
    lo = ((b & 0xF) ^ 8) - 8
    hi = (((b >> 4) & 0xF) ^ 8) - 8
    return torch.stack((lo, hi), dim=2).reshape(n, kh * 2).to(torch.int8)


def dequantize_weight(w: dict, dtype=torch.float32) -> torch.Tensor:
    """A quantized weight record (see ``QWeight``) back to dense [N, K]."""
    kind = w['kind']
    if kind == 'int8':
        return (w['q'].float() * w['scale'].float()[:, None]).to(dtype)
    if kind == 'int4':
        q = unpack_int4(w['q'], w.get('group', INT4_GROUP)).float()
        n, k = q.shape
        group = w.get('group', INT4_GROUP)
        return (q.reshape(n, k // group, group) * w['scale'].float()[..., None]).reshape(n, k).to(dtype)
    if kind in ('bf16', 'fp16', 'fp32'):
        return w['w'].to(dtype)
    raise ValueError(f'unknown weight kind {kind!r}')


# ----------------------------------------------------------------------------------------------------- matmuls

def matmul_w8a8(xq, xs, wq, ws, bias=None, out_dtype=torch.bfloat16):
    """Exact int8 x int8 product (fp64 accumulation reproduces int32), then both scales."""
    acc = xq.double() @ wq.double().T
    out = acc * xs.double()[:, None] * ws.double()[None, :]
    if bias is not None:
        out = out + bias.double()
    return out.to(out_dtype)


def matmul_w4a8(xq, xs, wpacked, wscale, bias=None, group: int = INT4_GROUP, out_dtype=torch.bfloat16):
    wq = unpack_int4(wpacked, group).double()
    m, k = xq.shape
    n = wq.shape[0]
    xg = xq.double().reshape(m, k // group, group)
    wg = wq.reshape(n, k // group, group)
    acc = torch.einsum('mgk,ngk->mng', xg, wg)  # exact per-group int products
    out = (acc * wscale.double()[None]).sum(-1) * xs.double()[:, None]
    if bias is not None:
        out = out + bias.double()
    return out.to(out_dtype)


def qlinear(xq, xs, w: dict, bias=None, out_dtype=torch.bfloat16):
    if w['kind'] == 'int8':
        return matmul_w8a8(xq, xs, w['q'], w['scale'], bias, out_dtype)
    if w['kind'] == 'int4':
        return matmul_w4a8(xq, xs, w['q'], w['scale'], bias, w.get('group', INT4_GROUP), out_dtype)
    raise ValueError(w['kind'])
