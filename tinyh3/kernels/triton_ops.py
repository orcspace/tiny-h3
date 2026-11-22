"""Triton kernels for Ampere-and-newer GPUs (tuned for SM86, RTX 30 series).

Semantics are defined by ``reference.py``; ``tests/test_kernels.py`` runs these under ``TRITON_INTERPRET=1`` on CPU
and compares. Three families:

* ``prep``: one pass over activation rows that optionally applies RMSNorm + AdaLN modulation (mode 1) or the SwiGLU
  product (mode 2), rotates every 256-wide group by H256 and writes int8 rows plus one fp32 scale per row. This
  replaces four to six full-tensor PyTorch passes (norm, mul, add, matmul by H, abs-max, divide, round, cast).
* ``w8a8``: int8 x int8 -> int32 on tensor cores, epilogue applies row and channel scales and the bias.
* ``w4a8``: int4 weights (two per byte, see reference.py) unpacked in registers to int8 and multiplied on the same
  int8 tensor cores; group scales are applied to the int32 partial sums of each 128-wide K group.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# The CPU interpreter has no driver to benchmark autotune candidates on; it runs one fixed configuration.
_INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'


def _autotune(configs, key):
    if _INTERPRET:
        return lambda fn: fn
    return triton.autotune(configs=configs, key=key)

from .reference import ROT_GROUP, INT4_GROUP, hadamard

_H16: dict = {}


def _h16(device) -> torch.Tensor:
    key = str(device)
    if key not in _H16:
        _H16[key] = hadamard(16, device, torch.float32).contiguous()
    return _H16[key]


@triton.jit
def _rotate256(x, h16, BM: tl.constexpr):
    # Channel c = 16 a + b of a 256-wide group: H256 = H16 (x) H16, so y = H16 X H16 over (a, b).
    z = tl.dot(tl.reshape(x, (BM * 16, 16)), h16, input_precision='ieee')
    z = tl.permute(tl.reshape(z, (BM, 16, 16)), (0, 2, 1))
    z = tl.dot(tl.reshape(z, (BM * 16, 16)), h16, input_precision='ieee')
    return tl.reshape(tl.permute(tl.reshape(z, (BM, 16, 16)), (0, 2, 1)), (BM, 256))


@triton.jit
def _rint(v):
    # Round half to even for |v| < 2**22: adding 1.5 * 2**23 forces the fp32 sum onto the integer grid with IEEE
    # rounding, which is exactly rint. Portable across backends (no libdevice), and Triton never reassociates it.
    magic = 12582912.0
    return (v + magic) - magic


@triton.jit
def _load_group(X, rows, live, group, SX, lane, MODE: tl.constexpr, K: tl.constexpr,
                NW, SC, SH, IDX, rstd):
    cols = group * 256 + lane
    if MODE == 2:
        # SwiGLU over [up | gate]: value = up * silu(gate)
        up = tl.load(X + rows[:, None] * SX + cols[None, :], live[:, None], other=0.).to(tl.float32)
        gate = tl.load(X + rows[:, None] * SX + K + cols[None, :], live[:, None], other=0.).to(tl.float32)
        x = up * gate * tl.sigmoid(gate)
    else:
        x = tl.load(X + rows[:, None] * SX + cols[None, :], live[:, None], other=0.).to(tl.float32)
        if MODE == 1:
            w = tl.load(NW + cols).to(tl.float32)
            r = tl.load(IDX + rows, live, other=0)
            sc = tl.load(SC + r[:, None] * K + cols[None, :], live[:, None], other=0.).to(tl.float32)
            sh = tl.load(SH + r[:, None] * K + cols[None, :], live[:, None], other=0.).to(tl.float32)
            x = x * rstd[:, None] * w[None, :] * (1. + sc) + sh
    return x


@triton.jit
def _prep_kernel(X, Q, S, H, M, SX, NW, SC, SH, IDX, eps,
                 K: tl.constexpr, MODE: tl.constexpr, BM: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    live = rows < M
    lane = tl.arange(0, 256)
    pair = tl.arange(0, 16)
    h16 = tl.load(H + pair[:, None] * 16 + pair[None, :])
    rstd = tl.zeros((BM,), tl.float32)
    if MODE == 1:
        acc = tl.zeros((BM,), tl.float32)
        for group in range(K // 256):
            x = tl.load(X + rows[:, None] * SX + group * 256 + lane[None, :], live[:, None], other=0.)
            x = x.to(tl.float32)
            acc += tl.sum(x * x, 1)
        rstd = 1. / tl.sqrt(acc / K + eps)
    largest = tl.zeros((BM,), tl.float32)
    for group in range(K // 256):
        x = _load_group(X, rows, live, group, SX, lane, MODE, K, NW, SC, SH, IDX, rstd)
        largest = tl.maximum(largest, tl.max(tl.abs(_rotate256(x, h16, BM)), 1))
    scale = tl.maximum(largest, 1e-12) / 127.
    tl.store(S + rows, scale, live)
    for group in range(K // 256):
        x = _load_group(X, rows, live, group, SX, lane, MODE, K, NW, SC, SH, IDX, rstd)
        v = _rint(_rotate256(x, h16, BM) / scale[:, None])
        v = tl.minimum(tl.maximum(v, -127.), 127.)
        tl.store(Q + rows[:, None] * K + group * 256 + lane[None, :], v.to(tl.int8), live[:, None])


def _prep(x, mode, k, norm_weight=None, eps=0., scale_table=None, shift_table=None, rows=None):
    x2 = x.reshape(-1, x.shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    if k % ROT_GROUP:
        raise ValueError(f'{k} channels is not a multiple of {ROT_GROUP}')
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    s = torch.empty(m, device=x.device, dtype=torch.float32)
    dummy = s
    bm = 16
    _prep_kernel[(triton.cdiv(m, bm),)](
        x2, q, s, _h16(x.device), m, x2.stride(0),
        norm_weight if norm_weight is not None else dummy,
        scale_table.contiguous() if scale_table is not None else dummy,
        shift_table.contiguous() if shift_table is not None else dummy,
        rows if rows is not None else dummy, float(eps),
        K=k, MODE=mode, BM=bm, num_warps=4)
    return q, s


def rotate_quantize(x):
    return _prep(x, 0, x.shape[-1])


def norm_modulate_rotate_quantize(x, norm_weight, eps, scale_table, shift_table, rows):
    return _prep(x, 1, x.shape[-1], norm_weight, eps, scale_table, shift_table, rows)


def swiglu_rotate_quantize(h, gate_first=False):
    if gate_first:
        raise NotImplementedError('the fused SwiGLU prologue expects the H3 [up, gate] layout')
    return _prep(h, 2, h.shape[-1] // 2)


# ------------------------------------------------------------------------------------------------------ GEMMs

def _mm_configs():
    return [
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 128}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 128}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 128}, num_warps=8, num_stages=2),
        triton.Config({'BM': 64, 'BN': 64, 'BK': 128}, num_warps=4, num_stages=4),
    ]


@triton.jit
def _tile_ids(M, N, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
    # Grouped ordering keeps a band of activation tiles hot in L2 while it sweeps the output columns.
    pid = tl.program_id(0)
    blocks_m = tl.cdiv(M, BM)
    blocks_n = tl.cdiv(N, BN)
    width = GROUP_M * blocks_n
    group = pid // width
    first_m = group * GROUP_M
    size_m = tl.minimum(blocks_m - first_m, GROUP_M)
    pid_m = first_m + (pid % width) % size_m
    pid_n = (pid % width) // size_m
    return pid_m, pid_n


@_autotune(_mm_configs(), ['N', 'K'])
@triton.jit
def _w8a8_kernel(A, W, SA, SW, B, O, M, N, K, SAM, SWN, SOM,
                 HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = _tile_ids(M, N, BM, BN, 8)
    rows = pid_m.to(tl.int64) * BM + tl.arange(0, BM)
    cols = pid_n.to(tl.int64) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for k0 in range(0, K, BK):
        kk = k0 + inner
        a = tl.load(A + rows[:, None] * SAM + kk[None, :], (rows[:, None] < M) & (kk[None, :] < K), other=0)
        w = tl.load(W + cols[None, :] * SWN + kk[:, None], (cols[None, :] < N) & (kk[:, None] < K), other=0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    sa = tl.load(SA + rows, rows < M, other=0.)
    sw = tl.load(SW + cols, cols < N, other=0.)
    out = acc.to(tl.float32) * sa[:, None] * sw[None, :]
    if HAS_BIAS:
        out += tl.load(B + cols, cols < N, other=0.).to(tl.float32)[None, :]
    tl.store(O + rows[:, None] * SOM + cols[None, :], out.to(O.dtype.element_ty),
             (rows[:, None] < M) & (cols[None, :] < N))


def _w4_configs():
    return [
        triton.Config({'BM': 128, 'BN': 128}, num_warps=8, num_stages=3),
        triton.Config({'BM': 128, 'BN': 64}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64, 'BN': 128}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64, 'BN': 64}, num_warps=4, num_stages=4),
    ]


@_autotune(_w4_configs(), ['N', 'K'])
@triton.jit
def _w4a8_kernel(A, W, SA, SW, B, O, M, N, K, SAM, SWN, SOM, SSN,
                 HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr):
    pid_m, pid_n = _tile_ids(M, N, BM, BN, 8)
    rows = pid_m.to(tl.int64) * BM + tl.arange(0, BM)
    cols = pid_n.to(tl.int64) * BN + tl.arange(0, BN)
    half = tl.arange(0, G // 2)
    acc = tl.zeros((BM, BN), tl.float32)
    for g in range(0, K // G):
        k0 = g * G
        rmask = rows[:, None] < M
        a_lo = tl.load(A + rows[:, None] * SAM + (k0 + half)[None, :], rmask, other=0)
        a_hi = tl.load(A + rows[:, None] * SAM + (k0 + G // 2 + half)[None, :], rmask, other=0)
        packed = tl.load(W + cols[None, :] * SWN + (g * (G // 2) + half)[:, None], cols[None, :] < N, other=0)
        packed = packed.to(tl.int8, bitcast=True)
        w_lo = (packed << 4) >> 4
        w_hi = packed >> 4
        part = tl.dot(a_lo, w_lo, out_dtype=tl.int32)
        part = tl.dot(a_hi, w_hi, part, out_dtype=tl.int32)
        sw = tl.load(SW + cols * SSN + g, cols < N, other=0.).to(tl.float32)
        acc += part.to(tl.float32) * sw[None, :]
    sa = tl.load(SA + rows, rows < M, other=0.)
    out = acc * sa[:, None]
    if HAS_BIAS:
        out += tl.load(B + cols, cols < N, other=0.).to(tl.float32)[None, :]
    tl.store(O + rows[:, None] * SOM + cols[None, :], out.to(O.dtype.element_ty),
             (rows[:, None] < M) & (cols[None, :] < N))


def _grid(meta, m, n):
    return (triton.cdiv(m, meta['BM']) * triton.cdiv(n, meta['BN']),)


def matmul_w8a8(xq, xs, wq, ws, bias=None, out_dtype=torch.bfloat16, out=None):
    m, k = xq.shape
    n = wq.shape[0]
    o = out if out is not None else torch.empty((m, n), device=xq.device, dtype=out_dtype)
    _w8a8_kernel[lambda meta: _grid(meta, m, n)](
        xq, wq, xs, ws, bias if bias is not None else xs, o, m, n, k,
        xq.stride(0), wq.stride(0), o.stride(0), HAS_BIAS=bias is not None,
        **(dict(BM=32, BN=32, BK=64) if _INTERPRET else {}))
    return o


def matmul_w4a8(xq, xs, wpacked, wscale, bias=None, group=INT4_GROUP, out_dtype=torch.bfloat16, out=None):
    m, k = xq.shape
    n = wpacked.shape[0]
    o = out if out is not None else torch.empty((m, n), device=xq.device, dtype=out_dtype)
    _w4a8_kernel[lambda meta: _grid(meta, m, n)](
        xq, wpacked, xs, wscale, bias if bias is not None else xs, o, m, n, k,
        xq.stride(0), wpacked.stride(0), o.stride(0), wscale.stride(0),
        HAS_BIAS=bias is not None, G=group, **(dict(BM=32, BN=32) if _INTERPRET else {}))
    return o
