"""Kernel dispatch: Triton on CUDA when available, otherwise PyTorch fallbacks with the same semantics.

``TINYH3_KERNELS=torch`` forces the fallbacks (useful to bisect a numerical problem on a new GPU).
"""
from __future__ import annotations

import os
import logging

import torch

from . import reference as ref
from .reference import ROT_GROUP, INT4_GROUP  # noqa: F401

log = logging.getLogger(__name__)

_TRITON = None


def get_triton():
    """The Triton module, or None when Triton is unusable here."""
    global _TRITON
    if _TRITON is None:
        mode = os.environ.get('TINYH3_KERNELS', 'auto')
        if mode == 'torch':
            _TRITON = False
        else:
            try:
                from . import triton_ops as t
                _TRITON = t
            except Exception as exc:  # pragma: no cover - depends on the install
                log.warning('Triton kernels unavailable (%s); using PyTorch fallbacks', exc)
                _TRITON = False
    return _TRITON or None


def _use_triton(t: torch.Tensor) -> bool:
    if not t.is_cuda:
        return os.environ.get('TRITON_INTERPRET') == '1' and get_triton() is not None
    return get_triton() is not None


# ---------------------------------------------------------------------------------------------- activation prep

def rotate_quantize(x):
    if _use_triton(x):
        return get_triton().rotate_quantize(x)
    return ref.rotate_quantize(x)


def norm_modulate_rotate_quantize(x, norm_weight, eps, scale_table, shift_table, rows):
    if _use_triton(x):
        return get_triton().norm_modulate_rotate_quantize(x, norm_weight, eps, scale_table, shift_table, rows)
    return ref.norm_modulate_rotate_quantize(x, norm_weight, eps, scale_table, shift_table, rows)


def swiglu_rotate_quantize(h, gate_first=False):
    if _use_triton(h) and not gate_first:
        return get_triton().swiglu_rotate_quantize(h)
    return ref.swiglu_rotate_quantize(h, gate_first)


# ------------------------------------------------------------------------------------------------------- GEMMs

def _int_mm_ok(m, n, k) -> bool:
    return m > 16 and k % 8 == 0 and n % 8 == 0


def qlinear(xq, xs, w, bias=None, out_dtype=torch.bfloat16, out=None):
    """int8 rows ``xq`` (with row scales ``xs``) times a quantized weight record ``w`` (see ``runtime.weights``)."""
    kind = w['kind']
    if xq.is_cuda or _use_triton(xq):
        t = get_triton() if _use_triton(xq) else None
        if t is not None:
            if kind == 'int8':
                return t.matmul_w8a8(xq, xs, w['q'], w['scale'], bias, out_dtype, out)
            if kind == 'int4':
                return t.matmul_w4a8(xq, xs, w['q'], w['scale'], bias, w.get('group', INT4_GROUP), out_dtype, out)
        if kind == 'int8' and _int_mm_ok(xq.shape[0], w['q'].shape[0], xq.shape[1]):
            acc = torch._int_mm(xq, w['q'].t())
            res = acc.float().mul_(xs[:, None]).mul_(w['scale'][None, :])
            if bias is not None:
                res += bias.float()
            res = res.to(out_dtype)
        else:
            dense = ref.dequantize_weight(w, torch.bfloat16)
            x = (xq.to(torch.bfloat16) * xs[:, None].to(torch.bfloat16))
            res = torch.nn.functional.linear(x, dense, None if bias is None else bias.to(torch.bfloat16))
            res = res.to(out_dtype)
        if out is not None:
            out.copy_(res)
            return out
        return res
    res = ref.qlinear(xq, xs, w, bias, out_dtype)
    if out is not None:
        out.copy_(res)
        return out
    return res


def dense_linear(x, w, bias=None):
    """Weight-only path (text encoder, small layers): dequantize to the activation dtype and matmul. Weights kept in
    the ConvRot basis (``rotated``) get the same rotation on the activations: (x H)(W H)^T = x W^T."""
    if w.get('rotated'):
        x = ref.rotate(x).to(x.dtype)
    if w['kind'] in ('bf16', 'fp16', 'fp32'):
        weight = w['w'].to(x.dtype)
    else:
        weight = ref.dequantize_weight(w, x.dtype if x.is_cuda else torch.float32).to(x.dtype)
    return torch.nn.functional.linear(x, weight, None if bias is None else bias.to(x.dtype))


def selftest() -> dict:
    """Run the fused kernels on a small CUDA tensor and compare with the reference: tells a freshly installed
    Triton (e.g. triton-windows) that does not match the installed PyTorch from one that works."""
    out = {'triton_import': get_triton() is not None, 'triton_ok': False, 'int_mm_ok': False, 'error': None}
    if not torch.cuda.is_available():
        out['error'] = 'CUDA недоступна'
        return out
    try:
        x = torch.randn(40, 512, device='cuda')
        q, s = rotate_quantize(x)
        q0, s0 = ref.rotate_quantize(x.cpu())
        out['triton_ok'] = bool(get_triton() is not None and (q.cpu().int() - q0.int()).abs().max() <= 1
                                and torch.allclose(s.cpu(), s0, rtol=1e-4))
        a = torch.randint(-5, 5, (32, 64), device='cuda', dtype=torch.int8)
        b = torch.randint(-5, 5, (64, 32), device='cuda', dtype=torch.int8)
        out['int_mm_ok'] = bool(torch.equal(torch._int_mm(a, b).cpu(), (a.cpu().int() @ b.cpu().int())))
    except Exception as exc:  # a broken Triton / driver shows up here, not in the middle of a generation
        out['error'] = f'{type(exc).__name__}: {exc}'[:300]
    return out
