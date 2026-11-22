"""Attention backends over (seq, heads, head_dim) tensors of one sequence (no batch, no mask)."""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
_SAGE = None


def _sage():
    global _SAGE
    if _SAGE is None:
        try:
            from sageattention import sageattn
            _SAGE = sageattn
        except Exception:
            _SAGE = False
    return _SAGE or None


def available() -> list[str]:
    names = ['sdpa']
    if torch.cuda.is_available() and _sage() is not None:
        names.insert(0, 'sage')
    return names


def resolve(name: str) -> str:
    if name == 'auto':
        return available()[0]
    if name == 'sage' and _sage() is None:
        log.warning('SageAttention is not installed; falling back to SDPA')
        return 'sdpa'
    return name


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str = 'sdpa') -> torch.Tensor:
    """q, k, v: (L, H, D) -> (L, H, D)."""
    if backend == 'sage' and q.is_cuda:
        out = _sage()(q[None], k[None], v[None], tensor_layout='NHD', is_causal=False)
        return out[0]
    qh, kh, vh = (t.transpose(0, 1)[None] for t in (q, k, v))
    if not q.is_cuda and q.dtype == torch.bfloat16:
        qh, kh, vh = qh.float(), kh.float(), vh.float()
    out = F.scaled_dot_product_attention(qh, kh, vh)
    return out[0].transpose(0, 1).to(q.dtype)
