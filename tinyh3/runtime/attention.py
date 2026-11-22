"""Attention backends over (seq, heads, head_dim) tensors of one packed sequence (no batch, no mask).

* ``spark``  – Spark-Attn block-sparse attention for MiniMax-H3 (https://github.com/zechengtang/Spark-H3, Apache-2.0):
  similar tokens are reblocked, the top ~10% block pairs are exact, the tail is reweighted. Only the target-video
  grid is sparse; text/audio/condition rows are exact K/V sinks. SM80/86/89/90/120 kernels. Requires all heads of
  a layer at once and keeps per-layer state across steps.
* ``sage16`` – SageAttention 2 with int8 QK and FP16 PV accumulation: the GeForce FP32-accumulate cap does not apply.
* ``sage``   – SageAttention, its default kernel for the GPU.
* ``sdpa``   – PyTorch scaled_dot_product_attention (FlashAttention 2 on CUDA).
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
_SAGE = None


def _sage():
    global _SAGE
    if _SAGE is None:
        try:
            import sageattention as sa
            _SAGE = sa
        except Exception:
            _SAGE = False
    return _SAGE or None


def spark_available() -> bool:
    try:
        import h3_sparse_attention  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.get_device_capability() in {
            (8, 0), (8, 6), (8, 9), (9, 0), (12, 0)}
    except Exception:
        return False


def available() -> list[str]:
    names = []
    if spark_available():
        names.append('spark')
    if torch.cuda.is_available() and _sage() is not None:
        names += ['sage16', 'sage']
    return names + ['sdpa']


def resolve(name: str, gpu_unlocked: bool | None = None) -> str:
    """Pick the dense backend for ``auto``; on a capped GeForce prefer FP16 PV accumulation."""
    have = available()
    if name == 'auto':
        dense = [n for n in have if n != 'spark']
        if gpu_unlocked and 'sage' in dense:
            return 'sage'
        return dense[0]
    if name not in have:
        fallback = [n for n in have if n != 'spark'][0]
        log.warning('attention backend %s unavailable; using %s', name, fallback)
        return fallback
    return name


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str = 'sdpa') -> torch.Tensor:
    """Dense attention: q, k, v (L, H, D) -> (L, H, D)."""
    if q.is_cuda and backend in ('sage', 'sage16') and _sage() is not None:
        sa = _sage()
        if backend == 'sage16' and hasattr(sa, 'sageattn_qk_int8_pv_fp16_cuda'):
            out = sa.sageattn_qk_int8_pv_fp16_cuda(q[None], k[None], v[None], tensor_layout='NHD', is_causal=False,
                                                   pv_accum_dtype='fp16+fp32')
        else:
            out = sa.sageattn(q[None], k[None], v[None], tensor_layout='NHD', is_causal=False)
        return out[0]
    qh, kh, vh = (t.transpose(0, 1)[None] for t in (q, k, v))
    if not q.is_cuda and q.dtype == torch.bfloat16:
        qh, kh, vh = qh.float(), kh.float(), vh.float()
    out = F.scaled_dot_product_attention(qh, kh, vh)
    return out[0].transpose(0, 1).to(q.dtype)


class SparkAttention:
    """Spark-Attn driven directly from the TinyH3 executor (no diffusers processors).

    ``mode='speed'``: every step sparse (pi-H3's recommended setting); ``'fidelity'``: the first 20% of steps dense.
    """

    def __init__(self, num_steps: int, mode: str = 'speed', topk_ratio: float = 0.1, dense_layers: int = 0,
                 dense_fallback: str = 'sdpa'):
        from h3_sparse_attention.processor import H3SparseAttentionConfig, _Controller
        warmup = 0.0 if mode == 'speed' else 0.2
        cfg = H3SparseAttentionConfig.spark(None, num_denoise_steps=num_steps, mode='full',
                                            warmup_mode='warmup_ratio', warmup_ratio=warmup,
                                            sol_dense_layers=dense_layers, sol_route_topk_ratio=topk_ratio)
        self.config = cfg
        self.controller = _Controller(cfg)
        self.dense_fallback = dense_fallback
        self.dense_calls = 0
        self.sparse_calls = 0

    def begin_step(self, req, t_index: torch.Tensor):
        from h3_sparse_attention.processor import _packed_layout
        c = self.controller
        c.evaluation_index += 1
        c.layout = _packed_layout(req.tags, req.layout.position_ids.to(req.tags.device),
                                  video_indices=req.video_idx, timestep_indices=t_index.to(req.tags.device),
                                  text_indices=req.text_idx, sparse_video_scope=self.config.sol_sparse_video_scope)

    def is_dense(self, layer: int, seq: int) -> bool:
        c, cfg = self.controller, self.config
        return (c.is_warmup or c.evaluation_index in cfg.sol_extra_dense_evaluations or layer < cfg.sol_dense_layers
                or layer in cfg.sol_extra_dense_layers or seq < cfg.sol_min_tokens)

    def permutation(self):
        layout = self.controller.layout
        return layout.permutation, layout.inverse_permutation

    def __call__(self, q, k, v, layer: int, permuted: bool = False) -> torch.Tensor:
        """q, k, v: (L, H, D) for *all* heads of ``layer``; with ``permuted`` they are already in Spark order and the
        output stays in that order."""
        if self.is_dense(layer, q.shape[0]):
            self.dense_calls += 1
            return attention(q, k, v, self.dense_fallback)
        from h3_sparse_attention.processor import _sol_attention
        from h3_sparse_attention.spark_integration import _sm80_low_memory_reblock
        layout = self.controller.layout
        if permuted:
            pq, pk, pv = q[None], k[None], v[None]
        else:
            perm = layout.permutation
            pq = q.index_select(0, perm)[None]
            pk = k.index_select(0, perm)[None]
            pv = v.index_select(0, perm)[None]
        native = self.config.sol_landmark_preprocess and _sm80_low_memory_reblock(pq)
        if native:
            out = _sol_attention(self.controller, pq, pk, pv, layout, layer, return_bthd=True, inputs_bthd=True)
        else:
            out = _sol_attention(self.controller, *(t.permute(0, 2, 1, 3) for t in (pq, pk, pv)), layout, layer,
                                 return_bthd=True, inputs_bthd=False)
        self.sparse_calls += 1
        out = out[0]
        return (out if permuted else out.index_select(0, layout.inverse_permutation)).to(q.dtype)

    def summary(self) -> dict:
        return {'sparse_calls': self.sparse_calls, 'dense_calls': self.dense_calls,
                'density': self.controller.sol_route_density}
