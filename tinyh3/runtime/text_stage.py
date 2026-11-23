"""Text stage: prompt -> refined text rows for the DiT, cached on disk.

1. Tokenize the prompt verbatim (no chat template, no special tokens), as MiniMax-H3 does for t2va.
2. Gather only the needed rows of the embedding table from ``text/embed.bin``.
3. Run Qwen3 decoder layers 0..49 streamed from the store (weight-only int8/int4), keep ``hidden_states[50]``.
   For text-only input Qwen3-VL's interleaved M-RoPE has equal t/h/w positions, i.e. plain 1-D RoPE.
4. Apply the DiT's ``context_embedder`` and token refiner once (they do not depend on the diffusion step).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import kernels as K
from ..config import TextEncoderConfig, TEXT, TEXT_TAG, cache_dir
from ..kernels import reference as ref
from ..store import Store, embedding_rows, linear_records
from .dit import plain_block, config_from_manifest

BF16 = torch.bfloat16


def tokenizer_for(store: Store):
    path = store.path('tokenizer/tokenizer.json')
    if path.exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(path))
        return lambda text: tok.encode(text, add_special_tokens=False).ids
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(store.path('tokenizer')))
    return lambda text: tok(text, add_special_tokens=False)['input_ids']


def text_config(store: Store) -> TextEncoderConfig:
    raw = store.manifest.get('text', {}).get('config')
    if not raw:
        return TEXT
    raw = dict(raw)
    if 'mrope_section' in raw:
        raw['mrope_section'] = tuple(raw['mrope_section'])
    return TextEncoderConfig(**raw)


def rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w.to(x.dtype)


def rope_1d(T: int, dim: int, theta: float, device):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    freqs = torch.arange(T, dtype=torch.float32, device=device)[:, None] * inv[None]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """HF rotate_half convention over the full head dim; x (T, H, D)."""
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos[:, None, :].to(x.dtype) + rot * sin[:, None, :].to(x.dtype))


def qwen3_layer(h, w, cfg: TextEncoderConfig, rope):
    T = h.shape[0]
    lin = w['linears']
    x = rms(h, w['input_norm'], cfg.rms_norm_eps)
    qkv = K.dense_linear(x, lin['qkv'])
    nq, nkv, d = cfg.num_heads * cfg.head_dim, cfg.num_kv_heads * cfg.head_dim, cfg.head_dim
    q = qkv[:, :nq].view(T, cfg.num_heads, d)
    k = qkv[:, nq:nq + nkv].view(T, cfg.num_kv_heads, d)
    v = qkv[:, nq + nkv:].view(T, cfg.num_kv_heads, d)
    q = apply_rope(rms(q, w['q_norm'], cfg.rms_norm_eps), *rope)
    k = apply_rope(rms(k, w['k_norm'], cfg.rms_norm_eps), *rope)
    rep = cfg.num_heads // cfg.num_kv_heads
    qh = q.transpose(0, 1)[None]
    kh = k.repeat_interleave(rep, dim=1).transpose(0, 1)[None]
    vh = v.repeat_interleave(rep, dim=1).transpose(0, 1)[None]
    if not h.is_cuda:
        qh, kh, vh = qh.float(), kh.float(), vh.float()
    a = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)[0].transpose(0, 1).reshape(T, nq).to(h.dtype)
    h = h + K.dense_linear(a, lin['o'])
    x = rms(h, w['post_norm'], cfg.rms_norm_eps)
    gu = K.dense_linear(x, lin['gate_up'])
    gate, up = gu.chunk(2, dim=-1)
    h = h + K.dense_linear(F.silu(gate) * up, lin['down'])
    return h


def _weights(tensors, unit):
    w = {k: v for k, v in tensors.items() if not k.startswith('__') and '.' not in k}
    w['linears'] = linear_records(tensors, unit)
    return w


def encode_tokens(store: Store, token_ids: list[int], device, *, tiers=None, progress=None,
                  dtype=None, vision_inputs: dict | None = None) -> torch.Tensor:
    """hidden_states[num_layers_used] of the Qwen3 text tower, (T, hidden) in ``dtype``. With ``vision_inputs``
    (processor outputs for the presentation's image / video blocks) the Qwen3-VL multimodal prefix is applied:
    vision embeddings, 3-axis M-RoPE and deepstack."""
    from .streamer import UnitStreamer
    cfg = text_config(store)
    dev = torch.device(device)
    dtype = dtype or (BF16 if dev.type == 'cuda' else torch.float32)
    mm = None
    if vision_inputs:
        from .vision import prepare, mrope_tables
        mm = prepare(store, token_ids, vision_inputs, dev, dtype)
        h = mm.embeds
        rope = mrope_tables(mm.positions, cfg.head_dim, cfg.rope_theta, cfg.mrope_section, dev)
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        emb = embedding_rows(store.path(store.manifest['text']['embed']['file']), token_ids, cfg.hidden_size)
        h = emb.to(dev, dtype)
        rope = rope_1d(len(token_ids), cfg.head_dim, cfg.rope_theta, dev)
    n = len(store.units('text'))
    streamer = UnitStreamer(store, 'text', device=dev, tiers=tiers or {i: 'disk' for i in range(n)}, slots=2,
                            passes=1)
    streamer.prime()
    units = store.units('text')
    try:
        for i in range(n):
            tensors = streamer.acquire(i)
            h = qwen3_layer(h, _weights(tensors, units[i]), cfg, rope)
            streamer.release(i, tensors)
            # transformers records a layer's output before adding deepstack features, so the last layer we keep
            # never gets them (H3 reads layer 50; deepstack touches layers 0-2)
            if mm is not None and mm.visual_mask is not None and i < len(mm.deepstack) and i < n - 1:
                h[mm.visual_mask] = h[mm.visual_mask] + mm.deepstack[i].to(h.dtype)
            if progress:
                progress(i + 1, n)
    finally:
        streamer.close()
    return h


def refine(store: Store, text_hidden: torch.Tensor, device, group: str = 'refiner',
           dit_group: str = 'dit') -> torch.Tensor:
    """context_embedder + token refiner + final norm -> (T, dit hidden) bf16. ``group``: the refiner of the
    transformer that will consume the rows ('refiner' for FL2VA, 'refiner_ref' for Ref2VA)."""
    cfg = config_from_manifest(store.manifest[dit_group].get('config'))
    dev = torch.device(device)
    dtype = BF16 if dev.type == 'cuda' else torch.float32
    u0 = store.load_unit(group, 0, dev)
    rec = linear_records(u0['tensors'], u0['unit'])
    x = K.dense_linear(text_hidden.to(dev, dtype), rec['ctx'], u0['tensors']['ctx_bias'])
    x = x.to(BF16) if dev.type == 'cuda' else x
    units = store.units(group)
    for i in range(1, len(units)):
        u = store.load_unit(group, i, dev)
        plain_block(x, _weights(u['tensors'], u['unit']), cfg=cfg, head_group=cfg.num_heads)
    return ref.rms_norm(x, u0['tensors']['final_norm'], cfg.final_norm_eps).to(BF16)


def prompt_key(store: Store, token_ids: list[int], extra: str = '') -> str:
    h = hashlib.sha256()
    h.update(json.dumps({'store': str(store.root.resolve()), 'created': store.manifest.get('created'),
                         'kind': store.manifest.get('text', {}).get('kind'), 'extra': extra}).encode())
    h.update(json.dumps(token_ids).encode())
    return h.hexdigest()[:32]


def text_stage(store: Store, prompt: str, device='cuda', *, tiers=None, progress=None, use_cache=True, cond=None,
               refiner: str = 'refiner', dit_group: str = 'dit'):
    """Returns ``(text_states (T, D) bf16 on CPU, token_tags (T,), cached: bool)``. ``cond``
    (``conditioning.Conditioning``) supplies the presentation of image / reference prompts."""
    if cond is None:
        token_ids = tokenizer_for(store)(prompt)
        tags = torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long)
        vision, extra = None, ''
    else:
        token_ids, tags = cond.token_ids, torch.as_tensor(cond.token_tags, dtype=torch.long)
        vision, extra = cond.vision_inputs or None, cond.media_digest
    if not token_ids:
        raise ValueError('пустой промпт')
    key = prompt_key(store, token_ids, extra + refiner)
    path = cache_dir() / 'prompts' / f'{key}.safetensors'
    if use_cache and path.exists():
        from safetensors.torch import load_file
        return load_file(str(path))['text_states'], tags, True
    hidden = encode_tokens(store, token_ids, device, tiers=tiers, progress=progress, vision_inputs=vision)
    states = refine(store, hidden, device, group=refiner, dit_group=dit_group).cpu()
    if use_cache:
        from safetensors.torch import save_file
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({'text_states': states.contiguous()}, str(path))
    return states, tags, False
