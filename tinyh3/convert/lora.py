"""LakonLab / diffusers adapters: LoRA pairs fused into base weights, plus full-weight overrides.

A pi-H3 adapter file holds two kinds of tensors (``load_lakonlab_adapter``): keys containing ``lora`` are a PEFT
LoRA (loaded with diffusers' ``load_lora_adapter``: rank from the shapes, ``lora_alpha = rank`` unless the file
says otherwise, i.e. scaling 1), everything else overrides or adds a module weight (the DWD head and its norm).
"""
from __future__ import annotations

import json
import re

import torch

LORA_KEY = re.compile(r'^(?:transformer\.)?(?P<module>.+?)\.lora_(?P<which>[AB])(?:\.default)?\.weight$')


class Adapter:
    def __init__(self, reader):
        self.reader = reader
        self.pairs: dict[str, dict] = {}
        self.overrides: dict[str, str] = {}
        alphas = {}
        for key in reader.keys():
            m = LORA_KEY.match(key)
            if m:
                self.pairs.setdefault(m['module'], {})[m['which']] = key
            elif key.endswith('.alpha') and 'lora' in key:
                alphas[key.rsplit('.lora', 1)[0].removeprefix('transformer.')] = key
            elif 'lora' not in key:
                self.overrides[_canonical(key)] = key
        self.alpha_keys = alphas
        meta = {}
        try:
            file_meta = reader.metadata() or {}
            raw = file_meta.get('lora_adapter_metadata')
            meta = json.loads(raw) if raw else {}
            if 'lora_alpha' not in meta and 'alpha' in file_meta:  # lightx2v: {'alpha': '8'} for every module
                meta['lora_alpha'] = float(file_meta['alpha'])
        except Exception:
            meta = {}
        self.meta = meta

    def scaling(self, module: str, rank: int) -> float:
        if module in self.alpha_keys:
            return float(self.reader.get(self.alpha_keys[module]).item()) / rank
        alpha = self.meta.get('lora_alpha')
        r = self.meta.get('r', rank) if 'r' in self.meta else rank
        if alpha is not None:
            alpha_pattern = self.meta.get('alpha_pattern') or {}
            alpha = alpha_pattern.get(module, alpha)
            return float(alpha) / float(r)
        return 1.0

    def delta(self, module: str) -> torch.Tensor | None:
        """``scale * B @ A`` in fp32 for a module path such as ``transformer_blocks.3.attn.to_q``."""
        pair = self.pairs.get(module)
        if not pair:
            return None
        a = self.reader.get(pair['A']).float()
        b = self.reader.get(pair['B']).float()
        return (b @ a) * self.scaling(module, a.shape[0])

    def factors(self, module: str):
        pair = self.pairs.get(module)
        if not pair:
            return None
        return self.reader.get(pair['A']).float(), self.reader.get(pair['B']).float()

    def override(self, key: str) -> torch.Tensor | None:
        src = self.overrides.get(key)
        return None if src is None else self.reader.get(src)

    def keys_with_prefix(self, prefix: str) -> list[str]:
        return sorted(k for k in self.overrides if k.startswith(prefix))


def _canonical(key: str) -> str:
    key = key.removeprefix('transformer.')
    # LakonLab names the pruned AdaLN's folded bias `linear.bias`; the checkpoint calls it `folded_bias`.
    key = re.sub(r'(^|\.)(adaln_proj|norm_out)\.linear\.bias$', r'\1\2.folded_bias', key)
    for old, new in (('pixel_upsampler', 'dwd_pixel_refiner'), ('dwd_pixel_refiner.pixel_proj', 'dwd_pixel_refiner.pixel_embed'),
                     ('first_frame_pixel_proj_out', 'first_frame_dwd_linear_head'), ('pixel_proj_out', 'dwd_linear_head')):
        key = key.replace(old, new)
    return key


def fused(base_weight: torch.Tensor, module: str, adapter: Adapter | None) -> torch.Tensor:
    """fp32 weight with the adapter's override or LoRA applied."""
    w = base_weight.float()
    if adapter is None:
        return w
    over = adapter.override(module + '.weight')
    if over is not None:
        w = over.float()
    d = adapter.delta(module)
    if d is not None:
        w = w + d
    return w


def project_adaln(adapter: Adapter, module: str, basis: torch.Tensor, mean: torch.Tensor, adaln_rank: int):
    """An AdaLN LoRA trained on the released 2688-wide projection, mapped onto the pruned 8 coordinates
    (MiniMaxH3PrunedTransformer3DModel.project_adaln_lora): returns (weight delta [out, 8], bias offset [out])."""
    f = adapter.factors(module)
    if f is None:
        return None, None
    a, b = f
    scale = adapter.scaling(module, a.shape[0])
    if a.shape[1] == adaln_rank:
        return (b @ a) * scale, None
    a64, b64 = a.double(), b.double()
    delta = (b64 @ (a64 @ basis.double().T)) * scale
    offset = (b64 @ (a64 @ mean.double())) * scale
    return delta.float(), offset.float()
