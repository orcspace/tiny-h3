"""Read Comfy-Org's prebuilt MiniMax-H3 files as if they were diffusers checkpoints.

``Comfy-Org/MiniMax-H3`` ships the pruned FL2VA DiT (21 GB) and the Qwen3-VL text tower (27 GB) as INT8 ConvRot:
int8 rows of ``W @ blockdiag(H256)`` with one fp32 scale per row, i.e. exactly TinyH3's W8A8 format. Starting from
them halves the download (~52 GB instead of ~95 GB of bf16) and skips most of the quantization error budget:
re-quantizing an untouched layer reproduces the same int8 values.

The views below expose diffusers key names (so the adapter's LoRA and overrides apply unchanged) and return fp32
weights in the *unrotated* basis; the converter rotates and quantizes again. Key mapping follows ComfyUI-piFlow's
``minimax_h3_to_diffusers`` (q/k/v packed into ``qkv_proj``, ``mlp.fc1`` stores the SwiGLU halves swapped).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..kernels import reference as ref

COMFY_REPO = 'Comfy-Org/MiniMax-H3'
COMFY_DIT = 'diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors'
COMFY_TEXT = 'text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors'


class _File:
    def __init__(self, path: Path):
        from safetensors import safe_open
        self.path = Path(path)
        self.h = safe_open(str(self.path), 'pt')
        self.names = set(self.h.keys())

    def get(self, key):
        return self.h.get_tensor(key)


def _swap_halves(t):
    a, b = t.chunk(2, dim=0)
    return torch.cat([b, a])


class _QuantView:
    """Shared machinery: dequantize + unrotate ConvRot int8 layers on demand (one-entry cache for fused qkv)."""

    def __init__(self, path: Path):
        self.f = _File(path)
        self.map: dict = {}
        self._cache = (None, None)

    def _layer(self, base: str) -> torch.Tensor:
        if self._cache[0] == base:
            return self._cache[1]
        w = self.f.get(base + '.weight')
        if base + '.weight_scale' in self.f.names:
            group = 256
            if base + '.comfy_quant' in self.f.names:
                cfg = json.loads(bytes(self.f.get(base + '.comfy_quant').tolist()).decode())
                if not cfg.get('convrot', False):
                    group = 0
                else:
                    group = int(cfg.get('convrot_groupsize', 256))
            w = w.float() * self.f.get(base + '.weight_scale').float().reshape(-1, 1)
            if group:
                if group != ref.ROT_GROUP:
                    raise ValueError(f'{base}: ConvRot group {group} is not supported')
                w = ref.rotate(w)  # H is its own inverse: back to the unrotated basis
        else:
            w = w.float()
        self._cache = (base, w)
        return w

    def keys(self):
        return self.map.keys()

    def __contains__(self, key):
        return key in self.map

    def metadata(self):
        return {}

    def get(self, key: str) -> torch.Tensor:
        target = self.map[key]
        if isinstance(target, str):
            if target.endswith('.weight') and target[:-7] + '.weight_scale' in self.f.names:
                return self._layer(target[:-7])
            return self.f.get(target)
        base, op = target
        w = self._layer(base)
        if op == 'swap':
            return _swap_halves(w)
        start, size = op
        return w[start:start + size]


class ComfyDiTView(_QuantView):
    def __init__(self, path: Path, num_layers=50, num_refiner_layers=2, inner_dim=56 * 128):
        super().__init__(path)
        m = self.map

        def block(src, dst, adaln):
            m[f'{src}.attn.to_q.weight'] = (f'{dst}.attn.qkv_proj', (0, inner_dim))
            m[f'{src}.attn.to_k.weight'] = (f'{dst}.attn.qkv_proj', (inner_dim, inner_dim))
            m[f'{src}.attn.to_v.weight'] = (f'{dst}.attn.qkv_proj', (2 * inner_dim, inner_dim))
            m[f'{src}.ff.net.0.proj.weight'] = (f'{dst}.mlp.fc1', 'swap')
            for a, b in (('norm1.weight', 'norm1.weight'), ('norm2.weight', 'norm2.weight'),
                         ('attn.norm_q.weight', 'attn.q_norm.weight'), ('attn.norm_k.weight', 'attn.k_norm.weight'),
                         ('attn.to_out.0.weight', 'attn.out_proj.weight'), ('ff.net.2.weight', 'mlp.fc2.weight')):
                m[f'{src}.{a}'] = f'{dst}.{b}'
            if adaln:
                m[f'{src}.adaln_proj.linear.weight'] = f'{dst}.adaln_proj.linear.weight'
                m[f'{src}.adaln_proj.folded_bias'] = f'{dst}.adaln_proj.linear.bias'

        for i in range(num_refiner_layers):
            block(f'token_refiner.refiner_blocks.{i}', f'token_refiner.blocks.{i}', False)
        for i in range(num_layers):
            block(f'transformer_blocks.{i}', f'blocks.{i}', True)
        for comfy_key, key in (
                ('video_patch_proj.weight', 'proj_in.weight'), ('video_patch_proj.bias', 'proj_in.bias'),
                ('audio_patch_proj.weight', 'audio_proj_in.weight'), ('audio_patch_proj.bias', 'audio_proj_in.bias'),
                ('condition_proj.weight', 'context_embedder.weight'), ('condition_proj.bias', 'context_embedder.bias'),
                ('adaln_t_table', 'time_embedder.table'),
                ('token_refiner.final_norm.weight', 'token_refiner.final_norm.weight'),
                ('final_layer.norm.weight', 'norm_out.norm.weight'),
                ('final_layer.adaln_proj.linear.weight', 'norm_out.linear.weight'),
                ('final_layer.adaln_proj.linear.bias', 'norm_out.folded_bias'),
                ('final_layer.video_out.weight', 'proj_out.weight'), ('final_layer.video_out.bias', 'proj_out.bias'),
                ('final_layer.audio_out.weight', 'audio_proj_out.weight'),
                ('final_layer.audio_out.bias', 'audio_proj_out.bias')):
            m[key] = comfy_key
        missing = [v if isinstance(v, str) else v[0] + '.weight' for v in m.values()]
        missing = [k for k in missing if k not in self.f.names]
        if missing:
            raise KeyError(f'{path}: not a MiniMax-H3 pruned ComfyUI checkpoint (missing {missing[:3]} …)')


class ComfyTextView(_QuantView):
    """Qwen3-VL text tower: diffusers/transformers names ``model.language_model.X`` -> Comfy ``model.X``."""

    def __init__(self, path: Path):
        super().__init__(path)
        for name in self.f.names:
            if name.startswith('model.') and not name.startswith('model.visual') and \
                    not name.endswith(('.weight_scale', '.comfy_quant')):
                self.map['model.language_model.' + name[len('model.'):]] = name

    def rotated(self, key: str) -> bool:
        target = self.map[key]
        return isinstance(target, str) and target[:-7] + '.weight_scale' in self.f.names


def download(target: Path, sampler: str = 'balanced', token=None) -> dict:
    """~52 GB: Comfy-Org int8 DiT + text tower, the pi-H3 adapter, the audio VAE and tokenizer."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from ..config import SAMPLERS, SOURCES
    target = Path(target)
    dit = hf_hub_download(COMFY_REPO, COMFY_DIT, local_dir=target / 'comfy', token=token)
    text = hf_hub_download(COMFY_REPO, COMFY_TEXT, local_dir=target / 'comfy', token=token)
    preset = SAMPLERS[sampler]
    adapter = snapshot_download(SOURCES['adapter_repo'], local_dir=target / 'adapter', token=token,
                                allow_patterns=[f'{preset.adapter}/*'])
    official = snapshot_download(SOURCES['official_repo'], local_dir=target / 'official', token=token,
                                 allow_patterns=['audio_vae/*', 'tokenizer/*'])
    return {'dit': dit, 'text': text, 'adapter': str(Path(adapter) / preset.adapter), 'official': official}


def download_ref(target: Path, token=None) -> dict:
    """Ref2VA: Comfy-Org's INT8 ConvRot Ref2VA transformer (21 GB) and lightx2v's Ref2VA turbo LoRA (1.4 GB)."""
    from huggingface_hub import hf_hub_download
    from ..config import REF_ADAPTER_FILE, REF_ADAPTER_REPO
    target = Path(target)
    dit = hf_hub_download(COMFY_REPO, 'diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors',
                          local_dir=target / 'comfy', token=token)
    lora = hf_hub_download(REF_ADAPTER_REPO, REF_ADAPTER_FILE, local_dir=target / 'ref_adapter', token=token)
    return {'dit': dit, 'adapter': lora}
