"""Latents -> RGB frames with the MiniMax-H3 video VAE decoder (used by Ref2VA, which has no DWD head).

The decoder is a 36-layer ViT (2.4B parameters, 4.8 GB in fp16). On cards with less than ~12 GB free it is streamed
block by block with diffusers' group offloading on a side stream, so only one block plus the tile activations live in
VRAM; the decode itself is diffusers' tiled ``AutoencoderKLMiniMaxH3.decode`` under fp16 autocast, as in the released
pipeline.
"""
from __future__ import annotations

import json

import torch

from ..config import VAE_SPATIAL
from ..store import Store

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def load_decoder(store: Store, device):
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLMiniMaxH3
    from safetensors.torch import load_file
    cfg = json.loads(store.path('vae/config.json').read_text())
    with init_empty_weights():
        vae = AutoencoderKLMiniMaxH3.from_config(cfg)
    vae.load_state_dict(load_file(str(store.path('vae/decoder.safetensors'))), strict=False, assign=True)
    vae.decoder.to(torch.float16)
    vae.post_quant_conv.to(torch.float16)
    dev = torch.device(device)
    if dev.type == 'cuda':
        free, _ = torch.cuda.mem_get_info()
        if free < 12 << 30:
            from diffusers.hooks import apply_group_offloading
            vae.post_quant_conv.to(dev)
            apply_group_offloading(vae.decoder, onload_device=dev, offload_device=torch.device('cpu'),
                                   offload_type='block_level', num_blocks_per_group=1, use_stream=True)
        else:
            vae.decoder.to(dev)
            vae.post_quant_conv.to(dev)
    else:
        vae.decoder.float()
        vae.post_quant_conv.float()
    return vae.eval(), cfg


@torch.no_grad()
def decode(store: Store, latents: torch.Tensor, device) -> torch.Tensor:
    """(1, C, T, h, w) normalized latents -> (F, H, W, 3) uint8 on CPU."""
    vae, cfg = load_decoder(store, device)
    dev = torch.device(device)
    mean = torch.tensor(cfg['latents_mean'], device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(cfg['latents_std'], device=dev).view(1, -1, 1, 1, 1)
    z = latents.to(dev).float() * std + mean
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
        video = vae.decode(z.to(next(vae.decoder.parameters()).dtype), return_dict=False)[0]
    pm = torch.tensor(PIXEL_MEAN, device=dev).view(1, -1, 1, 1, 1)
    ps = torch.tensor(PIXEL_STD, device=dev).view(1, -1, 1, 1, 1)
    video = (video.float() * ps + pm).clamp(0, 1)
    frames = (video[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu()
    del vae
    return frames
