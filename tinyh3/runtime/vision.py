"""Multimodal prefix of the Qwen3-VL conditioner: images / video clips inside the prompt presentation.

The vision tower (~0.4B parameters) and Qwen3-VL's 3-axis position indexing are run with transformers' own modules,
so they stay exact: only the ``visual.*`` weights are materialized (the 32B language model is built on the meta
device and never touched). The language model itself then runs in TinyH3's streamed decoder with

* ``inputs_embeds``: token embeddings with the vision embeddings scattered into the placeholder rows;
* 3-axis M-RoPE positions (interleaved t/h/w frequency sections);
* deepstack: the tower's intermediate features added to the visual rows after the first decoder layers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from ..store import Store, embedding_rows


@dataclass
class Multimodal:
    embeds: torch.Tensor                 # (T, D) input embeddings
    positions: torch.Tensor              # (3, T) M-RoPE positions
    visual_mask: torch.Tensor | None     # (T,) bool
    deepstack: list[torch.Tensor]        # per early layer: (n_visual, D)


def _qwen3vl(store: Store, device):
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig, Qwen3VLModel
    from safetensors.torch import load_file
    cfg = Qwen3VLConfig(**json.loads(store.path('text/config.json').read_text()))
    with init_empty_weights():
        model = Qwen3VLModel(cfg)
    weights = load_file(str(store.path('text/visual.safetensors')))
    model.visual.load_state_dict({k.removeprefix('visual.'): v for k, v in weights.items()}, strict=True,
                                 assign=True)
    model.visual.to(device).eval()
    return model, cfg


def mm_token_type_ids(token_ids: list[int], cfg) -> torch.Tensor:
    ids = torch.tensor(token_ids)
    out = torch.zeros_like(ids)
    out[ids == cfg.image_token_id] = 1
    out[ids == cfg.video_token_id] = 2
    return out


@torch.no_grad()
def prepare(store: Store, token_ids: list[int], vision_inputs: dict, device, dtype) -> Multimodal:
    """Vision embeddings, positions and deepstack features for one tokenized presentation."""
    model, cfg = _qwen3vl(store, device)
    dim = cfg.text_config.hidden_size
    input_ids = torch.tensor([token_ids], device=device)
    types = mm_token_type_ids(token_ids, cfg)[None].to(device)
    embeds = embedding_rows(store.path(store.manifest['text']['embed']['file']), token_ids, dim).to(device, dtype)
    image_grid = vision_inputs.get('image_grid_thw')
    video_grid = vision_inputs.get('video_grid_thw')
    masks, deep = [], []
    for kind, pix_key, grid in (('image', 'pixel_values', image_grid), ('video', 'pixel_values_videos', video_grid)):
        if grid is None:
            continue
        fn = model.get_image_features if kind == 'image' else model.get_video_features
        out = fn(vision_inputs[pix_key].to(device, model.visual.dtype), grid.to(device), return_dict=True)
        feats = torch.cat(out.pooler_output).to(dtype)
        token = cfg.image_token_id if kind == 'image' else cfg.video_token_id
        mask = torch.tensor([t == token for t in token_ids], device=device)
        if int(mask.sum()) != feats.shape[0]:
            raise ValueError(f'{kind}: {int(mask.sum())} placeholder tokens for {feats.shape[0]} features')
        embeds[mask] = feats
        masks.append(mask)
        deep.append([torch.cat(level).to(dtype) for level in out.deepstack_features])
    positions, _ = model.get_rope_index(input_ids, types, image_grid_thw=image_grid if image_grid is None else
                                        image_grid.to(device), video_grid_thw=video_grid if video_grid is None else
                                        video_grid.to(device), attention_mask=torch.ones_like(input_ids))
    visual_mask, deepstack = None, []
    if masks:
        visual_mask = masks[0] if len(masks) == 1 else masks[0] | masks[1]
        if len(masks) == 1:
            deepstack = deep[0]
        else:  # both modalities: interleave per row in sequence order, as transformers does
            for img, vid in zip(*deep):
                merged = img.new_zeros(int(visual_mask.sum()), dim)
                img_rows = masks[0][visual_mask]
                merged[img_rows] = img
                merged[~img_rows] = vid
                deepstack.append(merged)
    del model
    return Multimodal(embeds, positions[:, 0].to(device), visual_mask, deepstack)


def mrope_tables(positions: torch.Tensor, head_dim: int, theta: float, sections, device):
    """cos/sin (T, head_dim) of 3-axis positions with Qwen3-VL's interleaved frequency layout."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    freqs = positions.to(device).float()[..., None] * inv  # (3, T, head_dim / 2)
    out = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        length = sections[axis] * 3
        out[..., offset:length:3] = freqs[axis, ..., offset:length:3]
    emb = torch.cat((out, out), dim=-1)
    return emb.cos(), emb.sin()
