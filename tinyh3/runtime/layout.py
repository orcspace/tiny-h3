"""The packed ``[text | keyframe conditions | audio | video]`` sequence of one request.

A port of diffusers' ``MiniMaxH3PrepareLayoutStep.build_packed_sequence`` (t2va / first-frame) that also produces
everything the executor needs per request: rotary cos/sin, AdaLN row indices per step, and the row ranges of each
modality. ``tests/test_layout.py`` checks it against diffusers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import DIT, AUDIO_CHANNELS, VIDEO_TAG, AUDIO_TAG, Geometry, audio_latents

ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
ROPE_SPATIAL_SCALE = 32
KEYFRAME_NOISE_AUG = 0.999


def _spatial_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return torch.from_numpy(np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE)


def _temporal_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor([ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)]
                          for i in range(num_latent_frames)], dtype=torch.float64)
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _frame_grid(latent_h: int, latent_w: int, patch_h: int, patch_w: int):
    sqrt_area = np.sqrt(latent_h * latent_w)
    hg = _spatial_grid(latent_h, patch_h, sqrt_area)
    wg = _spatial_grid(latent_w, patch_w, sqrt_area)
    grids = torch.meshgrid(hg, wg, indexing='ij')
    return torch.stack([g.reshape(-1) for g in grids], dim=-1), wg


def build_packed_sequence(text_token_tags: torch.Tensor, num_latent_frames: int, latent_h: int, latent_w: int,
                          num_audio_latents: int, patch_size=DIT.patch_size, audio_channels=AUDIO_CHANNELS,
                          keyframe_anchors: tuple = ()):
    _, ph, pw = patch_size
    rows_per_frame = (latent_h // ph) * (latent_w // pw)
    n_text = text_token_tags.shape[0]
    n_cond = len(keyframe_anchors) * rows_per_frame
    n_audio = num_audio_latents * audio_channels
    n_video = num_latent_frames * rows_per_frame
    seq = n_text + n_cond + n_audio + n_video
    cond_start, audio_start = n_text, n_text + n_cond
    video_start = audio_start + n_audio

    pos = torch.zeros(seq, 3, dtype=torch.float64)
    pos[:n_text, 0] = torch.arange(n_text, dtype=torch.float64)
    frame_grid, width_grid = _frame_grid(latent_h, latent_w, ph, pw)
    for index, anchor in enumerate(keyframe_anchors):
        if anchor == 'first':
            t0 = float(n_text)
        elif anchor == 'last':
            spans = np.ones(num_latent_frames, dtype=np.float64) * ROPE_FRAME_RESCALE
            for off in range(len(ROPE_FRAMES_PER_LATENT)):
                spans[off::len(ROPE_FRAMES_PER_LATENT)] *= ROPE_FRAMES_PER_LATENT[off]
            t0 = float(n_text) + float(spans.sum()) - ROPE_FRAME_RESCALE
        else:
            raise ValueError(anchor)
        rows = slice(cond_start + index * rows_per_frame, cond_start + (index + 1) * rows_per_frame)
        pos[rows, 0] = t0
        pos[rows, 1:] = frame_grid

    audio_time = float(n_text) + torch.arange(num_audio_latents, dtype=torch.float64)
    pos[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    pos[audio_start:video_start, 2] = torch.cat([
        torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
        torch.full((n_audio - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64)])

    vpos = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    vpos[:, :, 0] = _temporal_grid(num_latent_frames, float(n_text))[:, None]
    vpos[:, :, 1:] = frame_grid[None]
    pos[video_start:] = vpos.reshape(-1, 3)

    video_idx = torch.cat([torch.arange(cond_start, audio_start), torch.arange(video_start, seq)])
    audio_idx = torch.arange(audio_start, video_start)
    text_idx = torch.arange(n_text)
    tags = torch.empty(seq, dtype=torch.long)
    tags[text_idx] = text_token_tags.long()
    tags[audio_idx] = AUDIO_TAG
    tags[video_idx] = VIDEO_TAG
    return pos, tags, video_idx, audio_idx, text_idx, n_cond


def rotary(position_ids: torch.Tensor, rope_freq_dim=DIT.rope_freq_dim, theta=DIT.rope_theta):
    """cos/sin (seq, 2 * 3 * rope_freq_dim) in fp32, as diffusers' MiniMaxH3RotaryPosEmbed."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)))
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    ft, fh, fw = freqs.unbind(dim=1)
    freqs = torch.cat((ft, fh, fw), dim=-1)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def row_timesteps(video_idx, audio_idx, n_cond_video: int, n_text: int, video_t: float, audio_t: float,
                  cond_t: float, n_cond_audio: int = 0, cond_audio_t: float = 1.0):
    """(distinct timesteps sorted, per-row index into them) — diffusers ``build_row_timesteps``. Reference audio rows
    (Ref2VA soundtracks) are clean: t = 1."""
    seq = int(video_idx.numel() + audio_idx.numel() + n_text)
    rt = torch.full((seq,), video_t, dtype=torch.float32)
    rt[video_idx[:n_cond_video]] = cond_t
    rt[audio_idx[n_cond_audio:]] = audio_t
    rt[audio_idx[:n_cond_audio]] = cond_audio_t
    return torch.unique(rt, sorted=True, return_inverse=True)


@dataclass
class Layout:
    geometry: Geometry
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_idx: torch.Tensor
    audio_idx: torch.Tensor
    text_idx: torch.Tensor
    n_cond: int
    n_cond_audio: int = 0

    @property
    def seq(self) -> int:
        return self.position_ids.shape[0]

    @classmethod
    def build(cls, geometry: Geometry, text_token_tags: torch.Tensor, keyframe_anchors: tuple = ()):
        pos, tags, v, a, t, n_cond = build_packed_sequence(
            text_token_tags, geometry.latent_t, geometry.latent_h, geometry.latent_w,
            audio_latents(geometry.num_frames), keyframe_anchors=keyframe_anchors)
        return cls(geometry, pos, tags, v, a, t, n_cond)

    @classmethod
    def from_packed(cls, geometry: Geometry, packed: tuple):
        """Wrap diffusers' ``build_*_packed_sequence`` output (position_ids, token_tags, video_indices, audio_indices,
        text_indices, n_cond_video, n_cond_audio), e.g. the Ref2VA layout."""
        pos, tags, v, a, t, n_cond, n_cond_audio = packed
        return cls(geometry, pos, tags, v, a, t, int(n_cond), int(n_cond_audio))

    def contiguous_ranges(self):
        """The layout puts text, conditions, audio, then video rows in one run each — return them as slices."""
        n_text = self.text_idx.numel()
        n_audio = self.audio_idx.numel()
        return {
            'text': slice(0, n_text),
            'cond': slice(n_text, n_text + self.n_cond),
            'audio': slice(n_text + self.n_cond, n_text + self.n_cond + n_audio),
            'video': slice(n_text + self.n_cond + n_audio, self.seq),
        }
