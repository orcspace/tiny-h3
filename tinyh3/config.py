"""Architecture constants, generation presets and paths.

Every number here is taken from the released configs (``multimodalart/MiniMax-H3-Pruned`` transformer and
text_encoder, ``Lakonik/pi-H3`` adapters, diffusers' ``MiniMaxH3ModularPipeline``). Keep them in one place: the
planner, the converter and the runtime all derive their shapes from this module.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True)
class DiTConfig:
    hidden_size: int = 5376
    num_heads: int = 56
    head_dim: int = 128
    ffn_dim: int = 14336
    num_layers: int = 50
    num_refiner_layers: int = 2
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple = (1, 2, 2)
    text_dim: int = 5120
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    adaln_rank: int = 8
    time_table_size: int = 1025
    modality_num: int = 3  # 0 video, 1 text, 2 audio

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def video_patch_dim(self) -> int:
        return self.in_channels * math.prod(self.patch_size)

    @property
    def rotary_dim(self) -> int:
        return 2 * 3 * self.rope_freq_dim

    def linear_shapes(self) -> dict:
        """(out_features, in_features) of the four quantized GEMMs of one block (QKV fused)."""
        d, inner, ffn = self.hidden_size, self.inner_dim, self.ffn_dim
        return {
            'qkv': (3 * inner, d),
            'out': (d, inner),
            'ff_in': (2 * ffn, d),
            'ff_out': (d, ffn),
        }

    def block_params(self) -> int:
        return sum(o * i for o, i in self.linear_shapes().values())


@dataclass(frozen=True)
class TextEncoderConfig:
    """Qwen3-VL-32B text tower as MiniMax-H3 uses it: ``hidden_states[50]``, i.e. the first 50 decoder layers."""
    hidden_size: int = 5120
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 25600
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    num_layers_used: int = 50

    def linear_shapes(self) -> dict:
        d, q, kv, ffn = self.hidden_size, self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim, \
            self.intermediate_size
        return {'qkv': (q + 2 * kv, d), 'o': (d, q), 'gate_up': (2 * ffn, d), 'down': (d, ffn)}

    def layer_params(self) -> int:
        return sum(o * i for o, i in self.linear_shapes().values())


@dataclass(frozen=True)
class SolConfig:
    """The Sol-H3 decoder layers the DWD head reuses (pi-H3 ``SolH3DWDHead``)."""
    num_heads: int = 32
    head_dim: int = 64
    ffn_mult: int = 4
    num_layers: int = 8
    total_layers: int = 24  # layer index offset: the head holds the last 8 of 24
    norm_eps: float = 1e-5
    num_register_tokens: int = 4
    rope_theta: float = 100.0
    window: int = 16
    block: int = 5
    t_shift: int = 2
    time_norm: int = 5
    patch: int = 16
    patch_t: int = 4
    out_ch: int = 3
    frames_to_trim: int = 3

    @property
    def dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * 0.75)


# Pipeline constants (diffusers MiniMaxH3ModularPipeline)
FPS = 24
VAE_SPATIAL = 16
VAE_FRAMES_PER_CHUNK = 17
VAE_LATENTS_PER_CHUNK = 5
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
AUDIO_SAMPLE_RATE = 32000
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
CANVAS_MULTIPLE = 32
MIN_SECONDS, MAX_SECONDS = 5.0, 15.0
TEXT_ENCODER_LAYER = 50

# The DWD head's RGB codec (pi-H3 app: RGBColorEncoder(use_affine_norm=True, ...))
DWD_RGB_MEAN = (-0.03, -0.088, -0.188)
DWD_RGB_STD = (0.458, 0.448, 0.45)


@dataclass(frozen=True)
class SamplerPreset:
    name: str
    steps: int
    shift: float
    audio_shift: float
    max_raw_t: float
    final_step_size_scale: float
    adapter: str


SAMPLERS = {
    'turbo': SamplerPreset('turbo', 2, 12.0, 3.0, 0.93, 0.6, 'pi_h3_2step_dwd_adapter'),
    'balanced': SamplerPreset('balanced', 4, 12.0, 3.0, 0.96, 0.6, 'pi_h3_4step_dwd_adapter'),
}

# Resolution presets as (height, width) for 16:9; other aspect ratios keep the pixel count.
RESOLUTIONS = {
    '360p': 352 * 640,
    '384p': 384 * 672,
    '448p': 448 * 800,
    '480p': 480 * 864,
    '512p': 512 * 896,
    '576p': 576 * 1024,
    '640p': 640 * 1120,
    '720p': 704 * 1248,
    '768p': 768 * 1344,
}
ASPECT_RATIOS = ('21:9', '16:9', '4:3', '1:1', '3:4', '9:16')

SOURCES = {
    'base_repo': 'multimodalart/MiniMax-H3-Pruned',
    'adapter_repo': 'Lakonik/pi-H3',
    'official_repo': 'MiniMaxAI/MiniMax-H3',
}

DIT = DiTConfig()
TEXT = TextEncoderConfig()
SOL = SolConfig()


def canvas_size(aspect_ratio: str, num_pixels: float, multiple: int = CANVAS_MULTIPLE) -> tuple[int, int]:
    """(height, width) for an aspect ratio at roughly ``num_pixels``, both multiples of ``multiple``."""
    aw, ah = map(float, aspect_ratio.split(':'))
    height = (num_pixels * ah / aw) ** 0.5
    width = (num_pixels * aw / ah) ** 0.5
    return (max(multiple, round(height / multiple) * multiple),
            max(multiple, round(width / multiple) * multiple))


def align_num_frames(num_frames: int) -> int:
    """Round up to the 17 n + 5 frame counts the video VAE geometry allows."""
    while num_frames % VAE_FRAMES_PER_CHUNK != VAE_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def frames_for_seconds(seconds: float) -> int:
    """The nearest 17 n + 5 frame count, n in [7, 20]: 124 frames (5.2 s) to 345 frames (14.4 s)."""
    n = round((seconds * FPS - VAE_LATENTS_PER_CHUNK) / VAE_FRAMES_PER_CHUNK)
    return min(max(n, 7), 20) * VAE_FRAMES_PER_CHUNK + VAE_LATENTS_PER_CHUNK


def latent_frames(num_frames: int) -> int:
    if num_frames % VAE_FRAMES_PER_CHUNK != VAE_LATENTS_PER_CHUNK:
        raise ValueError(f'{num_frames} frames is not of the form 17n+5')
    return (num_frames - VAE_LATENTS_PER_CHUNK) // VAE_FRAMES_PER_CHUNK * VAE_LATENTS_PER_CHUNK + 2


def audio_latents(num_frames: int) -> int:
    return int(round(num_frames / FPS * AUDIO_LATENTS_PER_SECOND))


@dataclass(frozen=True)
class Geometry:
    """Token geometry of one request."""
    height: int
    width: int
    num_frames: int
    text_tokens: int = 320

    @property
    def latent_t(self) -> int:
        return latent_frames(self.num_frames)

    @property
    def latent_h(self) -> int:
        return self.height // VAE_SPATIAL

    @property
    def latent_w(self) -> int:
        return self.width // VAE_SPATIAL

    @property
    def tokens_per_frame(self) -> int:
        return (self.latent_h // DIT.patch_size[1]) * (self.latent_w // DIT.patch_size[2])

    @property
    def video_rows(self) -> int:
        return self.latent_t * self.tokens_per_frame

    @property
    def audio_rows(self) -> int:
        return audio_latents(self.num_frames) * AUDIO_CHANNELS

    @property
    def sequence_length(self) -> int:
        return self.text_tokens + self.audio_rows + self.video_rows

    @property
    def seconds(self) -> float:
        return self.num_frames / FPS

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(latent_t=self.latent_t, latent_h=self.latent_h, latent_w=self.latent_w,
                 video_rows=self.video_rows, audio_rows=self.audio_rows, sequence_length=self.sequence_length,
                 seconds=round(self.seconds, 3))
        return d


def home() -> Path:
    return Path(os.environ.get('TINYH3_HOME', Path.home() / '.tinyh3'))


def models_dir() -> Path:
    return Path(os.environ.get('TINYH3_MODELS', home() / 'models'))


def outputs_dir() -> Path:
    return Path(os.environ.get('TINYH3_OUTPUTS', home() / 'outputs'))


def cache_dir() -> Path:
    return Path(os.environ.get('TINYH3_CACHE', home() / 'cache'))
