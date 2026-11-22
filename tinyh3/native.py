"""Bindings to the Rust core (``native/tinyh3-core``) with validated buffers and PyTorch fallbacks.

Build it once with ``pip install ./native/tinyh3-core`` (needs a Rust toolchain) or install a prebuilt wheel;
everything keeps working without it, just slower on the CPU-side paths (disk reads, 2K upscaling).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import tinyh3_core as _core
except ImportError:  # pragma: no cover - depends on the install
    _core = None


def available() -> bool:
    return _core is not None and os.environ.get('TINYH3_NATIVE', '1') != '0'


def version() -> str | None:
    return _core.version() if _core is not None else None


def _check_buffer(buf: torch.Tensor, nbytes: int):
    if buf.device.type != 'cpu' or buf.dtype != torch.uint8 or not buf.is_contiguous():
        raise ValueError('destination must be a contiguous CPU uint8 tensor')
    if buf.numel() < nbytes:
        raise ValueError(f'destination holds {buf.numel()} bytes, {nbytes} requested')


def read_into(path, offset: int, buf: torch.Tensor, nbytes: int, threads: int = 4) -> bool:
    """Fill ``buf[:nbytes]`` from ``path[offset:]`` with parallel positional reads. False when not available."""
    if not available():
        return False
    _check_buffer(buf, nbytes)
    _core.read_into(str(path), int(offset), buf.data_ptr(), int(nbytes), int(threads))
    return True


def upscale_frames(frames: torch.Tensor, out_h: int, out_w: int, sharpen: float = 0.4) -> torch.Tensor:
    """(F, H, W, 3) uint8 -> (F, out_h, out_w, 3) uint8: Lanczos-3 + contrast-adaptive sharpening."""
    if frames.dtype != torch.uint8 or frames.dim() != 4 or frames.shape[-1] != 3:
        raise ValueError('frames must be (F, H, W, 3) uint8')
    src = frames.contiguous().cpu()
    n, h, w, _ = src.shape
    if available():
        out = torch.empty((n, out_h, out_w, 3), dtype=torch.uint8)
        _core.upscale_rgb(src.data_ptr(), n, h, w, out.data_ptr(), out_h, out_w, float(sharpen), 0)
        return out
    # fallback: bicubic with antialiasing, frame chunks to bound memory, mild unsharp mask
    outs = []
    for f0 in range(0, n, 16):
        x = src[f0:f0 + 16].permute(0, 3, 1, 2).float()
        y = F.interpolate(x, size=(out_h, out_w), mode='bicubic', align_corners=False, antialias=True)
        if sharpen > 0:
            blur = F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
            y = y + sharpen * (y - blur)
        outs.append(y.clamp_(0, 255).round_().to(torch.uint8).permute(0, 2, 3, 1))
    return torch.cat(outs)
