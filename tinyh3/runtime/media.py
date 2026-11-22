"""Audio decoding (MiniMax-H3 audio VAE) and MP4 muxing (H.264 + AAC via PyAV)."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch


def decode_audio(store, audio_latents: torch.Tensor, device) -> torch.Tensor:
    """(2, 32, N) normalized latents -> (2, N * 800) float waveform in [-1, 1] on CPU."""
    from diffusers import AutoencoderKLMiniMaxH3Audio
    path = store.path('audio_vae')
    vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(path), torch_dtype=torch.float32).to(device).eval()
    cfg = vae.config
    n = audio_latents.shape[1]
    mean = torch.tensor(cfg.latents_mean if cfg.latents_mean is not None else [0.0] * n, device=device).view(1, -1, 1)
    std = torch.tensor(cfg.latents_std if cfg.latents_std is not None else [1.0] * n, device=device).view(1, -1, 1)
    with torch.no_grad():
        wave = vae.decode(audio_latents.to(device).float() * std + mean, return_dict=False)[0]
    del vae
    return wave[:, 0].float().cpu()


def write_mp4(path, frames: torch.Tensor, audio: torch.Tensor | None, fps: int, sample_rate: int, *, crf: int = 18,
              metadata: str | None = None) -> None:
    """frames (F, H, W, 3) uint8, audio (channels, samples) float in [-1, 1]."""
    import av
    path = Path(path)
    tmp = path.with_suffix('.part.mp4')
    container = av.open(str(tmp), mode='w')
    if metadata:
        container.metadata['comment'] = metadata
    vs = container.add_stream('libx264', rate=fps)
    vs.width, vs.height = int(frames.shape[2]), int(frames.shape[1])
    vs.pix_fmt = 'yuv420p'
    vs.options = {'crf': str(crf), 'preset': 'medium', 'movflags': '+faststart'}
    stream_a = None
    if audio is not None:
        stream_a = container.add_stream('aac', rate=sample_rate)
        stream_a.layout = 'stereo' if audio.shape[0] == 2 else 'mono'
        stream_a.bit_rate = 192_000
    arr = frames.numpy()
    for i in range(arr.shape[0]):
        frame = av.VideoFrame.from_ndarray(arr[i], format='rgb24')
        for packet in vs.encode(frame):
            container.mux(packet)
    for packet in vs.encode():
        container.mux(packet)
    if stream_a is not None:
        pcm = audio.clamp(-1, 1).numpy().astype(np.float32)
        chunk = 1024
        pts = 0
        for s0 in range(0, pcm.shape[1], chunk):
            block = np.ascontiguousarray(pcm[:, s0:s0 + chunk])
            frame = av.AudioFrame.from_ndarray(block, format='fltp', layout=stream_a.layout.name)
            frame.sample_rate = sample_rate
            frame.pts = pts
            frame.time_base = Fraction(1, sample_rate)
            pts += block.shape[1]
            for packet in stream_a.encode(frame):
                container.mux(packet)
        for packet in stream_a.encode():
            container.mux(packet)
    container.close()
    tmp.replace(path)


def read_metadata(path) -> dict | None:
    import av
    with av.open(str(path)) as c:
        raw = c.metadata.get('comment')
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None
