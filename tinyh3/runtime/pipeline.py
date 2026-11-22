"""Generation pipeline: text -> denoise -> DWD + audio -> MP4.

Each stage is a plain function that can run in-process or in its own subprocess (``worker.run_stage``). Running
stages in fresh processes is the default: the text tower's 24 GB of weights and the DiT's pinned blocks never share
the 16 GB of RAM, and every stage starts from an empty CUDA allocator.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

from ..config import (SAMPLERS, SAMPLER_ALIASES, UPSCALE, Geometry, canvas_size, frames_for_seconds, RESOLUTIONS,
                      FPS, AUDIO_CHANNELS, AUDIO_SAMPLE_RATE, SolConfig, outputs_dir, cache_dir, render_size,
                      output_size)
from ..store import Store


@dataclass
class GenRequest:
    prompt: str
    seconds: float = 5.0
    resolution: str = '480p'
    aspect: str = '16:9'
    sampler: str = 'fast'           # quality | balance | fast | fast-high (see config.SAMPLERS)
    seed: int = -1
    attention: str = 'auto'         # auto | spark | sage16 | sage | sdpa
    spark_mode: str | None = None   # override the mode's Spark setting: speed | fidelity
    upscale: str = 'none'           # none | 1080p | 2k: native Lanczos + CAS upscale of the output
    height: int | None = None
    width: int | None = None
    crf: int = 18
    model: str | None = None        # store directory; default: the first converted model
    output: str | None = None

    def resolve(self) -> 'GenRequest':
        r = GenRequest(**asdict(self))
        if r.seed is None or r.seed < 0:
            r.seed = int.from_bytes(os.urandom(4), 'little') & 0x7FFFFFFF
        if not (r.height and r.width):
            pixels = RESOLUTIONS.get(r.resolution)
            if pixels is None:
                raise ValueError(f'неизвестное разрешение {r.resolution!r}')
            r.height, r.width = canvas_size(r.aspect, pixels)
        r.sampler = SAMPLER_ALIASES.get(r.sampler, r.sampler)
        if r.sampler not in SAMPLERS:
            raise ValueError(f'режим должен быть одним из {list(SAMPLERS)}')
        if r.upscale not in UPSCALE:
            raise ValueError(f'upscale должен быть одним из {list(UPSCALE)}')
        return r

    @property
    def preset(self):
        return SAMPLERS[SAMPLER_ALIASES.get(self.sampler, self.sampler)]

    @property
    def num_frames(self) -> int:
        return frames_for_seconds(self.seconds)



def default_model(adapter_steps: int | None = None) -> Path | None:
    """The first converted model, preferring one built with the adapter for ``adapter_steps`` steps."""
    from ..config import models_dir
    root = models_dir()
    if not root.exists():
        return None
    stores = sorted(p.parent for p in root.glob('*/manifest.json'))
    if adapter_steps is not None:
        for st in stores:
            try:
                if json.loads((st / 'manifest.json').read_text()).get('recipe', {}).get('steps') == adapter_steps:
                    return st
            except (OSError, ValueError):
                continue
    return stores[0] if stores else None


# ---------------------------------------------------------------------------------------------------- stages

def stage_text(model: str, prompt: str, emit) -> dict:
    from .text_stage import text_stage
    store = Store(model)
    # Text layers are read exactly once per prompt, so pinning them first would only add a copy: every layer goes
    # through the disk reader, which keeps two layers ahead of the GPU (the OS page cache keeps re-used files warm).
    tiers = {i: 'disk' for i in range(len(store.units('text')))}
    t0 = time.time()
    states, tags, cached = text_stage(
        store, prompt, 'cuda' if torch.cuda.is_available() else 'cpu', tiers=tiers,
        progress=lambda i, total: emit(stage='text', step=i, total=total))
    path = cache_dir() / 'runs' / f'text-{os.getpid()}-{int(t0 * 1000)}.safetensors'
    path.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({'text_states': states.contiguous(), 'tags': tags}, str(path))
    return {'text': str(path), 'cached': cached, 'seconds': round(time.time() - t0, 2)}


def patchify(latents: torch.Tensor, patch=(1, 2, 2)) -> torch.Tensor:
    """(1, C, T, H, W) -> rows (T/pt * H/ph * W/pw, C * pt * ph * pw), frame-major then row-major."""
    pt, ph, pw = patch
    b, c, t, h, w = latents.shape
    x = latents.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw).permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * pt * ph * pw).contiguous()


def unpatchify(rows: torch.Tensor, t: int, h: int, w: int, channels: int, patch=(1, 2, 2)) -> torch.Tensor:
    """Inverse of ``patchify``: rows -> (1, C, T, H, W)."""
    pt, ph, pw = patch
    x = rows.reshape(1, t // pt, h // ph, w // pw, channels, pt, ph, pw).permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(1, channels, t, h, w)


def upsample_latents(x0: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(1, C, T, h0, w0) -> (1, C, T, h, w), bicubic per frame."""
    _, c, t, h0, w0 = x0.shape
    frames = x0[0].permute(1, 0, 2, 3).float()  # (T, C, h0, w0)
    up = torch.nn.functional.interpolate(frames, size=(h, w), mode='bicubic', align_corners=False)
    return up.permute(1, 0, 2, 3)[None]


def stage_denoise(model: str, req: dict, text_path: str, emit) -> dict:
    from safetensors.torch import load_file, save_file
    from ..hardware import probe
    from ..planner import make_plan, draft_geometry
    from .dit import DiT, Request
    from .layout import Layout, row_timesteps, KEYFRAME_NOISE_AUG
    from .scheduler import FlowEuler
    from .attention import SparkAttention, resolve

    r = GenRequest(**req)
    preset = r.preset
    store = Store(model)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text = load_file(text_path)
    text_states, tags = text['text_states'], text['tags']
    run_h, run_w = render_size(r.height, r.width, preset.final_scale)
    geom = Geometry(run_h, run_w, r.num_frames, text_tokens=text_states.shape[0])
    hw = probe()
    units = store.units('dit')
    kinds = store.manifest['dit'].get('kinds') or ['int8'] * len(units)
    plan = make_plan(hw, geom, sizes=[u['nbytes'] for u in units], kinds=kinds, steps=preset.steps,
                     attention=r.attention, mode=preset)
    if not plan.fits:
        raise MemoryError(plan.reason)
    emit(stage='plan', plan=plan.as_dict(), message=plan.summary())

    draft = preset.draft_scale > 0
    full_steps = preset.steps - (1 if draft else 0)
    spark = None
    dense = resolve('auto' if plan.attention == 'spark' else plan.attention, hw.unlocked)
    if plan.attention == 'spark':
        spark = SparkAttention(full_steps, mode=r.spark_mode or preset.spark_mode, topk_ratio=preset.topk,
                               dense_fallback=dense)
    t_load = time.time()
    dit = DiT(store, device=dev, tiers={i: plan.tier(i) for i in range(len(units))}, slots=plan.slots,
              head_group=plan.head_group, ffn_chunk=plan.ffn_chunk, attention_backend=dense, spark=None,
              progress=lambda i, n: emit(stage='load', step=i, total=n))
    load_s = time.time() - t_load
    cfg = dit.cfg

    layout = Layout.build(geom, tags)
    request = Request(layout, dev, cfg)
    gen = torch.Generator().manual_seed(int(r.seed))
    noise = torch.randn((1, cfg.in_channels, geom.latent_t, geom.latent_h, geom.latent_w), generator=gen)
    audio = torch.randn((layout.audio_idx.numel(), cfg.audio_in_channels), generator=gen).to(dev)
    noise_rows = patchify(noise, cfg.patch_size).to(dev)
    vs = FlowEuler(preset.steps, preset.shift, preset.max_raw_t, preset.final_step_size_scale)
    aus = FlowEuler(preset.steps, preset.audio_shift, preset.max_raw_t, preset.final_step_size_scale)
    n_text = layout.text_idx.numel()
    total_blocks = preset.steps * dit.num_layers
    step_times, features = [], None
    t_run = time.time()

    def progress(i, b):
        done = i * dit.num_layers + b + 1
        elapsed = time.time() - t_run
        emit(stage='denoise', step=done, total=total_blocks, substep=i + 1, steps=preset.steps,
             eta=round(elapsed / done * (total_blocks - done), 1))

    def run_step(i, req_i, lay_i, video_rows, features_wanted):
        tv, ta = vs.model_t(i), aus.model_t(i)
        t_unique, t_index = row_timesteps(lay_i.video_idx, lay_i.audio_idx, lay_i.n_cond, n_text, tv, ta,
                                          max(tv, KEYFRAME_NOISE_AUG))
        out_v, out_a, feats = dit.forward(req_i, video_rows, audio, text_states, t_unique, t_index,
                                          want_features=features_wanted, on_block=lambda b, _dt: progress(i, b))
        return -out_v.float(), -out_a.float(), feats

    video = noise_rows
    first = 0
    if draft:
        # Two-pass: step 0 on a smaller canvas (1/4 of the tokens at scale 0.5). Its clean estimate x0 is
        # upsampled and re-noised to the next noise level with the full-resolution noise; at sigma ~0.87 the
        # transformer re-draws all detail, so the upsampling blur does not survive.
        t0 = time.time()
        dgeom = draft_geometry(geom, preset.draft_scale)
        dlayout = Layout.build(dgeom, tags)
        dreq = Request(dlayout, dev, cfg)
        dnoise = torch.randn((1, cfg.in_channels, dgeom.latent_t, dgeom.latent_h, dgeom.latent_w), generator=gen)
        drows = patchify(dnoise, cfg.patch_size).to(dev)
        vel, avel, _ = run_step(0, dreq, dlayout, drows, False)
        sigma0, sigma1 = float(vs.sigmas[0]), float(vs.sigmas[1])
        x0 = drows.float() - sigma0 * vel
        x0 = upsample_latents(unpatchify(x0, dgeom.latent_t, dgeom.latent_h, dgeom.latent_w, cfg.in_channels,
                                         cfg.patch_size), geom.latent_h, geom.latent_w)
        video = (1 - sigma1) * patchify(x0, cfg.patch_size).to(dev) + sigma1 * noise_rows.float()
        audio = aus.step(0, avel, audio.float())
        del dreq, drows, vel, x0
        step_times.append(round(time.time() - t0, 2))
        first = 1
    dit.spark = spark
    for i in range(first, preset.steps):
        t0 = time.time()
        last = i == preset.steps - 1
        vel, avel, feats = run_step(i, request, layout, video, last)
        video = vs.step(i, vel, video.float())
        audio = aus.step(i, avel, audio.float())
        if last:
            features = feats.cpu()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        step_times.append(round(time.time() - t0, 2))
    dit.close()
    stats = dit.streamer.stats.as_dict()
    n_audio = layout.audio_idx.numel() // AUDIO_CHANNELS
    audio_latents = audio.reshape(AUDIO_CHANNELS, n_audio, -1).permute(0, 2, 1).contiguous().cpu()
    out = cache_dir() / 'runs' / f'denoise-{os.getpid()}-{int(time.time() * 1000)}.safetensors'
    save_file({'features': features.contiguous(), 'audio_latents': audio_latents}, str(out))
    peak = torch.cuda.max_memory_allocated() if dev.type == 'cuda' else 0
    return {'denoise': str(out), 'grid': [geom.latent_t, geom.latent_h // 2, geom.latent_w // 2],
            'render': [run_w, run_h], 'step_seconds': step_times, 'load_seconds': round(load_s, 2),
            'stream': stats, 'attention': plan.attention, 'spark': spark.summary() if spark else None,
            'vram_peak_gib': round(peak / (1 << 30), 2), 'plan': plan.as_dict()}


def stage_decode(model: str, req: dict, denoise_path: str, grid: list, output: str, emit) -> dict:
    from safetensors.torch import load_file
    from .dwd import DWDHead
    from .media import decode_audio, write_mp4
    r = GenRequest(**req)
    store = Store(model)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_file(denoise_path)
    t0 = time.time()
    head = DWDHead(store.load_safetensors(store.manifest['dwd']['file']),
                   SolConfig(**store.manifest['dwd']['config']), device=dev)
    frames = head.decode(data['features'], tuple(grid),
                         on_chunk=lambda i, n: emit(stage='decode', step=i, total=n))
    del head
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    frames = frames[:r.num_frames]
    out_h, out_w = output_size(r.height, r.width, r.upscale)
    upscaled = None
    if (frames.shape[1], frames.shape[2]) != (out_h, out_w):
        from .. import native
        t_up = time.time()
        emit(stage='upscale', step=0, total=1, message=f'{frames.shape[2]}×{frames.shape[1]} → {out_w}×{out_h}')
        frames = native.upscale_frames(frames, out_h, out_w, sharpen=0.35)
        upscaled = {'to': [out_w, out_h], 'seconds': round(time.time() - t_up, 2), 'native': native.available()}
        emit(stage='upscale', step=1, total=1)
    emit(stage='audio', step=0, total=1)
    audio = decode_audio(store, data['audio_latents'], dev)
    samples = int(round(frames.shape[0] / FPS * AUDIO_SAMPLE_RATE))
    audio = audio[..., :samples]
    meta = {'tinyh3': {k: v for k, v in asdict(r).items() if k not in ('output', 'model')}}
    write_mp4(output, frames, audio, FPS, AUDIO_SAMPLE_RATE, crf=r.crf, metadata=json.dumps(meta, ensure_ascii=False))
    emit(stage='audio', step=1, total=1)
    for p in (denoise_path,):
        try:
            os.remove(p)
        except OSError:
            pass
    return {'output': output, 'frames': int(frames.shape[0]), 'size': [int(frames.shape[2]), int(frames.shape[1])],
            'upscale': upscaled, 'seconds': round(time.time() - t0, 2)}


STAGES = {'text': stage_text, 'denoise': stage_denoise, 'decode': stage_decode}


# ------------------------------------------------------------------------------------------------ orchestration

def generate(request: GenRequest, on_event=None, *, subprocess: bool = True, cancel=None) -> dict:
    """Run the three stages; ``on_event(dict)`` receives progress. Returns a report with timings."""
    from .worker import run_stage
    on_event = on_event or (lambda e: None)
    r = request.resolve()
    found = default_model(r.preset.adapter_steps)
    model = r.model or (str(found) if found else None)
    if not model:
        raise FileNotFoundError('нет сконвертированной модели: запустите `tinyh3 setup`')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    output = r.output or str(outputs_dir() / f'tinyh3-{stamp}-{r.seed}.mp4')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    report = {'request': asdict(r), 'output': output, 'stages': {}}
    t0 = time.time()

    def call(name, *args):
        if cancel is not None and cancel.is_set():
            raise InterruptedError('отменено')
        s0 = time.time()
        on_event({'stage': name, 'event': 'start'})
        if subprocess:
            result = run_stage(name, args, on_event, cancel=cancel)
        else:
            result = STAGES[name](*args, lambda **e: on_event(e))
        result['wall_seconds'] = round(time.time() - s0, 2)
        report['stages'][name] = result
        on_event({'stage': name, 'event': 'done', 'result': result})
        return result

    text = call('text', model, r.prompt)
    den = call('denoise', model, asdict(r), text['text'])
    call('decode', model, asdict(r), den['denoise'], den['grid'], output)
    try:
        os.remove(text['text'])
    except OSError:
        pass
    report['total_seconds'] = round(time.time() - t0, 2)
    Path(output + '.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
    on_event({'stage': 'done', 'event': 'done', 'output': output, 'report': report})
    return report
