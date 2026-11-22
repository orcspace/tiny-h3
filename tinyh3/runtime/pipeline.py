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

from ..config import (SAMPLERS, Geometry, canvas_size, frames_for_seconds, RESOLUTIONS, FPS, AUDIO_CHANNELS,
                      AUDIO_SAMPLE_RATE, SolConfig, outputs_dir, cache_dir)
from ..store import Store


@dataclass
class GenRequest:
    prompt: str
    seconds: float = 5.0
    resolution: str = '480p'
    aspect: str = '16:9'
    sampler: str = 'balanced'       # 'balanced' (4 steps) | 'turbo' (2 steps)
    seed: int = -1
    attention: str = 'auto'         # auto | spark | sage16 | sage | sdpa
    spark_mode: str = 'speed'       # speed | fidelity
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
        if r.sampler not in SAMPLERS:
            raise ValueError(f'sampler должен быть одним из {list(SAMPLERS)}')
        return r

    @property
    def num_frames(self) -> int:
        return frames_for_seconds(self.seconds)



def default_model() -> Path | None:
    from ..config import models_dir
    root = models_dir()
    if not root.exists():
        return None
    stores = sorted(p.parent for p in root.glob('*/manifest.json'))
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


def stage_denoise(model: str, req: dict, text_path: str, emit) -> dict:
    from safetensors.torch import load_file, save_file
    from ..hardware import probe
    from ..planner import make_plan
    from .dit import DiT, Request
    from .layout import Layout, row_timesteps, KEYFRAME_NOISE_AUG
    from .scheduler import FlowEuler
    from .attention import SparkAttention, resolve

    r = GenRequest(**req)
    store = Store(model)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text = load_file(text_path)
    text_states, tags = text['text_states'], text['tags']
    geom = Geometry(r.height, r.width, r.num_frames, text_tokens=text_states.shape[0])
    preset = SAMPLERS[r.sampler]
    hw = probe()
    units = store.units('dit')
    kinds = store.manifest['dit'].get('kinds') or ['int8'] * len(units)
    plan = make_plan(hw, geom, sizes=[u['nbytes'] for u in units], kinds=kinds, steps=preset.steps,
                     attention=r.attention)
    if not plan.fits:
        raise MemoryError(plan.reason)
    emit(stage='plan', plan=plan.as_dict(), message=plan.summary())

    spark = None
    dense = resolve('auto' if plan.attention == 'spark' else plan.attention, hw.unlocked)
    if plan.attention == 'spark':
        spark = SparkAttention(preset.steps, mode=r.spark_mode, dense_fallback=dense)
    t_load = time.time()
    dit = DiT(store, device=dev, tiers={i: plan.tier(i) for i in range(len(units))}, slots=plan.slots,
              head_group=plan.head_group, ffn_chunk=plan.ffn_chunk, attention_backend=dense, spark=spark,
              progress=lambda i, n: emit(stage='load', step=i, total=n))
    load_s = time.time() - t_load

    layout = Layout.build(geom, tags)
    request = Request(layout, dev, dit.cfg)
    gen = torch.Generator().manual_seed(int(r.seed))
    latents = torch.randn((1, dit.cfg.in_channels, geom.latent_t, geom.latent_h, geom.latent_w), generator=gen)
    video = patchify(latents, dit.cfg.patch_size).to(dev)
    audio = torch.randn((layout.audio_idx.numel(), dit.cfg.audio_in_channels), generator=gen).to(dev)
    vs = FlowEuler(preset.steps, preset.shift, preset.max_raw_t, preset.final_step_size_scale)
    aus = FlowEuler(preset.steps, preset.audio_shift, preset.max_raw_t, preset.final_step_size_scale)
    n_text = layout.text_idx.numel()
    step_times, features = [], None
    total_blocks = preset.steps * dit.num_layers
    t_run = time.time()
    for i in range(preset.steps):
        tv, ta = vs.model_t(i), aus.model_t(i)
        t_unique, t_index = row_timesteps(layout.video_idx, layout.audio_idx, layout.n_cond, n_text, tv, ta,
                                          max(tv, KEYFRAME_NOISE_AUG))
        last = i == preset.steps - 1
        t0 = time.time()

        def on_block(b, _dt, i=i):
            done = i * dit.num_layers + b + 1
            elapsed = time.time() - t_run
            emit(stage='denoise', step=done, total=total_blocks, substep=i + 1, steps=preset.steps,
                 eta=round(elapsed / done * (total_blocks - done), 1))

        out_v, out_a, feats = dit.forward(request, video, audio, text_states, t_unique, t_index, want_features=last,
                                          on_block=on_block)
        video = vs.step(i, -out_v.float(), video.float())
        audio = aus.step(i, -out_a.float(), audio.float())
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
            'step_seconds': step_times, 'load_seconds': round(load_s, 2), 'stream': stats,
            'attention': plan.attention, 'spark': spark.summary() if spark else None,
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
    return {'output': output, 'frames': int(frames.shape[0]), 'seconds': round(time.time() - t0, 2)}


STAGES = {'text': stage_text, 'denoise': stage_denoise, 'decode': stage_decode}


# ------------------------------------------------------------------------------------------------ orchestration

def generate(request: GenRequest, on_event=None, *, subprocess: bool = True, cancel=None) -> dict:
    """Run the three stages; ``on_event(dict)`` receives progress. Returns a report with timings."""
    from .worker import run_stage
    on_event = on_event or (lambda e: None)
    r = request.resolve()
    model = r.model or (str(default_model()) if default_model() else None)
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
