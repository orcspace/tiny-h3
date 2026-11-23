"""HF checkpoints -> TinyH3 store.

Streams one tensor at a time (peak host memory ~2 GB regardless of model size), fuses the pi-Flow/DWD adapter in
fp32, folds the ConvRot Hadamard rotation into W8A8/W4A8 weights, quantizes, and writes one packed blob per unit.
Re-running resumes: finished units are skipped.

Groups written:

* ``dit``     – 50 blocks: fused QKV, out, FFN in/out (quantized), norms, pruned AdaLN weight + fp32 bias;
* ``refiner`` – unit 0: context_embedder + refiner final norm; units 1..: token refiner blocks (they run once per
  prompt in the text stage, not per step);
* ``text``    – Qwen3 decoder layers 0..49 (weight-only int8/int4) + ``embed.bin`` (raw bf16 rows);
* ``dwd/head.safetensors``, ``dit/globals.safetensors``, ``audio_vae/``, ``tokenizer/``.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ..config import DiTConfig, TextEncoderConfig, SolConfig, SAMPLERS
from ..kernels import reference as ref
from ..planner import block_kinds
from ..store import BlobWriter, FORMAT
from .lora import Adapter, fused, project_adaln
from .sources import ShardedReader

log = logging.getLogger(__name__)


def rotate_rows(w: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    out = torch.empty_like(w, dtype=torch.float32)
    for r0 in range(0, w.shape[0], chunk):
        out[r0:r0 + chunk] = ref.rotate(w[r0:r0 + chunk].float())
    return out


def quantize_linear(w: torch.Tensor, kind: str, *, rotate: bool = True, device='cpu') -> dict:
    w = w.to(device, torch.float32)
    n, k = w.shape
    if kind == 'bf16':
        return {'kind': 'bf16', 'shape': [n, k], 'w': w.to(torch.bfloat16).cpu()}
    if rotate:
        w = rotate_rows(w)
    if kind == 'int8':
        q, s = ref.quantize_weight_int8(w)
        rec = {'kind': 'int8', 'shape': [n, k], 'q': q.cpu(), 'scale': s.cpu()}
    elif kind == 'int4':
        q, s = ref.quantize_weight_int4(w)
        rec = {'kind': 'int4', 'group': ref.INT4_GROUP, 'shape': [n, k], 'q': q, 'scale': s}
    else:
        raise ValueError(kind)
    rec['rotated'] = rotate
    return rec


class _Units:
    """Writes units of one group, skipping those already finished (resume)."""

    def __init__(self, root: Path, group: str):
        self.dir = root / group
        self.dir.mkdir(parents=True, exist_ok=True)

    def done(self, index: int) -> dict | None:
        side = self.dir / f'unit_{index:03d}.json'
        if side.exists() and (self.dir / f'unit_{index:03d}.bin').exists():
            return json.loads(side.read_text())
        return None

    def write(self, index: int, writer: BlobWriter) -> dict:
        unit = writer.write(self.dir / f'unit_{index:03d}.bin')
        (self.dir / f'unit_{index:03d}.json').write_text(json.dumps(unit))
        return unit

    def collect(self, count: int) -> list[dict]:
        return [json.loads((self.dir / f'unit_{i:03d}.json').read_text()) for i in range(count)]


def _get(reader, adapter, key):
    if adapter is not None:
        over = adapter.override(key)
        if over is not None:
            return over
    return reader.get(key)


def _block_linears(writer, reader, adapter, prefix, kind, device):
    def w(name):
        return fused(reader.get(f'{prefix}.{name}.weight'), f'{prefix}.{name}', adapter)
    qkv = torch.cat([w('attn.to_q'), w('attn.to_k'), w('attn.to_v')])
    writer.add_linear('qkv', quantize_linear(qkv, kind, device=device))
    del qkv
    writer.add_linear('out', quantize_linear(w('attn.to_out.0'), kind, device=device))
    writer.add_linear('ff_in', quantize_linear(w('ff.net.0.proj'), kind, device=device))
    writer.add_linear('ff_out', quantize_linear(w('ff.net.2'), kind, device=device))
    for name, key in (('norm1', 'norm1.weight'), ('norm2', 'norm2.weight'),
                      ('norm_q', 'attn.norm_q.weight'), ('norm_k', 'attn.norm_k.weight')):
        writer.add(name, _get(reader, adapter, f'{prefix}.{key}').to(torch.bfloat16))


def convert_dit(reader, adapter, out: Path, cfg: DiTConfig, policy: str, device, progress, group='dit') -> dict:
    units = _Units(out, group)
    kinds = block_kinds(policy, cfg.num_layers)
    basis = reader.get('adaln_basis') if 'adaln_basis' in reader else None
    mean = reader.get('adaln_mean') if 'adaln_mean' in reader else None
    for i in range(cfg.num_layers):
        if units.done(i):
            progress(group, i + 1, cfg.num_layers)
            continue
        prefix = f'transformer_blocks.{i}'
        writer = BlobWriter()
        _block_linears(writer, reader, adapter, prefix, kinds[i], device)
        aw = _get(reader, adapter, f'{prefix}.adaln_proj.linear.weight').float()
        ab = _get(reader, adapter, f'{prefix}.adaln_proj.folded_bias').float()
        if adapter is not None:
            delta, offset = project_adaln(adapter, f'{prefix}.adaln_proj.linear', basis, mean, cfg.adaln_rank)
            if delta is not None:
                aw = aw + delta
            if offset is not None:
                ab = ab + offset
        writer.add('adaln_w', aw.to(torch.bfloat16))
        writer.add('adaln_b', ab.float())
        units.write(i, writer)
        progress(group, i + 1, cfg.num_layers)
    return {'config': asdict(cfg), 'kinds': kinds, 'units': units.collect(cfg.num_layers)}


def convert_dit_globals(reader, adapter, out: Path, cfg: DiTConfig, has_dwd: bool, group='dit') -> None:
    from safetensors.torch import save_file
    g = {}
    for key in ('proj_in.weight', 'proj_in.bias', 'audio_proj_in.weight', 'audio_proj_in.bias',
                'proj_out.bias', 'audio_proj_out.bias', 'norm_out.norm.weight'):
        g[key] = _get(reader, adapter, key).float()
    for module in ('proj_out', 'audio_proj_out', 'proj_in', 'audio_proj_in'):
        g[f'{module}.weight'] = fused(reader.get(f'{module}.weight'), module, adapter)
    g['time_table'] = _get(reader, adapter, 'time_embedder.table').float()
    nw = _get(reader, adapter, 'norm_out.linear.weight').float()
    nb = _get(reader, adapter, 'norm_out.folded_bias').float()
    if adapter is not None:
        basis = reader.get('adaln_basis') if 'adaln_basis' in reader else None
        mean = reader.get('adaln_mean') if 'adaln_mean' in reader else None
        delta, offset = project_adaln(adapter, 'norm_out.linear', basis, mean, cfg.adaln_rank)
        if delta is not None:
            nw = nw + delta
        if offset is not None:
            nb = nb + offset
    g['norm_out.linear.weight'] = nw.to(torch.bfloat16)
    g['norm_out.bias'] = nb
    if has_dwd:
        g['dwd_norm_out.linear.weight'] = fused(adapter.override('dwd_norm_out.linear.weight'),
                                                'dwd_norm_out.linear', adapter)
        g['dwd_norm_out.linear.bias'] = adapter.override('dwd_norm_out.linear.bias').float()
    (out / group).mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in g.items()}, str(out / group / 'globals.safetensors'))


def convert_refiner(reader, adapter, out: Path, cfg: DiTConfig, device, group='refiner') -> dict:
    units = _Units(out, group)
    count = 1 + cfg.num_refiner_layers
    if not units.done(0):
        writer = BlobWriter()
        ctx = fused(reader.get('context_embedder.weight'), 'context_embedder', adapter)
        writer.add_linear('ctx', quantize_linear(ctx, 'int8', rotate=False, device=device))
        writer.add('ctx_bias', _get(reader, adapter, 'context_embedder.bias').float())
        writer.add('final_norm', _get(reader, adapter, 'token_refiner.final_norm.weight').to(torch.bfloat16))
        units.write(0, writer)
    for i in range(cfg.num_refiner_layers):
        if units.done(i + 1):
            continue
        writer = BlobWriter()
        _block_linears(writer, reader, adapter, f'token_refiner.refiner_blocks.{i}', 'int8', device)
        units.write(i + 1, writer)
    return {'units': units.collect(count)}


def convert_dwd(adapter, out: Path) -> dict | None:
    from safetensors.torch import save_file
    keys = adapter.keys_with_prefix('dwd_pixel_refiner.') if adapter is not None else []
    if not keys:
        return None
    tensors = {}
    for key in keys:
        name = key[len('dwd_pixel_refiner.'):]
        t = adapter.override(key)
        module = key.rsplit('.', 1)[0]
        if key.endswith('.weight') and t.ndim == 2:
            t = fused(t, module, adapter)
        small = t.ndim < 2 or 'norm' in name or 'scale' in name or 'register' in name
        tensors[name] = t.float() if small else t.to(torch.bfloat16)
    (out / 'dwd').mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / 'dwd' / 'head.safetensors'))
    num_layers = 1 + max(int(k.split('.')[1]) for k in tensors if k.startswith('blocks.'))
    return {'file': 'dwd/head.safetensors', 'num_layers': num_layers}


def write_visual(reader, out: Path, config_dir: Path | None) -> None:
    """The Qwen3-VL vision tower (bf16, ~0.8 GB) and the conditioner's config, for image / video prompts."""
    from safetensors.torch import save_file
    target = out / 'text' / 'visual.safetensors'
    names = list(getattr(getattr(reader, 'f', None), 'names', []) or []) + list(reader.keys())
    vis = {}
    for name in names:
        for prefix in ('model.visual.', 'visual.'):
            if name.startswith(prefix) and not name.endswith(('.weight_scale', '.comfy_quant')):
                vis['visual.' + name[len(prefix):]] = name
    if vis and not target.exists():
        getter = reader.f.get if hasattr(reader, 'f') and not all(k in reader for k in vis.values()) else reader.get
        save_file({k: getter(v).to(torch.bfloat16).contiguous() for k, v in vis.items()}, str(target))
    if config_dir is not None and (Path(config_dir) / 'config.json').exists():
        shutil.copy(Path(config_dir) / 'config.json', out / 'text' / 'config.json')


def convert_text(reader, out: Path, cfg: TextEncoderConfig, kind: str, device, progress, rotate: bool = False,
                 config_dir: Path | None = None) -> dict:
    """``rotate``: keep the ConvRot basis (lossless when the source already is INT8 ConvRot); the runtime then
    rotates activations before the weight-only matmul."""
    units = _Units(out, 'text')
    prefix = 'model.language_model'
    embed_path = out / 'text' / 'embed.bin'
    if not embed_path.exists():
        table = reader.get(f'{prefix}.embed_tokens.weight')
        tmp = embed_path.with_suffix('.part')
        with open(tmp, 'wb') as f:
            for r0 in range(0, table.shape[0], 8192):
                f.write(table[r0:r0 + 8192].to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes())
        tmp.replace(embed_path)
        del table
    for i in range(cfg.num_layers_used):
        if units.done(i):
            progress('text', i + 1, cfg.num_layers_used)
            continue
        p = f'{prefix}.layers.{i}'
        writer = BlobWriter()
        qkv = torch.cat([reader.get(f'{p}.self_attn.{n}_proj.weight') for n in 'qkv'])
        writer.add_linear('qkv', quantize_linear(qkv, kind, rotate=rotate, device=device))
        writer.add_linear('o', quantize_linear(reader.get(f'{p}.self_attn.o_proj.weight'), kind, rotate=rotate,
                                               device=device))
        gu = torch.cat([reader.get(f'{p}.mlp.gate_proj.weight'), reader.get(f'{p}.mlp.up_proj.weight')])
        writer.add_linear('gate_up', quantize_linear(gu, kind, rotate=rotate, device=device))
        writer.add_linear('down', quantize_linear(reader.get(f'{p}.mlp.down_proj.weight'), kind, rotate=rotate,
                                                  device=device))
        for name, key in (('input_norm', 'input_layernorm'), ('post_norm', 'post_attention_layernorm'),
                          ('q_norm', 'self_attn.q_norm'), ('k_norm', 'self_attn.k_norm')):
            writer.add(name, reader.get(f'{p}.{key}.weight').to(torch.bfloat16))
        units.write(i, writer)
        progress('text', i + 1, cfg.num_layers_used)
    write_visual(reader, out, config_dir)
    return {'config': asdict(cfg), 'kind': kind, 'units': units.collect(cfg.num_layers_used),
            'embed': {'file': 'text/embed.bin', 'dim': cfg.hidden_size, 'dtype': 'bfloat16'}}


def write_vae(src: Path, out: Path, decoder: bool) -> dict:
    """Encoder (fp32) and optionally decoder (fp16) of a local diffusers video-VAE directory."""
    from safetensors.torch import save_file
    dst = out / 'vae'
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / 'config.json', dst / 'config.json')
    reader = ShardedReader(src)
    parts = {'encoder': (('encoder.', 'quant_conv.'), torch.float32)}
    if decoder:
        parts['decoder'] = (('decoder.', 'post_quant_conv.'), torch.float16)
    for name, (prefixes, dtype) in parts.items():
        target = dst / f'{name}.safetensors'
        if not target.exists():
            save_file({k: reader.get(k).to(dtype).contiguous() for k in reader.keys() if k.startswith(prefixes)},
                      str(target))
    return {'encoder': 'vae/encoder.safetensors', 'decoder': 'vae/decoder.safetensors' if decoder else None}


def _convert_ref(manifest: dict, out: Path, ref_dit: Path, ref_adapter: Path | None, policy, device, progress):
    ref_dit = Path(ref_dit)
    if ref_dit.is_file():
        from .comfy import ComfyDiTView
        cfg = DiTConfig()
        reader = ComfyDiTView(ref_dit, cfg.num_layers, cfg.num_refiner_layers, cfg.inner_dim)
    else:
        cfg = dit_config_from(ref_dit)
        reader = ShardedReader(ref_dit)
    adapter = Adapter(ShardedReader(Path(ref_adapter))) if ref_adapter is not None else None
    convert_dit_globals(reader, adapter, out, cfg, False, group='dit_ref')
    manifest['dit_ref'] = convert_dit(reader, adapter, out, cfg, policy, device, progress, group='dit_ref')
    manifest['refiner_ref'] = convert_refiner(reader, adapter, out, cfg, device, group='refiner_ref')
    manifest.setdefault('recipe', {})['ref_adapter'] = Path(ref_adapter).name if ref_adapter else None


def dit_config_from(path: Path) -> DiTConfig:
    raw = json.loads((path / 'config.json').read_text())
    raw = {{'num_attention_heads': 'num_heads', 'attention_head_dim': 'head_dim'}.get(k, k): v for k, v in raw.items()}
    fields = DiTConfig.__dataclass_fields__
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in fields}
    return DiTConfig(**kw)


def text_config_from(path: Path) -> TextEncoderConfig:
    raw = json.loads((path / 'config.json').read_text()).get('text_config', {})
    return TextEncoderConfig(
        hidden_size=raw.get('hidden_size', 5120), num_heads=raw.get('num_attention_heads', 64),
        num_kv_heads=raw.get('num_key_value_heads', 8), head_dim=raw.get('head_dim', 128),
        intermediate_size=raw.get('intermediate_size', 25600), vocab_size=raw.get('vocab_size', 151936),
        rms_norm_eps=raw.get('rms_norm_eps', 1e-6),
        rope_theta=raw.get('rope_theta', (raw.get('rope_parameters') or {}).get('rope_theta', 5e6)),
        num_layers_used=min(50, raw.get('num_hidden_layers', 51) - 1),
        mrope_section=tuple(((raw.get('rope_scaling') or raw.get('rope_parameters') or {}).get('mrope_section'))
                            or (24, 20, 20)))


def convert(base: Path | None, adapter_dir: Path | None, official: Path | None, out: Path, *, policy='int8',
            text_kind='int8', sampler='balanced', device=None, with_text=True, progress=None,
            comfy_dit: Path | None = None, comfy_text: Path | None = None, vae_dir: Path | None = None,
            vae_decoder: bool = False, ref_dit: Path | None = None, ref_adapter: Path | None = None,
            only_ref: bool = False) -> Path:
    """``base``: a diffusers MiniMax-H3-Pruned directory; or pass ``comfy_dit`` / ``comfy_text`` (Comfy-Org INT8
    ConvRot single files) instead.

    ``vae_dir``: a diffusers video-VAE directory; its encoder (keyframe / reference images) and, with
    ``vae_decoder``, its decoder (Ref2VA, which has no DWD head) are stored. ``ref_dit``: the Ref2VA transformer
    (a Comfy-Org single file or a diffusers ``transformer_ref`` directory) with ``ref_adapter`` (e.g. lightx2v's
    Ref2VA turbo LoRA) fused; stored as the ``dit_ref`` / ``refiner_ref`` groups of the same model. With
    ``only_ref`` an existing store gains the Ref2VA groups without touching the rest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else {}
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    progress = progress or (lambda stage, i, n: log.info('%s %d/%d', stage, i, n))
    t0 = time.time()

    if only_ref:
        manifest = previous
        _convert_ref(manifest, out, ref_dit, ref_adapter, policy, device, progress)
        if vae_dir is not None:
            manifest['vae'] = write_vae(Path(vae_dir), out, decoder=True)
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
        return out
    if comfy_dit is not None:
        from .comfy import ComfyDiTView
        cfg = DiTConfig()
        reader = ComfyDiTView(Path(comfy_dit), cfg.num_layers, cfg.num_refiner_layers, cfg.inner_dim)
    else:
        base = Path(base)
        cfg = dit_config_from(base / 'transformer')
        reader = ShardedReader(base / 'transformer')
    adapter = None
    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir)
        adapter = Adapter(ShardedReader(adapter_dir))
    has_dwd = adapter is not None and bool(adapter.keys_with_prefix('dwd_pixel_refiner.'))

    manifest = {'format': FORMAT, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'recipe': {'source': 'comfy-int8' if comfy_dit else 'diffusers', 'policy': policy, 'text_kind': text_kind, 'sampler': sampler,
                           'steps': SAMPLERS[sampler].adapter_steps if sampler in SAMPLERS else None,
                           'adapter': str(adapter_dir.name) if adapter_dir else None, 'dwd': has_dwd}}
    convert_dit_globals(reader, adapter, out, cfg, has_dwd)
    manifest['dit'] = convert_dit(reader, adapter, out, cfg, policy, device, progress)
    manifest['refiner'] = convert_refiner(reader, adapter, out, cfg, device)
    if has_dwd:
        dwd = convert_dwd(adapter, out)
        sol = SolConfig(num_layers=dwd['num_layers'])
        manifest['dwd'] = {**dwd, 'config': asdict(sol)}
    if with_text and comfy_text is not None:
        from .comfy import ComfyTextView
        te_cfg_dir = Path(official) / 'text_encoder' if official is not None else None
        manifest['text'] = convert_text(ComfyTextView(Path(comfy_text)), out, TextEncoderConfig(), 'int8', device,
                                        progress, rotate=True, config_dir=te_cfg_dir)
    elif with_text and base is not None and (Path(base) / 'text_encoder').exists():
        tcfg = text_config_from(Path(base) / 'text_encoder')
        manifest['text'] = convert_text(ShardedReader(Path(base) / 'text_encoder'), out, tcfg, text_kind, device,
                                        progress, config_dir=Path(base) / 'text_encoder')
    if vae_dir is not None:
        manifest['vae'] = write_vae(Path(vae_dir), out, decoder=vae_decoder)
    elif previous.get('vae'):
        manifest['vae'] = previous['vae']
    if ref_dit is not None:
        _convert_ref(manifest, out, ref_dit, ref_adapter, policy, device, progress)
    else:
        for key in ('dit_ref', 'refiner_ref'):
            if key in previous:
                manifest[key] = previous[key]
    if official is not None:
        official = Path(official)
        for sub in ('audio_vae', 'tokenizer', 'processor'):
            src = official / sub
            if src.exists() and not (out / sub).exists():
                shutil.copytree(src, out / sub)
    manifest['convert_seconds'] = round(time.time() - t0, 1)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return out
