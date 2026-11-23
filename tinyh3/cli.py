"""``tinyh3`` command line."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .config import (RESOLUTIONS, ASPECT_RATIOS, SAMPLERS, SAMPLER_ALIASES, UPSCALE, models_dir, canvas_size,
                     frames_for_seconds, Geometry, render_size)


def _bar(done, total, width=28):
    filled = int(width * done / max(total, 1))
    return '█' * filled + '░' * (width - filled)


STAGE_NAMES = {'text': 'Текст', 'load': 'Загрузка весов', 'denoise': 'Денойзинг', 'decode': 'DWD → кадры',
               'upscale': 'Апскейл (Rust)', 'audio': 'Звук и MP4', 'plan': 'План'}
MODES = list(SAMPLERS) + list(SAMPLER_ALIASES)


def _printer():
    last = {'line': ''}

    def on_event(e):
        stage = e.get('stage')
        if e.get('event') == 'done' and stage == 'done':
            sys.stdout.write('\n')
            return
        if stage == 'plan':
            sys.stdout.write(f"\n  план: {e.get('message')}\n")
            return
        if 'step' in e and 'total' in e:
            name = STAGE_NAMES.get(stage, stage)
            extra = f" шаг {e['substep']}/{e['steps']}" if 'substep' in e else ''
            eta = f" ~{e['eta']:.0f} с" if e.get('eta') else ''
            line = f"\r  {name:<16} {_bar(e['step'], e['total'])} {e['step']}/{e['total']}{extra}{eta}   "
            sys.stdout.write(line)
            sys.stdout.flush()
            last['line'] = line
        elif e.get('event') == 'done' and 'result' in e:
            sys.stdout.write(f"\n  ✓ {STAGE_NAMES.get(stage, stage)}: {e['result'].get('wall_seconds', 0):.1f} с\n")
    return on_event


def cmd_doctor(a):
    from .hardware import probe
    from .gpu_tuning import profile, advice
    from .runtime.attention import available
    hw = probe()
    print(f'TinyH3 Video {__version__}')
    print(f'GPU:  {hw.gpu_name}  CC {hw.compute_capability}  VRAM {hw.vram_free / 2**30:.1f}/{hw.vram_total / 2**30:.1f} ГБ')
    print(f'ОЗУ:  {hw.ram_available / 2**30:.1f}/{hw.ram_total / 2**30:.1f} ГБ доступно   ОС: {hw.os}')
    print(f'Внимание: {", ".join(available())}')
    try:
        import triton  # noqa: F401
        print('Triton: да (слитые ядра и int8 GEMM)')
    except ImportError:
        print('Triton: нет — будут использованы медленные PyTorch-фоллбэки')
    prof = profile(refresh=a.refresh) if hw.vram_total else None
    for tip in advice(prof, hw.os):
        print('•', tip)
    stores = sorted(p.parent.name for p in models_dir().glob('*/manifest.json')) if models_dir().exists() else []
    print(f'Модели ({models_dir()}): {", ".join(stores) or "нет — запустите `tinyh3 setup`"}')


def _geometry(a):
    h, w = (a.height, a.width) if getattr(a, 'height', None) else canvas_size(a.aspect, RESOLUTIONS[a.res])
    return h, w


def cmd_plan(a):
    from .planner import make_plan
    from .hardware import probe, simulated
    mode = SAMPLERS[SAMPLER_ALIASES.get(a.sampler, a.sampler)]
    steps = mode.steps
    h, w = render_size(*_geometry(a), mode.final_scale)
    hw = simulated(a.vram_gib, a.ram_gib, a.gpu, a.disk_gbps, unlocked=a.unlocked, spark=not a.no_spark) \
        if a.vram_gib else probe()
    plan = make_plan(hw, Geometry(h, w, frames_for_seconds(a.seconds)), policy=a.policy, steps=steps, mode=mode)
    if a.json:
        print(json.dumps(plan.as_dict(), ensure_ascii=False, indent=1))
        return
    g = Geometry(h, w, frames_for_seconds(a.seconds))
    print(f'режим {mode.title}: рендер {w}×{h}, {g.num_frames} кадров ({g.seconds:.1f} с), '
          f'{g.sequence_length} токенов, {steps} шага')
    print(plan.summary())
    print(f"  внимание: {plan.attention}   активации {plan.activation / 2**30:.2f} ГБ   "
          f"слоты {plan.slots}×{plan.slot_bytes / 2**20:.0f} МБ")
    for k, v in plan.est.items():
        print(f'  {k:>10}: {v}')
    for n in plan.notes:
        print('  •', n)


def cmd_setup(a):
    from .convert.convert import convert
    src = Path(a.sources)
    a.sampler = 'quality' if a.steps == 4 else 'balance'
    preset = SAMPLERS[a.sampler]
    out = models_dir() / (a.name or f'h3-pi{preset.adapter_steps}-dwd-{a.policy}')
    if a.source == 'comfy':
        from .convert.comfy import download
        print(f'Скачивание в {src}: INT8 ConvRot от Comfy-Org (~52 ГБ; можно удалить после конвертации)…')
        paths = download(src, sampler=a.sampler)
        print(f'Конвертация в {out}…')
        convert(None, Path(paths['adapter']), Path(paths['official']), out, policy=a.policy, sampler=a.sampler,
                comfy_dit=Path(paths['dit']), comfy_text=Path(paths['text']), progress=_convert_progress())
    else:
        from .convert.sources import download
        print(f'Скачивание в {src}: bf16 (~95 ГБ; можно удалить после конвертации)…')
        paths = download(src, sampler=a.sampler)
        print(f'Конвертация в {out}…')
        convert(Path(paths['base']), Path(paths['adapter']) / preset.adapter, Path(paths['official']), out,
                policy=a.policy, text_kind=a.text, sampler=a.sampler, progress=_convert_progress())
    _ensure_vae(out, decoder=False)
    if a.ref:
        _setup_ref(a, src, out)
    if a.delete_sources:
        import shutil
        shutil.rmtree(src, ignore_errors=True)
        print(f'\nИсходники удалены: {src}')
    print(f'\nГотово: {out}\nЗапуск: tinyh3 serve --open')


def _ensure_vae(out: Path, decoder: bool):
    """Video-VAE encoder (keyframes, reference images; ~0.7 GB) and optionally decoder (Ref2VA; ~4.8 GB fp16),
    read from the Hub with range requests: the 10 GB checkpoint is never downloaded as a whole."""
    from .convert.remote import fetch_vae
    from .config import SOURCES
    print('\nVAE (энкодер' + (' и декодер' if decoder else '') + ') …')
    fetch_vae(SOURCES['official_repo'], out, decoder=decoder)
    m = json.loads((out / 'manifest.json').read_text())
    m['vae'] = {'encoder': 'vae/encoder.safetensors',
                'decoder': 'vae/decoder.safetensors' if (out / 'vae' / 'decoder.safetensors').exists() else None}
    (out / 'manifest.json').write_text(json.dumps(m, indent=1))


def _setup_ref(a, src: Path, out: Path):
    from .convert.comfy import download_ref
    from .convert.convert import convert
    print(f'\nRef2VA: Comfy-Org INT8 (~21 ГБ) + LoRA Turbo 4 (~1.4 ГБ) в {src}…')
    paths = download_ref(src)
    _ensure_vae(out, decoder=True)
    print('Конвертация Ref2VA…')
    convert(None, None, None, out, policy=a.policy, only_ref=True, ref_dit=Path(paths['dit']),
            ref_adapter=Path(paths['adapter']), progress=_convert_progress())


def cmd_setup_ref(a):
    """Add Ref2VA (references → video) to an existing model."""
    stores = sorted(p.parent for p in models_dir().glob('*/manifest.json')) if models_dir().exists() else []
    if not stores:
        raise SystemExit('сначала `tinyh3 setup`')
    out = Path(a.model) if a.model else stores[0]
    _setup_ref(a, Path(a.sources), out)
    print(f'\nГотово: Ref2VA добавлен в {out}')


def _convert_progress():
    def p(stage, i, n):
        sys.stdout.write(f'\r  {stage:<6} {_bar(i, n)} {i}/{n}   ')
        sys.stdout.flush()
    return p


def cmd_convert(a):
    from .convert.convert import convert
    if not a.base and not a.comfy_dit:
        raise SystemExit('нужен --base или --comfy-dit')
    out = convert(Path(a.base) if a.base else None, Path(a.adapter) if a.adapter else None,
                  Path(a.official) if a.official else None, Path(a.out), policy=a.policy, text_kind=a.text,
                  sampler=a.sampler, device=a.device, with_text=not a.no_text, progress=_convert_progress(),
                  comfy_dit=Path(a.comfy_dit) if a.comfy_dit else None,
                  comfy_text=Path(a.comfy_text) if a.comfy_text else None)
    print(f'\nГотово: {out}')


REF_KINDS = {'.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.webp': 'image', '.bmp': 'image',
             '.mp4': 'video', '.mov': 'video', '.mkv': 'video', '.webm': 'video',
             '.wav': 'audio', '.mp3': 'audio', '.flac': 'audio', '.m4a': 'audio', '.ogg': 'audio'}


def cmd_generate(a):
    from .runtime.pipeline import GenRequest, generate
    refs = []
    for item in a.ref or []:
        kind, _, path = item.partition('=') if '=' in item and item.split('=')[0] in ('image', 'video', 'audio') \
            else (REF_KINDS.get(Path(item).suffix.lower(), ''), '', item)
        if kind not in ('image', 'video', 'audio'):
            raise SystemExit(f'не удалось определить тип референса: {item}')
        refs.append({'kind': kind, 'path': str(Path(path).expanduser())})
    task = a.task or ('ref2v' if refs else 'flf2v' if (a.first and a.last) else 'i2v' if a.first
                      else 'last2v' if a.last else 't2v')
    req = GenRequest(prompt=a.prompt, seconds=a.seconds, resolution=a.res, aspect=a.aspect, sampler=a.sampler,
                     seed=a.seed, attention=a.attention, spark_mode=a.spark_mode, model=a.model, output=a.out,
                     height=a.height, width=a.width, upscale=a.upscale, task=task, first=a.first, last=a.last,
                     references=refs)
    print(f'задача: {task}' + (f', референсов: {len(refs)}' if refs else ''))
    t0 = time.time()
    report = generate(req, _printer(), subprocess=not a.in_process)
    print(f"Видео: {report['output']}  ({time.time() - t0:.1f} с)")


def cmd_serve(a):
    from .server.app import serve
    serve(a.host, a.port, open_browser=a.open)


def cmd_bench(a):
    from .gpu_tuning import profile, measure_attention
    prof = profile(refresh=True)
    if prof is None:
        print('Нужна CUDA-видеокарта.')
        return
    print(prof.summary)
    for name, ms in measure_attention(a.tokens).items():
        print(f'  внимание {name:<7} {a.tokens} токенов × 8 голов: {ms:.1f} мс')


def main(argv=None):
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    p = argparse.ArgumentParser(prog='tinyh3', description='TinyH3 Video — MiniMax H3 на 8 ГБ видеопамяти')
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='cmd', required=True)

    d = sub.add_parser('doctor', help='проверить железо, ядра и модели')
    d.add_argument('--refresh', action='store_true', help='перемерить скорость тензорных ядер')
    d.set_defaults(fn=cmd_doctor)

    def geo(sp):
        sp.add_argument('--seconds', type=float, default=5.0)
        sp.add_argument('--res', default='480p', choices=list(RESOLUTIONS))
        sp.add_argument('--aspect', default='16:9', choices=ASPECT_RATIOS)
        sp.add_argument('--height', type=int)
        sp.add_argument('--width', type=int)
        sp.add_argument('--mode', '--sampler', dest='sampler', default='fast', choices=MODES,
                        help='quality (4 шага) | balance (2) | fast (черновик + 1) | fast-high (~2× к fast)')
        sp.add_argument('--upscale', default='none', choices=list(UPSCALE), help='апскейл выхода Rust-ядром')

    pl = sub.add_parser('plan', help='показать план памяти и оценку времени')
    geo(pl)
    pl.add_argument('--policy', default='int8', choices=['int8', 'mixed', 'int4'])
    pl.add_argument('--vram-gib', type=float, help='смоделировать машину вместо замера')
    pl.add_argument('--ram-gib', type=float, default=16)
    pl.add_argument('--gpu', default='RTX 3070')
    pl.add_argument('--disk-gbps', type=float, default=2.0)
    pl.add_argument('--unlocked', action='store_true')
    pl.add_argument('--no-spark', action='store_true')
    pl.add_argument('--json', action='store_true')
    pl.set_defaults(fn=cmd_plan)

    s = sub.add_parser('setup', help='скачать веса и сконвертировать модель')
    s.add_argument('--sources', default=str(Path.home() / '.tinyh3' / 'sources'))
    s.add_argument('--name')
    s.add_argument('--policy', default='int8', choices=['int8', 'mixed', 'int4'])
    s.add_argument('--steps', type=int, default=2, choices=[2, 4],
                   help='адаптер pi-H3: 2 шага (Баланс, Fast, Fast-high) или 4 шага (Качество)')
    s.add_argument('--text', default='int8', choices=['int8', 'int4'], help='только для --source hf')
    s.add_argument('--source', default='comfy', choices=['comfy', 'hf'],
                   help='comfy: готовые INT8 ConvRot от Comfy-Org (~52 ГБ); hf: bf16 оригиналы (~95 ГБ)')
    s.add_argument('--delete-sources', action='store_true', help='удалить скачанные исходники после конвертации')
    s.add_argument('--ref', action='store_true', help='сразу добавить Ref2VA (референсы → видео, +~27 ГБ)')
    s.set_defaults(fn=cmd_setup)

    sr = sub.add_parser('setup-ref', help='добавить Ref2VA (референсы → видео) к готовой модели')
    sr.add_argument('--sources', default=str(Path.home() / '.tinyh3' / 'sources'))
    sr.add_argument('--model', help='папка модели (по умолчанию первая)')
    sr.add_argument('--policy', default='int8', choices=['int8', 'mixed', 'int4'])
    sr.set_defaults(fn=cmd_setup_ref)

    c = sub.add_parser('convert', help='сконвертировать локальные HF-веса')
    c.add_argument('--base', help='папка MiniMax-H3-Pruned (transformer/, text_encoder/)')
    c.add_argument('--comfy-dit', help='minimax_h3_fl2va_pruned_int8_convrot.safetensors (вместо --base)')
    c.add_argument('--comfy-text', help='qwen3vl_32b_minimax_h3_int8_convrot.safetensors')
    c.add_argument('--adapter', help='папка адаптера pi-H3 (…_dwd_adapter)')
    c.add_argument('--official', help='папка MiniMax-H3 с audio_vae/ и tokenizer/')
    c.add_argument('--out', required=True)
    c.add_argument('--policy', default='int8', choices=['int8', 'mixed', 'int4'])
    c.add_argument('--sampler', default='balance', choices=MODES, help='какой адаптер конвертируется (2 или 4 шага)')
    c.add_argument('--text', default='int8', choices=['int8', 'int4'])
    c.add_argument('--device')
    c.add_argument('--no-text', action='store_true')
    c.set_defaults(fn=cmd_convert)

    g = sub.add_parser('generate', help='сгенерировать видео со звуком')
    g.add_argument('prompt')
    geo(g)
    g.add_argument('--seed', type=int, default=-1)
    g.add_argument('--attention', default='auto', choices=['auto', 'spark', 'sage16', 'sage', 'sdpa'])
    g.add_argument('--spark-mode', default=None, choices=['speed', 'fidelity'], help='переопределить режим Spark')
    g.add_argument('--model')
    g.add_argument('--out')
    g.add_argument('--task', choices=['t2v', 'i2v', 'last2v', 'flf2v', 'ref2v'],
                   help='по умолчанию выбирается по аргументам')
    g.add_argument('--first', help='первый кадр (изображение): image → video')
    g.add_argument('--last', help='последний кадр (изображение)')
    g.add_argument('--ref', action='append',
                   help='референс: путь (тип по расширению) или image=/video=/audio=путь; можно повторять')
    g.add_argument('--in-process', action='store_true', help='без отдельных процессов на стадии (отладка)')
    g.set_defaults(fn=cmd_generate)

    sv = sub.add_parser('serve', help='веб-интерфейс')
    sv.add_argument('--host', default='127.0.0.1')
    sv.add_argument('--port', type=int, default=7860)
    sv.add_argument('--open', action='store_true')
    sv.set_defaults(fn=cmd_serve)

    b = sub.add_parser('bench', help='замерить тензорные ядра и бэкенды внимания')
    b.add_argument('--tokens', type=int, default=16384)
    b.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
