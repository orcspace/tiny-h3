"""How much disk a converted model needs, and a free-space check before a long download / conversion starts."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import DIT, TEXT
from .planner import block_bytes, block_kinds, linear_bytes

GB = 1 << 30

# Parts that do not depend on the quantization policy (bytes).
FIXED = {
    'DWD-голова (кадры без VAE-декодера)': int(1.2 * GB),
    'refiner текста (2 блока int8)': int(2 * block_bytes('int8') + DIT.hidden_size * DIT.text_dim + 4e6),
    'аудио-VAE': int(0.61 * GB),
    'мелочь (globals, токенайзер, процессор)': int(0.15 * GB),
}
IMAGE_PARTS = {
    'VAE-энкодер (кадры, референсы-картинки)': int(0.72 * GB),
    'зрение Qwen3-VL (картинки в промпте)': int(0.8 * GB),
}
REF_PARTS = {'VAE-декодер fp16 (Ref2VA)': int(4.8 * GB), 'refiner Ref2VA': int(0.76 * GB)}


def text_bytes(kind: str) -> int:
    per_layer = sum(linear_bytes(kind, n, k) for n, k in TEXT.linear_shapes().values())
    return per_layer * TEXT.num_layers_used + TEXT.vocab_size * TEXT.hidden_size * 2  # + embedding table (bf16)


def estimate(policy: str = 'int8', text_kind: str = 'int8', *, images: bool = True, ref: bool = False) -> dict:
    """{'dit': bytes, 'text': bytes, 'other': bytes, 'total': bytes, 'parts': {name: bytes}}."""
    dit = sum(block_bytes(k) for k in block_kinds(policy))
    other = dict(FIXED)
    if images:
        other.update(IMAGE_PARTS)
    refs = {}
    if ref:
        refs = dict(REF_PARTS)
        refs['трансформер Ref2VA'] = dit
    text = text_bytes(text_kind)
    # the vision tower lives next to the text tower (text/visual.safetensors)
    text_side = text + (IMAGE_PARTS['зрение Qwen3-VL (картинки в промпте)'] if images else 0)
    other_main = sum(v for k, v in other.items() if k != 'зрение Qwen3-VL (картинки в промпте)')
    main = dit + other_main + sum(refs.values())
    return {'dit': dit, 'text': text_side, 'other': other_main, 'ref': sum(refs.values()),
            'main_drive': main, 'text_drive': text_side, 'total': main + text_side,
            'parts': {'DiT ' + policy: dit, 'текстовая башня ' + text_kind: text, **other, **refs}}


def free_bytes(path: Path) -> int:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def check_space(main_dir: Path, text_dir: Path | None, est: dict, margin: float = 1.05) -> list[str]:
    """Human-readable problems; an empty list means both drives have room."""
    problems = []
    need_main = est['main_drive'] if text_dir else est['total']
    if free_bytes(main_dir) < need_main * margin:
        problems.append(f'на диске с {main_dir} свободно {free_bytes(main_dir) / GB:.1f} ГБ, нужно ≈ '
                        f'{need_main * margin / GB:.1f} ГБ')
    if text_dir and free_bytes(text_dir) < est['text_drive'] * margin:
        problems.append(f'на диске с {text_dir} свободно {free_bytes(text_dir) / GB:.1f} ГБ, нужно ≈ '
                        f'{est["text_drive"] * margin / GB:.1f} ГБ')
    return problems


def link_dir(link: Path, target: Path) -> None:
    """Make ``link`` (inside the model) point at ``target`` (on another drive): a symlink, or a junction on Windows."""
    link, target = Path(link), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == target.resolve():
            return
        if link.is_dir() and not link.is_symlink() and not any(link.iterdir()):
            link.rmdir()
        else:
            raise FileExistsError(f'{link} уже существует и не является ссылкой на {target}')
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != 'nt':
            raise
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], check=True, capture_output=True)


def format_table(est: dict, text_dir: Path | None = None) -> str:
    lines = [f'  {name:<44}{size / GB:6.1f} ГБ' for name, size in est['parts'].items()]
    lines.append(f'  {"ИТОГО":<44}{est["total"] / GB:6.1f} ГБ')
    if text_dir:
        lines.append(f'  из них на основном диске{"":<20}{est["main_drive"] / GB:6.1f} ГБ')
        lines.append(f'  из них на {str(text_dir)[:30]:<35}{est["text_drive"] / GB:6.1f} ГБ')
    return '\n'.join(lines)
