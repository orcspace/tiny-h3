"""Interactive installer: ``python -m tinyh3 install`` (started by install.bat / install.sh).

Standard library only, so it runs before PyTorch is installed. It asks where to put things, checks disk space on every
drive involved, installs PyTorch (the CUDA build that matches the driver), the engine and the models, and writes the
launcher (start.bat / start.sh) that runs the local web server. Every step is resumable: a finished step is recorded
with a hash of its inputs in ``install-state.json``, so running the installer again after an error continues where it
stopped, and changing a choice redoes only what it affects.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import __version__, diskplan
from .diskplan import GB

PROFILES = {
    'compact': {'title': 'Компактный', 'policy': 'int4', 'text': 'int4',
                'about': 'минимум места: DiT и текстовая башня в int4 (качество int4 на настоящих весах не проверено)'},
    'balanced': {'title': 'Сбалансированный', 'policy': 'mixed', 'text': 'int4',
                 'about': 'края трансформера в int8, середина в int4: чуть аккуратнее, чуть больше места'},
    'max': {'title': 'Максимальное качество', 'policy': 'int8', 'text': 'int8',
            'about': 'int8 везде (~46 ГБ); часть блоков читается с диска на каждом шаге, нужен быстрый SSD'},
}
RUNTIME_GB = 7.0      # Python + PyTorch (CUDA) + libraries in the virtual environment
TEMP_GB = 3.0         # wheel cache while installing (removed at the end)
OUTPUT_GB = 2.0       # room for the first videos
SAFETY = 1.05

STEP_TITLES = {
    'prepare': 'Подготовка папок', 'venv': 'Python и окружение', 'torch': 'PyTorch (CUDA)', 'engine': 'Движок TinyH3',
    'models': 'Модели: скачивание и сжатие', 'quality4': 'Режим «Качество» (4 шага)', 'ref': 'Ref2VA: референсы',
    'verify': 'Проверка установки', 'launcher': 'Запуск и ярлыки', 'move-text': 'Перенос текстовой башни',
}
MODES = {
    'install': ['prepare', 'venv', 'torch', 'engine', 'models', 'quality4', 'ref', 'verify', 'launcher'],
    'repair': ['prepare', 'venv', 'torch', 'engine', 'models', 'verify', 'launcher'],
    'add-ref': ['ref', 'verify'],
    'add-quality': ['quality4', 'verify'],
    'move-text': ['move-text'],
}
WEIGHTS = {'prepare': 1, 'venv': 3, 'torch': 22, 'engine': 6, 'models': 60, 'quality4': 4, 'ref': 20, 'verify': 2,
           'launcher': 1, 'move-text': 10}
ALWAYS = {'verify', 'launcher', 'move-text'}           # cheap or action-like: run every time


@dataclass
class Choice:
    app_dir: str = ''
    models_dir: str = ''
    outputs_dir: str = ''
    text_dir: str = ''                 # '' = the text tower stays next to the model
    profile: str = 'compact'
    images: bool = True                # image prompts: VAE encoder + Qwen3-VL vision tower (+1.5 GB)
    ref: bool = False
    quality4: bool = False
    skip_models: bool = False
    python_spec: str = '3.12'
    system_site_packages: bool = False

    @property
    def policy(self) -> str:
        return PROFILES[self.profile]['policy']

    @property
    def text(self) -> str:
        return PROFILES[self.profile]['text']


# ------------------------------------------------------------------------------------------------------------ system

def detect_gpu(run=subprocess.run) -> dict:
    """{'present', 'name', 'vram_gb', 'driver', 'cuda'} from nvidia-smi (absent card or tool: present=False)."""
    none = {'present': False, 'name': None, 'vram_gb': None, 'driver': None, 'cuda': None}
    try:
        q = run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10, check=False)
        if q.returncode != 0:
            return none
        cards = []
        for line in q.stdout.splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3 and parts[-2].replace('.', '', 1).isdigit():
                cards.append({'name': ', '.join(parts[:-2]), 'vram_gb': round(float(parts[-2]) / 1024, 1),
                              'driver': parts[-1]})
        if not cards:
            return none
        best = max(cards, key=lambda c: c['vram_gb'])
        banner = run(['nvidia-smi'], capture_output=True, text=True, timeout=10, check=False).stdout
        m = re.search(r'CUDA Version:\s*([0-9]+\.[0-9]+)', banner)
        return {'present': True, **best, 'cuda': m.group(1) if m else None}
    except (OSError, subprocess.SubprocessError):
        return none


def cuda_tag(version: str | None) -> str | None:
    """PyTorch wheel index tag for the driver's CUDA version; None when the driver is too old."""
    m = re.match(r'(\d+)\.(\d+)', version or '')
    if not m:
        return 'cu128'
    v = int(m[1]) + int(m[2]) / 100
    return 'cu128' if v >= 12.08 else 'cu126' if v >= 12.06 else 'cu124' if v >= 12.04 else None


def total_ram_gb() -> float:
    try:
        if os.name == 'nt':
            class Status(ctypes.Structure):
                _fields_ = [('len', ctypes.c_ulong), ('load', ctypes.c_ulong), ('total', ctypes.c_ulonglong),
                            ('avail', ctypes.c_ulonglong), ('pt', ctypes.c_ulonglong), ('pa', ctypes.c_ulonglong),
                            ('vt', ctypes.c_ulonglong), ('va', ctypes.c_ulonglong), ('ve', ctypes.c_ulonglong)]
            st = Status()
            st.len = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.total / GB, 1)
        if Path('/proc/meminfo').exists():
            kb = int(re.search(r'MemTotal:\s+(\d+)', Path('/proc/meminfo').read_text())[1])
            return round(kb / 1024 / 1024, 1)
        return round(os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / GB, 1)
    except (OSError, ValueError, AttributeError, TypeError):
        return 0.0


def nearest_existing(p: str | Path) -> Path:
    cur = Path(p).resolve() if Path(p).exists() else Path(os.path.abspath(p))
    while not cur.exists() and cur != cur.parent:
        cur = cur.parent
    return cur


def free_gb(p) -> float:
    try:
        return round(shutil.disk_usage(nearest_existing(p)).free / GB, 2)
    except OSError:
        return 0.0


def device_of(p) -> str:
    try:
        return str(os.stat(nearest_existing(p)).st_dev)
    except OSError:
        return f'path:{p}'


def default_dirs(platform: str | None = None, env=None, home: Path | None = None) -> dict:
    platform = platform or sys.platform
    env = os.environ if env is None else env
    home = home or Path.home()
    if platform.startswith('win'):
        base = Path(env.get('LOCALAPPDATA') or home / 'AppData' / 'Local') / 'TinyH3'
        return {'app_dir': str(base), 'models_dir': str(base / 'models'), 'outputs_dir': str(home / 'Videos' / 'TinyH3')}
    if platform == 'darwin':
        base = home / 'Library' / 'Application Support' / 'TinyH3'
        return {'app_dir': str(base), 'models_dir': str(base / 'models'), 'outputs_dir': str(home / 'Movies' / 'TinyH3')}
    base = Path(env.get('XDG_DATA_HOME') or home / '.local' / 'share') / 'tinyh3'
    return {'app_dir': str(base), 'models_dir': str(base / 'models'), 'outputs_dir': str(home / 'Videos' / 'TinyH3')}


_FORBIDDEN_WIN = [re.compile(p, re.I) for p in (r'^[a-z]:\\windows(\\|$)', r'^[a-z]:\\program files( \(x86\))?(\\|$)',
                                                  r'^[a-z]:\\programdata(\\|$)')]
_FORBIDDEN_POSIX = ('/bin', '/boot', '/dev', '/etc', '/lib', '/lib64', '/proc', '/root', '/run', '/sbin', '/sys', '/usr',
                    '/var', '/System', '/Library', '/Applications')


def check_dir(p: str, platform: str | None = None) -> tuple[bool, str]:
    """(ok, reason): absolute, not a system folder or a drive root. Does not touch the disk."""
    platform = platform or sys.platform
    win = platform.startswith('win')
    if not p or not p.strip():
        return False, 'путь не указан'
    if '\0' in p:
        return False, 'недопустимые символы'
    import ntpath
    import posixpath
    mod = ntpath if win else posixpath
    if not mod.isabs(p):
        return False, 'нужен полный путь'
    norm = mod.normpath(p)
    if mod.splitdrive(norm)[1] in ('\\', '/', ''):
        return False, 'нельзя выбирать корень диска'
    if win:
        if norm.startswith('\\\\'):
            return False, 'сетевые пути не поддерживаются'
        if any(r.match(norm) for r in _FORBIDDEN_WIN):
            return False, 'системная папка'
        if len(norm) > 150:
            return False, 'слишком длинный путь (лимит Windows 260 знаков)'
    elif any(norm == f or norm.startswith(f + '/') for f in _FORBIDDEN_POSIX) and not norm.startswith(('/usr/local', '/var/tmp')):
        return False, 'системная папка'
    return True, ''


def ensure_writable(p: str) -> tuple[bool, str]:
    ok, why = check_dir(p)
    if not ok:
        return False, why
    try:
        Path(p).mkdir(parents=True, exist_ok=True)
        probe = Path(p) / f'.write-test-{os.getpid()}'
        probe.write_text('x')
        probe.unlink()
        return True, ''
    except OSError as e:
        return False, f'нет доступа на запись ({e.strerror or e})'


# ---------------------------------------------------------------------------------------------------------- the plan

def compute(choice: Choice) -> dict:
    """Sizes in GB: {'models', 'text', 'app', 'parts': [(name, gb)]}. ``text`` is the part that goes to text_dir."""
    est = diskplan.estimate(choice.policy, choice.text, images=choice.images, ref=choice.ref)
    models = est['main_drive'] / GB
    text = est['text_drive'] / GB
    parts = [] if choice.skip_models else [(n, v / GB) for n, v in est['parts'].items()]
    if choice.skip_models:
        models = text = 0.0
    elif choice.quality4:
        extra = (est['dit'] + est['other']) / GB              # a second DiT + decoders; shares the text tower
        models += extra
        parts.append(('второй DiT для режима «Качество»', extra))
    parts += [('Python, PyTorch и библиотеки', RUNTIME_GB), ('временные файлы установки (удаляются)', TEMP_GB)]
    if not choice.text_dir:
        models, text = models + text, 0.0
    return {'models': models, 'text': text, 'app': RUNTIME_GB + TEMP_GB, 'parts': parts}


def requirements(choice: Choice, device=None) -> list[dict]:
    """Disk needed per physical drive (paths on one drive add up)."""
    device = device or device_of
    c = compute(choice)
    need: dict[str, dict] = {}

    def bump(path: str, gb: float, what: str):
        if not path or gb <= 0:
            return
        e = need.setdefault(device(path), {'dev': device(path), 'need_gb': 0.0, 'path': path, 'items': []})
        e['need_gb'] += gb
        e['items'].append(what)

    bump(choice.app_dir, c['app'], 'приложение')
    bump(choice.models_dir, c['models'], 'модели')
    bump(choice.text_dir, c['text'], 'текстовая башня')
    bump(choice.outputs_dir, OUTPUT_GB, 'видео')
    for e in need.values():
        e['need_gb'] = round(e['need_gb'] * SAFETY, 1)
    return list(need.values())


def validate(choice: Choice, gpu: dict, ram_gb: float, platform: str | None = None, free=None,
             device=None) -> list[tuple[str, str]]:
    """[(level, message)] with level 'error' (blocks) or 'warn'."""
    free = free or free_gb
    device = device or device_of
    out: list[tuple[str, str]] = []
    for label, p in (('приложения', choice.app_dir), ('моделей', choice.models_dir), ('видео', choice.outputs_dir)):
        ok, why = check_dir(p, platform)
        if not ok:
            out.append(('error', f'Папка {label}: {why}'))
    if choice.text_dir:
        ok, why = check_dir(choice.text_dir, platform)
        if not ok:
            out.append(('error', f'Папка текстовой башни: {why}'))
    for e in requirements(choice, device):
        have = free(e['path'])
        if have < e['need_gb']:
            out.append(('error', f'Не хватает места ({", ".join(e["items"])}): нужно ≈ {e["need_gb"]} ГБ, свободно {have} ГБ '
                                 f'(диск с {e["path"]})'))
    if not gpu.get('present'):
        out.append(('warn', 'Видеокарта NVIDIA не найдена: установка возможна, но генерировать видео без неё практически нельзя.'))
    else:
        if gpu.get('vram_gb') and gpu['vram_gb'] < 6:
            out.append(('error', f'Видеопамяти {gpu["vram_gb"]} ГБ: для H3 нужно не меньше 6 ГБ (рекомендуется 8).'))
        if gpu.get('cuda') and cuda_tag(gpu['cuda']) is None:
            out.append(('error', f'Драйвер NVIDIA слишком старый (CUDA {gpu["cuda"]}): обновите его до версии с CUDA 12.4 или новее.'))
    if ram_gb and ram_gb < 12:
        out.append(('warn', f'ОЗУ {ram_gb} ГБ: рекомендуется 16 ГБ, иначе больше блоков будет читаться с диска.'))
    if (platform or sys.platform) == 'darwin':
        out.append(('warn', 'macOS: движок рассчитан на видеокарты NVIDIA (CUDA): установка возможна, генерация недоступна.'))
    if choice.profile == 'max':
        out.append(('warn', 'Профиль «Максимум»: нужен быстрый NVMe SSD, часть блоков читается с диска на каждом шаге.'))
    return out


def steps_for(mode: str, choice: Choice) -> list[str]:
    ids = list(MODES[mode])
    if mode in ('install', 'repair'):
        if choice.skip_models:
            ids = [i for i in ids if i not in ('models', 'quality4', 'ref')]
        if not choice.quality4:
            ids = [i for i in ids if i != 'quality4']
        if not choice.ref:
            ids = [i for i in ids if i != 'ref']
    return ids


def model_name(steps: int, policy: str) -> str:
    return f'h3-pi{steps}-dwd-{policy}'


# --------------------------------------------------------------------------------------------------------- launchers

def engine_env(c: Choice, base=None, **extra) -> dict:
    app = Path(c.app_dir)
    env = dict(os.environ if base is None else base)
    env.update({'TINYH3_HOME': str(app / 'data'), 'TINYH3_MODELS': c.models_dir, 'TINYH3_OUTPUTS': c.outputs_dir,
                'TINYH3_CACHE': str(app / 'cache'), 'HF_HOME': str(app / 'cache' / 'hf'), 'HF_HUB_DISABLE_TELEMETRY': '1',
                'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'})
    env.update(extra)
    for k in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'):
        env.pop(k, None)
    return env


def venv_python(app_dir: str, platform: str | None = None) -> Path:
    return Path(app_dir) / 'venv' / ('Scripts/python.exe' if (platform or sys.platform).startswith('win') else 'bin/python')


def render_launchers(c: Choice, platform: str | None = None) -> dict[str, str]:
    """{filename: content} of the scripts written into the app folder."""
    win = (platform or sys.platform).startswith('win')
    app = Path(c.app_dir)
    py = venv_python(c.app_dir, platform)
    env_vars = {'TINYH3_HOME': str(app / 'data'), 'TINYH3_MODELS': c.models_dir, 'TINYH3_OUTPUTS': c.outputs_dir,
                'TINYH3_CACHE': str(app / 'cache'), 'HF_HOME': str(app / 'cache' / 'hf'), 'PYTHONUTF8': '1',
                'PYTHONIOENCODING': 'utf-8'}
    if win:
        def esc(s: str) -> str:
            return str(s).replace('%', '%%')
        sets = '\n'.join(f'set "{k}={esc(v)}"' for k, v in env_vars.items())
        head = '@echo off\nchcp 65001 >nul\nsetlocal\n' + sets + '\n'
        start = (head + 'title TinyH3 Video\necho.\necho  TinyH3 Video: запуск локального сервера.\n'
                 'echo  Браузер откроется сам. Чтобы остановить, закройте это окно.\necho.\n'
                 f'"{esc(py)}" -m tinyh3 serve --open\necho.\necho  Сервер остановлен. Нажмите любую клавишу.\npause >nul\n')
        console = (head + 'title TinyH3 Video: консоль\n'
                   f'set "PATH={esc(py.parent)};%PATH%"\necho  Команды: tinyh3 doctor, tinyh3 plan, tinyh3 generate ...\n'
                   'cmd /k\n')
        rm = [f'rmdir /s /q "{esc(c.models_dir)}"']
        text_rm = f'rmdir /s /q "{esc(c.text_dir)}"' if c.text_dir else ''
        uninstall = ('@echo off\nchcp 65001 >nul\nsetlocal\ncd /d "%TEMP%"\n'
                     'echo  Будет удалено:\necho    приложение: ' + esc(c.app_dir) + '\necho    модели:     ' + esc(c.models_dir) + '\n'
                     + (f'echo    текстовая башня: {esc(c.text_dir)}\n' if c.text_dir else '') +
                     f'echo    (папка с видео {esc(c.outputs_dir)} не удаляется)\necho.\n'
                     'set /p ok=Удалить всё это? Введите YES: \nif /i not "%ok%"=="YES" exit /b 1\n'
                     + (text_rm + '\n' if text_rm else '') + rm[0] + '\n'
                     f'rmdir /s /q "{esc(c.app_dir)}"\necho  Готово.\npause\n')
        return {'start.bat': start, 'console.bat': console, 'uninstall.bat': uninstall}

    def q(s) -> str:
        return "'" + str(s).replace("'", "'\\''") + "'"
    exports = '\n'.join(f'export {k}={q(v)}' for k, v in env_vars.items())
    start = (f'#!/bin/sh\n{exports}\necho "TinyH3 Video: запуск локального сервера (Ctrl+C — остановить)"\n'
             f'exec {q(py)} -m tinyh3 serve --open\n')
    console = f'#!/bin/sh\n{exports}\nexport PATH={q(py.parent)}:"$PATH"\nexec "${{SHELL:-/bin/sh}}"\n'
    return {'start.sh': start, 'console.sh': console}


# ---------------------------------------------------------------------------------------------------------- running

class Bar:
    """A text progress bar that also works when the output is a log file (prints on every 10%)."""

    def __init__(self, out=print, tty: bool | None = None):
        self.out = out
        self.tty = sys.stdout.isatty() if tty is None else tty
        self.last_decile = -1
        self.active = False

    def update(self, fraction: float | None, text: str = ''):
        text = text[:70]
        if fraction is None:
            if self.tty:
                sys.stdout.write(f'\r  {"." * 3} {text:<70}')
                sys.stdout.flush()
                self.active = True
            return
        f = max(0.0, min(1.0, fraction))
        if self.tty:
            filled = int(30 * f)
            sys.stdout.write(f'\r  [{"#" * filled}{"-" * (30 - filled)}] {f * 100:5.1f}%  {text:<60}')
            sys.stdout.flush()
            self.active = True
        elif int(f * 10) != self.last_decile:
            self.last_decile = int(f * 10)
            self.out(f'  {f * 100:3.0f}%  {text}')

    def done(self):
        if self.tty and self.active:
            sys.stdout.write('\n')
        self.active = False
        self.last_decile = -1


class StepError(Exception):
    def __init__(self, message: str, tail: list[str] | None = None):
        super().__init__(message)
        self.tail = tail or []


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/pid', str(proc.pid), '/T', '/F'], capture_output=True, check=False)
        else:
            proc.terminate()
    except OSError:
        pass


def run_stream(cmd: list[str], env: dict, on_line, cwd: str | None = None) -> tuple[int, list[str]]:
    """Run a command, feeding every output line (stdout+stderr, progress-bar redraws included) to ``on_line``."""
    tail: list[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                            errors='replace', env=env, cwd=cwd,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        for raw in proc.stdout:
            line = raw.rstrip('\r\n')
            if not line.strip():
                continue
            tail.append(line)
            del tail[:-40]
            on_line(line)
        proc.wait()
    except KeyboardInterrupt:
        kill_tree(proc)
        raise
    return proc.returncode, tail


class Installer:
    def __init__(self, choice: Choice, *, mode: str = 'install', uv: str | None = None, source: str | None = None,
                 gpu: dict | None = None, platform: str | None = None, runner=run_stream, out=print, tty: bool | None = None,
                 python: str | None = None, force: bool = False, move_text_dest: str = '', shortcut: bool = True):
        self.c, self.mode, self.force = choice, mode, force
        self.platform = platform or sys.platform
        self.win = self.platform.startswith('win')
        self.gpu = gpu or {'present': False}
        self.run_cmd, self.out, self.bar = runner, out, Bar(out, tty)
        self.app = Path(choice.app_dir)
        self.py = Path(python) if python else venv_python(choice.app_dir, self.platform)
        self.uv = uv or os.environ.get('TINYH3_UV') or self._find_uv()
        self.source = source or self._find_source()
        self.move_text_dest, self.shortcut = move_text_dest, shortcut
        self.steps = steps_for(mode, choice)
        self.report: dict | None = None
        self.log_path: Path | None = None
        self._log_file = None
        self.state_path = self.app / 'install-state.json'
        self.overall = 0.0

    # -- environment lookups -----------------------------------------------------------------------------------
    def _find_uv(self) -> str | None:
        local = self.app / 'bin' / ('uv.exe' if self.win else 'uv')
        if local.exists():
            return str(local)
        return shutil.which('uv')

    @staticmethod
    def _find_source() -> str | None:
        root = Path(__file__).resolve().parents[1]
        return str(root) if (root / 'pyproject.toml').exists() else None

    # -- logging / progress ------------------------------------------------------------------------------------
    def log(self, line: str) -> None:
        if self._log_file is None and self.app.exists():
            (self.app / 'logs').mkdir(parents=True, exist_ok=True)
            self.log_path = self.app / 'logs' / f'install-{time.strftime("%Y%m%d-%H%M%S")}.log'
            self._log_file = open(self.log_path, 'a', encoding='utf-8')
        if self._log_file:
            self._log_file.write(line + '\n')
            self._log_file.flush()

    def _progress(self, step: str, fraction: float | None, text: str = '') -> None:
        weights = [WEIGHTS.get(s, 1) for s in self.steps]
        i = self.steps.index(step)
        f = 0.0 if fraction is None else max(0.0, min(1.0, fraction))
        overall = (sum(weights[:i]) + weights[i] * f) / sum(weights)
        self.overall = max(self.overall, overall)
        self.bar.update(None if fraction is None else fraction, f'{text}' if text else '')

    def _exec(self, step: str, cmd: list[str], env: dict, on_line=None, allow_fail: bool = False, cwd=None):
        self.log('$ ' + ' '.join(str(x) for x in cmd))

        def handle(line: str):
            self.log(line)
            if on_line:
                on_line(line)
        code, tail = self.run_cmd([str(x) for x in cmd], env, handle, cwd)
        if code != 0 and not allow_fail:
            raise StepError(f'команда завершилась с кодом {code}: {Path(str(cmd[0])).name} {" ".join(map(str, cmd[1:4]))}', tail)
        return code, tail

    def _uv_env(self) -> dict:
        env = engine_env(self.c)
        env.update({'UV_CACHE_DIR': str(self.app / '.uv-cache'), 'UV_PYTHON_INSTALL_DIR': str(self.app / 'python'),
                    'UV_NO_PROGRESS': '1', 'UV_NO_CONFIG': '1'})
        return env

    # -- state -------------------------------------------------------------------------------------------------
    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return {'done': {}}

    def _save_state(self, st: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(st, indent=2), encoding='utf-8')

    def _inputs(self, step: str) -> str:
        c = self.c
        tag = cuda_tag(self.gpu.get('cuda')) if self.gpu.get('present') else 'cpu'
        table = {
            'prepare': [c.app_dir, c.models_dir, c.outputs_dir, c.text_dir],
            'venv': [c.python_spec, c.system_site_packages],
            'torch': [tag, self.platform],
            'engine': [__version__, self.source],
            'models': [c.policy, c.text, c.images, c.text_dir, c.models_dir],
            'quality4': [c.policy, c.text, c.images, c.text_dir, c.models_dir],
            'ref': [c.policy, c.models_dir],
        }
        return hashlib.sha1(json.dumps(table.get(step, []), sort_keys=True).encode()).hexdigest()

    # -- the run -----------------------------------------------------------------------------------------------
    def run(self) -> bool:
        ok, why = ensure_writable(self.c.app_dir)
        if not ok:
            self.out(f'  Папка приложения: {why}')
            return False
        st = self._load_state()
        total = len(self.steps)
        try:
            for n, step in enumerate(self.steps, 1):
                title = STEP_TITLES[step]
                inputs = self._inputs(step)
                rec = st['done'].get(step)
                if rec and rec.get('inputs') == inputs and not (self.force and step != 'prepare') and step not in ALWAYS:
                    self.out(f'[{n}/{total}] {title}: уже выполнено')
                    continue
                self.out(f'[{n}/{total}] {title}')
                self._progress(step, 0.0)
                getattr(self, 'step_' + step.replace('-', '_'))()
                self.bar.done()
                st['done'][step] = {'inputs': inputs, 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
                self._save_state(st)
        except StepError as e:
            self.bar.done()
            self.log(f'ОШИБКА: {e}')
            self.out(f'\n  ОШИБКА на шаге «{STEP_TITLES[step]}»: {e}')
            for line in e.tail[-12:]:
                self.out(f'    {line[:160]}')
            self.out(f'\n  Подробности: {self.log_path}\n  Запустите установщик снова: он продолжит с этого шага.')
            return False
        except KeyboardInterrupt:
            self.bar.done()
            self.out('\n  Прервано. Запустите установщик снова: он продолжит с этого шага.')
            return False
        finally:
            if self._log_file:
                self._log_file.close()
                self._log_file = None
        return True

    # -- steps -------------------------------------------------------------------------------------------------
    def step_prepare(self):
        c = self.c
        for d in (c.models_dir, c.outputs_dir, c.text_dir):
            if d:
                ok, why = ensure_writable(d)
                if not ok:
                    raise StepError(f'{d}: {why}')
        for sub in ('bin', 'logs', 'data', 'cache', 'cache/sources'):
            (self.app / sub).mkdir(parents=True, exist_ok=True)
        local = self.app / 'bin' / ('uv.exe' if self.win else 'uv')
        if self.uv and Path(self.uv).is_file() and not local.exists():      # keep uv for «Починить» later
            try:
                shutil.copy2(self.uv, local)
                self.uv = str(local)
            except OSError:
                pass
        self.log(f'Папки: приложение {c.app_dir}; модели {c.models_dir}; видео {c.outputs_dir}; текст {c.text_dir or "—"}')

    def step_venv(self):
        if self.py.exists():
            self.log(f'окружение уже есть: {self.py}')
            return
        if not self.uv:
            raise StepError('не найден uv (менеджер пакетов): запустите install.bat / install.sh, он скачает его')
        spec = self.c.python_spec
        env = self._uv_env()
        managed = not re.search(r'[\\/]', spec)
        if managed:
            self._exec('venv', [self.uv, 'python', 'install', spec], env, lambda l: self._progress('venv', 0.4, l))
        args = [self.uv, 'venv', str(self.app / 'venv'), '--python', spec, '--allow-existing']
        if managed:
            args += ['--python-preference', 'only-managed']
        if self.c.system_site_packages:
            args.append('--system-site-packages')
        self._exec('venv', args, env)

    def step_torch(self):
        tag = cuda_tag(self.gpu.get('cuda')) if self.gpu.get('present') else 'cpu'
        if self.gpu.get('present') and self.gpu.get('cuda') and tag is None:
            raise StepError('драйвер NVIDIA слишком старый (нужна поддержка CUDA 12.4+): обновите драйвер и повторите')
        args = [self.uv, 'pip', 'install', '--python', self.py, 'torch', 'torchvision']
        if self.platform != 'darwin':
            args += ['--index', f'https://download.pytorch.org/whl/{tag or "cpu"}']
        self.out(f'  PyTorch: {tag or "cpu"}' + (f' (CUDA драйвера {self.gpu.get("cuda")})' if self.gpu.get('present') else ' (видеокарта не найдена)'))
        started = time.time()
        self._exec('torch', args, self._uv_env(),
                   lambda l: self._progress('torch', None, f'{l[:55]}  {int(time.time() - started)} с'))

    def step_engine(self):
        if not self.source:
            raise StepError('не найден каталог с исходниками TinyH3 (pyproject.toml)')
        env = self._uv_env()
        self._exec('engine', [self.uv, 'pip', 'install', '--python', self.py, self.source], env,
                   lambda l: self._progress('engine', None, l))
        wheels = Path(self.source) / 'wheels'
        tag = 'win' if self.win else 'macosx' if self.platform == 'darwin' else 'manylinux'
        found = sorted(wheels.glob('tinyh3_core*.whl')) if wheels.exists() else []
        found = [w for w in found if tag in w.name]
        if found:
            self._exec('engine', [self.uv, 'pip', 'install', '--python', self.py, found[0]], env, allow_fail=True)
        else:
            self.log('Rust-ядра нет: используются запасные пути на Python (медленнее чтение весов и апскейл)')
        if self.win:
            code, _ = self._exec('engine', [self.uv, 'pip', 'install', '--python', self.py, 'triton-windows'], env,
                                 allow_fail=True)
            if code != 0:
                self.out('  triton-windows не установился: ядра будут работать через запасные пути PyTorch (медленнее)')

    def _engine_run(self, step: str, args: list[str], on_event) -> None:
        env = engine_env(self.c, TINYH3_PROGRESS='json')

        def handle(line: str):
            if line.startswith('{'):
                try:
                    on_event(json.loads(line))
                    return
                except ValueError:
                    pass
            if re.search(r'\d{1,3}%\|', line):
                self._progress(step, None, line)
        self._exec(step, [self.py, '-m', 'tinyh3'] + args, env, handle, cwd=str(self.app))

    def _setup_events(self, step: str):
        frac = {'dit': 0.0, 'text': 0.0, 'vae': 0.0}
        names = {'dit': 'трансформер', 'text': 'текстовая башня', 'dit_ref': 'Ref2VA'}

        def on_event(e: dict):
            if e.get('event') == 'progress':
                if e['stage'] in frac:
                    frac[e['stage']] = e['i'] / e['n'] if e.get('n') else 0.0
                f = min(0.99, 0.35 * frac['dit'] + 0.55 * frac['text'] + 0.10 * frac['vae'])
                self._progress(step, f, f'{names.get(e["stage"], e["stage"])}: {e["i"]}/{e["n"]}')
            elif e.get('event') == 'phase':
                if e.get('stage') == 'vae':
                    frac['vae'] = 0.5
                self._progress(step, None, e.get('message', ''))
        return on_event

    def _setup_args(self, steps: int, text_dir: str) -> list[str]:
        c = self.c
        a = ['setup', '--low-disk', '--policy', c.policy, '--text', c.text, '--steps', str(steps),
             '--sources', str(self.app / 'cache' / 'sources')]
        if not c.images:
            a.append('--no-images')
        if text_dir:
            a += ['--text-dir', text_dir]
        return a

    def step_models(self):
        self._engine_run('models', self._setup_args(2, self.c.text_dir), self._setup_events('models'))

    def step_quality4(self):
        first_text = self.c.text_dir or str(Path(self.c.models_dir) / model_name(2, self.c.policy) / 'text')
        self._engine_run('quality4', self._setup_args(4, first_text), self._setup_events('quality4'))

    def step_ref(self):
        args = ['setup-ref', '--low-disk', '--policy', self.c.policy, '--sources', str(self.app / 'cache' / 'sources')]
        self._engine_run('ref', args, self._setup_events('ref'))

    def step_move_text(self):
        if not self.move_text_dest:
            raise StepError('не указана папка назначения')
        ok, why = ensure_writable(self.move_text_dest)
        if not ok:
            raise StepError(f'{self.move_text_dest}: {why}')
        self._engine_run('move-text', ['move-text', self.move_text_dest], lambda e: None)

    def step_verify(self):
        info = None

        def handle(line: str):
            nonlocal info
            if line.startswith('{'):
                try:
                    info = json.loads(line)
                except ValueError:
                    pass
        code, tail = self._exec('verify', [self.py, '-m', 'tinyh3', 'doctor', '--json'], engine_env(self.c), handle,
                                allow_fail=True, cwd=str(self.app))
        if not info:
            raise StepError('проверка не вернула отчёт: движок не запускается', tail)
        warnings = []
        if not info.get('cuda_available'):
            warnings.append('PyTorch не видит CUDA: без видеокарты NVIDIA генерация недоступна')
        st = info.get('selftest')
        if st and not st.get('triton_ok'):
            warnings.append(f'ядра Triton не прошли самопроверку ({st.get("error") or "ошибка"}): будут запасные пути (медленнее)')
        info['warnings'] = warnings
        self.report = info
        gpu = (info.get('hardware') or {}).get('gpu_name')
        self.out(f'  GPU: {gpu or "—"} · PyTorch {info.get("torch")} · внимание: {", ".join(info.get("attention", []))}'
                 f' · Rust-ядро: {info.get("native") or "нет"}')
        for w in warnings:
            self.out(f'  ! {w}')

    def step_launcher(self):
        files = render_launchers(self.c, self.platform)
        for name, content in files.items():
            path = self.app / name
            path.write_text(content, encoding='utf-8', newline='\r\n' if self.win else '\n')
            if not self.win:
                path.chmod(0o755)
        (self.app / 'install.json').write_text(json.dumps({
            'version': __version__, 'choice': asdict(self.c), 'installed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'report': {'warnings': (self.report or {}).get('warnings', []),
                       'models': (self.report or {}).get('models', [])}}, indent=2, ensure_ascii=False), encoding='utf-8')
        shutil.rmtree(self.app / '.uv-cache', ignore_errors=True)         # wheel cache: gigabytes nobody needs again
        if self.shortcut:
            make_shortcut(self.app, self.win, self.platform)
        self.out(f'  Запуск: {self.app / ("start.bat" if self.win else "start.sh")}')


def make_shortcut(app: Path, win: bool, platform: str) -> None:
    """Best effort: a desktop shortcut (Windows .lnk) or a menu entry (Linux .desktop)."""
    try:
        if win:
            desktop = Path(os.environ.get('USERPROFILE', str(Path.home()))) / 'Desktop'
            ps = ('$s=(New-Object -ComObject WScript.Shell).CreateShortcut($args[0]);$s.TargetPath=$args[1];'
                  '$s.WorkingDirectory=$args[2];$s.Description="TinyH3 Video";$s.Save()')
            subprocess.run(['powershell', '-NoProfile', '-Command', ps, str(desktop / 'TinyH3 Video.lnk'),
                            str(app / 'start.bat'), str(app)], capture_output=True, check=False, timeout=30)
        elif platform.startswith('linux'):
            d = Path.home() / '.local' / 'share' / 'applications'
            d.mkdir(parents=True, exist_ok=True)
            (d / 'tinyh3-video.desktop').write_text(
                f'[Desktop Entry]\nType=Application\nName=TinyH3 Video\nExec={app / "start.sh"}\nTerminal=true\n'
                'Categories=AudioVideo;\n', encoding='utf-8')
    except (OSError, subprocess.SubprocessError):
        pass


# ------------------------------------------------------------------------------------------------------ interaction

def pick_folder(title: str = 'Выберите папку') -> str | None:
    """A native folder dialog where one exists (Windows PowerShell, macOS osascript, Linux zenity); else None."""
    try:
        if os.name == 'nt':
            ps = ('Add-Type -AssemblyName System.Windows.Forms;$f=New-Object System.Windows.Forms.FolderBrowserDialog;'
                  '$f.Description=$args[0];$f.ShowNewFolderButton=$true;'
                  'if($f.ShowDialog() -eq "OK"){[Console]::OutputEncoding=[Text.Encoding]::UTF8;Write-Output $f.SelectedPath}')
            r = subprocess.run(['powershell', '-NoProfile', '-STA', '-Command', ps, title], capture_output=True, text=True,
                               encoding='utf-8', timeout=600, check=False)
        elif sys.platform == 'darwin':
            r = subprocess.run(['osascript', '-e', f'POSIX path of (choose folder with prompt "{title}")'],
                               capture_output=True, text=True, timeout=600, check=False)
        elif shutil.which('zenity'):
            r = subprocess.run(['zenity', '--file-selection', '--directory', f'--title={title}'], capture_output=True,
                               text=True, timeout=600, check=False)
        else:
            return None
        path = r.stdout.strip()
        return path or None
    except (OSError, subprocess.SubprocessError):
        return None


class Prompts:
    """Console questions; ``inp`` is injectable so the wizard can be tested with scripted answers."""

    def __init__(self, inp=input, out=print, picker=pick_folder):
        self.inp, self.out, self.picker = inp, out, picker

    def ask(self, text: str, default: str = '') -> str:
        shown = f' [{default}]' if default else ''
        ans = self.inp(f'{text}{shown}: ').strip()
        return ans or default

    def yes(self, text: str, default: bool) -> bool:
        ans = self.inp(f'{text} [{"Д/н" if default else "д/Н"}]: ').strip().lower()
        if not ans:
            return default
        return ans in ('д', 'да', 'y', 'yes', '1')

    def choose(self, text: str, options: list[tuple[str, str]], default: str) -> str:
        for key, label in options:
            self.out(f'    {key}) {label}')
        keys = {k for k, _ in options}
        while True:
            ans = self.ask(text, default)
            if ans in keys:
                return ans
            self.out('    Введите один из вариантов.')

    def directory(self, text: str, default: str, allow_empty: bool = False) -> str:
        """Type a path, press Enter for the default, or type ? for a folder dialog."""
        while True:
            hint = '  (Enter — по умолчанию, ? — окно выбора' + (', - — не использовать' if allow_empty else '') + ')'
            self.out(f'  {text}{hint}')
            ans = self.ask('  Путь', default).strip().strip('"')
            if allow_empty and ans in ('-', ''):
                return ''
            if ans == '?':
                picked = self.picker(text)
                if not picked:
                    self.out('    Окно выбора недоступно или закрыто: введите путь вручную.')
                    continue
                ans = picked
            ok, why = check_dir(ans)
            if ok:
                return os.path.normpath(ans)
            self.out(f'    {why}')


def describe_system(gpu: dict, ram: float, out=print) -> None:
    out(f'  Видеокарта: {gpu["name"]}, {gpu["vram_gb"]} ГБ, драйвер {gpu["driver"]} (CUDA {gpu["cuda"] or "?"})' if gpu.get('present')
        else '  Видеокарта NVIDIA: не найдена')
    out(f'  ОЗУ: {ram} ГБ   ОС: {sys.platform}')


def show_plan(choice: Choice, problems: list[tuple[str, str]], out=print) -> None:
    c = compute(choice)
    out('\n  Что будет установлено:')
    for name, gb in c['parts']:
        out(f'    {name:<46}{gb:6.1f} ГБ')
    out('\n  Куда:')
    out(f'    приложение  {choice.app_dir}\n    модели      {choice.models_dir}')
    if choice.text_dir:
        out(f'    текст. башня {choice.text_dir}')
    out(f'    видео       {choice.outputs_dir}')
    for e in requirements(choice):
        out(f'    диск с {e["path"]}: нужно ≈ {e["need_gb"]} ГБ, свободно {free_gb(e["path"])} ГБ')
    for level, msg in problems:
        out(f'  {"ОШИБКА" if level == "error" else "Внимание"}: {msg}')


def interactive_choice(base: Choice, p: Prompts, gpu: dict, ram: float, fixed_app: bool) -> Choice:
    """Ask for everything that is not already set. Returns the final choice."""
    c = base
    p.out('\nTinyH3 Video: установка')
    describe_system(gpu, ram, p.out)
    p.out('\n  Папки. Модели занимают от 13 до 46 ГБ: выберите диск с достаточным местом.')
    if not fixed_app:
        c.app_dir = p.directory('Приложение (Python, библиотеки, логи; ~10 ГБ)', c.app_dir)
    c.models_dir = p.directory('Модели', c.models_dir if c.models_dir else str(Path(c.app_dir) / 'models'))
    c.text_dir = p.directory('Текстовая башня (13-24 ГБ) на ДРУГОМ диске, если на основном мало места',
                             c.text_dir, allow_empty=True)
    c.outputs_dir = p.directory('Готовые видео', c.outputs_dir)
    p.out('\n  Профиль качества и размера:')
    default = c.profile if c.profile in PROFILES else 'compact'
    keys = {'1': 'compact', '2': 'balanced', '3': 'max'}
    for k, name in keys.items():
        prof = PROFILES[name]
        est = compute(Choice(**{**asdict(c), 'profile': name, 'text_dir': '', 'skip_models': False, 'ref': False,
                                'quality4': False}))
        p.out(f'    {k}) {prof["title"]} (~{est["models"]:.0f} ГБ): {prof["about"]}')
    pick = p.ask('  Профиль', next(k for k, v in keys.items() if v == default))
    c.profile = keys.get(pick, default)
    c.images = p.yes('  Картинки в промпте (изображение → видео, референсы; +1,5 ГБ)', c.images)
    c.ref = p.yes('  Ref2VA (референсы → видео; +15-24 ГБ, свой трансформер)', c.ref)
    c.quality4 = p.yes('  Режим «Качество» (4 шага; ещё один трансформер, +9-18 ГБ)', c.quality4)
    c.skip_models = not p.yes('  Скачать модели сейчас (иначе позже: install.bat → «Починить»)', not c.skip_models)
    return c


def existing_install(app_dir: str) -> dict | None:
    f = Path(app_dir) / 'install.json'
    try:
        return json.loads(f.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def launch(app: Path, win: bool) -> None:
    script = app / ('start.bat' if win else 'start.sh')
    if win:
        subprocess.Popen(['cmd', '/c', 'start', '', str(script)], cwd=str(app))
    else:
        subprocess.Popen([str(script)], cwd=str(app), start_new_session=True)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog='tinyh3 install', description='Установщик TinyH3 Video')
    d = default_dirs()
    ap.add_argument('--app-dir', default=None, help=f'папка приложения (по умолчанию {d["app_dir"]})')
    ap.add_argument('--models-dir')
    ap.add_argument('--outputs-dir')
    ap.add_argument('--text-dir', help='текстовая башня на другом диске')
    ap.add_argument('--profile', choices=list(PROFILES), default=None)
    ap.add_argument('--no-images', action='store_true')
    ap.add_argument('--ref', action='store_true')
    ap.add_argument('--quality4', action='store_true')
    ap.add_argument('--skip-models', action='store_true', help='установить только приложение, модели позже')
    ap.add_argument('--mode', choices=list(MODES), default=None)
    ap.add_argument('--move-text-to', default='')
    ap.add_argument('--yes', '-y', action='store_true', help='без вопросов: брать значения по умолчанию и аргументы')
    ap.add_argument('--uv', help='путь к uv')
    ap.add_argument('--source', help='каталог с исходниками TinyH3 (pyproject.toml)')
    ap.add_argument('--python-spec', default='3.12')
    ap.add_argument('--system-site-packages', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--no-shortcut', action='store_true')
    ap.add_argument('--no-launch', action='store_true')
    a = ap.parse_args(argv)

    win = sys.platform.startswith('win')
    out = print
    p = Prompts()
    gpu = detect_gpu()
    ram = total_ram_gb()
    app_dir = a.app_dir or d['app_dir']
    prev = existing_install(app_dir)
    mode = a.mode or 'install'

    if prev and not a.yes and not a.mode:
        c0 = Choice(**{k: v for k, v in prev['choice'].items() if k in Choice.__dataclass_fields__})
        out(f'\nНайдена установка TinyH3 {prev.get("version", "?")} в {app_dir}')
        sel = p.choose('Что сделать?', [('1', 'Запустить TinyH3'), ('2', 'Починить / обновить'), ('3', 'Добавить Ref2VA'),
                                        ('4', 'Добавить режим «Качество» (4 шага)'), ('5', 'Перенести текстовую башню на другой диск'),
                                        ('0', 'Выход')], '1')
        if sel == '0':
            return 0
        if sel == '1':
            launch(Path(app_dir), win)
            return 0
        mode = {'2': 'repair', '3': 'add-ref', '4': 'add-quality', '5': 'move-text'}[sel]
        if sel == '5':
            a.move_text_to = p.directory('Куда перенести текстовую башню (папка на другом диске)', '')
        choice = c0
        choice.ref = choice.ref or sel == '3'
        choice.quality4 = choice.quality4 or sel == '4'
    else:
        choice = Choice(app_dir=app_dir, models_dir=a.models_dir or str(Path(app_dir) / 'models'),
                        outputs_dir=a.outputs_dir or d['outputs_dir'], text_dir=a.text_dir or '',
                        profile=a.profile or 'compact', images=not a.no_images, ref=a.ref, quality4=a.quality4,
                        skip_models=a.skip_models, python_spec=a.python_spec, system_site_packages=a.system_site_packages)
        if prev and a.mode:
            choice = Choice(**{**asdict(choice), **{k: v for k, v in prev['choice'].items() if k in Choice.__dataclass_fields__ and not getattr(a, k, None)}})
        if not a.yes and mode == 'install':
            choice = interactive_choice(choice, p, gpu, ram, fixed_app=bool(a.app_dir))
        elif mode == 'add-ref':
            choice.ref = True
        elif mode == 'add-quality':
            choice.quality4 = True

    if mode in ('install', 'repair'):
        problems = validate(choice, gpu, ram)
        show_plan(choice, problems, out)
        if any(level == 'error' for level, _ in problems):
            out('\n  Исправьте ошибки выше и запустите установщик снова.')
            return 2
        if not a.yes and not p.yes('\n  Начать установку?', True):
            return 1
    out('')
    inst = Installer(choice, mode=mode, uv=a.uv, source=a.source, gpu=gpu, force=a.force or mode == 'repair',
                     move_text_dest=a.move_text_to, shortcut=not a.no_shortcut)
    ok = inst.run()
    if not ok:
        return 1
    out('\n  Готово.')
    if mode == 'install' and not a.no_launch and not a.yes and p.yes('  Запустить TinyH3 сейчас?', True):
        launch(Path(choice.app_dir), win)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
