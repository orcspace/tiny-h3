import json
from pathlib import Path

import pytest

from tinyh3 import installer as I


def mk(tmp_path, **kw):
    base = dict(app_dir=str(tmp_path / 'app'), models_dir=str(tmp_path / 'models'), outputs_dir=str(tmp_path / 'out'))
    base.update(kw)
    return I.Choice(**base)


GPU = {'present': True, 'name': 'RTX 3070', 'vram_gb': 8.0, 'driver': '560', 'cuda': '12.6'}


def test_cuda_tag():
    assert I.cuda_tag('12.8') == 'cu128' and I.cuda_tag('12.6') == 'cu126' and I.cuda_tag('12.4') == 'cu124'
    assert I.cuda_tag('11.8') is None and I.cuda_tag(None) == 'cu128'


def test_check_dir():
    assert I.check_dir('D:\\TinyH3', 'win32')[0]
    assert not I.check_dir('D:\\', 'win32')[0]
    assert not I.check_dir('C:\\Windows\\x', 'win32')[0]
    assert not I.check_dir('rel/path', 'linux')[0]
    assert not I.check_dir('/usr/lib/x', 'linux')[0]
    assert I.check_dir('/home/u/tinyh3', 'linux')[0]


def test_defaults_per_platform(tmp_path):
    w = I.default_dirs('win32', {'LOCALAPPDATA': 'C:\\L'}, tmp_path)
    assert w['app_dir'].endswith('TinyH3') and 'L' in w['app_dir']
    assert I.default_dirs('linux', {}, tmp_path)['app_dir'].endswith('tinyh3')


def test_requirements_group_by_drive(tmp_path):
    c = mk(tmp_path)
    same = I.requirements(c, device=lambda p: 'A')
    assert len(same) == 1
    c2 = mk(tmp_path, text_dir=str(tmp_path / 'text'))
    dev = lambda p: 'B' if 'text' in p else 'A'
    two = I.requirements(c2, device=dev)
    assert {e['dev'] for e in two} == {'A', 'B'}
    assert sum(e['need_gb'] for e in two) == pytest.approx(sum(e['need_gb'] for e in same), abs=1.0)


def test_validate_space_and_gpu(tmp_path):
    c = mk(tmp_path)
    errs = I.validate(c, GPU, 16, free=lambda p: 5.0, device=lambda p: 'A')
    assert any(l == 'error' and 'места' in m for l, m in errs)
    ok = I.validate(c, GPU, 16, free=lambda p: 500.0, device=lambda p: 'A')
    assert not [m for l, m in ok if l == 'error']
    small = I.validate(c, {**GPU, 'vram_gb': 4.0}, 16, free=lambda p: 500.0, device=lambda p: 'A')
    assert any('6 ГБ' in m for l, m in small)
    old = I.validate(c, {**GPU, 'cuda': '11.4'}, 16, free=lambda p: 500.0, device=lambda p: 'A')
    assert any('старый' in m for l, m in old)


def test_profile_sizes_monotonic(tmp_path):
    sizes = [I.compute(mk(tmp_path, profile=p))['models'] for p in ('compact', 'balanced', 'max')]
    assert sizes[0] < sizes[1] < sizes[2]


def test_launchers_windows(tmp_path):
    c = mk(tmp_path, app_dir='D:\\Tiny H3\\100%', models_dir='E:\\m')
    f = I.render_launchers(c, 'win32')
    assert set(f) == {'start.bat', 'console.bat', 'uninstall.bat'}
    s = f['start.bat']
    assert 'tinyh3 serve --open' in s and 'TINYH3_MODELS=E:\\m' in s and '100%%' in s
    assert 'YES' in f['uninstall.bat']


def test_launchers_posix_quotes(tmp_path):
    c = mk(tmp_path, app_dir=str(tmp_path / "it's app"))
    s = I.render_launchers(c, 'linux')['start.sh']
    assert "'\\''" in s and 'serve --open' in s


def test_steps_for(tmp_path):
    c = mk(tmp_path)
    assert 'ref' not in I.steps_for('install', c) and 'quality4' not in I.steps_for('install', c)
    c.ref = c.quality4 = True
    assert I.steps_for('install', c)[-4:-1] == ['quality4', 'ref', 'verify'][0:3] or 'ref' in I.steps_for('install', c)
    c.skip_models = True
    assert 'models' not in I.steps_for('install', c)


class Fake:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, cmd, env, on_line, cwd=None):
        self.calls.append(cmd)
        joined = ' '.join(cmd)
        if self.fail_on and self.fail_on in joined:
            self.fail_on = None
            return 1, ['boom']
        if 'doctor' in joined:
            on_line(json.dumps({'torch': '2.9', 'cuda_available': True, 'attention': ['sdpa'], 'native': None,
                                'hardware': {'gpu_name': 'x'}, 'selftest': {'triton_ok': False, 'error': 'e'}, 'models': []}))
        if 'python' in cmd[0] or cmd[0].endswith('python'):
            pass
        return 0, []


def run_installer(tmp_path, runner, **kw):
    c = mk(tmp_path, **kw)
    src = tmp_path / 'src'
    src.mkdir(exist_ok=True)
    (src / 'pyproject.toml').write_text('')
    py = tmp_path / 'py'
    py.write_text('')
    inst = I.Installer(c, uv='uv', source=str(src), gpu=GPU, platform='linux', runner=runner, out=lambda *a: None,
                       tty=False, python=str(py), shortcut=False)
    return inst, inst.run()


def test_full_run_and_resume(tmp_path):
    r = Fake(fail_on='setup --low-disk')
    inst, ok = run_installer(tmp_path, r)
    assert not ok
    st = json.loads((tmp_path / 'app' / 'install-state.json').read_text())
    assert 'torch' in st['done'] and 'models' not in st['done']
    assert any('cu126' in ' '.join(c) for c in r.calls)
    n_before = len(r.calls)
    r2 = Fake()
    inst, ok = run_installer(tmp_path, r2)
    assert ok
    flat = [' '.join(c) for c in r2.calls]
    assert not any('torch' in f and 'pip install' in f for f in flat)       # torch step skipped on resume
    assert any('setup --low-disk' in f for f in flat)
    assert (tmp_path / 'app' / 'start.sh').exists() and (tmp_path / 'app' / 'install.json').exists()
    assert inst.report['warnings']                                          # triton selftest warning surfaced


def test_changed_choice_redoes_models_only(tmp_path):
    run_installer(tmp_path, Fake())
    r = Fake()
    run_installer(tmp_path, r, profile='max')
    flat = [' '.join(c) for c in r.calls]
    assert any('--policy int8' in f for f in flat)
    assert not any('pip install' in f and 'torch' in f for f in flat)


def test_text_dir_and_quality4_args(tmp_path):
    r = Fake()
    run_installer(tmp_path, r, text_dir=str(tmp_path / 'tx'), quality4=True, ref=True)
    setups = [' '.join(c) for c in r.calls if 'setup' in c or 'setup-ref' in c]
    assert sum('--steps 2' in s for s in setups) == 1 and sum('--steps 4' in s for s in setups) == 1
    assert all(f'--text-dir {tmp_path / "tx"}' in s for s in setups if 'setup-ref' not in s)
    assert any('setup-ref' in s for s in setups)


def test_scripted_prompts(tmp_path):
    answers = iter([str(tmp_path / 'a'), str(tmp_path / 'm'), '-', str(tmp_path / 'o'), '2', 'д', 'н', 'н', 'д'])
    p = I.Prompts(inp=lambda _: next(answers), out=lambda *a: None)
    base = mk(tmp_path)
    c = I.interactive_choice(base, p, GPU, 16, fixed_app=False)
    assert c.models_dir == str(tmp_path / 'm') and c.text_dir == '' and c.profile == 'balanced'
    assert c.images and not c.ref and not c.skip_models


def test_directory_prompt_rejects_then_accepts(tmp_path):
    answers = iter(['relative', '/usr/lib/x', str(tmp_path / 'ok')])
    p = I.Prompts(inp=lambda _: next(answers), out=lambda *a: None)
    assert p.directory('x', '/home/d') == str(tmp_path / 'ok')


def test_main_non_interactive_rejects_no_space(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(I, 'free_gb', lambda p: 1.0)
    monkeypatch.setattr(I, 'detect_gpu', lambda: GPU)
    rc = I.main(['--yes', '--app-dir', str(tmp_path / 'a'), '--outputs-dir', str(tmp_path / 'o')])
    assert rc == 2 and 'места' in capsys.readouterr().out


def test_bootstrap_files():
    root = Path(__file__).resolve().parents[1]
    assert (root / 'installer' / 'bootstrap.ps1').read_bytes()[:3] == b'\xef\xbb\xbf'
    assert 'bootstrap.ps1' in (root / 'install.bat').read_text(encoding='utf-8')
    assert (root / 'install.sh').stat().st_mode & 0o111
