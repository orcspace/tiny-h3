"""`python -m tinyh3 generate` runs the stages in spawned child processes. A spawned child re-imports the parent's
main module, so an unguarded ``tinyh3/__main__.py`` would re-run the CLI in every child (what Windows and macOS
do for every multiprocessing child, and what this project always uses)."""
import os
import subprocess
import sys

import pytest

pytest.importorskip('diffusers')
from tests.test_pipeline_e2e import tiny_store  # noqa: F401  (fixture)


def test_cli_generate_with_spawned_stages(tiny_store, tmp_path):
    env = dict(os.environ, TINYH3_HOME=str(tmp_path / 'home'))
    out = tmp_path / 'cli.mp4'
    proc = subprocess.run(
        [sys.executable, '-m', 'tinyh3', 'generate', 'w1 w2 w3', '--model', str(tiny_store), '--height', '64',
         '--width', '96', '--seconds', '5', '--mode', 'fast', '--seed', '3', '--out', str(out)],
        env=env, capture_output=True, text=True, timeout=400, check=False)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert out.exists() and out.stat().st_size > 1000
    assert proc.stdout.count('Видео:') == 1          # the CLI ran once, not once per child process
