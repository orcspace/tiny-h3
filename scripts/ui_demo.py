"""UI demo server without a GPU or a model: simulated RTX 3070 8 GB + 16 GB RAM and a stand-in generator that emits
the real progress events. Finished "videos" are slow pans over the pi-H3 demo images (muxed by TinyH3's own MP4
writer), clearly not model output. For UI development and screenshots:

    python scripts/ui_demo.py --port 7861
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TINYH3_SIMULATE', '8,16')
os.environ.setdefault('TINYH3_HOME', tempfile.mkdtemp(prefix='tinyh3-demo-'))
_demo_model = Path(os.environ['TINYH3_HOME']) / 'models' / 'h3-pi2-dwd-int8'
_demo_model.mkdir(parents=True, exist_ok=True)
(_demo_model / 'manifest.json').write_text('{"format": "tinyh3-store-v1", "recipe": {"steps": 2, "demo": true}}')

import torch  # noqa: E402

from tinyh3.config import SAMPLERS, render_size, output_size  # noqa: E402
from tinyh3.server import app as server  # noqa: E402

IMAGES = sorted((ROOT / 'pi-H3' / 'demo' / 'examples' / 'minimax_h3').glob('*.webp'))


def _pan(image: Path, w: int, h: int, frames: int) -> torch.Tensor:
    from PIL import Image
    img = Image.open(image).convert('RGB')
    scale = max(w / img.width, h / img.height) * 1.15
    img = img.resize((int(img.width * scale), int(img.height * scale)))
    src = torch.from_numpy(__import__('numpy').asarray(img).copy())
    out = torch.empty(frames, h, w, 3, dtype=torch.uint8)
    for i in range(frames):
        t = i / max(frames - 1, 1)
        x = int((src.shape[1] - w) * (0.5 + 0.5 * math.sin(t * math.pi - math.pi / 2)))
        y = int((src.shape[0] - h) * t)
        out[i] = src[y:y + h, x:x + w]
    return out


def fake_generate(req, on_event, subprocess=True, cancel=None):
    from tinyh3.runtime.media import write_mp4
    r = req.resolve()
    mode = SAMPLERS[r.sampler]
    rh, rw = render_size(r.height, r.width, mode.final_scale)
    oh, ow = output_size(r.height, r.width, r.upscale)
    slow = r.seed == 14  # the demo's long-running job
    plan = server.estimate_plan(r)  # the real planner on the simulated machine
    on_event({'stage': 'text', 'event': 'start'})
    for i in range(1, 51, 7):
        on_event({'stage': 'text', 'step': i, 'total': 50})
        time.sleep(0.02)
    on_event({'stage': 'text', 'event': 'done', 'result': {'wall_seconds': 11.4}})
    on_event({'stage': 'plan', 'plan': plan.as_dict(), 'message': plan.summary()})
    for i in range(1, 51, 5):
        on_event({'stage': 'load', 'step': i, 'total': 50})
        time.sleep(0.01)
    on_event({'stage': 'load', 'step': 50, 'total': 50})
    total = mode.steps * 50
    stop_at = int(total * 0.62) if slow else total
    for b in range(1, stop_at + 1, 3):
        on_event({'stage': 'denoise', 'step': b, 'total': total, 'substep': (b - 1) // 50 + 1, 'steps': mode.steps,
                  'eta': round((total - b) * plan.est['denoise_s'] / total, 1)})
        time.sleep(0.004)
    if slow:
        while cancel is None or not cancel.is_set():
            time.sleep(0.2)
        raise InterruptedError
    on_event({'stage': 'denoise', 'event': 'done', 'result': {'wall_seconds': plan.est['denoise_s']}})
    for i in range(1, 9):
        on_event({'stage': 'decode', 'step': i, 'total': 8})
    on_event({'stage': 'decode', 'event': 'done', 'result': {'wall_seconds': plan.est['dwd_s']}})
    out = Path(os.environ['TINYH3_HOME']) / 'outputs' / f'demo-{r.seed}.mp4'
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = _pan(IMAGES[r.seed % len(IMAGES)], min(ow, 960), min(oh, 540), 48)
    write_mp4(out, frames, torch.zeros(2, 32000 * 2), 24, 32000, crf=24)
    report = {'request': {**r.__dict__}, 'output': str(out), 'total_seconds': plan.est['total_s'], 'stages': {}}
    on_event({'stage': 'done', 'event': 'done', 'output': str(out), 'report': report})
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=7861)
    a = p.parse_args()
    server.generate = fake_generate
    import uvicorn
    app = server.create_app(subprocess=False)
    print(f'UI demo: http://127.0.0.1:{a.port}  (TINYH3_HOME={os.environ["TINYH3_HOME"]})')
    uvicorn.run(app, host='127.0.0.1', port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
