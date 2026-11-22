"""Local web server: job queue, live progress over WebSocket, gallery, pre-flight estimates."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, fields
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .. import __version__
from ..config import RESOLUTIONS, ASPECT_RATIOS, SAMPLERS, SAMPLER_ALIASES, UPSCALE, MIN_SECONDS, home, models_dir, \
    canvas_size, frames_for_seconds, Geometry, render_size, output_size
from ..runtime.pipeline import GenRequest, generate

UI = Path(__file__).resolve().parent.parent / 'ui'


MODE_INFO = {
    'quality': '4 шага, первый плотный. Лучшее качество, адаптер на 4 шага.',
    'balance': '2 шага на полном разрешении, первый шаг с плотным вниманием.',
    'fast': 'Черновой шаг на ½ разрешения, финальный на полном. ~1.6× к «Балансу».',
    'fast-high': 'Черновик + финальный шаг на ¾ разрешения + апскейл на Rust-ядре. ~2× к Fast, чуть мягче.',
}


class JobIn(BaseModel):
    prompt: str
    seconds: float = 5.0
    resolution: str = '480p'
    aspect: str = '16:9'
    sampler: str = 'fast'
    seed: int = -1
    attention: str = 'auto'
    spark_mode: str | None = None
    upscale: str = 'none'
    model: str | None = None


class EstimateIn(BaseModel):
    seconds: float = 5.0
    resolution: str = '480p'
    aspect: str = '16:9'
    sampler: str = 'fast'
    attention: str = 'auto'
    upscale: str = 'none'


class Jobs:
    """SQLite-backed job records; progress lives in memory."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL, status TEXT, '
                        'request TEXT, report TEXT, error TEXT, output TEXT)')
        self.db.execute("UPDATE jobs SET status='failed', error='сервер был перезапущен' "
                        "WHERE status IN ('queued','running')")
        self.db.commit()
        self.progress: dict[str, dict] = {}

    def add(self, req: dict) -> dict:
        job = {'id': uuid.uuid4().hex[:12], 'created': time.time(), 'status': 'queued', 'request': req}
        with self.lock:
            self.db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?)',
                            (job['id'], job['created'], 'queued', json.dumps(req, ensure_ascii=False), None, None,
                             None))
            self.db.commit()
        return job

    def update(self, job_id: str, **kw):
        cols = ', '.join(f'{k}=?' for k in kw)
        vals = [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in kw.values()]
        with self.lock:
            self.db.execute(f'UPDATE jobs SET {cols} WHERE id=?', (*vals, job_id))
            self.db.commit()

    def get(self, job_id: str) -> dict | None:
        rows = self.list(where='id=?', args=(job_id,))
        return rows[0] if rows else None

    def list(self, limit=200, where='1=1', args=()) -> list[dict]:
        with self.lock:
            cur = self.db.execute(f'SELECT id, created, status, request, report, error, output FROM jobs '
                                  f'WHERE {where} ORDER BY created DESC LIMIT ?', (*args, limit))
            rows = cur.fetchall()
        out = []
        for r in rows:
            job = {'id': r[0], 'created': r[1], 'status': r[2], 'request': json.loads(r[3] or '{}'),
                   'report': json.loads(r[4]) if r[4] else None, 'error': r[5], 'output': r[6]}
            job['progress'] = self.progress.get(job['id'])
            out.append(job)
        return out

    def delete(self, job_id: str):
        with self.lock:
            self.db.execute('DELETE FROM jobs WHERE id=?', (job_id,))
            self.db.commit()


class Hub:
    """Fan-out of events to every connected WebSocket."""

    def __init__(self):
        self.clients: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, event: dict):
        if self.loop is None:
            return
        for q in list(self.clients):
            self.loop.call_soon_threadsafe(q.put_nowait, event)


def estimate_plan(r: GenRequest):
    """The planner's view of a resolved request on this machine (render size, mode, live budgets)."""
    from ..hardware import probe
    from ..planner import make_plan
    mode = r.preset
    rh, rw = render_size(r.height, r.width, mode.final_scale)
    return make_plan(probe(), Geometry(rh, rw, frames_for_seconds(r.seconds)), steps=mode.steps,
                     attention=r.attention, mode=mode)


def native_version():
    from .. import native
    return native.version() if native.available() else None


def create_app(*, subprocess: bool = True) -> FastAPI:
    from contextlib import asynccontextmanager
    hub = Hub()

    @asynccontextmanager
    async def lifespan(_app):
        hub.loop = asyncio.get_running_loop()
        yield

    app = FastAPI(title='TinyH3 Video', version=__version__, lifespan=lifespan)
    jobs = Jobs(home() / 'jobs.sqlite3')
    queue: list[str] = []
    cv = threading.Condition()
    cancels: dict[str, threading.Event] = {}

    def run_job(job_id: str):
        job = jobs.get(job_id)
        if job is None or job['status'] != 'queued':
            return
        cancel = cancels.setdefault(job_id, threading.Event())
        jobs.update(job_id, status='running')
        hub.publish({'job': job_id, 'type': 'status', 'status': 'running'})
        state = {'stages': {}, 'started': time.time()}

        def on_event(e):
            stage = e.get('stage')
            if stage:
                st = state['stages'].setdefault(stage, {})
                st.update({k: v for k, v in e.items() if k in ('step', 'total', 'substep', 'steps', 'eta', 'message',
                                                              'event')})
                if e.get('event') == 'done' and 'result' in e:
                    st['seconds'] = e['result'].get('wall_seconds')
                if stage == 'plan':
                    state['plan'] = {k: e['plan'].get(k) for k in ('attention', 'head_group', 'est', 'notes')}
                    state['plan_summary'] = e.get('message')
            jobs.progress[job_id] = state
            safe = {k: v for k, v in e.items() if k not in ('result', 'report', 'plan')}
            hub.publish({'job': job_id, 'type': 'progress', **safe})

        try:
            req = GenRequest(**{f.name: job['request'][f.name] for f in fields(GenRequest) if f.name in job['request']})
            report = generate(req, on_event, subprocess=subprocess, cancel=cancel)
            jobs.update(job_id, status='done', report=report, output=report['output'])
            hub.publish({'job': job_id, 'type': 'status', 'status': 'done'})
        except InterruptedError:
            jobs.update(job_id, status='cancelled')
            hub.publish({'job': job_id, 'type': 'status', 'status': 'cancelled'})
        except Exception as exc:  # reported to the UI; the server keeps running
            info = getattr(exc, 'info', None) or {'message': str(exc), 'type': type(exc).__name__}
            msg = info.get('message', str(exc))
            if info.get('oom'):
                msg = 'не хватило памяти: ' + msg + ' — уменьшите разрешение или длительность'
            jobs.update(job_id, status='failed', error=msg)
            hub.publish({'job': job_id, 'type': 'status', 'status': 'failed', 'error': msg})
        finally:
            cancels.pop(job_id, None)

    def worker():
        while True:
            with cv:
                while not queue:
                    cv.wait()
                job_id = queue.pop(0)
            run_job(job_id)

    threading.Thread(target=worker, daemon=True, name='tinyh3-jobs').start()

    @app.get('/', response_class=HTMLResponse)
    def index():
        return HTMLResponse((UI / 'index.html').read_text(encoding='utf-8'))

    @app.get('/api/info')
    def info():
        from ..hardware import probe
        from ..gpu_tuning import load, advice
        from ..runtime.attention import available
        hw = probe()
        prof = load()
        models = sorted(p.parent.name for p in models_dir().glob('*/manifest.json')) if models_dir().exists() else []
        return {'version': __version__, 'hardware': hw.as_dict(), 'attention': available(),
                'gpu_profile': prof.summary if prof else None, 'advice': advice(prof, hw.os) if prof else [],
                'models': models, 'resolutions': list(RESOLUTIONS), 'aspects': list(ASPECT_RATIOS),
                'samplers': {k: {'title': v.title, 'steps': v.steps, 'about': MODE_INFO.get(k, '')}
                             for k, v in SAMPLERS.items()},
                'upscale': list(UPSCALE), 'native': native_version(), 'min_seconds': 5.2, 'max_seconds': 14.4}

    @app.post('/api/estimate')
    def estimate(body: EstimateIn):
        from ..hardware import probe
        from ..planner import make_plan
        mode = SAMPLERS[SAMPLER_ALIASES.get(body.sampler, body.sampler)]
        h, w = canvas_size(body.aspect, RESOLUTIONS[body.resolution])
        rh, rw = render_size(h, w, mode.final_scale)
        oh, ow = output_size(h, w, body.upscale)
        geom = Geometry(rh, rw, frames_for_seconds(body.seconds))
        plan = make_plan(probe(), geom, steps=mode.steps, attention=body.attention, mode=mode)
        return {'width': ow, 'height': oh, 'render': [rw, rh], 'frames': geom.num_frames,
                'seconds': round(geom.seconds, 2),
                'tokens': geom.sequence_length, 'fits': plan.fits, 'reason': plan.reason,
                'summary': plan.summary(), 'attention': plan.attention, 'est': plan.est, 'notes': plan.notes,
                'tiers': {'vram': len(plan.resident), 'pinned': len(plan.pinned), 'disk': len(plan.disk)}}

    @app.post('/api/jobs')
    def submit(body: JobIn):
        if not body.prompt.strip():
            raise HTTPException(400, 'пустой промпт')
        if body.resolution not in RESOLUTIONS:
            raise HTTPException(400, 'неизвестное разрешение')
        if SAMPLER_ALIASES.get(body.sampler, body.sampler) not in SAMPLERS or body.upscale not in UPSCALE:
            raise HTTPException(400, 'неизвестный режим или апскейл')
        job = jobs.add(body.model_dump())
        with cv:
            queue.append(job['id'])
            cv.notify()
        hub.publish({'job': job['id'], 'type': 'status', 'status': 'queued'})
        return job

    @app.get('/api/jobs')
    def list_jobs():
        return jobs.list()

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        with cv:
            if job_id in queue:
                queue.remove(job_id)
                jobs.update(job_id, status='cancelled')
                hub.publish({'job': job_id, 'type': 'status', 'status': 'cancelled'})
                return {'ok': True}
        ev = cancels.get(job_id)
        if ev is None:
            raise HTTPException(404, 'задача не выполняется')
        ev.set()
        return {'ok': True}

    @app.delete('/api/jobs/{job_id}')
    def delete(job_id: str, remove_file: bool = False):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404)
        if job['status'] in ('queued', 'running'):
            raise HTTPException(409, 'сначала отмените задачу')
        if remove_file and job.get('output'):
            for p in (Path(job['output']), Path(job['output'] + '.json')):
                p.unlink(missing_ok=True)
        jobs.delete(job_id)
        return {'ok': True}

    @app.get('/api/videos/{job_id}')
    def video(job_id: str):
        job = jobs.get(job_id)
        if job is None or not job.get('output') or not Path(job['output']).exists():
            raise HTTPException(404)
        return FileResponse(job['output'], media_type='video/mp4', filename=Path(job['output']).name)

    @app.websocket('/api/events')
    async def events(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue()
        hub.clients.add(q)
        try:
            while True:
                await ws.send_text(json.dumps(await q.get(), ensure_ascii=False))
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(q)

    return app


def serve(host='127.0.0.1', port=7860, open_browser=False):
    import uvicorn
    if open_browser:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f'http://{host}:{port}')).start()
    print(f'TinyH3 Video: http://{host}:{port}')
    uvicorn.run(create_app(), host=host, port=port, log_level='warning')
