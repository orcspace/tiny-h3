"""Local web server: job queue, live progress over WebSocket, gallery, pre-flight estimates."""
from __future__ import annotations

import asyncio
import os
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import fields
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .. import __version__
from ..config import RESOLUTIONS, ASPECT_RATIOS, SAMPLERS, SAMPLER_ALIASES, REF_SAMPLER, UPSCALE, home, models_dir, \
    canvas_size, frames_for_seconds, Geometry, render_size, output_size
from ..runtime.pipeline import GenRequest, generate

UI = Path(__file__).resolve().parent.parent / 'ui'


TASK_INFO = {
    't2v': {'title': 'Текст', 'about': 'Видео со звуком по описанию.'},
    'i2v': {'title': 'Изображение', 'about': 'Оживить кадр: он становится первым кадром видео.'},
    'flf2v': {'title': 'Первый + последний', 'about': 'Видео, которое начинается и заканчивается вашими кадрами.'},
    'ref2v': {'title': 'Референсы', 'about': 'До 9 изображений, 3 видео и 3 аудио как опора: персонажи, предметы, '
                                              'стиль, голос. Свой трансформер, режим «Turbo 4».'},
}

MODE_INFO = {
    'quality': '4 шага, первый плотный. Лучшее качество, адаптер на 4 шага.',
    'balance': '2 шага на полном разрешении, первый шаг с плотным вниманием.',
    'fast': 'Черновой шаг на ½ разрешения, финальный на полном. ~1.6× к «Балансу».',
    'fast-high': 'Черновик + финальный шаг на ¾ разрешения + апскейл на Rust-ядре. ~2× к Fast, чуть мягче.',
}


class RefIn(BaseModel):
    kind: str          # image | video | audio
    id: str            # an id returned by POST /api/uploads


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
    task: str = 't2v'
    first: str | None = None        # upload id
    last: str | None = None         # upload id
    references: list[RefIn] = []


class EstimateIn(BaseModel):
    seconds: float = 5.0
    resolution: str = '480p'
    aspect: str = '16:9'
    sampler: str = 'fast'
    attention: str = 'auto'
    upscale: str = 'none'
    task: str = 't2v'
    references: list[RefIn] = []


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


UPLOAD_KINDS = {'.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.webp': 'image', '.bmp': 'image',
                '.mp4': 'video', '.mov': 'video', '.mkv': 'video', '.webm': 'video',
                '.wav': 'audio', '.mp3': 'audio', '.flac': 'audio', '.m4a': 'audio', '.ogg': 'audio'}
MAX_UPLOAD = 1 << 30
LIMITS = {'image': 9, 'video': 3, 'audio': 3, 'total': 12}  # MiniMax-H3 Ref2VA


def reference_rows(kinds: list[str], tokens_per_frame: int) -> int:
    """Rough number of extra sequence rows the references add (for the pre-flight estimate)."""
    rows = 0
    for k in kinds:
        rows += {'image': tokens_per_frame, 'video': 8 * tokens_per_frame, 'audio': 160}.get(k, 0)
    return rows


def estimate_plan(r: GenRequest):
    """The planner's view of a resolved request on this machine (render size, mode, live budgets)."""
    from ..hardware import probe
    from ..planner import make_plan
    mode = r.preset
    rh, rw = render_size(r.height, r.width, mode.final_scale)
    geom = Geometry(rh, rw, frames_for_seconds(r.seconds))
    extra = 0
    if r.task in ('i2v', 'last2v', 'flf2v'):
        extra = (1 + (r.task == 'flf2v')) * geom.tokens_per_frame
    elif r.task == 'ref2v':
        extra = reference_rows([x.get('kind') for x in r.references], geom.tokens_per_frame)
    geom = Geometry(rh, rw, geom.num_frames, text_tokens=geom.text_tokens + extra)
    return make_plan(probe(), geom, steps=mode.steps, attention=r.attention, mode=mode)


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

    @app.middleware('http')
    async def same_origin_only(request: Request, call_next):
        """State-changing requests must come from this server's own pages: browsers send ``Origin`` on cross-site
        POST / DELETE, and a web page you visit must not be able to start jobs or delete videos on localhost."""
        origin = request.headers.get('origin')
        if origin and request.method not in ('GET', 'HEAD', 'OPTIONS'):
            from urllib.parse import urlparse
            if urlparse(origin).netloc != request.headers.get('host', ''):
                from fastapi.responses import JSONResponse
                return JSONResponse({'detail': 'cross-origin request refused'}, status_code=403)
        return await call_next(request)
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
            req.references = [{'kind': x['kind'], 'path': x['path']} for x in job['request'].get('references') or []]
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

    uploads_dir = home() / 'uploads'

    def upload_path(upload_id: str) -> Path:
        if not upload_id or '/' in upload_id or '\\' in upload_id or upload_id.startswith('.'):
            raise HTTPException(400, 'некорректный id файла')
        path = uploads_dir / upload_id
        if not path.is_file():
            raise HTTPException(404, f'файл {upload_id} не найден')
        return path

    @app.post('/api/uploads')
    async def upload(request: Request, name: str = 'file'):
        ext = Path(name).suffix.lower()
        if ext not in UPLOAD_KINDS:
            raise HTTPException(400, f'неподдерживаемый формат {ext or "без расширения"}')
        uploads_dir.mkdir(parents=True, exist_ok=True)
        upload_id = f'{uuid.uuid4().hex[:12]}{ext}'
        size = 0
        tmp = uploads_dir / (upload_id + '.part')
        try:
            with open(tmp, 'wb') as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, 'файл больше 1 ГБ')
                    f.write(chunk)
            if size == 0:
                raise HTTPException(400, 'пустой файл')
            tmp.replace(uploads_dir / upload_id)
        finally:
            tmp.unlink(missing_ok=True)
        return {'id': upload_id, 'name': Path(name).name, 'kind': UPLOAD_KINDS[ext], 'size': size}

    @app.get('/api/uploads/{upload_id}')
    def upload_file(upload_id: str):
        return FileResponse(upload_path(upload_id))

    @app.get('/api/info')
    def info():
        from ..hardware import probe
        from ..gpu_tuning import load, advice
        from ..runtime.attention import available
        hw = probe()
        prof = load()
        models = sorted(p.parent.name for p in models_dir().glob('*/manifest.json')) if models_dir().exists() else []
        has_ref = False
        for m in models:
            try:
                has_ref |= 'dit_ref' in json.loads((models_dir() / m / 'manifest.json').read_text())
            except (OSError, ValueError):
                pass
        return {'version': __version__, 'hardware': hw.as_dict(), 'attention': available(),
                'gpu_profile': prof.summary if prof else None, 'advice': advice(prof, hw.os) if prof else [],
                'models': models, 'resolutions': list(RESOLUTIONS), 'aspects': list(ASPECT_RATIOS),
                'samplers': {k: {'title': v.title, 'steps': v.steps, 'about': MODE_INFO.get(k, '')}
                             for k, v in SAMPLERS.items()},
                'upscale': list(UPSCALE), 'native': native_version(), 'min_seconds': 5.2, 'max_seconds': 14.4,
                'tasks': TASK_INFO, 'ref_available': has_ref, 'ref_limits': LIMITS}

    @app.post('/api/estimate')
    def estimate(body: EstimateIn):
        mode = SAMPLERS[SAMPLER_ALIASES.get(body.sampler, body.sampler)]
        if body.task == 'ref2v':
            mode = REF_SAMPLER
        h, w = canvas_size(body.aspect, RESOLUTIONS[body.resolution])
        rh, rw = render_size(h, w, mode.final_scale)
        oh, ow = output_size(h, w, body.upscale)
        r = GenRequest(prompt='-', seconds=body.seconds, height=h, width=w, sampler=body.sampler,
                       attention=body.attention, task=body.task,
                       references=[{'kind': x.kind} for x in body.references])
        plan = estimate_plan(r)
        geom = Geometry(rh, rw, frames_for_seconds(body.seconds))
        return {'width': ow, 'height': oh, 'render': [rw, rh], 'frames': geom.num_frames, 'steps': mode.steps,
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
        if body.task not in TASK_INFO:
            raise HTTPException(400, 'неизвестная задача')
        data = body.model_dump()
        data['first'] = str(upload_path(body.first)) if body.first else None
        data['last'] = str(upload_path(body.last)) if body.last else None
        if body.task in ('i2v', 'flf2v') and not data['first']:
            raise HTTPException(400, 'нужен первый кадр')
        if body.task in ('last2v', 'flf2v') and not data['last']:
            raise HTTPException(400, 'нужен последний кадр')
        refs = []
        if body.task == 'ref2v':
            kinds = [x.kind for x in body.references]
            if not kinds:
                raise HTTPException(400, 'добавьте хотя бы один референс')
            if set(kinds) == {'audio'}:
                raise HTTPException(400, 'аудио нужно дополнить изображением или видео')
            for k in ('image', 'video', 'audio'):
                if kinds.count(k) > LIMITS[k]:
                    raise HTTPException(400, f'не больше {LIMITS[k]} референсов типа {k}')
            if len(kinds) > LIMITS['total']:
                raise HTTPException(400, f'не больше {LIMITS["total"]} референсов')
            for x in body.references:
                path = upload_path(x.id)
                if UPLOAD_KINDS.get(path.suffix.lower()) != x.kind:
                    raise HTTPException(400, f'файл {x.id} не является {x.kind}')
                refs.append({'kind': x.kind, 'path': str(path), 'id': x.id})
        data['references'] = refs
        job = jobs.add(data)
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


def _port_free(host: str, port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0 if os.name == 'nt' else 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(host: str, preferred: int, tries: int = 20) -> int:
    """``preferred`` if it is free, otherwise the next free port (a second copy, or another program, holds it)."""
    for port in range(preferred, preferred + tries):
        if _port_free(host, port):
            return port
    raise OSError(f'нет свободного порта в диапазоне {preferred}-{preferred + tries - 1}')


def _open_when_ready(host: str, port: int, timeout: float = 120.0) -> None:
    import socket
    import webbrowser
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(f'http://{host}:{port}')
                return
        except OSError:
            time.sleep(0.3)


def serve(host='127.0.0.1', port=7860, open_browser=False):
    import uvicorn
    chosen = pick_port(host, port)
    if chosen != port:
        print(f'Порт {port} занят, использую {chosen}')
    if open_browser:
        threading.Thread(target=_open_when_ready, args=(host, chosen), daemon=True).start()
    print(f'TinyH3 Video: http://{host}:{chosen}   (закройте окно или нажмите Ctrl+C, чтобы остановить)', flush=True)
    uvicorn.run(create_app(), host=host, port=chosen, log_level='warning')
