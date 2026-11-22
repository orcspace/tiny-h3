"""Run one pipeline stage in a fresh process and relay its progress events.

A stage process owns all of its memory: when it exits, the OS takes back every pinned page and the driver every
byte of VRAM, which no amount of ``empty_cache`` guarantees inside a long-lived process. A crash (including an
out-of-memory) is reported as an exception in the parent; the server survives it.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback


def _child(name, args, q):
    try:
        import logging
        logging.basicConfig(level=logging.WARNING)
        from .pipeline import STAGES

        def emit(**event):
            q.put(('event', event))
        result = STAGES[name](*args, emit)
        q.put(('result', result))
    except BaseException as exc:  # noqa: BLE001 - everything is reported to the parent
        oom = 'out of memory' in str(exc).lower() or isinstance(exc, MemoryError)
        q.put(('error', {'type': type(exc).__name__, 'message': str(exc), 'oom': oom,
                         'traceback': traceback.format_exc()}))


class StageError(RuntimeError):
    def __init__(self, stage, info):
        self.stage, self.info = stage, info
        super().__init__(f'{stage}: {info["type"]}: {info["message"]}')


def run_stage(name: str, args: tuple, on_event, cancel=None, poll: float = 0.25) -> dict:
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    proc = ctx.Process(target=_child, args=(name, args, q), name=f'tinyh3-{name}', daemon=True)
    proc.start()
    try:
        while True:
            if cancel is not None and cancel.is_set():
                proc.terminate()
                proc.join(5)
                raise InterruptedError('отменено')
            try:
                kind, payload = q.get(timeout=poll)
            except queue.Empty:
                if not proc.is_alive():
                    # drain anything sent just before exit
                    try:
                        kind, payload = q.get(timeout=1.0)
                    except queue.Empty:
                        raise StageError(name, {'type': 'Crash', 'message': f'процесс завершился с кодом '
                                                f'{proc.exitcode}', 'oom': False, 'traceback': ''})
                else:
                    continue
            if kind == 'event':
                payload.setdefault('time', time.time())
                on_event(payload)
            elif kind == 'result':
                return payload
            elif kind == 'error':
                raise StageError(name, payload)
    finally:
        if proc.is_alive():
            proc.join(10)
            if proc.is_alive():
                proc.kill()
