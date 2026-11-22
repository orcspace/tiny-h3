"""Tiered weight streaming.

Every unit (a DiT block, a text encoder layer) lives in one of three tiers chosen by the planner:

* ``vram``   – loaded once into its own GPU buffer;
* ``pinned`` – held in page-locked host memory and copied into a GPU slot each time it is needed;
* ``disk``   – read every pass by a background thread into a ring of pinned bounce buffers, then copied.

Streamed units flow through a ring of ``slots`` GPU buffers. All copies run on a dedicated copy stream; the compute
stream waits on a per-slot ``ready`` event, and a copy into a slot waits on that slot's ``free`` event, recorded when
the compute that read the previous occupant has been enqueued. The CPU never blocks on the GPU in steady state.

The order of units is the same every pass, so the schedule is cyclic: while the last blocks of step ``s`` compute,
the first streamed blocks of step ``s + 1`` are already being copied (``passes`` > 1).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import torch

from ..store import Store, views


@dataclass
class _Slot:
    buffer: torch.Tensor
    ready: object = None      # cuda.Event: copy finished
    free: object = None       # cuda.Event: compute that used it finished
    unit: int = -1
    serial: int = -1          # position in the cyclic fetch sequence


@dataclass
class StreamStats:
    copies: int = 0
    copied_bytes: int = 0
    disk_reads: int = 0
    disk_bytes: int = 0
    disk_wait_s: float = 0.0
    compute_wait_s: float = 0.0
    load_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class _DiskReader(threading.Thread):
    """Reads disk-tier units in the cyclic order into a ring of pinned buffers."""

    def __init__(self, store: Store, group: str, order: list[int], nbytes: int, depth: int, passes: int | None,
                 stats: StreamStats):
        super().__init__(daemon=True, name=f'tinyh3-disk-{group}')
        self.store, self.group, self.order, self.passes = store, group, order, passes
        self.stats = stats
        pin = torch.cuda.is_available()
        self.buffers = [torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin) for _ in range(depth)]
        self.released = [None] * depth  # cuda events of the last copy out of each buffer
        self.free = queue.Queue()
        for i in range(depth):
            self.free.put(i)
        self.filled = queue.Queue()
        self.stop = threading.Event()
        self.error = None

    def run(self):
        try:
            p = 0
            while not self.stop.is_set() and (self.passes is None or p < self.passes):
                for unit in self.order:
                    while True:
                        try:
                            slot = self.free.get(timeout=0.2)
                            break
                        except queue.Empty:
                            if self.stop.is_set():
                                return
                    event = self.released[slot]
                    if event is not None:
                        event.synchronize()
                    t0 = time.perf_counter()
                    self.store.read_into(self.group, unit, self.buffers[slot])
                    self.stats.disk_reads += 1
                    self.stats.disk_bytes += self.store.unit_nbytes(self.group, unit)
                    self.stats.extra['disk_read_s'] = self.stats.extra.get('disk_read_s', 0.0) + \
                        time.perf_counter() - t0
                    self.filled.put((unit, slot))
                p += 1
        except BaseException as exc:  # surfaced to the consumer
            self.error = exc
            self.filled.put((None, None))

    def take(self, unit: int):
        t0 = time.perf_counter()
        got, slot = self.filled.get()
        self.stats.disk_wait_s += time.perf_counter() - t0
        if got is None:
            raise RuntimeError(f'disk reader failed: {self.error!r}')
        if got != unit:
            raise RuntimeError(f'disk reader out of order: expected unit {unit}, got {got}')
        return self.buffers[slot], slot

    def give_back(self, slot: int, event):
        self.released[slot] = event
        self.free.put(slot)

    def close(self):
        self.stop.set()


class UnitStreamer:
    """Serves the units of one store group in a fixed cyclic order."""

    def __init__(self, store: Store, group: str, *, device, tiers: dict, slots: int = 2, passes: int | None = None,
                 order: list[int] | None = None, disk_depth: int = 2, progress=None):
        self.store, self.group = store, group
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        n = len(store.units(group))
        self.order = order if order is not None else list(range(n))
        self.tiers = {i: tiers.get(i, 'disk') for i in self.order}
        self.stats = StreamStats()
        self.units = store.units(group)
        t0 = time.perf_counter()

        self.resident: dict[int, dict] = {}
        self.pinned: dict[int, torch.Tensor] = {}
        resident = [i for i in self.order if self.tiers[i] == 'vram']
        pinned = [i for i in self.order if self.tiers[i] == 'pinned']
        streamed = [i for i in self.order if self.tiers[i] != 'vram']
        self.streamed = streamed
        bounce = None
        if self.cuda and (resident or pinned):
            bounce = torch.empty(max(store.unit_nbytes(group, i) for i in resident + pinned), dtype=torch.uint8,
                                 pin_memory=True)
        done = 0
        total = len(resident) + len(pinned)
        for i in resident:
            nbytes = store.unit_nbytes(group, i)
            if self.cuda:
                store.read_into(group, i, bounce)
                buf = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
                buf.copy_(bounce[:nbytes])
            else:
                buf = torch.empty(nbytes, dtype=torch.uint8)
                store.read_into(group, i, buf)
            self.resident[i] = views(buf, self.units[i])
            self.resident[i]['__buffer__'] = buf
            done += 1
            if progress:
                progress(done, total)
        for i in pinned:
            nbytes = store.unit_nbytes(group, i)
            buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=self.cuda)
            store.read_into(group, i, buf)
            self.pinned[i] = buf
            done += 1
            if progress:
                progress(done, total)
        del bounce

        self.slot_bytes = max((store.unit_nbytes(group, i) for i in streamed), default=0)
        self.disk_units = [i for i in self.order if self.tiers[i] == 'disk']
        self.reader = None
        if self.disk_units and self.cuda:
            self.reader = _DiskReader(store, group, self.disk_units,
                                      max(store.unit_nbytes(group, i) for i in self.disk_units),
                                      disk_depth, passes, self.stats)
            self.reader.start()
        self.slots: list[_Slot] = []
        if streamed and self.cuda:
            self.copy_stream = torch.cuda.Stream(device=self.device)
            for _ in range(min(slots, len(streamed))):
                self.slots.append(_Slot(torch.empty(self.slot_bytes, dtype=torch.uint8, device=self.device),
                                        torch.cuda.Event(), None))
        self.serial_issued = -1      # last serial whose copy has been issued
        self.serial_used = -1        # last serial acquired by compute
        self.passes = passes
        self.stats.load_s = time.perf_counter() - t0
        self._cpu_cache: dict[int, dict] = {}

    # ------------------------------------------------------------------------------------------- scheduling
    def _unit_of_serial(self, serial: int) -> int:
        return self.streamed[serial % len(self.streamed)]

    def _issue(self, serial: int):
        if self.passes is not None and serial >= self.passes * len(self.streamed):
            return
        slot = self.slots[serial % len(self.slots)]
        unit = self._unit_of_serial(serial)
        nbytes = self.store.unit_nbytes(self.group, unit)
        with torch.cuda.stream(self.copy_stream):
            if slot.free is not None:
                self.copy_stream.wait_event(slot.free)
            if self.tiers[unit] == 'pinned':
                slot.buffer[:nbytes].copy_(self.pinned[unit][:nbytes], non_blocking=True)
            else:
                host, ring = self.reader.take(unit)
                slot.buffer[:nbytes].copy_(host[:nbytes], non_blocking=True)
                done = torch.cuda.Event()
                done.record(self.copy_stream)
                self.reader.give_back(ring, done)
            slot.ready.record(self.copy_stream)
        slot.unit, slot.serial = unit, serial
        self.stats.copies += 1
        self.stats.copied_bytes += nbytes
        self.serial_issued = serial

    def _fill_ahead(self):
        # Slots hold serials used .. used + K - 1; the slot of `used` itself is only refilled after `release`
        # recorded its free event, i.e. on the next acquire.
        target = self.serial_used + len(self.slots) - 1
        while self.serial_issued < target:
            nxt = self.serial_issued + 1
            if self.passes is not None and nxt >= self.passes * len(self.streamed):
                break
            self._issue(nxt)

    def acquire(self, unit: int) -> dict:
        """Tensor views of ``unit`` usable on the current stream."""
        if unit in self.resident:
            return self.resident[unit]
        if not self.cuda:
            return self._cpu_unit(unit)
        serial = self.serial_used + 1
        if self._unit_of_serial(serial) != unit:
            raise RuntimeError(f'units must be acquired in plan order; expected {self._unit_of_serial(serial)}, '
                               f'got {unit}')
        self.serial_used = serial
        self._fill_ahead()
        slot = self.slots[serial % len(self.slots)]
        torch.cuda.current_stream(self.device).wait_event(slot.ready)
        tensors = views(slot.buffer, self.units[unit])
        tensors['__slot__'] = slot
        return tensors

    def release(self, unit: int, tensors: dict):
        """Mark the views returned by ``acquire`` as no longer needed once the enqueued compute finishes."""
        slot = tensors.get('__slot__')
        if slot is None:
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        slot.free = event

    def prime(self):
        """Start copying the first streamed units before compute asks for them."""
        if self.cuda and self.slots:
            self._fill_ahead()

    def _cpu_unit(self, unit: int) -> dict:
        if unit not in self._cpu_cache:
            if self.tiers[unit] != 'pinned':
                buf = torch.empty(self.store.unit_nbytes(self.group, unit), dtype=torch.uint8)
                self.store.read_into(self.group, unit, buf)
                self.stats.disk_reads += 1
                return views(buf, self.units[unit])
            self._cpu_cache[unit] = views(self.pinned[unit], self.units[unit])
        return self._cpu_cache[unit]

    def close(self):
        if self.reader is not None:
            self.reader.close()
        self.slots.clear()
        self.resident.clear()
        self.pinned.clear()
        self._cpu_cache.clear()
