"""Measure the machine: GPU, VRAM, host memory (cgroup-aware), disk read speed."""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path

GiB = 1 << 30

# Dense tensor throughput by GPU name fragment: (int8 TOPS, fp16/bf16-with-fp32-accumulate TFLOPS).
# GeForce parts run fp16 with fp32 accumulation at half rate, which is why int8 GEMMs matter so much on them.
KNOWN_GPUS = {
    'RTX 3050': (73, 18), 'RTX 3060 Ti': (130, 32), 'RTX 3060': (102, 25), 'RTX 3070 Ti': (174, 44),
    'RTX 3070': (163, 41), 'RTX 3080 Ti': (273, 68), 'RTX 3080': (238, 60), 'RTX 3090': (284, 71),
    'RTX 4060 Ti': (177, 44), 'RTX 4060': (121, 30), 'RTX 4070 Ti': (321, 80), 'RTX 4070': (233, 58),
    'RTX 4080': (390, 97), 'RTX 4090': (660, 165), 'RTX 5060': (190, 48), 'RTX 5070': (247, 62),
    'RTX 5080': (450, 113), 'RTX 5090': (838, 209), 'A100': (624, 312), 'H100': (1979, 989),
    'L40': (724, 181), 'A6000': (310, 155), 'A4000': (153, 77),
}


@dataclass
class Hardware:
    gpu_name: str = 'none'
    compute_capability: tuple = (0, 0)
    vram_total: int = 0
    vram_free: int = 0
    ram_total: int = 0
    ram_available: int = 0
    os: str = platform.system()
    int8_tops: float = 0.0
    fp16_tflops: float = 0.0
    pcie_gbps: float = 12.0
    disk_gbps: float = 1.5
    fp16acc_tflops: float = 0.0   # FP16 with FP16 accumulation: not capped on GeForce
    unlocked: bool | None = None  # gpu-unlock state measured by gpu_tuning (None: unknown / not GeForce)
    spark: bool = False           # Spark-Attn kernels importable for this GPU
    sage: bool = False            # SageAttention importable
    attention_ms: dict | None = None  # measured by `tinyh3 bench`: backend -> ms at 16384 tokens x 8 heads

    def as_dict(self) -> dict:
        d = asdict(self)
        d['vram_total_gib'] = round(self.vram_total / GiB, 2)
        d['vram_free_gib'] = round(self.vram_free / GiB, 2)
        d['ram_total_gib'] = round(self.ram_total / GiB, 2)
        d['ram_available_gib'] = round(self.ram_available / GiB, 2)
        return d


def throughput_for(name: str, cc: tuple) -> tuple[float, float]:
    for key in sorted(KNOWN_GPUS, key=len, reverse=True):
        if key.lower() in name.lower():
            return KNOWN_GPUS[key]
    if cc >= (8, 0):
        return (100.0, 25.0)
    return (0.0, 15.0)


def _cgroup_limit() -> int | None:
    for path in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            raw = Path(path).read_text().strip()
            if raw != 'max' and int(raw) < 1 << 60:
                return int(raw)
        except (OSError, ValueError):
            continue
    return None


def _cgroup_usage() -> int | None:
    for path in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def host_memory() -> tuple[int, int]:
    """(total, available) bytes, bounded by a cgroup limit when one applies."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        total, available = vm.total, vm.available
    except ImportError:  # pragma: no cover
        pages = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        total = available = pages
    limit = _cgroup_limit()
    if limit:
        usage = _cgroup_usage() or 0
        total = min(total, limit)
        available = min(available, max(limit - usage, 0))
    return total, available


def probe(measure_disk_at: Path | None = None) -> Hardware:
    sim = os.environ.get('TINYH3_SIMULATE')  # "vram_gib,ram_gib[,gpu]" — UI/plan demos without the hardware
    if sim:
        parts = sim.split(',')
        return simulated(float(parts[0]), float(parts[1]), parts[2] if len(parts) > 2 else 'RTX 3070')
    hw = Hardware()
    hw.ram_total, hw.ram_available = host_memory()
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            hw.gpu_name = props.name
            hw.compute_capability = (props.major, props.minor)
            hw.vram_free, hw.vram_total = torch.cuda.mem_get_info(0)
    except Exception:  # pragma: no cover - no torch / broken driver
        pass
    hw.int8_tops, hw.fp16_tflops = throughput_for(hw.gpu_name, hw.compute_capability)
    hw.fp16acc_tflops = 2 * hw.fp16_tflops if 'GeForce' in hw.gpu_name else hw.fp16_tflops
    try:
        from .gpu_tuning import load
        prof = load()
        if prof is not None and prof.gpu == hw.gpu_name:
            hw.int8_tops, hw.fp16_tflops = prof.int8_tops, prof.bf16_tflops
            hw.fp16acc_tflops, hw.unlocked = prof.fp16acc_tflops, prof.unlocked
            import json as _json
            from .gpu_tuning import _path
            hw.attention_ms = (_json.loads(_path().read_text()).get('attention_ms') or {}).get('16384')
    except Exception:
        pass
    try:
        from .runtime.attention import spark_available, _sage
        hw.spark = spark_available()
        hw.sage = _sage() is not None
    except Exception:
        pass
    cached = disk_speed_cache()
    if measure_disk_at is not None:
        hw.disk_gbps = measure_disk(measure_disk_at) or hw.disk_gbps
    elif cached:
        hw.disk_gbps = cached
    return hw


def _speed_file() -> Path:
    from .config import home
    return home() / 'disk_speed.json'


def disk_speed_cache() -> float | None:
    try:
        return float(json.loads(_speed_file().read_text())['gbps'])
    except (OSError, ValueError, KeyError):
        return None


def measure_disk(path: Path, nbytes: int = 512 << 20) -> float | None:
    """Sequential read speed of an existing large file under ``path`` (page cache may inflate it; we keep the
    lower of this and any earlier measurement)."""
    path = Path(path)
    files = sorted((p for p in path.rglob('*.bin') if p.stat().st_size >= 64 << 20), key=lambda p: -p.stat().st_size)
    if not files:
        return None
    buf = bytearray(16 << 20)
    done, start = 0, time.perf_counter()
    for f in files:
        with open(f, 'rb', buffering=0) as fh:
            while done < nbytes:
                n = fh.readinto(buf)
                if not n:
                    break
                done += n
        if done >= nbytes:
            break
    elapsed = time.perf_counter() - start
    gbps = done / max(elapsed, 1e-6) / 1e9
    previous = disk_speed_cache()
    gbps = min(gbps, previous) if previous else gbps
    try:
        _speed_file().parent.mkdir(parents=True, exist_ok=True)
        _speed_file().write_text(json.dumps({'gbps': gbps, 'measured_at': time.time()}))
    except OSError:
        pass
    return gbps


def simulated(vram_gib: float, ram_gib: float, gpu: str = 'RTX 3070', disk_gbps: float = 2.0,
              unlocked: bool = False, spark: bool = True, sage: bool = True) -> Hardware:
    """A hypothetical machine for ``tinyh3 plan --vram-gib --ram-gib``."""
    cc = (8, 6) if '30' in gpu else (8, 9)
    tops, tflops = throughput_for(gpu, cc)
    return Hardware(gpu_name=f'NVIDIA GeForce {gpu} (simulated)', compute_capability=cc,
                    vram_total=int(vram_gib * GiB), vram_free=int((vram_gib - 0.45) * GiB),
                    ram_total=int(ram_gib * GiB), ram_available=int((ram_gib - 3.5) * GiB),
                    int8_tops=tops, fp16_tflops=tflops * (2 if unlocked else 1), disk_gbps=disk_gbps,
                    fp16acc_tflops=2 * tflops, unlocked=unlocked, spark=spark, sage=sage)
