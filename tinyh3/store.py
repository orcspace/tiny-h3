"""The TinyH3 weight store.

A converted model is a directory with ``manifest.json`` and *packed blobs*: every streamed unit (a DiT block, a text
encoder layer) is one file holding all of its tensors back to back, each aligned to 256 bytes, the file padded to
4 KiB. Streaming a unit is then one sequential ``readinto`` into a pinned buffer and one host-to-device copy into a
GPU slot; the tensors are views into the slot (``views``), so no per-tensor allocation or copy ever happens.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

ALIGN = 256
FILE_ALIGN = 4096
FORMAT = 'tinyh3-store-v1'

_DTYPES = {
    'int8': torch.int8, 'uint8': torch.uint8, 'bfloat16': torch.bfloat16, 'float16': torch.float16,
    'float32': torch.float32, 'int32': torch.int32, 'int64': torch.int64,
}
_NAMES = {v: k for k, v in _DTYPES.items()}


def _round_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def tensor_bytes(t: torch.Tensor) -> memoryview:
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return memoryview(t.view(torch.int16).numpy()).cast('B')
    return memoryview(t.numpy()).cast('B')


class BlobWriter:
    """Accumulates tensors and writes them as one aligned blob."""

    def __init__(self):
        self.entries: list[tuple[str, torch.Tensor]] = []
        self.linears: dict = {}

    def add(self, name: str, tensor: torch.Tensor):
        self.entries.append((name, tensor.detach().contiguous().cpu()))

    def add_linear(self, name: str, record: dict):
        """Store a weight record from ``quant.quantize_linear`` under ``name``."""
        meta = {k: v for k, v in record.items() if not isinstance(v, torch.Tensor)}
        for key, value in record.items():
            if isinstance(value, torch.Tensor):
                self.add(f'{name}.{key}', value)
        self.linears[name] = meta

    def write(self, path: Path) -> dict:
        layout, offset = {}, 0
        for name, t in self.entries:
            offset = _round_up(offset, ALIGN)
            nbytes = t.numel() * t.element_size()
            layout[name] = {'dtype': _NAMES[t.dtype], 'shape': list(t.shape), 'offset': offset, 'nbytes': nbytes}
            offset += nbytes
        total = _round_up(max(offset, 1), FILE_ALIGN)
        tmp = Path(str(path) + '.part')
        with open(tmp, 'wb') as f:
            pos = 0
            for name, t in self.entries:
                start = layout[name]['offset']
                if start > pos:
                    f.write(b'\0' * (start - pos))
                data = tensor_bytes(t)
                f.write(data)
                pos = start + len(data)
            f.write(b'\0' * (total - pos))
        os.replace(tmp, path)
        return {'file': Path(path).name, 'nbytes': total, 'tensors': layout, 'linears': self.linears}


def views(buffer: torch.Tensor, unit: dict) -> dict:
    """Tensor views (on the buffer's device) of a unit laid out in a uint8 ``buffer``."""
    out = {}
    for name, spec in unit['tensors'].items():
        dtype = _DTYPES[spec['dtype']]
        start, nbytes = spec['offset'], spec['nbytes']
        flat = buffer[start:start + nbytes]
        out[name] = flat.view(dtype).view(spec['shape']) if nbytes else torch.empty(spec['shape'], dtype=dtype)
    return out


def linear_records(tensors: dict, unit: dict) -> dict:
    """Group views into weight records: ``{'qkv': {'kind': 'int4', 'q': ..., 'scale': ..., ...}}``."""
    records = {}
    for name, meta in unit.get('linears', {}).items():
        rec = dict(meta)
        prefix = name + '.'
        for key, value in tensors.items():
            if key.startswith(prefix) and '.' not in key[len(prefix):]:
                rec[key[len(prefix):]] = value
        records[name] = rec
    return records


class Store:
    """Read side of a converted model directory."""

    def __init__(self, root):
        self.root = Path(root)
        with open(self.root / 'manifest.json') as f:
            self.manifest = json.load(f)
        if self.manifest.get('format') != FORMAT:
            raise ValueError(f'{self.root} is not a {FORMAT} store')

    # -- units -------------------------------------------------------------------------------------------
    def units(self, group: str) -> list[dict]:
        return self.manifest[group]['units']

    def unit_path(self, group: str, index: int) -> Path:
        return self.root / group / self.units(group)[index]['file']

    def unit_nbytes(self, group: str, index: int) -> int:
        return self.units(group)[index]['nbytes']

    def read_into(self, group: str, index: int, buffer: torch.Tensor, chunk: int = 64 << 20) -> None:
        """Read one unit into a CPU uint8 ``buffer`` (pinned or not) with large sequential reads."""
        nbytes = self.unit_nbytes(group, index)
        from . import native
        if buffer.is_contiguous() and native.read_into(self.unit_path(group, index), 0, buffer, nbytes):
            return
        target = buffer[:nbytes].numpy()
        with open(self.unit_path(group, index), 'rb', buffering=0) as f:
            view = memoryview(target)
            done = 0
            while done < nbytes:
                n = f.readinto(view[done:min(nbytes, done + chunk)])
                if not n:
                    raise IOError(f'short read in {self.unit_path(group, index)}')
                done += n

    def load_unit(self, group: str, index: int, device='cpu') -> dict:
        buf = torch.empty(self.unit_nbytes(group, index), dtype=torch.uint8)
        self.read_into(group, index, buf)
        if device != 'cpu':
            buf = buf.to(device)
        unit = self.units(group)[index]
        return {'tensors': views(buf, unit), 'unit': unit, 'buffer': buf}

    # -- small whole-file components ----------------------------------------------------------------------
    def load_safetensors(self, rel: str, device='cpu') -> dict:
        from safetensors.torch import load_file
        return load_file(str(self.root / rel), device=str(device))

    def path(self, rel: str) -> Path:
        return self.root / rel

    @property
    def recipe(self) -> dict:
        return self.manifest.get('recipe', {})


def embedding_rows(path: Path, rows, dim: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Gather ``rows`` of a raw row-major [V, dim] table without reading the rest of the file."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    table = np.memmap(path, dtype=np.int16 if itemsize == 2 else np.float32, mode='r').reshape(-1, dim)
    picked = torch.from_numpy(np.ascontiguousarray(table[np.asarray(rows, dtype=np.int64)]))
    return picked.view(dtype) if itemsize == 2 else picked
