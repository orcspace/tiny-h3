"""Source checkpoints: download from the Hub and read sharded safetensors one tensor at a time."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from ..config import OFFICIAL_PATTERNS, SOURCES, SAMPLERS

log = logging.getLogger(__name__)


def download(target: Path, sampler: str = 'balanced', text_encoder: bool = True, token=None) -> dict:
    """Fetch exactly the files TinyH3 converts (~45 GB DiT + ~50 GB text encoder, bf16) into ``target``."""
    from huggingface_hub import snapshot_download
    target = Path(target)
    preset = SAMPLERS[sampler]
    base = snapshot_download(
        SOURCES['base_repo'], local_dir=target / 'base', token=token,
        allow_patterns=['transformer/*'] + (['text_encoder/*'] if text_encoder else []))
    adapter = snapshot_download(
        SOURCES['adapter_repo'], local_dir=target / 'adapter', token=token,
        allow_patterns=[f'{preset.adapter}/*'])
    official = snapshot_download(
        SOURCES['official_repo'], local_dir=target / 'official', token=token,
        allow_patterns=OFFICIAL_PATTERNS)
    return {'base': base, 'adapter': adapter, 'official': official}


class ShardedReader:
    """Lazy access to tensors of a (possibly sharded) safetensors checkpoint directory."""

    def __init__(self, root: Path):
        from safetensors import safe_open
        root = Path(root)
        if root.is_file():  # a single .safetensors file
            self.root, self.map = root.parent, {}
            with safe_open(str(root), 'pt') as h:
                for k in h.keys():
                    self.map[k] = root.name
            self._open, self._safe_open = {}, safe_open
            return
        self.root = root
        index = next(iter(sorted(self.root.glob('*.safetensors.index.json'))), None)
        if index is not None:
            self.map = json.loads(index.read_text())['weight_map']
        else:
            files = sorted(self.root.glob('*.safetensors'))
            if not files:
                raise FileNotFoundError(f'no safetensors in {self.root}')
            self.map = {}
            for f in files:
                with safe_open(str(f), 'pt') as h:
                    for k in h.keys():
                        self.map[k] = f.name
        self._open = {}
        self._safe_open = safe_open

    def keys(self):
        return self.map.keys()

    def __contains__(self, key):
        return key in self.map

    def get(self, key: str) -> torch.Tensor:
        name = self.map[key]
        if name not in self._open:
            # keep at most a few shard handles open (mmap-backed; pages are reclaimable)
            if len(self._open) > 4:
                self._open.pop(next(iter(self._open)))
            self._open[name] = self._safe_open(str(self.root / name), 'pt')
        return self._open[name].get_tensor(key)

    def metadata(self):
        out = {}
        for name in set(self.map.values()):
            with self._safe_open(str(self.root / name), 'pt') as h:
                out.update(h.metadata() or {})
        return out
