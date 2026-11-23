"""Fetch selected tensors of a remote safetensors checkpoint with HTTP range requests.

The video VAE ships as 10.4 GB of fp32, of which the keyframe / reference encoder is 0.72 GB: reading only those
tensors (the header, then one range per contiguous run) avoids downloading the 2.4B-parameter decoder for tasks that
do not need it.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

_DT = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16, 'I8': torch.int8, 'U8': torch.uint8,
       'I64': torch.int64, 'I32': torch.int32}


def _session(token=None):
    import requests
    from huggingface_hub import get_token
    s = requests.Session()
    tok = token or get_token()
    if tok:
        s.headers['Authorization'] = f'Bearer {tok}'
    return s


def _range(session, url, start, end) -> bytes:
    r = session.get(url, headers={'Range': f'bytes={start}-{end}'}, timeout=600, allow_redirects=True)
    r.raise_for_status()
    return r.content


def fetch_tensors(repo: str, filename: str, wanted, token=None, revision='main') -> dict:
    """{name: tensor} for every tensor of ``repo/filename`` whose name satisfies ``wanted(name)``."""
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(repo, filename, revision=revision)
    s = _session(token)
    n = struct.unpack('<Q', _range(s, url, 0, 7))[0]
    header = json.loads(_range(s, url, 8, 7 + n))
    header.pop('__metadata__', None)
    base = 8 + n
    picked = sorted(((k, v) for k, v in header.items() if wanted(k)), key=lambda kv: kv[1]['data_offsets'][0])
    out, run = {}, []

    def flush():
        if not run:
            return
        a, b = run[0][1]['data_offsets'][0], run[-1][1]['data_offsets'][1]
        blob = _range(s, url, base + a, base + b - 1)
        for k, v in run:
            o0, o1 = v['data_offsets']
            t = torch.frombuffer(bytearray(blob[o0 - a:o1 - a]), dtype=_DT[v['dtype']]).reshape(v['shape'])
            out[k] = t.clone()
        run.clear()

    for k, v in picked:
        if run and v['data_offsets'][0] != run[-1][1]['data_offsets'][1]:
            flush()
        run.append((k, v))
        if run[-1][1]['data_offsets'][1] - run[0][1]['data_offsets'][0] > 512 << 20:
            flush()
    flush()
    return out


def fetch_vae(repo: str, out: Path, *, decoder: bool, token=None) -> None:
    """``vae/config.json`` + ``vae/encoder.safetensors`` (fp32) and, with ``decoder``, ``vae/decoder.safetensors``
    (fp16) of the MiniMax-H3 video VAE."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    out = Path(out) / 'vae'
    out.mkdir(parents=True, exist_ok=True)
    cfg = hf_hub_download(repo, 'vae/config.json', token=token)
    (out / 'config.json').write_text(Path(cfg).read_text())
    index = json.loads(Path(hf_hub_download(repo, 'vae/diffusion_pytorch_model.safetensors.index.json',
                                            token=token)).read_text())['weight_map']
    parts = {'encoder': ('encoder.', 'quant_conv.')}
    if decoder:
        parts['decoder'] = ('decoder.', 'post_quant_conv.')
    for name, prefixes in parts.items():
        target = out / f'{name}.safetensors'
        if target.exists():
            continue
        tensors = {}
        for shard in sorted({f for k, f in index.items() if k.startswith(prefixes)}):
            tensors.update(fetch_tensors(repo, f'vae/{shard}', lambda k: k.startswith(prefixes), token))
        dtype = torch.float32 if name == 'encoder' else torch.float16
        save_file({k: v.to(dtype).contiguous() for k, v in tensors.items()}, str(target))


class RemoteFile:
    """A remote .safetensors file read tensor by tensor with range requests (same interface as ``comfy._File``).

    Nothing is written to disk: the converter streams a tensor, quantizes it and drops it. The whole 21 GB DiT or
    27 GB text tower is therefore converted with a few GB of RAM and zero temporary disk space.
    """

    def __init__(self, url: str, token=None, retries: int = 6):
        self.url, self.retries = url, retries
        self.session = _session(token)
        n = struct.unpack('<Q', self._range(0, 7))[0]
        header = json.loads(self._range(8, 7 + n))
        self.meta = header.pop('__metadata__', None) or {}
        self.header, self.base = header, 8 + n
        self.names = set(header)

    # the interface of ``sources.ShardedReader``, so a LoRA / DWD adapter can be read remotely too
    def keys(self):
        return self.names

    def __contains__(self, key):
        return key in self.names

    def metadata(self) -> dict:
        return self.meta

    def _range(self, start: int, end: int) -> bytes:
        import time
        last = None
        for attempt in range(self.retries):
            try:
                data = _range(self.session, self.url, start, end)
                if len(data) != end - start + 1:
                    raise IOError(f'short range read: {len(data)} of {end - start + 1} bytes')
                return data
            except Exception as exc:  # network hiccups on a 50 GB transfer are expected: back off and retry
                last = exc
                time.sleep(min(2 ** attempt, 30))
        raise IOError(f'{self.url}: {last}')

    def get(self, key: str) -> torch.Tensor:
        meta = self.header[key]
        o0, o1 = meta['data_offsets']
        if o1 == o0:
            return torch.empty(meta['shape'], dtype=_DT[meta['dtype']])
        blob = self._range(self.base + o0, self.base + o1 - 1)
        return torch.frombuffer(bytearray(blob), dtype=_DT[meta['dtype']]).reshape(meta['shape']).clone()


def hub_url(repo: str, filename: str, revision: str = 'main') -> str:
    from huggingface_hub import hf_hub_url
    return hf_hub_url(repo, filename, revision=revision)
