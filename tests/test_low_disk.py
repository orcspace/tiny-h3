"""Low-disk setup: tensors streamed over HTTP range requests, int4 text tower, text tower on another drive."""
import functools
import http.server
import json
import re
import threading
from pathlib import Path

import pytest
import torch

pytest.importorskip('diffusers')

from tests.test_comfy_source import _comfy_int8, _to_comfy_dit
from tests.test_dit_parity import tiny, _inputs, _reference, _rel  # noqa: F401 (fixture)
from tinyh3 import diskplan


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus single-range GETs (what the Hub's CDN answers)."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        rng = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not rng or not path.is_file():
            return super().do_GET()
        m = re.match(r'bytes=(\d+)-(\d+)', rng)
        start, end = int(m[1]), int(m[2])
        size = path.stat().st_size
        end = min(end, size - 1)
        with open(path, 'rb') as f:
            f.seek(start)
            body = f.read(end - start + 1)
        self.send_response(206)
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def server(tmp_path):
    handler = functools.partial(RangeHandler, directory=str(tmp_path))
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield tmp_path, f'http://127.0.0.1:{httpd.server_port}'
    httpd.shutdown()


def test_remote_file_reads_tensors_by_range(server):
    from safetensors.torch import save_file
    from tinyh3.convert.remote import RemoteFile
    root, base = server
    t = {'a': torch.randn(7, 5), 'b': torch.arange(10, dtype=torch.int8), 'c.d': torch.randn(3, dtype=torch.bfloat16)}
    save_file(t, str(root / 'x.safetensors'), metadata={'alpha': '8'})
    f = RemoteFile(base + '/x.safetensors')
    assert f.names == set(t) and f.metadata() == {'alpha': '8'}
    for k, v in t.items():
        assert torch.equal(f.get(k), v)


def test_convert_from_urls_int4(tiny, server, tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from tests.fixtures.modeling_minimax_h3_pruned import MiniMaxH3PrunedTransformer3DModel
    from tinyh3.config import DiTConfig
    from tinyh3.convert import convert as conv
    from tinyh3.runtime.dit import DiT, Request
    from tinyh3.runtime.text_stage import refine
    from tinyh3.store import Store
    root, model = tiny
    srv, base = server
    clean = MiniMaxH3PrunedTransformer3DModel.from_pretrained(root / 'base' / 'transformer').eval()
    save_file(_to_comfy_dit(clean), str(srv / 'dit.safetensors'))
    (srv / 'adapter').mkdir()
    (srv / 'adapter' / 'diffusion_pytorch_model.safetensors').write_bytes(
        (root / 'adapter' / 'diffusion_pytorch_model.safetensors').read_bytes())
    tiny_cfg = DiTConfig(num_heads=4, head_dim=128, hidden_size=512, num_layers=3, num_refiner_layers=1,
                         ffn_dim=1024, text_dim=256, time_table_size=33)
    monkeypatch.setattr(conv, 'DiTConfig', lambda: tiny_cfg)
    out = conv.convert(None, base + '/adapter/diffusion_pytorch_model.safetensors', None, tmp_path / 'store',
                       policy='int4', device='cpu', comfy_dit=base + '/dit.safetensors', with_text=False)
    store = Store(out)
    m = json.loads((out / 'manifest.json').read_text())
    assert m['dit']['kinds'] == ['int4'] * 3 and m['recipe']['adapter'] == 'adapter'
    geom, layout, vrows, arows, enc, t_unique, t_index = _inputs(model)
    ref_v, _ = _reference(model, layout, vrows, arows, enc, t_unique, t_index)
    dit = DiT(store, device='cpu', head_group=2)
    got_v, _, _ = dit.forward(Request(layout, torch.device('cpu'), dit.cfg), vrows, arows,
                              refine(store, enc, 'cpu'), t_unique, t_index)
    assert _rel(got_v, ref_v) < 0.3   # int4 on random weights: coarse, but the whole path is wired right


def test_text_tower_int4_rotated(tmp_path):
    pytest.importorskip('transformers')
    from safetensors.torch import save_file
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from tinyh3.config import TextEncoderConfig
    from tinyh3.convert.comfy import ComfyTextView
    from tinyh3.convert.convert import convert_text
    from tinyh3.runtime.text_stage import encode_tokens
    from tinyh3.store import FORMAT, Store
    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
        text_config=dict(hidden_size=512, intermediate_size=1024, num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=128, vocab_size=1000, rope_theta=5000000.0,
                         rope_scaling={'rope_type': 'default', 'mrope_section': [24, 20, 20], 'mrope_interleaved': True}),
        vision_config=dict(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=512,
                           deepstack_visual_indexes=[0]))).eval()
    comfy = {}
    for k, v in model.state_dict().items():
        if not k.startswith('model.language_model.'):
            continue
        name = 'model.' + k[len('model.language_model.'):]
        if k.endswith('_proj.weight'):
            _comfy_int8(comfy, name[:-7], v)
        else:
            comfy[name] = v.contiguous()
    save_file(comfy, str(tmp_path / 'te.safetensors'))
    cfg = TextEncoderConfig(hidden_size=512, num_heads=4, num_kv_heads=2, head_dim=128, intermediate_size=1024,
                            vocab_size=1000, num_layers_used=2, mrope_section=(24, 20, 20))
    out = tmp_path / 'store'
    out.mkdir()
    sizes = {}
    for kind in ('int8', 'int4'):
        text = convert_text(ComfyTextView(tmp_path / 'te.safetensors'), out / kind, cfg, kind, 'cpu',
                            lambda *a: None, rotate=True) if (out / kind).mkdir() is None else None
        (out / kind / 'manifest.json').write_text(json.dumps({'format': FORMAT, 'text': text, 'dit': {}}))
        sizes[kind] = sum(f.stat().st_size for f in (out / kind / 'text').glob('unit_*.bin'))
    assert sizes['int4'] < 0.6 * sizes['int8']
    ids = [4, 8, 15, 16, 23, 42]
    with torch.no_grad():
        t = torch.tensor([ids])
        want = model.model(input_ids=t, attention_mask=torch.ones_like(t), mm_token_type_ids=torch.zeros_like(t),
                           use_cache=False, output_hidden_states=True).hidden_states[2][0]
    errs = {k: ((encode_tokens(Store(out / k), ids, 'cpu') - want).norm() / want.norm()).item() for k in sizes}
    assert errs['int8'] < 0.03 and errs['int4'] < 0.35, errs


def test_disk_estimates_and_text_dir(tmp_path):
    full = diskplan.estimate('int8', 'int8')
    low = diskplan.estimate('int4', 'int4')
    no_img = diskplan.estimate('int4', 'int4', images=False)
    assert low['total'] < 0.6 * full['total'] and no_img['total'] < low['total']
    assert abs(low['main_drive'] + low['text_drive'] - low['total']) < 1
    assert low['main_drive'] < 14 * diskplan.GB          # the main drive keeps DiT + decoders only
    assert diskplan.estimate('int4', 'int4', ref=True)['total'] > low['total'] + 10 * diskplan.GB
    assert diskplan.check_space(tmp_path, None, {'main_drive': 1, 'text_drive': 1, 'total': 1}) == []
    assert diskplan.check_space(tmp_path, None, {'main_drive': 1, 'text_drive': 1, 'total': 1 << 62})

    model, other = tmp_path / 'model', tmp_path / 'other_drive' / 'text'
    model.mkdir()
    diskplan.link_dir(model / 'text', other)
    (model / 'text' / 'unit.bin').write_bytes(b'x' * 10)
    assert (other / 'unit.bin').exists() and (model / 'text').is_symlink()
    diskplan.link_dir(model / 'text', other)              # idempotent
    with pytest.raises(FileExistsError):
        diskplan.link_dir(model / 'text', tmp_path / 'elsewhere')
