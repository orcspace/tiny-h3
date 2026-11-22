"""End to end on a tiny random MiniMax-H3 (pruned AdaLN): HF checkpoint + LoRA adapter -> tinyh3 convert ->
streamed quantized DiT, compared with the diffusers reference forward."""
import json
import math

import pytest
import torch

pytest.importorskip('diffusers')

from tinyh3.config import DiTConfig, Geometry, TEXT_TAG
from tinyh3.convert.convert import convert
from tinyh3.runtime.dit import DiT, Request
from tinyh3.runtime.layout import Layout, row_timesteps
from tinyh3.runtime.text_stage import refine
from tinyh3.store import Store

TINY = dict(num_attention_heads=4, attention_head_dim=128, hidden_size=512, num_layers=3, num_refiner_layers=1,
            ffn_dim=1024, text_dim=256, time_embed_dim=64, time_embed_hidden_dim=64, adaln_rank=8,
            time_table_size=33)


def _randomize(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if not p.is_floating_point() or name.endswith('inv_freq'):
                continue
            if 'norm' in name and name.endswith('weight') and p.ndim == 1:
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith('table'):
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                fan_in = p.shape[-1] if p.ndim > 1 else 64
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(fan_in))
            if 'folded_bias' in name:
                p.mul_(0.2)


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    from tests.fixtures.modeling_minimax_h3_pruned import MiniMaxH3PrunedTransformer3DModel
    root = tmp_path_factory.mktemp('tiny')
    torch.manual_seed(0)
    model = MiniMaxH3PrunedTransformer3DModel(**TINY).eval()
    _randomize(model)
    model.save_pretrained(root / 'base' / 'transformer', safe_serialization=True)
    # an adapter: LoRA on two linears of block 1 and one of the refiner, scaling 1 (diffusers default)
    from safetensors.torch import save_file
    lora = {}
    g = torch.Generator().manual_seed(1)
    targets = {'transformer_blocks.1.attn.to_q': (512, 512), 'transformer_blocks.1.ff.net.2': (512, 1024),
               'token_refiner.refiner_blocks.0.attn.to_v': (512, 512)}
    for module, (out_f, in_f) in targets.items():
        lora[f'transformer.{module}.lora_A.weight'] = torch.randn(4, in_f, generator=g) * 0.05
        lora[f'transformer.{module}.lora_B.weight'] = torch.randn(out_f, 4, generator=g) * 0.05
    (root / 'adapter').mkdir()
    save_file(lora, str(root / 'adapter' / 'diffusion_pytorch_model.safetensors'))
    # the reference model with the LoRA merged by hand
    with torch.no_grad():
        for module in targets:
            a = lora[f'transformer.{module}.lora_A.weight']
            b = lora[f'transformer.{module}.lora_B.weight']
            model.get_submodule(module).weight += b @ a
    return root, model


def _inputs(model, seed=3):
    geom = Geometry(height=64, width=96, num_frames=124, text_tokens=11)
    g = torch.Generator().manual_seed(seed)
    tags = torch.full((geom.text_tokens,), TEXT_TAG)
    layout = Layout.build(geom, tags)
    vrows = torch.randn(geom.video_rows, 96, generator=g)
    arows = torch.randn(layout.audio_idx.numel(), 32, generator=g)
    enc = torch.randn(geom.text_tokens, 256, generator=g)
    t_unique, t_index = row_timesteps(layout.video_idx, layout.audio_idx, 0, geom.text_tokens, 0.37, 0.61, 0.999)
    return geom, layout, vrows, arows, enc, t_unique, t_index


def _reference(model, layout, vrows, arows, enc, t_unique, t_index):
    with torch.no_grad():
        out = model(hidden_states=vrows[None], audio_hidden_states=arows[None], encoder_hidden_states=enc[None],
                    timestep=t_unique, timestep_indices=t_index, token_tags=layout.token_tags,
                    position_ids=layout.position_ids, video_indices=layout.video_idx,
                    audio_indices=layout.audio_idx, text_indices=layout.text_idx, return_dict=False)
    return out[0][0].float(), out[1][0].float()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize('policy,tol', [('int8', 0.03), ('mixed', 0.10), ('int4', 0.25)])
def test_dit_matches_diffusers(tiny, tmp_path, policy, tol):
    root, model = tiny
    out = convert(root / 'base', root / 'adapter', None, tmp_path / 'store', policy=policy, device='cpu',
                  with_text=False)
    store = Store(out)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['dit']['kinds'][0] == ('int4' if policy == 'int4' else 'int8')

    geom, layout, vrows, arows, enc, t_unique, t_index = _inputs(model)
    ref_v, ref_a = _reference(model, layout, vrows, arows, enc, t_unique, t_index)

    text_states = refine(store, enc, 'cpu')
    tiers = {0: 'vram', 1: 'pinned', 2: 'disk'}  # exercise every tier
    dit = DiT(store, device='cpu', tiers=tiers, head_group=2, ffn_chunk=200)
    req = Request(layout, torch.device('cpu'), dit.cfg)
    got_v, got_a, _ = dit.forward(req, vrows, arows, text_states, t_unique, t_index)
    assert got_v.shape == ref_v.shape and got_a.shape == ref_a.shape
    assert _rel(got_v, ref_v) < tol, _rel(got_v, ref_v)
    assert _rel(got_a, ref_a) < tol, _rel(got_a, ref_a)
