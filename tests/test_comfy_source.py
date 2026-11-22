"""Comfy-Org INT8 ConvRot single files as the conversion source (DiT and text tower)."""
import json

import pytest
import torch

pytest.importorskip('diffusers')

from tinyh3.kernels import reference as ref
from tests.test_dit_parity import tiny, _inputs, _reference, _rel  # noqa: F401 (fixture)

QUANT = json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256}).encode()


def _comfy_int8(out: dict, name: str, w: torch.Tensor):
    q, s = ref.quantize_weight_int8(ref.rotate(w.float()))
    out[name + '.weight'] = q
    out[name + '.weight_scale'] = s.reshape(-1, 1)
    out[name + '.comfy_quant'] = torch.tensor(list(QUANT), dtype=torch.uint8)


def _to_comfy_dit(model) -> dict:
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    out = {}

    def block(src, dst, adaln):
        _comfy_int8(out, f'{dst}.attn.qkv_proj', torch.cat([sd[f'{src}.attn.to_{n}.weight'] for n in 'qkv']))
        _comfy_int8(out, f'{dst}.attn.out_proj', sd[f'{src}.attn.to_out.0.weight'])
        up, gate = sd[f'{src}.ff.net.0.proj.weight'].chunk(2)
        _comfy_int8(out, f'{dst}.mlp.fc1', torch.cat([gate, up]))
        _comfy_int8(out, f'{dst}.mlp.fc2', sd[f'{src}.ff.net.2.weight'])
        out[f'{dst}.norm1.weight'] = sd[f'{src}.norm1.weight']
        out[f'{dst}.norm2.weight'] = sd[f'{src}.norm2.weight']
        out[f'{dst}.attn.q_norm.weight'] = sd[f'{src}.attn.norm_q.weight']
        out[f'{dst}.attn.k_norm.weight'] = sd[f'{src}.attn.norm_k.weight']
        if adaln:
            out[f'{dst}.adaln_proj.linear.weight'] = sd[f'{src}.adaln_proj.linear.weight']
            out[f'{dst}.adaln_proj.linear.bias'] = sd[f'{src}.adaln_proj.folded_bias']

    block('token_refiner.refiner_blocks.0', 'token_refiner.blocks.0', False)
    for i in range(3):
        block(f'transformer_blocks.{i}', f'blocks.{i}', True)
    for c, d in (('video_patch_proj.weight', 'proj_in.weight'), ('video_patch_proj.bias', 'proj_in.bias'),
                 ('audio_patch_proj.weight', 'audio_proj_in.weight'), ('audio_patch_proj.bias', 'audio_proj_in.bias'),
                 ('condition_proj.weight', 'context_embedder.weight'), ('condition_proj.bias', 'context_embedder.bias'),
                 ('adaln_t_table', 'time_embedder.table'), ('token_refiner.final_norm.weight', 'token_refiner.final_norm.weight'),
                 ('final_layer.norm.weight', 'norm_out.norm.weight'),
                 ('final_layer.adaln_proj.linear.weight', 'norm_out.linear.weight'),
                 ('final_layer.adaln_proj.linear.bias', 'norm_out.folded_bias'),
                 ('final_layer.video_out.weight', 'proj_out.weight'), ('final_layer.video_out.bias', 'proj_out.bias'),
                 ('final_layer.audio_out.weight', 'audio_proj_out.weight'), ('final_layer.audio_out.bias', 'audio_proj_out.bias')):
        out[c] = sd[d]
    return {k: v.contiguous() for k, v in out.items()}


def test_comfy_dit_source(tiny, tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from tinyh3.config import DiTConfig
    from tinyh3.convert import convert as conv
    from tinyh3.runtime.dit import DiT, Request
    from tinyh3.runtime.text_stage import refine
    from tinyh3.store import Store
    from tests.test_dit_parity import TINY
    root, model = tiny  # reference already has the adapter's LoRA merged
    path = tmp_path / 'comfy.safetensors'
    # the Comfy file holds the *base* weights: unmerge the fixture's LoRA by converting a clean copy
    from tests.fixtures.modeling_minimax_h3_pruned import MiniMaxH3PrunedTransformer3DModel
    base = MiniMaxH3PrunedTransformer3DModel.from_pretrained(root / 'base' / 'transformer').eval()
    save_file(_to_comfy_dit(base), str(path))
    tiny_cfg = DiTConfig(num_heads=4, head_dim=128, hidden_size=512, num_layers=3, num_refiner_layers=1,
                         ffn_dim=1024, text_dim=256, time_table_size=33)
    monkeypatch.setattr(conv, 'DiTConfig', lambda: tiny_cfg)
    out = conv.convert(None, root / 'adapter', None, tmp_path / 'store', policy='int8', device='cpu',
                       comfy_dit=path, with_text=False)
    store = Store(out)
    assert store.manifest['recipe']['source'] == 'comfy-int8'
    geom, layout, vrows, arows, enc, t_unique, t_index = _inputs(model)
    ref_v, ref_a = _reference(model, layout, vrows, arows, enc, t_unique, t_index)
    dit = DiT(store, device='cpu', head_group=2)
    got_v, got_a, _ = dit.forward(Request(layout, torch.device('cpu'), dit.cfg), vrows, arows,
                                  refine(store, enc, 'cpu'), t_unique, t_index)
    assert _rel(got_v, ref_v) < 0.04 and _rel(got_a, ref_a) < 0.04


def test_comfy_int8_roundtrip_is_lossless():
    """An untouched Comfy int8 layer, unrotated and re-quantized, gives back the same integers."""
    from tinyh3.convert.convert import quantize_linear
    torch.manual_seed(0)
    w = torch.randn(64, 512)
    q, s = ref.quantize_weight_int8(ref.rotate(w))
    unrot = ref.rotate(q.float() * s[:, None])
    rec = quantize_linear(unrot, 'int8')
    assert torch.equal(rec['q'], q)


def test_comfy_text_source(tmp_path):
    pytest.importorskip('transformers')
    from safetensors.torch import save_file
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from tinyh3.convert.comfy import ComfyTextView
    from tinyh3.convert.convert import convert_text
    from tinyh3.config import TextEncoderConfig
    from tinyh3.store import FORMAT, Store
    from tinyh3.runtime.text_stage import encode_tokens
    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
        text_config=dict(hidden_size=256, intermediate_size=512, num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=64, vocab_size=1000, rope_theta=5000000.0,
                         rope_scaling={'rope_type': 'default', 'mrope_section': [12, 10, 10], 'mrope_interleaved': True}),
        vision_config=dict(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=256,
                           deepstack_visual_indexes=[0]))).eval()
    sd = model.state_dict()
    comfy = {}
    for k, v in sd.items():
        if not k.startswith('model.language_model.'):
            continue
        name = 'model.' + k[len('model.language_model.'):]
        if k.endswith('_proj.weight'):
            _comfy_int8(comfy, name[:-7], v)
        else:
            comfy[name] = v.contiguous()
    save_file(comfy, str(tmp_path / 'te.safetensors'))
    cfg = TextEncoderConfig(hidden_size=256, num_heads=4, num_kv_heads=2, head_dim=64, intermediate_size=512,
                            vocab_size=1000, num_layers_used=2)
    out = tmp_path / 'store'
    out.mkdir()
    text = convert_text(ComfyTextView(tmp_path / 'te.safetensors'), out, cfg, 'int8', 'cpu', lambda *a: None,
                        rotate=True)
    (out / 'manifest.json').write_text(json.dumps({'format': FORMAT, 'text': text, 'dit': {}}))
    ids = [4, 8, 15, 16, 23, 42]
    got = encode_tokens(Store(out), ids, 'cpu')
    with torch.no_grad():
        t = torch.tensor([ids])
        want = model.model(input_ids=t, attention_mask=torch.ones_like(t), mm_token_type_ids=torch.zeros_like(t),
                           use_cache=False, output_hidden_states=True).hidden_states[2][0]
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 0.02, rel
