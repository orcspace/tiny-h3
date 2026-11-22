"""The whole pipeline on CPU with tiny random components: convert -> text -> denoise -> DWD + audio -> MP4.

Checks plumbing, shapes, file formats, caching and determinism; numerical parity of each part is covered by the
other test files.
"""
import json
import os

import pytest
import torch

pytest.importorskip('diffusers')
pytest.importorskip('av')


@pytest.fixture(scope='module')
def tiny_store(tmp_path_factory):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from tests.fixtures.modeling_minimax_h3_pruned import MiniMaxH3PrunedTransformer3DModel
    from tests.test_dit_parity import TINY, _randomize
    from tinyh3.convert.convert import convert

    root = tmp_path_factory.mktemp('e2e')
    torch.manual_seed(0)
    dit = MiniMaxH3PrunedTransformer3DModel(**TINY).eval()
    _randomize(dit)
    dit.save_pretrained(root / 'base' / 'transformer', safe_serialization=True)

    te = Qwen3VLForConditionalGeneration(Qwen3VLConfig(
        text_config=dict(hidden_size=256, intermediate_size=512, num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=64, vocab_size=1000, rope_theta=5000000.0,
                         rope_scaling={'rope_type': 'default', 'mrope_section': [12, 10, 10],
                                       'mrope_interleaved': True}),
        vision_config=dict(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=256,
                           deepstack_visual_indexes=[0]))).eval()
    te.save_pretrained(root / 'base' / 'text_encoder', safe_serialization=True)

    # adapter: DWD head (reduced-width Sol, 2 layers) + DWD norm + a LoRA
    g = torch.Generator().manual_seed(2)
    D, d = 512, 128
    r = lambda *s: torch.randn(*s, generator=g) * 0.05
    adapter = {'transformer.dwd_norm_out.linear.weight': r(2 * D, 8), 'transformer.dwd_norm_out.linear.bias': r(2 * D),
               'transformer.transformer_blocks.0.attn.to_k.lora_A.weight': r(4, D),
               'transformer.transformer_blocks.0.attn.to_k.lora_B.weight': r(D, 4)}
    pre = 'transformer.dwd_pixel_refiner.'
    adapter.update({pre + 'feature_expand.weight': r(d * 4, D), pre + 'feature_expand.bias': r(d * 4),
                    pre + 'register_tokens': r(1, 4, d), pre + 'norm_out.weight': 1 + r(d), pre + 'norm_out.bias': r(d),
                    pre + 'proj_out.weight': r(3 * 4 * 256, d), pre + 'proj_out.bias': r(3 * 4 * 256),
                    pre + 'first_frame_proj_out.weight': r(3 * 256, d), pre + 'first_frame_proj_out.bias': r(3 * 256)})
    for i in range(2):
        b = f'{pre}blocks.{i}.'
        adapter.update({b + 'norm1.weight': 1 + r(d), b + 'norm2.weight': 1 + r(d), b + 'scale1': 0.5 + r(d),
                        b + 'scale2': 0.5 + r(d), b + 'attn.to_qkv.weight': r(3 * d, d), b + 'attn.to_qkv.bias': r(3 * d),
                        b + 'attn.to_out.weight': r(d, d), b + 'attn.to_out.bias': r(d),
                        b + 'ff.w1.weight': r(8 * d, d), b + 'ff.w1.bias': r(8 * d),
                        b + 'ff.w2.weight': r(d, 4 * d), b + 'ff.w2.bias': r(d)})
    (root / 'adapter').mkdir()
    save_file(adapter, str(root / 'adapter' / 'diffusion_pytorch_model.safetensors'))

    official = root / 'official'
    AutoencoderKLMiniMaxH3Audio(encoder_dim=8, encoder_rates=(2, 4, 4, 5, 5), latent_dim=64, num_attention_heads=2,
                                decoder_dim=256).save_pretrained(official / 'audio_vae')
    tok = Tokenizer(models.WordLevel({f'w{i}': i for i in range(1000)} | {'[UNK]': 999}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    (official / 'tokenizer').mkdir(parents=True)
    tok.save(str(official / 'tokenizer' / 'tokenizer.json'))

    out = convert(root / 'base', root / 'adapter', official, root / 'store', policy='mixed', device='cpu')
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['dwd']['config'].update(num_heads=2, head_dim=64)  # the reduced Sol width of this fixture
    (out / 'manifest.json').write_text(json.dumps(manifest))
    return out


def test_generate_end_to_end(tiny_store, tmp_path, monkeypatch):
    monkeypatch.setenv('TINYH3_HOME', str(tmp_path / 'home'))
    from tinyh3.runtime.pipeline import GenRequest, generate
    from tinyh3.runtime.media import read_metadata
    import tinyh3.hardware as hw

    fake = hw.simulated(8, 16)
    fake.gpu_name = 'cpu-test'
    monkeypatch.setattr(hw, 'probe', lambda *a, **k: fake)
    events = []
    req = GenRequest(prompt='w1 w2 w3 w4 w5', seconds=5.2, height=64, width=96, sampler='turbo', seed=123,
                     model=str(tiny_store), output=str(tmp_path / 'out.mp4'))
    report = generate(req, events.append, subprocess=False)
    assert os.path.getsize(tmp_path / 'out.mp4') > 1000
    stages = {e['stage'] for e in events}
    assert {'text', 'plan', 'denoise', 'decode', 'audio', 'done'} <= stages
    assert report['stages']['decode']['frames'] == 124
    meta = read_metadata(tmp_path / 'out.mp4')
    assert meta['tinyh3']['seed'] == 123
    import av
    with av.open(str(tmp_path / 'out.mp4')) as c:
        kinds = sorted(s.type for s in c.streams)
        v = c.streams.video[0]
        assert (v.width, v.height) == (96, 64)
    assert kinds == ['audio', 'video']

    # the second run hits the prompt cache and reproduces the same video bytes for the same seed
    report2 = generate(GenRequest(**{**req.__dict__, 'output': str(tmp_path / 'out2.mp4')}), subprocess=False)
    assert report2['stages']['text']['cached'] is True


def test_generate_in_subprocesses(tiny_store, tmp_path, monkeypatch):
    monkeypatch.setenv('TINYH3_HOME', str(tmp_path / 'home'))
    from tinyh3.runtime.pipeline import GenRequest, generate
    events = []
    report = generate(GenRequest(prompt='w7 w8', seconds=5, height=64, width=64, sampler='turbo', seed=5,
                                 model=str(tiny_store), output=str(tmp_path / 'sp.mp4')), events.append)
    assert os.path.getsize(tmp_path / 'sp.mp4') > 1000
    assert any(e.get('stage') == 'denoise' and e.get('step') for e in events)
    assert report['stages']['denoise']['step_seconds']
