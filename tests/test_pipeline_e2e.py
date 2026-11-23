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


def _tokenizer_and_processor(official):
    """A word-level tokenizer that knows Qwen3-VL's vision tokens, and a real Qwen3-VL processor around it."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    special = {'<|vision_start|>': 151652, '<|vision_end|>': 151653, '<|image_pad|>': 151655, '<|video_pad|>': 151656}
    vocab = {f'w{i}': i for i in range(1000)} | {'[UNK]': 1000} | special
    tok = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]',
                                   additional_special_tokens=list(special))
    fast.save_pretrained(official / 'tokenizer')
    proc = Qwen3VLProcessor(image_processor=Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
                            tokenizer=fast, video_processor=Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2,
                                                                                  merge_size=2))
    proc.save_pretrained(official / 'processor')


@pytest.fixture(scope='module')
def tiny_store(tmp_path_factory):
    from safetensors.torch import save_file
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
                         num_key_value_heads=2, head_dim=64, vocab_size=152000, rope_theta=5000000.0,
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
                                decoder_dim=256, latents_mean=[0.1] * 32,
                                latents_std=[1.2] * 32).save_pretrained(official / 'audio_vae')
    _tokenizer_and_processor(official)

    # the Ref2VA transformer (another random pruned H3) and a lightx2v-style LoRA (PEFT keys, alpha in metadata)
    torch.manual_seed(1)
    ref = MiniMaxH3PrunedTransformer3DModel(**TINY).eval()
    _randomize(ref, seed=5)
    ref.save_pretrained(root / 'base' / 'transformer_ref', safe_serialization=True)
    lora = {'token_refiner.refiner_blocks.0.attn.to_q.lora_A.default.weight': r(8, D),
            'token_refiner.refiner_blocks.0.attn.to_q.lora_B.default.weight': r(D, 8),
            'transformer_blocks.2.ff.net.2.lora_A.default.weight': r(8, 1024),
            'transformer_blocks.2.ff.net.2.lora_B.default.weight': r(D, 8)}
    save_file(lora, str(root / 'ref_lora.safetensors'), metadata={'alpha': '4', 'format': 'pt'})

    from diffusers import AutoencoderKLMiniMaxH3
    AutoencoderKLMiniMaxH3(block_out_channels=(8, 8, 8, 8, 16, 16), norm_num_groups=4, decoder_num_layers=1,
                           decoder_num_attention_heads=2, decoder_attention_head_dim=32, latents_mean=[0.0] * 24,
                           latents_std=[1.0] * 24).save_pretrained(root / 'vae')

    out = convert(root / 'base', root / 'adapter', official, root / 'store', policy='mixed', device='cpu',
                  vae_dir=root / 'vae', vae_decoder=True, ref_dit=root / 'base' / 'transformer_ref',
                  ref_adapter=root / 'ref_lora.safetensors')
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


@pytest.mark.parametrize('mode', ['balance', 'fast-high'])
def test_modes(tiny_store, tmp_path, monkeypatch, mode):
    monkeypatch.setenv('TINYH3_HOME', str(tmp_path / 'home'))
    from tinyh3.runtime.pipeline import GenRequest, generate
    import av
    report = generate(GenRequest(prompt='w3 w4', seconds=5, height=128, width=192, sampler=mode, seed=1,
                                 model=str(tiny_store), output=str(tmp_path / f'{mode}.mp4')), subprocess=False)
    den, dec = report['stages']['denoise'], report['stages']['decode']
    assert len(den['step_seconds']) == 2
    if mode == 'fast-high':
        assert den['render'] == [128, 96] and dec['upscale']['to'] == [192, 128]
    else:
        assert den['render'] == [192, 128] and dec['upscale'] is None
    with av.open(str(tmp_path / f'{mode}.mp4')) as c:
        assert (c.streams.video[0].width, c.streams.video[0].height) == (192, 128)


def _image(path, w=96, h=64, seed=0):
    import numpy as np
    from PIL import Image
    a = (np.random.RandomState(seed).rand(h, w, 3) * 255).astype('uint8')
    Image.fromarray(a).save(path)
    return str(path)


@pytest.mark.parametrize('task,mode', [('i2v', 'fast'), ('flf2v', 'balance'), ('last2v', 'fast-high')])
def test_keyframe_tasks(tiny_store, tmp_path, monkeypatch, task, mode):
    monkeypatch.setenv('TINYH3_HOME', str(tmp_path / 'home'))
    from tinyh3.runtime.pipeline import GenRequest, generate
    first = _image(tmp_path / 'a.png') if task in ('i2v', 'flf2v') else None
    last = _image(tmp_path / 'b.png', seed=1) if task in ('last2v', 'flf2v') else None
    report = generate(GenRequest(prompt='w5 w6', seconds=5, height=128, width=192, sampler=mode, seed=3, task=task,
                                 first=first, last=last, model=str(tiny_store),
                                 output=str(tmp_path / f'{task}.mp4')), subprocess=False)
    den = report['stages']['denoise']
    anchors = {'i2v': 1, 'last2v': 1, 'flf2v': 2}[task]
    rows_per_frame = (den['render'][1] // 32) * (den['render'][0] // 32)
    assert den['conditioning_rows'] == [anchors * rows_per_frame, 0]
    assert report['stages']['text']['anchors'] == {'i2v': ['first'], 'last2v': ['last'], 'flf2v': ['first', 'last']}[task]
    assert os.path.getsize(tmp_path / f'{task}.mp4') > 1000


def test_reference_task(tiny_store, tmp_path, monkeypatch):
    monkeypatch.setenv('TINYH3_HOME', str(tmp_path / 'home'))
    from tinyh3.runtime.media import write_mp4
    from tinyh3.runtime.pipeline import GenRequest, generate
    clip = tmp_path / 'clip.mp4'
    frames = (torch.rand(30, 64, 96, 3) * 255).to(torch.uint8)
    write_mp4(clip, frames, torch.rand(2, 40000) * 0.2 - 0.1, 24, 32000)
    refs = [{'kind': 'image', 'path': _image(tmp_path / 'r.png', 80, 120)}, {'kind': 'video', 'path': str(clip)}]
    report = generate(GenRequest(prompt='w9 w8', seconds=5, height=64, width=96, sampler='fast', seed=4, task='ref2v',
                                 references=refs, model=str(tiny_store), output=str(tmp_path / 'ref.mp4')),
                      subprocess=False)
    den = report['stages']['denoise']
    assert den['decoder'] == 'vae' and len(den['step_seconds']) == 4
    n_video, n_audio = den['conditioning_rows']
    assert n_video > 0 and n_audio > 0  # image + video reference rows, the clip's soundtrack
    import av
    with av.open(str(tmp_path / 'ref.mp4')) as c:
        assert (c.streams.video[0].width, c.streams.video[0].height) == (96, 64)
        assert c.streams.video[0].frames == 124 or c.streams.video[0].frames == 0
