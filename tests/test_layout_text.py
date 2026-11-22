"""Packed layout vs diffusers, and the streamed Qwen3 text tower vs transformers' Qwen3-VL."""
import json

import pytest
import torch

from tinyh3.config import Geometry, frames_for_seconds, latent_frames, TextEncoderConfig
from tinyh3.runtime.layout import build_packed_sequence, rotary


def test_frames_for_seconds():
    assert frames_for_seconds(5.2) == 124
    assert frames_for_seconds(10) == 243
    assert frames_for_seconds(20) == 345
    assert latent_frames(124) == 37


@pytest.mark.parametrize('anchors', [(), ('first',)])
def test_layout_matches_diffusers(anchors):
    pytest.importorskip('diffusers')
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3PrepareLayoutStep
    tags = torch.ones(17, dtype=torch.long)
    want = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=tags, num_latent_frames=12, latent_height=6, latent_width=10, num_audio_latents=33,
        patch_size=(1, 2, 2), audio_channels=2, audio_tag=2, video_tag=0, keyframe_anchors=anchors)
    got = build_packed_sequence(tags, 12, 6, 10, 33, keyframe_anchors=anchors)
    for a, b in zip(got[:5], want[:5]):
        assert torch.equal(a, b)
    assert got[5] == want[5]


def test_rotary_matches_diffusers():
    pytest.importorskip('diffusers')
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
    pos = torch.rand(50, 3, dtype=torch.float64) * 40
    cos, sin = MiniMaxH3RotaryPosEmbed(16, 10000.0)(pos)
    c2, s2 = rotary(pos)
    assert torch.allclose(cos, c2) and torch.allclose(sin, s2)


def test_text_tower_matches_transformers(tmp_path):
    transformers = pytest.importorskip('transformers')
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    cfg = Qwen3VLConfig(
        text_config=dict(hidden_size=256, intermediate_size=512, num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=64, vocab_size=1000, rms_norm_eps=1e-6,
                         rope_theta=5000000.0, max_position_embeddings=4096,
                         rope_scaling={'rope_type': 'default', 'mrope_section': [12, 10, 10],
                                       'mrope_interleaved': True}),
        vision_config=dict(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=256,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2, deepstack_visual_indexes=[0]))
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 and 'norm' in name:
                p.copy_(1 + 0.1 * torch.randn_like(p))
    model.save_pretrained(tmp_path / 'te', safe_serialization=True)

    from tinyh3.convert.convert import convert_text, text_config_from
    from tinyh3.convert.sources import ShardedReader
    from tinyh3.store import FORMAT, Store
    from tinyh3.runtime.text_stage import encode_tokens
    tcfg = text_config_from(tmp_path / 'te')
    assert tcfg.num_layers_used == 2 and tcfg.hidden_size == 256
    out = tmp_path / 'store'
    out.mkdir()
    text = convert_text(ShardedReader(tmp_path / 'te'), out, tcfg, 'int8', 'cpu', lambda *a: None)
    (out / 'manifest.json').write_text(json.dumps({'format': FORMAT, 'text': text, 'dit': {}}))
    store = Store(out)

    ids = [5, 17, 99, 3, 250, 7, 7, 42]
    got = encode_tokens(store, ids, 'cpu')
    with torch.no_grad():
        input_ids = torch.tensor([ids])
        hs = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                         mm_token_type_ids=torch.zeros_like(input_ids), use_cache=False,
                         output_hidden_states=True).hidden_states
    want = hs[tcfg.num_layers_used][0].float()
    rel = ((got.float() - want).norm() / want.norm()).item()
    assert rel < 0.02, rel
