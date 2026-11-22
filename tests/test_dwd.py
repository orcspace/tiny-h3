"""DWD head vs pi-H3's SolH3DWDHead (LakonLab) on a reduced-width Sol stack."""
import os
import sys

import pytest
import torch

from tinyh3.config import SolConfig
from tinyh3.runtime.dwd import DWDHead

PI_H3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pi-H3')


def test_dwd_head_matches_lakonlab(monkeypatch):
    if not os.path.isdir(os.path.join(PI_H3, 'lakonlab')):
        pytest.skip('pi-H3 reference checkout not present')
    pytest.importorskip('diffusers')
    monkeypatch.syspath_prepend(PI_H3)
    import lakonlab.models.architectures.sol_h3.model as m
    import lakonlab.models.architectures.sol_h3.sol_h3 as s
    for mod in (m, s):
        for name, value in (('NUM_HEADS', 2), ('HEAD_DIM', 64), ('DIM', 128)):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, value)
    torch.manual_seed(0)
    ref = s.SolH3DWDHead(in_channels=64, src_patch_size=(4, 32, 32), first_frame_src_patch_size=(1, 32, 32),
                         first_frame_period=5, num_layers=2).float().eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if 'norm' in name and name.endswith('weight'):
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif 'scale' in name:
                p.copy_(0.5 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) / (p.shape[-1] ** 0.5 if p.ndim > 1 else 4))
    t, h, w = 7, 10, 12   # crosses a 5-latent block; 20x24 Sol tokens -> shifted windows straddle edges
    feats = torch.randn(1, t * h * w, 64)
    with torch.no_grad():
        want = ref(None, feats, (t, h, w))[0]  # (F, 3, 16H, 16W)
    cfg = SolConfig(num_heads=2, head_dim=64, num_layers=2)
    head = DWDHead({k: v.detach() for k, v in ref.state_dict().items()}, cfg, device='cpu')
    got = head.decode(feats[0], (t, h, w), raw=True)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert got.shape[0] == 1 + 4 * 4 + 1 + 4  # 17 frames per 5 latents, then 5 for the 2-latent tail
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 1e-4, rel
