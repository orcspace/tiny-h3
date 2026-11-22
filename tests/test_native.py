"""Rust core bindings (skipped when the extension is not built) and the PyTorch fallbacks."""
import pytest
import torch

from tinyh3 import native


def test_upscale_fallback_and_native_agree_roughly(monkeypatch):
    y, x = torch.meshgrid(torch.linspace(0, 6, 24), torch.linspace(0, 9, 40), indexing='ij')
    img = torch.stack([128 + 90 * torch.sin(x), 128 + 90 * torch.cos(y), 64 + 3 * (x + y) * 8], -1)
    frames = img.clamp(0, 255).to(torch.uint8)[None].repeat(3, 1, 1, 1)
    up = native.upscale_frames(frames, 45, 75, sharpen=0.0)
    assert up.shape == (3, 45, 75, 3) and up.dtype == torch.uint8
    monkeypatch.setenv('TINYH3_NATIVE', '0')
    fb = native.upscale_frames(frames, 45, 75, sharpen=0.0)
    assert fb.shape == up.shape
    # Lanczos-3 and antialiased bicubic differ slightly; both must track the source closely
    assert (up.float() - fb.float()).abs().mean() < 2.5


@pytest.mark.skipif(not native.available(), reason='tinyh3_core not built')
def test_native_read_into(tmp_path):
    data = bytes(range(256)) * 50000
    p = tmp_path / 'blob.bin'
    p.write_bytes(data)
    buf = torch.empty(len(data) - 100, dtype=torch.uint8)
    assert native.read_into(p, 100, buf, buf.numel(), threads=4)
    assert bytes(buf.numpy()) == data[100:]
    with pytest.raises(ValueError):
        native.read_into(p, 0, buf[::2], 10)


@pytest.mark.skipif(not native.available(), reason='tinyh3_core not built')
def test_native_upscale_constant_and_identity():
    c = torch.full((2, 9, 16, 3), 200, dtype=torch.uint8)
    assert torch.all(native.upscale_frames(c, 18, 32, 0.5) == 200)
    x = (torch.rand(1, 11, 13, 3) * 255).to(torch.uint8)
    assert torch.equal(native.upscale_frames(x, 11, 13, 0.0), x)
