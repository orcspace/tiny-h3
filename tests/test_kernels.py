"""Kernel semantics: the reference definitions, and Triton (under TRITON_INTERPRET=1 on CPU) against them."""
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from tinyh3.kernels import reference as ref


def test_hadamard_is_symmetric_orthogonal():
    h = ref.hadamard(256, dtype=torch.float64)
    assert torch.allclose(h, h.T)
    assert torch.allclose(h @ h, torch.eye(256, dtype=torch.float64), atol=1e-12)


def test_rotation_preserves_products():
    torch.manual_seed(0)
    x = torch.randn(7, 512, dtype=torch.float64)
    w = torch.randn(9, 512, dtype=torch.float64)
    assert torch.allclose(ref.rotate(x) @ ref.rotate(w).T, x @ w.T, atol=1e-9)


def test_int4_pack_roundtrip():
    q = torch.randint(-7, 8, (6, 256), dtype=torch.int8)
    assert torch.equal(ref.unpack_int4(ref.pack_int4(q)), q)


def test_int4_layout_halves():
    # byte j of a group holds element j (low nibble) and j + 64 (high nibble)
    q = torch.zeros(1, 128, dtype=torch.int8)
    q[0, 3], q[0, 67] = 5, -2
    packed = ref.pack_int4(q)
    assert packed[0, 3].item() == (5 | ((-2 & 0xF) << 4))


def test_w4a8_matches_dequantized():
    torch.manual_seed(1)
    w = torch.randn(64, 512)
    x = torch.randn(33, 512)
    wp, ws = ref.quantize_weight_int4(w)
    xq, xs = ref.quantize_rows(x)
    got = ref.matmul_w4a8(xq, xs, wp, ws, out_dtype=torch.float64)
    dense = ref.dequantize_weight({'kind': 'int4', 'q': wp, 'scale': ws}, torch.float64)
    want = (xq.double() * xs.double()[:, None]) @ dense.T
    assert torch.allclose(got, want, rtol=1e-9, atol=1e-9)


def test_quantized_linear_close_to_dense():
    torch.manual_seed(2)
    w = torch.randn(256, 1024) * 0.02
    x = torch.randn(64, 1024)
    x[:, 5] *= 40  # an outlier channel, the case the rotation exists for
    want = x @ w.T
    xq, xs = ref.rotate_quantize(x)
    w8 = ref.quantize_weight_int8(ref.rotate(w))
    got8 = ref.matmul_w8a8(xq, xs, *w8, out_dtype=torch.float32)
    w4 = ref.quantize_weight_int4(ref.rotate(w))
    got4 = ref.matmul_w4a8(xq, xs, *w4, out_dtype=torch.float32)
    rel = lambda a: ((a - want).norm() / want.norm()).item()
    assert rel(got8) < 0.02
    assert rel(got4) < 0.12


TRITON_CHECK = textwrap.dedent('''
    import torch
    from tinyh3.kernels import reference as ref
    from tinyh3.kernels import triton_ops as t
    torch.manual_seed(0)
    x = torch.randn(37, 512)
    q1, s1 = t.rotate_quantize(x); q0, s0 = ref.rotate_quantize(x)
    assert torch.allclose(s1, s0, rtol=1e-5), 'scale'
    assert (q1.int() - q0.int()).abs().max() <= 1, 'rq'
    nw = torch.randn(512); sc = torch.randn(5, 512) * .1; sh = torch.randn(5, 512) * .1
    rows = torch.randint(0, 5, (37,))
    q1, s1 = t.norm_modulate_rotate_quantize(x, nw, 1e-5, sc, sh, rows)
    q0, s0 = ref.norm_modulate_rotate_quantize(x, nw, 1e-5, sc, sh, rows)
    assert torch.allclose(s1, s0, rtol=1e-4), 'nm scale'
    assert (q1.int() - q0.int()).abs().max() <= 1, 'nm'
    h = torch.randn(37, 1024)
    q1, s1 = t.swiglu_rotate_quantize(h); q0, s0 = ref.swiglu_rotate_quantize(h)
    assert torch.allclose(s1, s0, rtol=1e-4) and (q1.int() - q0.int()).abs().max() <= 1, 'swiglu'
    xq, xs = ref.quantize_rows(torch.randn(70, 256))
    wq, ws = ref.quantize_weight_int8(torch.randn(96, 256))
    b = torch.randn(96)
    got = t.matmul_w8a8(xq, xs, wq, ws, b, out_dtype=torch.float32)
    want = ref.matmul_w8a8(xq, xs, wq, ws, b, out_dtype=torch.float32)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-3), 'w8a8'
    wp, wsc = ref.quantize_weight_int4(torch.randn(96, 256))
    got = t.matmul_w4a8(xq, xs, wp, wsc, b, out_dtype=torch.float32)
    want = ref.matmul_w4a8(xq, xs, wp, wsc, b, out_dtype=torch.float32)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-3), 'w4a8'
    print('TRITON_OK')
''')


@pytest.mark.slow
def test_triton_kernels_interpreted():
    pytest.importorskip('triton')
    env = dict(os.environ, TRITON_INTERPRET='1')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run([sys.executable, '-c', TRITON_CHECK], env=env, cwd=root,
                          capture_output=True, text=True, timeout=900)
    assert 'TRITON_OK' in proc.stdout, proc.stderr[-4000:]
