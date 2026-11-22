"""DWD pixel head: the last 8 Sol-H3-VAE decoder layers on the DiT's final-step features -> RGB frames.

Same math as pi-H3's ``SolH3DWDHead`` (shifted 5-latent x 16x16-token windows, per-5-latent-block global tokens,
window-local 3-axis RoPE on 48 of 64 head channels), restructured for 8 GB:

* attention runs one temporal window slab at a time: Q/K/V are only ever materialized for <= 5 latent frames;
* global tokens attend to their native 5-latent block, computed block by block before any token is updated;
* the FFN runs in row chunks; pixels are projected per 5-latent chunk and leave the GPU as uint8.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..config import SolConfig, SOL, DWD_RGB_MEAN, DWD_RGB_STD

BF16 = torch.bfloat16


def _rope_tables(T, H, W, s, st, cfg: SolConfig, device):
    """cos/sin (T, H, W, rope_dim // 2) of window-local (t, y, x) coordinates for shifts (st, s)."""
    lt = ((torch.arange(T, device=device) + st) % cfg.block).float()
    ly = ((torch.arange(H, device=device) + s) % cfg.window).float()
    lx = ((torch.arange(W, device=device) + s) % cfg.window).float()
    t = (lt + 0.5) / cfg.time_norm * 2 - 1
    hy = (ly + 0.5) / cfg.window * 2 - 1
    wx = (lx + 0.5) / cfg.window * 2 - 1
    grid = torch.stack(torch.meshgrid(t, hy, wx, indexing='ij'), -1)  # (T, H, W, 3)
    inv = 1.0 / cfg.rope_theta ** torch.arange(0, 1, 6 / cfg.rope_dim, dtype=torch.float32, device=device)
    angles = (2.0 * math.pi * grid[..., None] * inv).flatten(-2)  # (T, H, W, 3 * len(inv))
    return angles.cos(), angles.sin()


def _rotate(x, cos, sin):
    """Rotate the leading 2 * half channels of (..., heads, dh) with cos/sin (..., half), out of place."""
    half = cos.shape[-1]
    c, s = cos.to(x.dtype)[..., None, :], sin.to(x.dtype)[..., None, :]
    first, second = x[..., :half], x[..., half:2 * half]
    return torch.cat((first * c - second * s, second * c + first * s, x[..., 2 * half:]), dim=-1)


class DWDHead:
    def __init__(self, tensors: dict, cfg: SolConfig = SOL, device='cuda', dtype=BF16):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type == 'cuda' else torch.float32
        t = {k: v.to(self.device) for k, v in tensors.items()}
        self.t = t
        self.layers = sorted({int(k.split('.')[1]) for k in t if k.startswith('blocks.')})
        self.offset = cfg.total_layers - len(self.layers)
        w = t['feature_expand.weight']
        self.expansion = int(round(math.sqrt(w.shape[0] // cfg.dim)))

    def _w(self, name):
        return self.t[name].to(self.dtype)

    def _lin(self, x, name):
        b = self.t.get(name + '.bias')
        return F.linear(x, self._w(name + '.weight'), None if b is None else b.to(self.dtype))

    def _norm(self, x, weight=None):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.cfg.norm_eps)
        return (y * weight.float() if weight is not None else y).to(self.dtype)

    def _qkv(self, x, p):
        """x (..., D) -> q, k, v (..., heads, dh) with q/k RMS-normalized (no affine)."""
        c = self.cfg
        qkv = self._lin(self._norm(x, self.t[p + 'norm1.weight']), p + 'attn.to_qkv')
        qkv = qkv.view(*x.shape[:-1], c.num_heads, 3, c.head_dim)
        q = self._norm(qkv[..., 0, :])
        k = self._norm(qkv[..., 1, :])
        return q, k, qkv[..., 2, :]

    @torch.no_grad()
    def layer(self, x, g, li: int, row_chunk: int = 16384):
        """x (T, H, W, D), g (nb, S, D) updated in place by layer ``li`` (absolute Sol index)."""
        c = self.cfg
        p = f'blocks.{li - self.offset}.'
        T, H, W, D = x.shape
        nb, S = g.shape[:2]
        shifted = li % 2 == 1
        s, st = (c.window // 2, c.t_shift) if shifted else (0, 0)
        cos, sin = _rope_tables(T, H, W, s, st, c, x.device)
        gq, gk, gv = self._qkv(g, p)  # (nb, S, heads, dh)

        # 1. global tokens over their native 5-latent block (from the pre-update tokens)
        og = torch.empty_like(gq)
        for b in range(nb):
            t0, t1 = b * c.block, min((b + 1) * c.block, T)
            _, k, v = self._qkv(x[t0:t1], p)
            k = _rotate(k, cos[t0:t1], sin[t0:t1]).reshape(-1, c.num_heads, c.head_dim)
            v = v.reshape(-1, c.num_heads, c.head_dim)
            keys = torch.cat([k, gk[b]])
            vals = torch.cat([v, gv[b]])
            og[b] = F.scaled_dot_product_attention(gq[b].transpose(0, 1)[None], keys.transpose(0, 1)[None],
                                                   vals.transpose(0, 1)[None])[0].transpose(0, 1)
        # 2. windows, one temporal slab at a time (slabs are disjoint, so updating x slab by slab is exact)
        ws = c.window
        Hp = ((H + s + ws - 1) // ws) * ws
        Wp = ((W + s + ws - 1) // ws) * ws
        nt = ((T + st + c.block - 1) // c.block)
        nh, nw = Hp // ws, Wp // ws
        for j in range(nt):
            a, e = max(j * c.block - st, 0), min((j + 1) * c.block - st, T)
            if a >= e:
                continue
            q, k, v = self._qkv(x[a:e], p)
            q = _rotate(q, cos[a:e], sin[a:e])
            k = _rotate(k, cos[a:e], sin[a:e])
            Tw = c.block
            off = j * c.block - st  # original frame index of the window's first (padded) frame

            def windows(z):
                zp = z.new_zeros(Tw, Hp, Wp, *z.shape[3:])
                zp[a - off:e - off, s:s + H, s:s + W] = z
                zp = zp.view(Tw, nh, ws, nw, ws, *z.shape[3:]).permute(1, 3, 0, 2, 4, *range(5, 5 + z.dim() - 3))
                return zp.reshape(nh * nw, Tw * ws * ws, *z.shape[3:])

            valid = torch.zeros(Tw, Hp, Wp, dtype=torch.bool, device=x.device)
            valid[a - off:e - off, s:s + H, s:s + W] = True
            vmask = valid.view(Tw, nh, ws, nw, ws).permute(1, 3, 0, 2, 4).reshape(nh * nw, -1)
            qw, kw, vw = windows(q), windows(k), windows(v)
            b0 = a // c.block
            b1 = (e - 1) // c.block
            gks = [gk[b0]] + ([gk[b1]] if b1 != b0 else [])
            gvs = [gv[b0]] + ([gv[b1]] if b1 != b0 else [])
            ng = sum(t.shape[0] for t in gks)
            kw = torch.cat([kw, torch.cat(gks)[None].expand(kw.shape[0], -1, -1, -1)], 1)
            vw = torch.cat([vw, torch.cat(gvs)[None].expand(vw.shape[0], -1, -1, -1)], 1)
            kmask = torch.cat([vmask, vmask.new_ones(vmask.shape[0], ng)], 1)
            o = F.scaled_dot_product_attention(qw.transpose(1, 2), kw.transpose(1, 2), vw.transpose(1, 2),
                                               attn_mask=kmask[:, None, None, :])
            o = o.transpose(1, 2)  # (nwin, Tw*ws*ws, heads, dh)
            o = o.reshape(nh, nw, Tw, ws, ws, c.num_heads, c.head_dim).permute(2, 0, 3, 1, 4, 5, 6)
            o = o.reshape(Tw, Hp, Wp, D)[a - off:e - off, s:s + H, s:s + W]
            x[a:e] += self._lin(o.reshape(-1, D), p + 'attn.to_out').view(e - a, H, W, D) * \
                self._w(p + 'scale1')
            del q, k, v, qw, kw, vw, o
        g += self._lin(og.reshape(nb, S, D), p + 'attn.to_out') * self._w(p + 'scale1')
        # 3. FFN in row chunks
        flat = x.view(-1, D)
        for r0 in range(0, flat.shape[0], row_chunk):
            r1 = min(r0 + row_chunk, flat.shape[0])
            flat[r0:r1] += self._ffn(flat[r0:r1], p) * self._w(p + 'scale2')
        g += self._ffn(g, p) * self._w(p + 'scale2')

    def _ffn(self, x, p):
        h = self._lin(self._norm(x, self.t[p + 'norm2.weight']), p + 'ff.w1')
        gate, up = h.chunk(2, dim=-1)
        return self._lin(F.silu(gate) * up, p + 'ff.w2')

    @torch.no_grad()
    def decode(self, features: torch.Tensor, grid: tuple, on_chunk=None, raw: bool = False) -> torch.Tensor:
        """features (t*h*w, D_dit) on a (t, h, w) token grid -> uint8 frames (F, 16H, 16W, 3) on CPU, or with
        ``raw`` the head's native output (F, 3, 16H, 16W) before the RGB codec."""
        c = self.cfg
        t, h, w = grid
        e = self.expansion
        x = self._lin(features.to(self.device, self.dtype), 'feature_expand')
        x = x.view(t, h, w, e, e, c.dim).permute(0, 1, 3, 2, 4, 5).reshape(t, h * e, w * e, c.dim).contiguous()
        nb = (t + c.block - 1) // c.block
        reg = self.t['register_tokens'].to(self.dtype)[0]
        g = torch.cat([reg, reg.new_zeros(1, c.dim)])[None].expand(nb, -1, -1).contiguous()
        for li in range(self.offset, self.offset + len(self.layers)):
            self.layer(x, g, li)
        frames = []
        mean = torch.tensor(DWD_RGB_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(DWD_RGB_STD, device=self.device).view(1, 3, 1, 1)
        for b in range(nb):
            t0, t1 = b * c.block, min((b + 1) * c.block, t)
            pix = self._project(x[t0:t1])  # (F, 3, 16H, 16W)
            if raw:
                frames.append(pix.float().cpu())
                continue
            rgb = (pix.float() * std + mean).add_(1).mul_(127.5).clamp_(0, 255).round_()
            frames.append(rgb.to(torch.uint8).permute(0, 2, 3, 1).cpu())
            if on_chunk:
                on_chunk(b + 1, nb)
        return torch.cat(frames)

    def _project(self, x):
        """(n, H, W, D) latent frames of one 5-latent chunk -> (F, 3, 16H, 16W)."""
        c = self.cfg
        n, H, W, D = x.shape
        P = c.patch
        y = F.layer_norm(x.float(), (D,), self.t['norm_out.weight'].float(), self.t['norm_out.bias'].float(),
                         c.norm_eps).to(self.dtype)
        first_t = c.patch_t - c.frames_to_trim
        first = self._lin(y[:1], 'first_frame_proj_out').view(1, H, W, c.out_ch, first_t, P, P)
        first = first.permute(0, 4, 3, 1, 5, 2, 6).reshape(first_t, c.out_ch, H * P, W * P)
        if n == 1:
            return first
        rest = self._lin(y[1:], 'proj_out').view(n - 1, H, W, c.out_ch, c.patch_t, P, P)
        rest = rest.permute(0, 4, 3, 1, 5, 2, 6).reshape((n - 1) * c.patch_t, c.out_ch, H * P, W * P)
        return torch.cat([first, rest])
