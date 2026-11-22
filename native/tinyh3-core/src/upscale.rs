//! Lanczos-3 resampling and contrast-adaptive sharpening (CAS) of RGB8 clips.
//!
//! Separable resampling with precomputed, normalized tap tables (edge-clamped), in f32. When downscaling the kernel
//! is stretched by the scale factor (proper anti-aliasing). Sharpening follows AMD FidelityFX CAS: a per-pixel
//! negative-lobe cross filter whose strength adapts to local contrast, so flat areas and strong edges are left alone.

use rayon::prelude::*;

const A: f32 = 3.0;

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        let p = std::f32::consts::PI * x;
        p.sin() / p
    }
}

fn lanczos(x: f32) -> f32 {
    if x.abs() >= A {
        0.0
    } else {
        sinc(x) * sinc(x / A)
    }
}

/// For each output index: (first source index, normalized weights).
pub fn taps(n_in: usize, n_out: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = n_out as f32 / n_in as f32;
    let stretch = if scale < 1.0 { 1.0 / scale } else { 1.0 };
    let support = A * stretch;
    (0..n_out)
        .map(|o| {
            let center = (o as f32 + 0.5) / scale - 0.5;
            let lo = (center - support).floor() as isize + 1;
            let hi = (center + support).floor() as isize;
            let mut idx = Vec::new();
            let mut w = Vec::new();
            for i in lo..=hi {
                let weight = lanczos((i as f32 - center) / stretch);
                if weight != 0.0 {
                    idx.push(i.clamp(0, n_in as isize - 1) as usize);
                    w.push(weight);
                }
            }
            let sum: f32 = w.iter().sum();
            for x in w.iter_mut() {
                *x /= sum;
            }
            // Edge clamping can repeat a source index; fold into a contiguous window starting at idx[0].
            let first = *idx.iter().min().unwrap();
            let last = *idx.iter().max().unwrap();
            let mut dense = vec![0.0f32; last - first + 1];
            for (i, x) in idx.iter().zip(w) {
                dense[i - first] += x;
            }
            (first, dense)
        })
        .collect()
}

fn resample_frame(src: &[u8], h: usize, w: usize, dst: &mut [f32], oh: usize, ow: usize,
                  tx: &[(usize, Vec<f32>)], ty: &[(usize, Vec<f32>)]) {
    // horizontal: (h, w, 3) u8 -> (h, ow, 3) f32
    let mut tmp = vec![0.0f32; h * ow * 3];
    for y in 0..h {
        let row = &src[y * w * 3..(y + 1) * w * 3];
        let out = &mut tmp[y * ow * 3..(y + 1) * ow * 3];
        for (x, (first, wts)) in tx.iter().enumerate() {
            let (mut r, mut g, mut b) = (0.0f32, 0.0f32, 0.0f32);
            for (k, wt) in wts.iter().enumerate() {
                let p = (first + k) * 3;
                r += row[p] as f32 * wt;
                g += row[p + 1] as f32 * wt;
                b += row[p + 2] as f32 * wt;
            }
            out[x * 3] = r;
            out[x * 3 + 1] = g;
            out[x * 3 + 2] = b;
        }
    }
    // vertical: (h, ow, 3) -> (oh, ow, 3)
    let stride = ow * 3;
    for (y, (first, wts)) in ty.iter().enumerate() {
        let out = &mut dst[y * stride..(y + 1) * stride];
        out.iter_mut().for_each(|v| *v = 0.0);
        for (k, wt) in wts.iter().enumerate() {
            let row = &tmp[(first + k) * stride..(first + k + 1) * stride];
            for (o, s) in out.iter_mut().zip(row) {
                *o += s * wt;
            }
        }
    }
}

/// CAS on an (h, w, 3) f32 frame in 0..255, written as RGB8.
fn sharpen_frame(src: &[f32], h: usize, w: usize, dst: &mut [u8], amount: f32) {
    let peak = -1.0 / (8.0 + (5.0 - 8.0) * amount); // lerp(8, 5, amount)
    let at = |y: usize, x: usize, c: usize| (src[(y * w + x) * 3 + c] / 255.0).clamp(0.0, 1.0);
    for y in 0..h {
        let (yu, yd) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for x in 0..w {
            let (xl, xr) = (x.saturating_sub(1), (x + 1).min(w - 1));
            for c in 0..3 {
                let (n, s, e, wv, m) = (at(yu, x, c), at(yd, x, c), at(y, xr, c), at(y, xl, c), at(y, x, c));
                let mn = n.min(s).min(e).min(wv).min(m);
                let mx = n.max(s).max(e).max(wv).max(m);
                let amp = if mx > 0.0 { ((mn.min(1.0 - mx)) / mx).clamp(0.0, 1.0).sqrt() } else { 0.0 };
                let k = amp * peak;
                let v = ((n + s + e + wv) * k + m) / (1.0 + 4.0 * k);
                dst[(y * w + x) * 3 + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn upscale_rgb(src: &[u8], frames: usize, h: usize, w: usize, dst: &mut [u8], oh: usize, ow: usize,
                   sharpen: f32, threads: usize) {
    let tx = taps(w, ow);
    let ty = taps(h, oh);
    let work = |(f, out): (usize, &mut [u8])| {
        let frame = &src[f * h * w * 3..(f + 1) * h * w * 3];
        let mut res = vec![0.0f32; oh * ow * 3];
        resample_frame(frame, h, w, &mut res, oh, ow, &tx, &ty);
        if sharpen > 0.0 {
            sharpen_frame(&res, oh, ow, out, sharpen);
        } else {
            for (o, v) in out.iter_mut().zip(&res) {
                *o = (v.clamp(0.0, 255.0) + 0.5) as u8;
            }
        }
    };
    let chunks = dst.par_chunks_mut(oh * ow * 3).enumerate();
    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| chunks.for_each(work));
    } else {
        chunks.for_each(work);
    }
    let _ = frames;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taps_are_normalized() {
        for (n_in, n_out) in [(10, 19), (19, 10), (768, 1440), (7, 7)] {
            for (_, w) in taps(n_in, n_out) {
                let s: f32 = w.iter().sum();
                assert!((s - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn constant_frames_stay_constant() {
        let (h, w, oh, ow) = (9, 16, 20, 37);
        let src = vec![137u8; 2 * h * w * 3];
        let mut dst = vec![0u8; 2 * oh * ow * 3];
        upscale_rgb(&src, 2, h, w, &mut dst, oh, ow, 0.8, 2);
        assert!(dst.iter().all(|&v| v == 137));
    }

    #[test]
    fn identity_size_without_sharpening_is_exact() {
        let (h, w) = (11, 13);
        let src: Vec<u8> = (0..h * w * 3).map(|i| (i * 31 % 256) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        upscale_rgb(&src, 1, h, w, &mut dst, h, w, 0.0, 1);
        assert_eq!(src, dst);
    }
}
