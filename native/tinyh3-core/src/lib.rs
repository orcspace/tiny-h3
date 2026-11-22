//! TinyH3 native core.
//!
//! The GPU hot path of TinyH3 is already native (Triton kernels compiled to SASS, Spark's CuTe kernels); Python only
//! enqueues them. What still ran on the CPU in Python is here:
//!
//! * [`io::read_into`]: positional, multi-threaded reads of a weight unit straight into a caller-owned (pinned)
//!   buffer. Several requests in flight keep an NVMe queue busy; the GIL is released for the whole read.
//! * [`upscale::upscale_rgb`]: Lanczos-3 resampling + contrast-adaptive sharpening of whole clips (e.g. 768p -> 2K),
//!   parallel over frames.
//!
//! The Python bindings take raw addresses + sizes (from `tensor.data_ptr()` / `ndarray.ctypes.data`); the Python
//! wrapper in `tinyh3/native.py` validates dtype, contiguity and lengths before calling in.

pub mod io;
pub mod upscale;

#[cfg(feature = "python")]
mod python {
    use pyo3::exceptions::{PyIOError, PyValueError};
    use pyo3::prelude::*;

    /// Read `nbytes` of `path` starting at `offset` into the memory at `ptr`, using `threads` parallel requests.
    #[pyfunction]
    #[pyo3(signature = (path, offset, ptr, nbytes, threads=4, chunk=8 << 20))]
    fn read_into(py: Python<'_>, path: String, offset: u64, ptr: usize, nbytes: usize, threads: usize,
                 chunk: usize) -> PyResult<()> {
        if ptr == 0 && nbytes > 0 {
            return Err(PyValueError::new_err("null destination"));
        }
        // SAFETY: the Python wrapper guarantees `ptr..ptr+nbytes` is a live, writable, exclusively owned buffer.
        let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, nbytes) };
        py.allow_threads(|| crate::io::read_into(&path, offset, dst, threads, chunk))
            .map_err(|e| PyIOError::new_err(format!("{path}: {e}")))
    }

    /// Resample `frames` RGB8 frames (h, w) at `src` into (out_h, out_w) at `dst`, then sharpen (0 = off, 1 = max).
    #[pyfunction]
    #[pyo3(signature = (src, frames, h, w, dst, out_h, out_w, sharpen=0.4, threads=0))]
    #[allow(clippy::too_many_arguments)]
    fn upscale_rgb(py: Python<'_>, src: usize, frames: usize, h: usize, w: usize, dst: usize, out_h: usize,
                   out_w: usize, sharpen: f32, threads: usize) -> PyResult<()> {
        if h == 0 || w == 0 || out_h == 0 || out_w == 0 {
            return Err(PyValueError::new_err("empty frame"));
        }
        if !(0.0..=1.0).contains(&sharpen) {
            return Err(PyValueError::new_err("sharpen must be in [0, 1]"));
        }
        // SAFETY: validated by the Python wrapper (sizes, contiguity, non-overlap).
        let s = unsafe { std::slice::from_raw_parts(src as *const u8, frames * h * w * 3) };
        let d = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, frames * out_h * out_w * 3) };
        py.allow_threads(|| crate::upscale::upscale_rgb(s, frames, h, w, d, out_h, out_w, sharpen, threads));
        Ok(())
    }

    #[pyfunction]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    #[pymodule]
    fn tinyh3_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(read_into, m)?)?;
        m.add_function(wrap_pyfunction!(upscale_rgb, m)?)?;
        m.add_function(wrap_pyfunction!(version, m)?)?;
        Ok(())
    }
}
