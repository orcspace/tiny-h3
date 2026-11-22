//! Parallel positional reads into a caller-owned buffer.

use rayon::prelude::*;
use std::fs::File;
use std::io;

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.read_at(buf, off)
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_read(buf, off)
}

fn read_exact_at(f: &File, mut buf: &mut [u8], mut off: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match read_at(f, buf, off) {
            Ok(0) => return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read")),
            Ok(n) => {
                buf = &mut buf[n..];
                off += n as u64;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Fill `dst` with `dst.len()` bytes of `path` from `offset`, `threads` requests in flight, `chunk` bytes each.
pub fn read_into(path: &str, offset: u64, dst: &mut [u8], threads: usize, chunk: usize) -> io::Result<()> {
    let file = File::open(path)?;
    let len = file.metadata()?.len();
    if offset + dst.len() as u64 > len {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "range past end of file"));
    }
    let chunk = chunk.max(1 << 20);
    let threads = threads.clamp(1, 32);
    if threads == 1 || dst.len() <= chunk {
        return read_exact_at(&file, dst, offset);
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()
        .map_err(|e| io::Error::other(e.to_string()))?;
    pool.install(|| {
        dst.par_chunks_mut(chunk).enumerate().try_for_each(|(i, part)| {
            read_exact_at(&file, part, offset + (i * chunk) as u64)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn reads_exact_ranges_in_parallel() {
        let dir = std::env::temp_dir().join(format!("tinyh3-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob.bin");
        let data: Vec<u8> = (0..(5 << 20) + 123).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::File::create(&path).unwrap().write_all(&data).unwrap();
        let mut out = vec![0u8; data.len() - 1000];
        read_into(path.to_str().unwrap(), 1000, &mut out, 4, 1 << 20).unwrap();
        assert_eq!(&out[..], &data[1000..]);
        let mut past = vec![0u8; 10];
        assert!(read_into(path.to_str().unwrap(), data.len() as u64 - 5, &mut past, 2, 1 << 20).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
