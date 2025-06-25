pub mod background;
pub mod cfg;
pub mod cif;

pub mod element;
pub mod io;
pub mod math;
pub mod noise;
pub mod pattern;
pub mod preferred_orientation;
pub mod site;
pub mod species;
pub mod structure;
pub mod symop;

#[cfg(not(feature = "cpu-only"))]
pub mod discretize_cuda;

#[allow(clippy::uninit_vec)]
pub(crate) unsafe fn uninit_vec<T>(len: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(len);
    unsafe { v.set_len(len) };
    v
}
