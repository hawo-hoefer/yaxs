pub mod background;
pub mod cfg;
pub mod cif;
pub mod util;

pub mod element;
pub mod io;
pub mod lattice;
pub mod math;
pub mod noise;
pub mod pattern;
pub mod peak_sim;
pub mod preferred_orientation;
pub mod scatter;
pub mod site;
pub mod species;
pub mod strain;
pub mod structure;
pub mod symop;

#[cfg(feature = "use-gpu")]
pub mod discretize_cuda;

#[cfg(feature = "use-gpu")]
pub mod peak_sim_cuda;

#[cfg(feature = "use-gpu")]
pub mod cuda_common;

#[allow(clippy::uninit_vec)]
pub(crate) unsafe fn uninit_vec<T>(len: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(len);
    unsafe { v.set_len(len) };
    v
}
