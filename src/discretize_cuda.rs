use std::cell::UnsafeCell;
use std::sync::Arc;

use log::{debug, error};

use crate::background::Background;
use crate::noise::Noise;
use crate::pattern::{Discretizer, PeakRenderParams};
use crate::uninit_vec;

use self::ffi::Uniform;

mod ffi {
    #[link(name = "discretize_cuda")]
    extern "C" {
        pub fn render_peaks_and_background(
            soa: PeakSOA<f32>,
            pat_info: *const CUDAPattern,
            intensities: *mut f32,
            two_thetas: *const f32,
            n_patterns: usize,
            pat_len: usize,
            noise: Noise,
            rng_state: *const u64,
            background_kind: BkgKind,
            bkg_data: *const f32,
            bkg_degree_if_poly: usize,
            bkg_scale_if_not_none: *const f32,
            normalize: bool,
        ) -> bool;
    }

    // FFI structs for interaction with cuda

    #[repr(C)]
    #[derive(Eq, PartialEq)]
    pub enum NoiseKind {
        NoiseNone,
        Gaussian,
        Uniform,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Uniform {
        pub min: *const f64,
        pub max: *const f64,
    }

    pub enum BkgSOA {
        None,
        Polynomial { degree: usize, all_coef: Vec<f32> },
        Exponential(Vec<f32>),
    }

    #[repr(C)]
    pub union NoiseVal {
        pub none: *const f64,
        pub gaussian: *const f64,
        pub uniform: Uniform,
    }

    #[repr(C)]
    pub struct Noise {
        pub v: NoiseVal,
        pub kind: NoiseKind,
    }

    #[repr(C)]
    pub enum BkgKind {
        BkgNone,
        Exponential,
        Polynomial,
    }

    #[repr(C)]
    pub struct PeakSOA<T> {
        pub intensity: *const T,
        pub pos: *const T,
        pub fwhm: *const T,
        pub eta: *const T,
        pub n_peaks_tot: usize,
    }

    #[repr(C)]
    pub struct CUDAPattern {
        pub start_idx: usize,
        pub n_peaks: usize,
    }
}

struct RenderCtx {
    inner: UnsafeCell<Inner>,
}

struct Inner {
    fwhm: Vec<f32>,
    eta: Vec<f32>,
    pos: Vec<f32>,
    intens: Vec<f32>,
}

impl RenderCtx {
    pub fn new(peak_cap: usize) -> Self {
        Self {
            inner: UnsafeCell::new(Inner {
                intens: unsafe { uninit_vec(peak_cap) },
                pos: unsafe { uninit_vec(peak_cap) },
                eta: unsafe { uninit_vec(peak_cap) },
                fwhm: unsafe { uninit_vec(peak_cap) },
            }),
        }
    }

    pub unsafe fn set_at(&self, idx: usize, p: PeakRenderParams) {
        let Inner {
            fwhm,
            eta,
            pos,
            intens,
        } = unsafe { &mut *(self.inner.get() as *mut Inner) };

        fwhm[idx] = p.fwhm as f32;
        eta[idx] = p.eta as f32;
        pos[idx] = p.pos;
        intens[idx] = p.intensity;
    }

    pub fn to_soa(&mut self) -> ffi::PeakSOA<f32> {
        let Inner {
            fwhm,
            eta,
            pos,
            intens,
        } = self.inner.get_mut();

        ffi::PeakSOA {
            intensity: intens.as_ptr(),
            pos: pos.as_ptr(),
            fwhm: fwhm.as_ptr(),
            eta: eta.as_ptr(),
            n_peaks_tot: fwhm.len(),
        }
    }
}

unsafe impl Sync for RenderCtx {}

pub fn discretize_peaks_cuda<'a, T>(jobs: Vec<T>, two_thetas: &[f32]) -> Vec<f32>
where
    T: Discretizer + Send + Sync + 'static,
{
    use self::ffi::BkgSOA;

    let n_peaks_tot: usize = jobs.iter().map(|job| job.n_peaks_tot()).sum();

    let mut patterns = Vec::<ffi::CUDAPattern>::with_capacity(jobs.len());
    let mut ctx = Arc::new(RenderCtx::new(n_peaks_tot));

    let mut rng_state = Vec::<u64>::with_capacity(jobs.len() * 4);

    let (noise_kind, mut noise_data) = match jobs.first().expect("at least one job").noise() {
        Some(Noise::Uniform { .. }) => {
            let mut data = Vec::<f64>::with_capacity(2 * jobs.len());
            data.resize(2 * jobs.len(), 0.0);
            (ffi::NoiseKind::Uniform, data)
        }
        Some(Noise::Gaussian { .. }) => {
            let mut data = Vec::<f64>::with_capacity(jobs.len());
            data.resize(jobs.len(), 0.0);
            (ffi::NoiseKind::Gaussian, data)
        }
        None => (ffi::NoiseKind::NoiseNone, Vec::new()),
    };

    let mut bkg_scales = Vec::new();
    let (mut bkg_soa, normalize) = {
        let fjob = jobs.first().expect("at least one discretization job");
        let soa = match fjob.bkg() {
            Background::None => BkgSOA::None,
            Background::Polynomial {
                poly_coef: ref coef,
                ..
            } => {
                bkg_scales.reserve_exact(jobs.len());
                BkgSOA::Polynomial {
                    degree: coef.len(),
                    all_coef: Vec::with_capacity(coef.len() * jobs.len()),
                }
            }
            Background::Exponential { .. } => {
                bkg_scales.reserve_exact(jobs.len());
                BkgSOA::Exponential(Vec::with_capacity(jobs.len()))
            }
        };
        (soa, fjob.normalize())
    };

    // build SOA for peak and background rendering
    for (i, job) in jobs.iter().enumerate() {
        use crate::background::Background;
        match (job.bkg(), &mut bkg_soa) {
            (Background::None, BkgSOA::None) => (),
            (
                Background::Polynomial { poly_coef, scale },
                BkgSOA::Polynomial {
                    degree,
                    ref mut all_coef,
                },
            ) if *degree == poly_coef.len() => {
                bkg_scales.push(*scale);
                all_coef.extend(poly_coef);
            }
            (Background::Exponential { slope, scale }, BkgSOA::Exponential(ref mut all_coef)) => {
                bkg_scales.push(*scale);
                all_coef.push(*slope);
            }
            (_, _) => {
                unimplemented!("rendering of varying backgrounds CUDA backend.")
            }
        }

        if job.normalize() && normalize || (!job.normalize() && !normalize) {
            // all good; do nothing
            //
            // make sure all patterns are normalized or all aren't
            // this is currently an invariant in the code, but may change in the future
            // we want to crash if this ever happens, especially because it is unclear
            // right now why you would want mixed normalized and unnormalized XRD patterns in the
            // same dataset
        } else {
            unimplemented!(
                "Rendering with varying normalization in CUDA backend is not implemented."
            )
        }

        match job.noise() {
            Some(Noise::Gaussian { sigma }) if noise_kind == ffi::NoiseKind::Gaussian => {
                // all good
                //
                // in
                noise_data[i] = *sigma;
            }
            Some(Noise::Uniform { min, max }) if noise_kind == ffi::NoiseKind::Uniform => {
                // all good, append uniform noise parameters to SOA
                // c-style polymorphism going on here (noise_kind and noise_data are basically a
                // tagged union)
                //
                // in case of uniform noise, noise_data is an array of length 2 * jobs.len()
                // the first half contains each pattern's minimum noise amplitude
                // the second half contains each pattern's maximum noise amplitude
                // 0                        jobs.len()               2 * jobs.len()
                // |                        |                        |
                // v                        v                        v
                // +------------------------+------------------------+
                // | minima                 | maxima                 |
                // +------------------------+------------------------+
                noise_data[i] = *min;
                noise_data[i + jobs.len()] = *max;
            }
            None if noise_kind == ffi::NoiseKind::NoiseNone => {
                // do nothing - no noise
            }
            _ => {
                unimplemented!(
                    "Rendering with varying noise kind in CUDA backend is not implemented."
                )
            }
        }

        if noise_kind != ffi::NoiseKind::NoiseNone {
            let seed: [u64; 4] =
                unsafe { core::mem::transmute(crate::noise::get_xoshiro256_seed(job.seed())) };
            rng_state.extend_from_slice(&seed);
        }
    }

    let mut start_idx = 0;
    for job in jobs.iter() {
        let n_peaks = job.n_peaks_tot();
        patterns.push(ffi::CUDAPattern { start_idx, n_peaks });
        start_idx += n_peaks;
    }

    let patterns = Arc::new(patterns);
    let jobs = Arc::new(jobs);

    let n_threads: usize = std::thread::available_parallelism()
        .map(|x| x.into())
        .unwrap_or(1);

    let mut chunk_size = jobs.len() / n_threads;
    if chunk_size % jobs.len() != 0 {
        chunk_size += 1;
    }
    assert!(chunk_size * n_threads >= jobs.len());

    let mut handles = Vec::new();
    for i in 0..n_threads {
        let start = chunk_size * i;
        let end = (chunk_size * (i + 1)).min(jobs.len());
        let patterns = Arc::clone(&patterns);
        let ctx = Arc::clone(&ctx);
        let jobs = Arc::clone(&jobs);
        let handle = std::thread::spawn(move || {
            for idx in start..end {
                for (i, p) in jobs[idx].peak_info_iterator().enumerate() {
                    let pat = &patterns[idx];
                    assert!(i < pat.n_peaks, "error in peak number computation");
                    let peak_idx = pat.start_idx + i;

                    unsafe { ctx.set_at(peak_idx, p) };
                }
            }
            debug!("peak info generation thread {i} exiting");
        });
        handles.push(handle);
    }

    for (i, handle) in handles.drain(..).enumerate() {
        handle
            .join()
            .map_err(|err| {
                error!("could not join peak info generation thread {i}: '{err:?}'. Exiting...");
                std::process::exit(1);
            })
            .expect("error is handled inside");
    }

    let ctx = Arc::get_mut(&mut ctx).expect("all threads owning ctx have stopped");
    let soa = ctx.to_soa();
    let mut intensities = Vec::<f32>::with_capacity(patterns.len() * two_thetas.len());
    unsafe {
        // SAFETY: this is OK because discretize_peaks **sets** every single f64-sized
        // slot in intensities, and the length is set to the capacity. Therefore we are
        // not touching memory outside of the allocated bounds, and after the call to
        // discretize_peaks, we are only dealing with initialized memory
        intensities.set_len(intensities.capacity());

        let noise_val = match noise_kind {
            ffi::NoiseKind::NoiseNone => ffi::NoiseVal {
                none: core::ptr::null(),
            },
            ffi::NoiseKind::Gaussian => ffi::NoiseVal {
                gaussian: noise_data.as_ptr(),
            },
            ffi::NoiseKind::Uniform => ffi::NoiseVal {
                uniform: Uniform {
                    min: noise_data.as_ptr(),
                    max: noise_data[patterns.len()..].as_ptr(),
                },
            },
        };

        let rng_state_ptr = if noise_kind != ffi::NoiseKind::NoiseNone {
            rng_state.as_ptr()
        } else {
            core::ptr::null()
        };

        let noise = ffi::Noise {
            v: noise_val,
            kind: noise_kind,
        };

        let ret = ffi::render_peaks_and_background(
            soa,
            patterns.as_ptr(),
            intensities.as_mut_ptr(),
            two_thetas.as_ptr(),
            patterns.len(),
            two_thetas.len(),
            noise,
            rng_state_ptr,
            match &bkg_soa {
                BkgSOA::None => ffi::BkgKind::BkgNone,
                BkgSOA::Polynomial { .. } => ffi::BkgKind::Polynomial,
                BkgSOA::Exponential(_) => ffi::BkgKind::Exponential,
            },
            match &bkg_soa {
                BkgSOA::None => 0 as *const f32,
                BkgSOA::Polynomial { all_coef, .. } => all_coef.as_ptr(),
                BkgSOA::Exponential(vec) => vec.as_ptr(),
            },
            match &bkg_soa {
                BkgSOA::None => 0,
                BkgSOA::Polynomial { degree, .. } => *degree,
                BkgSOA::Exponential(_) => 1,
            },
            match &bkg_soa {
                BkgSOA::None => 0 as *const f32,
                BkgSOA::Polynomial { .. } | BkgSOA::Exponential(_) => bkg_scales.as_ptr(),
            },
            normalize,
        );
        assert!(ret);
    };

    intensities
}
