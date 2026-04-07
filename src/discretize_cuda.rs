use std::cell::UnsafeCell;
use std::sync::Arc;

use ahash::HashMapExt;
use log::debug;
use log::info;
use ordered_float::NotNan;

use crate::background::cheb2poly;
use crate::background::Background;
use crate::noise::Noise;
use crate::pattern::{DiscretizeSample, Discretizer, PeakRenderParams};
use crate::uninit_vec;

use self::ffi::BkgSOA;
use self::ffi::CUDAPattern;
use self::ffi::Uniform;

mod ffi {
    #[link(name = "cuda_lib")]
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
            chunk_idx: usize,
            n_chunks: usize,
            device_id: std::ffi::c_int,
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

    #[derive(Clone, Debug)]
    #[repr(C)]
    pub struct CUDAPattern {
        pub start_idx: usize,
        pub n_peaks: usize,
    }
}

struct RenderCtx {
    inner: UnsafeCell<Option<Inner>>,
}

struct Inner {
    fwhm: Vec<f32>,
    eta: Vec<f32>,
    pos: Vec<f32>,
    intens: Vec<f32>,
}

struct CudaPatterns(UnsafeCell<Vec<CUDAPattern>>);
unsafe impl Sync for CudaPatterns {}

impl RenderCtx {
    pub fn empty() -> Self {
        Self { inner: None.into() }
    }

    pub fn initialize(&self, peak_cap: usize) {
        assert!(unsafe { &*self.inner.get() }.is_none());

        *unsafe { &mut *self.inner.get() } = Some(Inner {
            intens: unsafe { uninit_vec(peak_cap) },
            pos: unsafe { uninit_vec(peak_cap) },
            eta: unsafe { uninit_vec(peak_cap) },
            fwhm: unsafe { uninit_vec(peak_cap) },
        });
    }

    pub unsafe fn set_at(&self, idx: usize, p: PeakRenderParams) {
        let Inner {
            fwhm,
            eta,
            pos,
            intens,
        } = unsafe { &mut *self.inner.get() }
            .as_mut()
            .expect("must be initialized");

        fwhm[idx] = p.fwhm;
        eta[idx] = p.eta;
        pos[idx] = p.pos;
        intens[idx] = p.intensity;
    }

    pub fn as_soa(&mut self) -> ffi::PeakSOA<f32> {
        let Inner {
            fwhm,
            eta,
            pos,
            intens,
        } = self.inner.get_mut().as_mut().expect("is initialized");

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

pub fn prepare_cuda_discretize<T>(
    mut jobs: Vec<DiscretizeSample<T>>,
    two_thetas: Vec<f32>,
    chunk_idx: usize,
    n_chunks: usize,
) -> Result<PreparedCudaBatch, String>
where
    T: Discretizer + Send + Sync + 'static,
{
    debug!(
        "(Chunk {} / {n_chunks}) Collecting peak rendering info for CUDA-based rendering",
        chunk_idx + 1
    );
    use self::ffi::BkgSOA;

    let num_peak_sets = jobs.iter().map(|job| job.n_patterns()).sum();

    let jobs = {
        let mut jobs_flat = Vec::with_capacity(num_peak_sets);
        for job in jobs.drain(..) {
            match job {
                DiscretizeSample::Standard(j) => jobs_flat.push(j),
                DiscretizeSample::TextureMeasurement(mut items) => {
                    jobs_flat.extend(items.drain(..));
                }
            }
        }
        jobs_flat
    };

    let mut patterns = Vec::<ffi::CUDAPattern>::with_capacity(num_peak_sets);
    let ctx = Arc::new(RenderCtx::empty());

    let mut rng_state = Vec::<u64>::with_capacity(num_peak_sets * 4);

    let fjob = jobs.first().expect("at least one discretization job");

    let (noise_kind, mut noise_data) = match fjob.noise() {
        Some(Noise::Uniform { .. }) => {
            let mut data = Vec::<f64>::with_capacity(2 * num_peak_sets);
            data.resize(2 * num_peak_sets, 0.0);
            (ffi::NoiseKind::Uniform, data)
        }
        Some(Noise::Gaussian { .. }) => {
            let mut data = Vec::<f64>::with_capacity(num_peak_sets);
            data.resize(num_peak_sets, 0.0);
            (ffi::NoiseKind::Gaussian, data)
        }
        None => (ffi::NoiseKind::NoiseNone, Vec::new()),
    };

    let mut bkg_scales = Vec::new();
    let (mut bkg_soa, normalize) = {
        let soa = match fjob.bkg() {
            Background::None => BkgSOA::None,
            Background::Chebyshev { ref coef, .. } => {
                bkg_scales.reserve_exact(num_peak_sets);
                let poly = cheb2poly(coef);
                BkgSOA::Polynomial {
                    degree: poly.len(),
                    all_coef: Vec::with_capacity(coef.len() * num_peak_sets),
                }
            }
            Background::Exponential { .. } => {
                bkg_scales.reserve_exact(num_peak_sets);
                BkgSOA::Exponential(Vec::with_capacity(num_peak_sets))
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
                Background::Chebyshev { coef, scale },
                BkgSOA::Polynomial {
                    degree: _,
                    ref mut all_coef,
                },
            ) => {
                let poly_coef = cheb2poly(coef);
                // TODO: check for scale matching
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
                // in case of uniform noise, noise_data is an array of length 2 * num_peak_sets
                // the first half contains each pattern's minimum noise amplitude
                // the second half contains each pattern's maximum noise amplitude
                // 0                        num_peak_sets               2 * num_peak_sets
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

    // TODO: check if this is correct for texture measurement. do we want n_peak_sets?
    patterns.resize(
        jobs.len(),
        ffi::CUDAPattern {
            start_idx: 0,
            n_peaks: 0,
        },
    );

    let patterns = Arc::new(CudaPatterns(UnsafeCell::new(patterns)));
    let jobs = Arc::new(jobs);
    let n_jobs = jobs.len();

    let mut n_threads: usize = std::thread::available_parallelism()
        .map(|x| x.into())
        .unwrap_or(1);

    let mut chunk_size = n_jobs / n_threads;
    if chunk_size % n_jobs != 0 || chunk_size == 0 {
        chunk_size += 1;
    }
    assert!(chunk_size * n_threads >= n_jobs);
    if chunk_size < 10 {
        info!("(Chunk {} / {n_chunks}) Small amount of jobs ({n_jobs}). Falling back to single threaded CUDA input preparation.",
            chunk_idx + 1,
        );
        n_threads = 1;
        chunk_size = n_jobs;
    }
    info!(
        "(Chunk {} / {n_chunks}) Generating CUDA inputs using {n_threads} threads",
        chunk_idx + 1
    );

    // TODO: allocate after each thread has processed its data
    // and only then write to correct position
    // currently, we overallocate by about a factor of 4, and we
    // need to move things around after compression

    // let mut handles = Vec::new();
    let finish_computation = std::sync::Barrier::new(n_threads + 1);
    let start_ctx_write = std::sync::Barrier::new(n_threads + 1);
    std::thread::scope(|s| {
        for thread_idx in 0..n_threads {
            let start = chunk_size * thread_idx;
            let end = (chunk_size * (thread_idx + 1)).min(n_jobs);

            let patterns = Arc::clone(&patterns.clone());
            let ctx = Arc::clone(&ctx);
            let jobs = Arc::clone(&jobs);

            let finish_computation = &finish_computation;
            let start_ctx_write = &start_ctx_write;

            s.spawn(move || {
                debug!("PREP {}: starting peak compression", thread_idx + 1);
                let mut compressed_by_job = Vec::new();
                let mut n = 0;
                for job_idx in start..end {
                    let mut peak_idx_in_pattern = 0;

                    let mut compressed = ahash::HashMap::with_capacity(jobs[job_idx].n_peaks_tot() / 2);
                    for (p, _) in jobs[job_idx].peak_info_iterator() {
                        n += 1;
                        let pos = NotNan::try_from(p.pos).expect("peak position is not nan");
                        let fwhm = NotNan::try_from(p.fwhm).expect("peak position is not nan");
                        let eta = NotNan::try_from(p.eta).expect("peak position is not nan");

                        use std::collections::hash_map::Entry;
                        match compressed.entry((pos, fwhm, eta)) {
                            Entry::Vacant(vacant) => {
                                vacant.insert((p.intensity, peak_idx_in_pattern));
                                peak_idx_in_pattern += 1;
                            },
                            Entry::Occupied(mut occ) => {
                                occ.get_mut().0 += p.intensity;
                            },
                        }
                    }

                    compressed_by_job.push(compressed);
                }

                let mut n_compressed = 0;
                for (i, c) in compressed_by_job.iter().enumerate() {
                    let job_idx = start + i;
                    unsafe {(&mut *patterns.0.get())[job_idx].n_peaks = c.len()};
                    n_compressed += c.len();
                }

                debug!("PREP {}: waiting for computation to finish", thread_idx + 1);
                finish_computation.wait();

                // main thread fixes pattern starts here

                start_ctx_write.wait();

                for (i, mut compressed) in compressed_by_job.drain(..).enumerate() {
                    let job_idx = i + start;
                    for ((pos, fwhm, eta), (intensity, peak_idx_in_pattern)) in compressed.drain() {
                        let pat = unsafe {&(&*patterns.0.get())[job_idx]};
                        let peak_idx = pat.start_idx + peak_idx_in_pattern;
                        unsafe {
                            ctx.set_at(
                                peak_idx,
                                PeakRenderParams {
                                    pos: *pos,
                                    intensity: intensity,
                                    fwhm: *fwhm,
                                    eta: *eta,
                                },
                            )
                        };
                    }
                }

                debug!(
                "(Chunk {} / {n_chunks}) Peak info generation thread {thread_idx} exiting (compressed from {} to {} peaks)",
                chunk_idx + 1,
                n,
                n_compressed,
            );
            });

            if end >= n_jobs {
                debug!("finished starting up all prep threads");
                break;
            }
        }

        {
            // fix up pattern starts
            debug!("main thread: waiting for processing in peak info gen threads to finish");
            finish_computation.wait();
            debug!("fixing pattern starts");
            let mut peaks_total = 0;
            for p in unsafe { &mut *patterns.0.get() }.iter_mut() {
                p.start_idx = peaks_total;
                peaks_total += p.n_peaks;
            }

            ctx.initialize(peaks_total);
            start_ctx_write.wait();
            debug!("done fixing pattern starts");
        }
    });

    let ctx = Arc::into_inner(ctx).expect("all threads owning ctx have stopped");

    let patterns = Arc::into_inner(patterns)
        .expect("only reference left")
        .0
        .into_inner();

    info!("Prepared {} patterns", patterns.len());

    Ok(PreparedCudaBatch {
        ctx,
        noise_kind,
        noise_data,
        patterns,
        rng_state,
        two_thetas,
        bkg_soa,
        bkg_scales,
        normalize,
    })
}

pub struct PreparedCudaBatch {
    ctx: RenderCtx,
    noise_kind: ffi::NoiseKind,
    noise_data: Vec<f64>,
    rng_state: Vec<u64>,
    patterns: Vec<ffi::CUDAPattern>,
    two_thetas: Vec<f32>,
    bkg_soa: BkgSOA,
    bkg_scales: Vec<f32>,
    normalize: bool,
}

pub fn render_with_cuda(
    PreparedCudaBatch {
        mut ctx,
        noise_kind,
        noise_data,
        rng_state,
        patterns,
        two_thetas,
        bkg_soa,
        bkg_scales,
        normalize,
    }: PreparedCudaBatch,
    chunk_idx: usize,
    n_chunks: usize,
    device_id: usize,
) -> Result<Vec<f32>, String> {
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

    let mut intensities = unsafe { uninit_vec(patterns.len() * two_thetas.len()) };

    let soa = ctx.as_soa();

    unsafe {
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
                BkgSOA::None => core::ptr::null(),
                BkgSOA::Polynomial { all_coef, .. } => all_coef.as_ptr(),
                BkgSOA::Exponential(vec) => vec.as_ptr(),
            },
            match &bkg_soa {
                BkgSOA::None => 0,
                BkgSOA::Polynomial { degree, .. } => *degree,
                BkgSOA::Exponential(_) => 1,
            },
            match &bkg_soa {
                BkgSOA::None => core::ptr::null(),
                BkgSOA::Polynomial { .. } | BkgSOA::Exponential(_) => bkg_scales.as_ptr(),
            },
            normalize,
            chunk_idx,
            n_chunks,
            device_id
                .try_into()
                .expect("device_id is not that large and greater than 0"),
        );
        if !ret {
            return Err(
                "An error has happened in the cuda backend. See the above log for details. Out of memory issues may be mitigated by using a smaller chunk size (or any, if you aren't)"
                    .to_string(),
            );
        }
    }

    Ok(intensities)
}
