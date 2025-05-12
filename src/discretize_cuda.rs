use crate::background::Background;
use crate::math::{caglioti, scherrer_broadening};
use crate::pattern::{DiscretizationJob, PatternMeta};

pub fn discretize_peaks_cuda(jobs: &[DiscretizationJob], two_thetas: &[f32]) -> Vec<f32> {
    #[link(name = "discretize_cuda")]
    extern "C" {
        fn render_peaks_and_background(
            soa: PeakSOA<f32>,
            pat_info: *const CUDAPattern,
            intensities: *mut f32,
            two_thetas: *const f32,
            n_patterns: usize,
            pat_len: usize,
            background_kind: BkgKind,
            bkg_data: *const f32,
            bkg_degree_if_poly: usize,
            bkg_scale_if_not_none: *const f32,
            normalize: bool,
        ) -> bool;
    }

    // FFI structs for interaction with cuda

    #[repr(C)]
    enum BkgKind {
        None,
        Exponential,
        Polynomial,
    }

    #[repr(C)]
    struct PeakSOA<T> {
        intensity: *const T,
        pos: *const T,
        weight: *const T,
        fwhm: *const T,
        eta: *const T,
        n_peaks_tot: usize,
    }

    #[repr(C)]
    struct CUDAPattern {
        start_idx: usize,
        n_peaks: usize,
    }

    let n_peaks_tot: usize = jobs
        .iter()
        .map(|job| {
            job.all_simulated_peaks
                .iter()
                .zip(&job.indices)
                .map(|(phase_peaks, idx)| phase_peaks[*idx].peaks.len() * job.emission_lines.len())
                .sum::<usize>()
        })
        .sum();

    let mut patterns = Vec::<CUDAPattern>::with_capacity(jobs.len());

    let mut ffi_peak_intensity = Vec::with_capacity(n_peaks_tot);
    let mut ffi_peak_pos = Vec::with_capacity(n_peaks_tot);
    let mut ffi_peak_info_fwhm = Vec::with_capacity(n_peaks_tot);
    let mut ffi_peak_info_eta = Vec::with_capacity(n_peaks_tot);
    let mut ffi_peak_info_weight = Vec::with_capacity(n_peaks_tot);

    let mut start_idx = 0;

    enum BkgSOA {
        None,
        Polynomial { degree: usize, all_coef: Vec<f32> },
        Exponential(Vec<f32>),
    }

    let mut bkg_scales = Vec::new();
    let (mut bkg_soa, normalize) = {
        let fjob = jobs.first().expect("at least one discretization job");
        let soa = match fjob.meta.background {
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
        (soa, fjob.normalize)
    };

    // build SOA for peak and background rendering
    for job in jobs.iter() {
        let PatternMeta {
            vol_fractions,
            eta,
            mean_ds_nm,
            u,
            v,
            w,
            background,
            ..
        } = &job.meta;

        use crate::background::Background;
        match (background, &mut bkg_soa) {
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
        if job.normalize && normalize || (!job.normalize && !normalize) {
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

        let mut n_peaks = 0;
        for (((phase_peaks, idx), vf), phase_mean_ds_nm) in job
            .all_simulated_peaks
            .iter()
            .zip(&job.indices)
            .zip(vol_fractions)
            .zip(mean_ds_nm)
        {
            let peaks = &phase_peaks[*idx];
            for emission_line in job.emission_lines {
                for peak in &peaks.peaks {
                    let cpeak =
                        peak.convert(peaks.wavelength_nm, emission_line.wavelength_ams / 10.0);

                    let theta_pos_rad = peak.pos.to_radians() / 2.0;
                    let fwhm = caglioti(*u, *v, *w, theta_pos_rad)
                        + scherrer_broadening(
                            peaks.wavelength_nm,
                            theta_pos_rad,
                            *phase_mean_ds_nm,
                        );
                    ffi_peak_info_weight.push((emission_line.weight * vf) as f32);
                    ffi_peak_info_fwhm.push(fwhm as f32);
                    ffi_peak_info_eta.push(*eta as f32);
                    ffi_peak_pos.push(cpeak.pos as f32);
                    ffi_peak_intensity.push(cpeak.intensity as f32);
                    n_peaks += 1;
                }
            }
        }

        patterns.push(CUDAPattern { start_idx, n_peaks });
        start_idx += n_peaks;

        // TODO: normalization
        // if self.normalize {
        //     let f = *pat.first().unwrap();
        //     let vmin = pat.iter().fold(f, |a, b| f64::min(a, *b));
        //     let vmax = pat.iter().fold(f, |a, b| f64::max(a, *b));
        //     pat.iter_mut().for_each(|x| {
        //         *x = (*x - vmin) / (vmax - vmin);
        //     });
        // }
    }

    let mut intensities = Vec::<f32>::with_capacity(patterns.len() * two_thetas.len());
    unsafe {
        // SAFETY: this is OK because discretize_peaks **sets** every single f64-sized
        // slot in intensities, and the length is set to the capacity. Therefore we are
        // not touching memory outside of the allocated bounds, and after the call to
        // discretize_peaks, we are only dealing with initialized memory
        intensities.set_len(intensities.capacity());

        let soa = PeakSOA {
            intensity: ffi_peak_intensity.as_ptr(),
            pos: ffi_peak_pos.as_ptr(),
            weight: ffi_peak_info_weight.as_ptr(),
            fwhm: ffi_peak_info_fwhm.as_ptr(),
            eta: ffi_peak_info_eta.as_ptr(),
            n_peaks_tot,
        };

        let ret = render_peaks_and_background(
            soa,
            patterns.as_ptr(),
            intensities.as_mut_ptr(),
            two_thetas.as_ptr(),
            patterns.len(),
            two_thetas.len(),
            match &bkg_soa {
                BkgSOA::None => BkgKind::None,
                BkgSOA::Polynomial { .. } => BkgKind::Polynomial,
                BkgSOA::Exponential(_) => BkgKind::Exponential,
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
