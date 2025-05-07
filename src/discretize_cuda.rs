use crate::math::{caglioti, scherrer_broadening};
use crate::pattern::{DiscretizationJob, PatternMeta, Peak};

pub fn discretize_peaks_cuda(jobs: Iterator<DiscretizationJob>, two_thetas: &[f32]) -> Vec<f32> {
    #[link(name = "discretize_cuda")]
    extern "C" {
        fn discretize_peaks(
            soa: PeakSOA<f32>,
            pat_info: *const CUDAPattern,
            intensities: *mut f32,
            two_thetas: *const f32,
            n_patterns: usize,
            pat_len: usize,
        ) -> bool;
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
                .map(|(phase_peaks, idx)| phase_peaks[*idx].peaks.len())
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
    for job in jobs.iter() {
        let n_peaks = job
            .all_simulated_peaks
            .iter()
            .zip(&job.indices)
            .map(|(phase_peaks, idx)| phase_peaks[*idx].peaks.len())
            .sum();

        let PatternMeta {
            vol_fractions,
            eta,
            mean_ds_nm,
            u,
            v,
            w,
            ..
            // background,
        } = &job.meta;

        for ((phase_peaks, idx), vf) in job
            .all_simulated_peaks
            .iter()
            .zip(&job.indices)
            .zip(vol_fractions)
        {
            let peaks = &phase_peaks[*idx];
            // * `pat`: target pattern
            // * `two_thetas`: two theta values of pattern's intensities in degrees
            // * `wavelength`: wavelength of the x-rays in nanometers
            // * `mean_ds`: mean domain size used for scherrer broadening
            // * `u`: caglioti parameter u
            // * `v`: caglioti parameter v
            // * `w`: caglioti parameter w
            for emission_line in job.emission_lines {
                for peak in &peaks.peaks {
                    let cpeak =
                        peak.convert(peaks.wavelength_nm, emission_line.wavelength_ams / 10.0);

                    let theta_pos_rad = peak.pos.to_radians() / 2.0;
                    let fwhm = caglioti(*u, *v, *w, theta_pos_rad)
                        + scherrer_broadening(peaks.wavelength_nm, theta_pos_rad, *mean_ds_nm);
                    ffi_peak_info_weight.push((emission_line.weight * vf) as f32);
                    ffi_peak_info_fwhm.push(fwhm as f32);
                    ffi_peak_info_eta.push(*eta as f32);
                    ffi_peak_pos.push(cpeak.pos as f32);
                    ffi_peak_intensity.push(cpeak.intensity as f32);
                }
            }
        }

        patterns.push(CUDAPattern { start_idx, n_peaks });
        start_idx += n_peaks;
        // background.render(pat, two_thetas);

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

        let ret = discretize_peaks(
            soa,
            patterns.as_ptr(),
            intensities.as_mut_ptr(),
            two_thetas.as_ptr(),
            patterns.len(),
            two_thetas.len(),
        );
        assert!(ret);
    };
    intensities
}
