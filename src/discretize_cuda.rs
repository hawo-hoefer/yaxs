use std::mem::MaybeUninit;

use crate::math::{caglioti, scherrer_broadening};
use crate::pattern::{DiscretizationJob, PatternMeta, Peak};

pub fn discretize_peaks_cuda(jobs: &[DiscretizationJob], two_thetas: &[f64]) -> Vec<f64> {
    #[link(name = "discretize_cuda")]
    extern "C" {
        fn discretize_peaks(
            pat_info: *const CUDAPattern,
            intensities: *mut f64,
            two_thetas: *const f64,
            n_patterns: usize,
            pat_len: usize,
        ) -> bool;
    }

    #[repr(C)]
    struct CUDAPattern {
        peaks: *const Peak,
        peak_info: *const PeakInfo,
        n_peaks: usize,
    };

    #[repr(C)]
    struct PeakInfo {
        weight: f64,
        fwhm: f64,
        eta: f64,
    }

    let mut patterns = Vec::<CUDAPattern>::with_capacity(jobs.len());
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
            background,
        } = &job.meta;

        let mut ffi_peaks = Vec::with_capacity(n_peaks);
        let mut ffi_peak_info = Vec::with_capacity(n_peaks);

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
            // for emission_line in job.emission_lines {
            for peak in &peaks.peaks {
                // let cpeak =
                //     peak.convert(peaks.wavelength_nm, emission_line.wavelength_ams / 10.0);

                let theta_pos_rad = peak.pos.to_radians() / 2.0;
                let fwhm = caglioti(*u, *v, *w, theta_pos_rad)
                    + scherrer_broadening(peaks.wavelength_nm, theta_pos_rad, *mean_ds_nm);
                ffi_peak_info.push(PeakInfo {
                    weight: 1.0 * vf, // TODO: add emission line intensity back in
                    fwhm,
                    eta: *eta,
                });
                ffi_peaks.push(*peak)
            }
            // }
        }

        patterns.push(CUDAPattern {
            peaks: ffi_peaks.as_ptr(),
            peak_info: ffi_peak_info.as_ptr(),
            n_peaks: ffi_peaks.len(),
        });
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

    let mut intensities = Vec::<f64>::with_capacity(patterns.len() * two_thetas.len());
    unsafe {
        // SAFETY: this is OK because discretize_peaks **sets** every single f64-sized
        // slot in intensities, and the length is set to the capacity. Therefore we are
        // not touching memory outside of the allocated bounds, and after the call to
        // discretize_peaks, we are only dealing with initialized memory
        intensities.set_len(intensities.capacity());

        let ret = discretize_peaks(
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
