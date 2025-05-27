use crate::pattern::{lorentz_factor, render_peak};
use crate::structure::Strain;

use super::Peaks;

#[derive(Clone, Debug, PartialEq)]
pub struct EDXRDMeta {
    pub vol_fractions: Box<[f64]>,
    pub mean_ds_nm: Box<[f64]>,
    pub eta: f64,
    pub theta_rad: f64,
}

pub struct DiscretizeEnergyDispersive<'a> {
    // all simulated peaks for all phases in order [structure, structure permutations]
    pub all_simulated_peaks: &'a Vec<Vec<Peaks>>,
    pub all_strains: &'a Vec<Vec<Strain>>,
    // indices to select from simulated peaks, length is number of structures
    pub indices: Vec<usize>,
    pub normalize: bool,
    pub meta: EDXRDMeta,
}

impl<'a> DiscretizeEnergyDispersive<'a> {
    pub fn discretize_into(&self, intensities: &mut [f32], energies_kev: &[f32], abstol: f32) {
        let EDXRDMeta {
            vol_fractions,
            eta,
            mean_ds_nm,
            theta_rad,
        } = &self.meta;

        let f_lorentz = lorentz_factor(*theta_rad);

        // hardcoded for now :)
        fn beamline_intensity(e_kev: f64) -> f64 {
            10.0f64.powf(12.30 - e_kev * 0.7 / 100.0)
            // 1.0
        }

        for (((phase_peaks, idx), vf), phase_mean_ds_nm) in self
            .all_simulated_peaks
            .iter()
            .zip(&self.indices)
            .zip(vol_fractions)
            .zip(mean_ds_nm)
        {
            let peaks = &phase_peaks[*idx];
            for peak in peaks.iter() {
                let (e_hkl_kev, peak_weight, fwhm) = peak.get_edxrd_render_params(
                    *theta_rad,
                    f_lorentz,
                    *phase_mean_ds_nm,
                    *vf,
                    beamline_intensity,
                );
                render_peak(
                    e_hkl_kev,
                    peak_weight,
                    fwhm,
                    *eta as f32,
                    abstol,
                    energies_kev,
                    intensities,
                )
            }
        }

        if self.normalize {
            // TODO: check for NaNs and normalization
            let f = *intensities.first().unwrap();
            let vmin = intensities.iter().fold(f, |a, b| f32::min(a, *b));
            let vmax = intensities.iter().fold(f, |a, b| f32::max(a, *b));
            intensities.iter_mut().for_each(|x| {
                *x = (*x - vmin) / (vmax - vmin);
            });
        }
    }
}
