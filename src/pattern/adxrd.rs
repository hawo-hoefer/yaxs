use super::{Peaks, render_peak};
use crate::background::Background;
use crate::structure::Strain;

#[derive(Clone, Debug, PartialEq)]
pub struct ADXRDMeta {
    pub vol_fractions: Box<[f64]>,
    pub mean_ds_nm: Box<[f64]>,
    pub eta: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub background: Background,
}

#[derive(serde::Deserialize, serde::Serialize, PartialEq, Debug, Clone)]
#[repr(C)]
pub struct EmissionLine {
    // wavelength in amstrong
    pub wavelength_ams: f64,
    // wavelength relative weight
    pub weight: f64,
}

impl EmissionLine {
    /// create a new emission line from wavelength and weight
    ///
    /// * `wavelength`: wavelength in amstrong
    /// * `weight`: intensity of the emission line relative to other emission lines in the spectrum
    pub fn new(wavelength: f64, weight: f64) -> Self {
        Self {
            wavelength_ams: wavelength,
            weight,
        }
    }
}

pub struct DiscretizeAngleDisperse<'a> {
    // all simulated peaks for all phases in order [structure, structure permutations]
    pub all_simulated_peaks: &'a Vec<Vec<Peaks>>,
    pub all_strains: &'a Vec<Vec<Strain>>,
    // indices to select from simulated peaks, length is number of structures
    pub indices: Vec<usize>,
    pub emission_lines: &'a [EmissionLine],
    pub normalize: bool,
    pub meta: ADXRDMeta,
}

impl<'a> DiscretizeAngleDisperse<'a> {
    pub fn discretize_into(&self, pat: &mut [f32], two_thetas: &[f32], abstol: f32) {
        let ADXRDMeta {
            vol_fractions,
            eta,
            mean_ds_nm,
            u,
            v,
            w,
            background,
        } = &self.meta;
        for (((phase_peaks, idx), vf), phase_mean_ds_nm) in self
            .all_simulated_peaks
            .iter()
            .zip(&self.indices)
            .zip(vol_fractions)
            .zip(mean_ds_nm)
        {
            let peaks = &phase_peaks[*idx];
            // * `pat`: target pattern
            // * `two_thetas`: two theta values of pattern's intensities in degrees
            // * `wavelength`: wavelength of the x-rays in nanometers
            // * `mean_ds`: mean domain size used for scherrer broadening
            // * `u`: caglioti parameter u
            // * `v`: caglioti parameter v
            // * `w`: caglioti parameter w
            for emission_line in self.emission_lines {
                let wavelength_nm = emission_line.wavelength_ams / 10.0;
                for peak in peaks.iter() {
                    let (two_theta_hkl_deg, peak_weight, fwhm) = peak.get_adxrd_render_params(
                        wavelength_nm,
                        *u,
                        *v,
                        *w,
                        *phase_mean_ds_nm,
                        vf * emission_line.weight,
                    );
                    render_peak(
                        two_theta_hkl_deg,
                        peak_weight,
                        fwhm,
                        *eta as f32,
                        abstol,
                        two_thetas,
                        pat,
                    )
                }
            }
        }
        background.render(pat, two_thetas);

        if self.normalize {
            // TODO: check for NaNs and normalization
            let f = *pat.first().unwrap();
            let vmin = pat.iter().fold(f, |a, b| f32::min(a, *b));
            let vmax = pat.iter().fold(f, |a, b| f32::max(a, *b));
            pat.iter_mut().for_each(|x| {
                *x = (*x - vmin) / (vmax - vmin);
            });
        }
    }
}
