use crate::background::Background;
use crate::io::PatternMeta;
use crate::pattern::lorentz_factor;
use crate::preferred_orientation::MarchDollase;
use crate::structure::Strain;

use super::{Discretizer, PeakRenderParams, Peaks};

#[derive(Clone, Debug, PartialEq)]
pub struct EDXRDMeta {
    pub vol_fractions: Box<[f64]>,
    pub mean_ds_nm: Box<[f64]>,
    pub eta: f64,
    pub theta_rad: f64,
}

#[derive(Debug)]
pub struct DiscretizeEnergyDispersive<'a> {
    // all simulated peaks for all phases in order [structure, structure permutations]
    pub all_simulated_peaks: &'a Vec<Vec<Peaks>>,
    pub all_strains: &'a Vec<Vec<Strain>>,
    pub all_preferred_orientations: &'a Vec<Vec<Option<MarchDollase>>>,
    // indices to select from simulated peaks, length is number of structures
    pub indices: Vec<usize>,
    pub normalize: bool,
    pub meta: EDXRDMeta,
}

// pub struct DiscretizeIter<'a, T, V>
// where
//     T: Iterator<Item = PhaseInfo<'a>>,
//     V: Iterator<Item = &'a PeakRenderParams>,
// {
//     structure_iter: T,
//     peak_iter: V,
//     idx: usize,
// }

// struct PhaseInfo<'a> {
//     peaks: &'a [Peak],
//     indices: &'a [usize],
// }

// impl Iterator for DiscretizeIter<'a> {
//     type Item = PeakRenderParams;

//     fn next(&mut self) -> Option<Self::Item> {
//         fn beamline_intensity(e_kev: f64) -> f64 {
//             10.0f64.powf(12.30 - e_kev * 0.7 / 100.0)
//             // 1.0
//         }
//         let f_lorentz = lorentz_factor(self.meta.theta_rad);

//         let EDXRDMeta {
//             vol_fractions,
//             mean_ds_nm,
//             eta,
//             theta_rad,
//         } = &self.meta;

//         self.all_simulated_peaks
//             .iter()
//             .zip(self.indices)
//             .zip(vol_fractions)
//             .zip(mean_ds_nm)
//             .map(|(((phase_peaks, idx), vf), phase_mean_ds_nm)| {
//                 phase_peaks[idx].iter().map(move |peak| {
//                     let (e_hkl_kev, peak_weight, fwhm) = peak.get_edxrd_render_params(
//                         *theta_rad,
//                         f_lorentz,
//                         *phase_mean_ds_nm,
//                         *vf,
//                         beamline_intensity,
//                     );
//                     PeakRenderParams {
//                         pos: e_hkl_kev,
//                         intensity: peak_weight,
//                         fwhm,
//                         eta: *eta as f32,
//                     }
//                 })
//             })
//             .flatten()
//     }
// }

impl Discretizer for DiscretizeEnergyDispersive<'_> {
    fn peak_info_iterator(&self) -> impl Iterator<Item = PeakRenderParams> {
        fn beamline_intensity(e_kev: f64) -> f64 {
            10.0f64.powf(12.30 - e_kev * 0.7 / 100.0)
            // 1.0
        }
        let f_lorentz = lorentz_factor(self.meta.theta_rad);

        let EDXRDMeta {
            vol_fractions,
            mean_ds_nm,
            eta,
            theta_rad,
        } = &self.meta;

        itertools::izip!(
            self.all_simulated_peaks,
            self.indices.clone(), // TODO: get rid of this clone
            vol_fractions,
            mean_ds_nm
        )
        .map(move |(phase_peaks, idx, vf, phase_mean_ds_nm)| {
            phase_peaks[idx].iter().map(move |peak| {
                let (e_hkl_kev, peak_weight, fwhm) = peak.get_edxrd_render_params(
                    *theta_rad,
                    f_lorentz,
                    *phase_mean_ds_nm,
                    *vf,
                    beamline_intensity,
                );
                PeakRenderParams {
                    pos: e_hkl_kev,
                    intensity: peak_weight,
                    fwhm,
                    eta: *eta as f32,
                }
            })
        })
        .flatten()
    }

    fn n_peaks_tot(&self) -> usize {
        self.all_simulated_peaks
            .iter()
            .zip(&self.indices)
            .map(|(phase_peaks, idx)| phase_peaks[*idx].len())
            .sum::<usize>()
    }

    fn bkg(&self) -> &Background {
        &Background::None
    }

    fn normalize(&self) -> bool {
        self.normalize
    }

    fn write_meta_data(&self, key: &mut PatternMeta, pat_id: usize) {
        todo!()
    }

    fn init_meta_data(n_patterns: usize, n_phases: usize) -> Vec<PatternMeta> {
        todo!()
    }
}
