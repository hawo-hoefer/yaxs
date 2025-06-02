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

    fn write_meta_data(&self, data: &mut PatternMeta, pat_id: usize) {
        use PatternMeta::*;
        let n_phases = self.all_simulated_peaks.len();
        match data {
            VolumeFractions(ref mut dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.vol_fractions[i] as f32;
                }
            }
            Strains(ref mut dst) => {
                for i in 0..n_phases {
                    let strain = &self.all_strains[i][self.indices[i]];

                    for j in 0..6 {
                        dst[(pat_id, i, j)] = strain.0[j] as f32;
                    }
                }
            }
            Etas(dst) => {
                dst[pat_id] = self.meta.eta as f32;
            }
            MeanDsNm(dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.mean_ds_nm[i] as f32;
                }
            }
            CagliotiParams(_) => unreachable!(),
            MarchParameter(dst) => {
                for i in 0..n_phases {
                    let po = &self.all_preferred_orientations[i][self.indices[i]];
                    dst[(pat_id, i)] = po.as_ref().map_or(1.0, |x| x.r) as f32;
                }
            }
        }
    }

    fn init_meta_data(n_patterns: usize, n_phases: usize) -> Vec<PatternMeta> {
        use ndarray::{Array1, Array2, Array3};
        use PatternMeta::*;
        vec![
            Strains(Array3::<f32>::zeros((n_patterns, n_phases, 6))),
            Etas(Array1::<f32>::zeros(n_patterns)),
            MeanDsNm(Array2::<f32>::zeros((n_patterns, n_phases))),
            VolumeFractions(Array2::<f32>::zeros((n_patterns, n_phases))),
            MarchParameter(Array2::<f32>::zeros((n_patterns, n_phases))),
        ]
    }
}
