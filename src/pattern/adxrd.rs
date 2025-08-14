use super::{
    render_peak, DiscretizeJobGenerator, Discretizer, PeakRenderParams, RenderCommon, VFGenerator,
};
use crate::background::Background;
use crate::cfg::{AngleDispersive, SimulationParameters, ToDiscretize};
use crate::io::PatternMeta;
use crate::noise::Noise;
use itertools::Itertools;
use log::debug;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct ADXRDMeta {
    pub vol_fractions: Box<[f64]>,
    pub weight_fractions: Option<Box<[f64]>>,
    pub mean_ds_nm: Box<[f64]>,
    pub eta: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub sample_displacement_mu_m: f64,
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

pub struct DiscretizeAngleDispersive {
    pub common: RenderCommon,
    pub emission_lines: Box<[EmissionLine]>,
    pub normalize: bool,
    pub meta: ADXRDMeta,
    pub goniometer_radius_mm: f64,
}

impl Discretizer for DiscretizeAngleDispersive {
    fn peak_info_iterator(&self) -> impl Iterator<Item = PeakRenderParams> {
        let ADXRDMeta {
            vol_fractions,
            mean_ds_nm,
            eta,
            u,
            v,
            w,
            sample_displacement_mu_m,
            background: _,
            weight_fractions: _,
        } = &self.meta;

        itertools::izip!(0..self.common.n_phases(), vol_fractions, mean_ds_nm,)
            .cartesian_product(&self.emission_lines)
            .flat_map(move |((phase_idx, vf, phase_mean_ds_nm), emission_line)| {
                let wavelength_nm = emission_line.wavelength_ams / 10.0;
                let idx = self.common.idx(phase_idx);
                self.common.sim_res.all_simulated_peaks[idx]
                    .iter()
                    .map(move |peak| {
                        let (two_theta_hkl_deg, peak_weight, fwhm) = peak.get_adxrd_render_params(
                            wavelength_nm,
                            *u,
                            *v,
                            *w,
                            *phase_mean_ds_nm,
                            vf * emission_line.weight,
                            *sample_displacement_mu_m,
                            self.goniometer_radius_mm,
                        );
                        PeakRenderParams {
                            pos: two_theta_hkl_deg,
                            intensity: peak_weight,
                            fwhm,
                            eta: *eta as f32,
                        }
                    })
            })
            .chain(
                self.common
                    .impurity_peaks
                    .iter()
                    .cartesian_product(&self.emission_lines)
                    .map(move |(ip, emission_line)| {
                        let wavelength_nm = emission_line.wavelength_ams / 10.0;
                        let (two_theta_hkl_deg, _, fwhm) = ip.peak.get_adxrd_render_params(
                            wavelength_nm,
                            *u,
                            *v,
                            *w,
                            ip.mean_ds_nm,
                            emission_line.weight,
                            0.0,
                            self.goniometer_radius_mm,
                        );
                        let peak_weight = ip.peak.i_hkl * emission_line.weight;
                        PeakRenderParams {
                            pos: two_theta_hkl_deg,
                            intensity: peak_weight as f32,
                            fwhm,
                            eta: ip.eta as f32,
                        }
                    }),
            )
    }

    fn n_peaks_tot(&self) -> usize {
        ((0..self.common.n_phases())
            .map(|i| self.common.sim_res.all_simulated_peaks[self.common.idx(i)].len())
            .sum::<usize>()
            + self.common.impurity_peaks.len())
            * self.emission_lines.len()
    }

    fn bkg(&self) -> &Background {
        &self.meta.background
    }

    fn normalize(&self) -> bool {
        self.normalize
    }

    fn write_meta_data(&self, data: &mut PatternMeta, pat_id: usize) {
        use PatternMeta::*;
        let n_phases = self.common.n_phases();
        match data {
            VolumeFractions(ref mut dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.vol_fractions[i] as f32;
                }
            }
            Strains(ref mut dst) => {
                for i in 0..n_phases {
                    let flat_idx = self.common.idx(i);
                    let strain = &self.common.sim_res.all_strains[flat_idx];

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
            CagliotiParams(dst) => {
                dst[(pat_id, 0)] = self.meta.u as f32;
                dst[(pat_id, 1)] = self.meta.v as f32;
                dst[(pat_id, 2)] = self.meta.w as f32;
            }
            ImpuritySum(dst) => {
                dst[pat_id] = self
                    .common
                    .impurity_peaks
                    .iter()
                    .map(|x| x.peak.i_hkl as f32)
                    .sum();
            }
            MarchParameter(dst) => {
                for i in 0..n_phases {
                    let flat_idx = self.common.idx(i);
                    let po = &self.common.sim_res.all_preferred_orientations[flat_idx];
                    dst[(pat_id, i)] = po.as_ref().map_or(1.0, |x| x.r) as f32;
                }
            }
            WeightFractions(dst) => {
                let Some(ref wfs) = self.meta.weight_fractions else {
                    panic!("Can only call this if weight fractions were computed before.");
                };
                for i in 0..n_phases {
                    dst[(pat_id, i)] = wfs[i] as f32;
                }

            }
        }
    }

    fn init_meta_data(
        n_patterns: usize,
        n_phases: usize,
        with_weight_fractions: bool,
    ) -> Vec<PatternMeta> {
        use ndarray::{Array1, Array2, Array3};
        use PatternMeta::*;
        let mut v = vec![
            Strains(Array3::<f32>::zeros((n_patterns, n_phases, 6))),
            Etas(Array1::<f32>::zeros(n_patterns)),
            CagliotiParams(Array2::<f32>::zeros((n_patterns, 3))),
            MeanDsNm(Array2::<f32>::zeros((n_patterns, n_phases))),
            VolumeFractions(Array2::<f32>::zeros((n_patterns, n_phases))),
            MarchParameter(Array2::<f32>::zeros((n_patterns, n_phases))),
            ImpuritySum(Array1::<f32>::zeros(n_patterns)),
        ];
        if with_weight_fractions {
            v.push(WeightFractions(Array2::<f32>::zeros((
                n_patterns, n_phases,
            ))))
        }
        v
    }

    fn seed(&self) -> u64 {
        self.common.random_seed
    }

    fn noise(&self) -> &Option<Noise> {
        &self.common.noise
    }

    fn discretize_into(&self, intensities: &mut [f32], positions: &[f32], abstol: f32) {
        for PeakRenderParams {
            pos,
            intensity,
            fwhm,
            eta,
        } in self.peak_info_iterator()
        {
            render_peak(pos, intensity, fwhm, eta, abstol, positions, intensities)
        }

        self.bkg().render(intensities, positions);
        if let Some(noise) = self.noise() {
            noise.apply(intensities, self.seed());
        }

        if self.normalize() {
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

pub struct JobGen<T> {
    cfg: AngleDispersive,
    discretize_info: ToDiscretize,
    sim_params: SimulationParameters,
    vf_generator: VFGenerator,
    two_thetas: Vec<f32>,
    n: usize,
    rng: T,
}

impl<T> JobGen<T> {
    pub fn new(
        cfg: AngleDispersive,
        discretize_info: ToDiscretize,
        sim_params: SimulationParameters,
        vf_generator: VFGenerator,
        rng: T,
    ) -> Self {
        let mut two_thetas = Vec::with_capacity(cfg.n_steps);
        two_thetas.resize(two_thetas.capacity(), 0.0f32);
        for (i, t) in two_thetas.iter_mut().enumerate() {
            let r = cfg.two_theta_range;
            *t = (r.0 + (r.1 - r.0) * (i as f64 / (cfg.n_steps as f64 - 1.0))) as f32;
        }

        Self {
            cfg,
            discretize_info,
            sim_params,
            vf_generator,
            two_thetas,
            n: 0,
            rng,
        }
    }
}

impl<T> DiscretizeJobGenerator for JobGen<T>
where
    T: Rng,
{
    type Item = DiscretizeAngleDispersive;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n >= self.sim_params.n_patterns {
            return None;
        }

        let job = self.discretize_info.generate_adxrd_job(
            &self.vf_generator,
            &self.cfg,
            &self.sim_params,
            &mut self.rng,
        );

        self.n += 1;

        Some(job)
    }

    fn remaining(&self) -> usize {
        self.sim_params.n_patterns - self.n
    }

    fn xs(&self) -> &[f32] {
        &self.two_thetas
    }

    fn n_phases(&self) -> usize {
        self.discretize_info.structures.len()
    }

    fn abstol(&self) -> f32 {
        self.sim_params.abstol
    }

    fn with_weight_fractions(&self) -> bool {
        self.discretize_info
            .structures
            .iter()
            .all(|s| s.density.is_some())
    }
}
