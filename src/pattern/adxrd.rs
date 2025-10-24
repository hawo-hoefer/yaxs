use super::{
    render_peak, DiscretizeJobGenerator, DiscretizeSample, Discretizer, JobParams,
    PeakRenderParams, RenderCommon, VFGenerator,
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

#[derive(Clone)]
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
                            *sample_displacement_mu_m,
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
            BinghamODFParams { orientations, ks } => {
                for (i, bingham_odf) in (0..n_phases)
                    .filter_map(|i| {
                        let flat_idx = self.common.idx(i);
                        self.common.sim_res.all_preferred_orientations[flat_idx].as_ref()
                    })
                    .enumerate()
                {
                    orientations[(pat_id, i, 0)] = bingham_odf.orientation[0] as f32;
                    orientations[(pat_id, i, 1)] = bingham_odf.orientation[1] as f32;
                    orientations[(pat_id, i, 2)] = bingham_odf.orientation[2] as f32;
                    orientations[(pat_id, i, 3)] = bingham_odf.orientation[3] as f32;

                    let phase_ks = &bingham_odf.ks;

                    ks[(pat_id, i, 0)] = phase_ks[0] as f32;
                    ks[(pat_id, i, 1)] = phase_ks[1] as f32;
                    ks[(pat_id, i, 2)] = phase_ks[2] as f32;
                    ks[(pat_id, i, 3)] = phase_ks[3] as f32;
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
            SampleDisplacementMuM(dst) => dst[pat_id] = self.meta.sample_displacement_mu_m as f32,
        }
    }

    fn init_meta_data(n_samples: usize, p: &JobParams) -> Vec<PatternMeta> {
        use ndarray::{Array1, Array2, Array3};
        use PatternMeta::*;
        let mut v = vec![
            Strains(Array3::<f32>::zeros((n_samples, p.n_phases, 6))),
            Etas(Array1::<f32>::zeros(n_samples)),
            CagliotiParams(Array2::<f32>::zeros((n_samples, 3))),
            MeanDsNm(Array2::<f32>::zeros((n_samples, p.n_phases))),
            VolumeFractions(Array2::<f32>::zeros((n_samples, p.n_phases))),
            ImpuritySum(Array1::<f32>::zeros(n_samples)),
            SampleDisplacementMuM(Array1::<f32>::zeros(n_samples)),
        ];

        if p.has_weight_fracs {
            v.push(WeightFractions(Array2::<f32>::zeros((
                n_samples, p.n_phases,
            ))))
        }

        if let Some(n) = p.textured_phases {
            v.push(BinghamODFParams {
                orientations: Array3::zeros((n_samples, n, 4)),
                ks: Array3::zeros((n_samples, n, 4)),
            })
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
    to_discretize: ToDiscretize,
    sim_params: SimulationParameters,
    vf_generator: VFGenerator,
    two_thetas: Vec<f32>,
    n: usize,
    cur_job: Option<DiscretizeAngleDispersive>,
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
            to_discretize: discretize_info,
            sim_params,
            vf_generator,
            two_thetas,
            n: 0,
            cur_job: None,
            rng,
        }
    }
}

impl<T> DiscretizeJobGenerator for JobGen<T>
where
    T: Rng,
{
    type Item = DiscretizeAngleDispersive;

    fn next(&mut self) -> Option<DiscretizeSample<Self::Item>> {
        if self.n >= self.sim_params.n_patterns {
            return None;
        }
        let mut job = self.to_discretize.generate_adxrd_job(
            &self.vf_generator,
            &self.cfg,
            &self.sim_params,
            &mut self.rng,
        );

        let ret = match self.sim_params.texture_measurement {
            Some(t) => {
                let mut ret = Vec::new();
                for _ in 0..t.stride() {
                    for idx in job.common.indices.iter_mut() {
                        *idx += 1;
                    }
                    ret.push(job.clone());
                }
                DiscretizeSample::TextureMeasurement(ret)
            }
            None => DiscretizeSample::Standard(job),
        };

        self.n += 1;

        Some(ret)
    }

    fn remaining(&self) -> usize {
        self.sim_params.n_patterns - self.n
    }

    fn xs(&self) -> &[f32] {
        &self.two_thetas
    }

    fn get_job_params(&self) -> JobParams {
        let textured_phases = self.sim_params.texture_measurement.as_ref().map(|_| {
            self.to_discretize
                .sample_parameters
                .structures
                .iter()
                .map(|x| x.preferred_orientation.as_ref().map(|_| 1).unwrap_or(0))
                .sum()
        });

        JobParams {
            n_phases: self.to_discretize.structures.len(),
            abstol: self.sim_params.abstol,
            has_weight_fracs: self
                .to_discretize
                .structures
                .iter()
                .all(|s| s.density.is_some()),
            textured_phases,
            texture_measurement: self.sim_params.texture_measurement,
        }
    }
}
