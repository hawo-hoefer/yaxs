use super::{
    DiscretizeJobGenerator, DiscretizeSample, Discretizer, JobParams, PeakRenderParams,
    RenderCommon, VFGenerator,
};
use crate::background::Background;
use crate::cfg::{AngleDispersive, SimulationParameters, ToDiscretize};
use crate::io::PatternMeta;
use crate::noise::Noise;
use itertools::Itertools;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct ADXRDMeta {
    pub vol_fractions: Box<[f64]>,
    pub weight_fractions: Option<Box<[f64]>>,
    pub mean_ds_nm: Box<[f64]>,
    pub ds_eta: Box<[f64]>,
    pub mustrain: Box<[f64]>,
    pub mustrain_eta: Box<[f64]>,
    pub instrument_parameters: InstrumentParameters,
    pub sample_displacement_mu_m: f64,
    pub background: Background,
}

#[repr(C)]
#[derive(serde::Deserialize, serde::Serialize, PartialEq, Debug, Clone)]
#[serde(deny_unknown_fields)]
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

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct InstrumentParameters {
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl InstrumentParameters {
    pub fn new(u: f64, v: f64, w: f64, x: f64, y: f64, z: f64) -> Self {
        InstrumentParameters { u, v, w, x, y, z }
    }
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Calculate gaussian line broadening (Caglioti)
    /// $FWHM(\theta)^2 = u \tan(\theta)^2 + v \tan(\theta) + w$
    ///
    /// * `theta`: theta in radians
    pub fn gauss_broadening(&self, theta: f64) -> f64 {
        self.u * theta.tan().powi(2) + self.v * theta.tan() + self.w
    }

    /// Calculate lorentzian line broadening
    pub fn lorentz_broadening(&self, theta: f64) -> f64 {
        self.x / theta.cos() + self.y * theta.tan() + self.z
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
            instrument_parameters,
            sample_displacement_mu_m,
            background: _,
            weight_fractions: _,
            mean_ds_nm,
            ds_eta,
            mustrain,
            mustrain_eta,
        } = &self.meta;

        itertools::izip!(
            0..self.common.n_phases(),
            vol_fractions,
            mean_ds_nm,
            ds_eta,
            mustrain,
            mustrain_eta
        )
        .cartesian_product(&self.emission_lines)
        .flat_map(
            move |(
                (phase_idx, vf, phase_mean_ds_nm, phase_ds_eta, phase_mustrain, phase_mustrain_eta),
                emission_line,
            )| {
                let wavelength_nm = emission_line.wavelength_ams / 10.0;
                let idx = self.common.idx(phase_idx);
                self.common.sim_res.all_simulated_peaks[idx]
                    .iter()
                    .map(move |peak| {
                        peak.get_adxrd_render_params(
                            wavelength_nm,
                            instrument_parameters,
                            *phase_mean_ds_nm,
                            *phase_ds_eta,
                            *phase_mustrain,
                            *phase_mustrain_eta,
                            vf * emission_line.weight,
                            *sample_displacement_mu_m,
                            self.goniometer_radius_mm,
                        )
                    })
            },
        )
        .chain(
            self.common
                .impurity_peaks
                .iter()
                .cartesian_product(&self.emission_lines)
                .map(move |(ip, emission_line)| {
                    let wavelength_nm = emission_line.wavelength_ams / 10.0;
                    ip.peak.get_adxrd_render_params(
                        wavelength_nm,
                        instrument_parameters,
                        ip.mean_ds_nm,
                        ip.eta,
                        0.0, // impurity peaks only have one source of
                        0.0, // peak broadening for now.
                        emission_line.weight,
                        *sample_displacement_mu_m,
                        self.goniometer_radius_mm,
                    )
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
            MeanDsNm(dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.mean_ds_nm[i] as f32;
                }
            }
            DsEtas(dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.ds_eta[i] as f32;
                }
            }
            InstrumentParameters(dst) => {
                dst[(pat_id, 0)] = self.meta.instrument_parameters.u as f32;
                dst[(pat_id, 1)] = self.meta.instrument_parameters.v as f32;
                dst[(pat_id, 2)] = self.meta.instrument_parameters.w as f32;
                dst[(pat_id, 3)] = self.meta.instrument_parameters.x as f32;
                dst[(pat_id, 4)] = self.meta.instrument_parameters.y as f32;
                dst[(pat_id, 5)] = self.meta.instrument_parameters.z as f32;
            }
            ImpuritySum(dst) => {
                dst[pat_id] = self
                    .common
                    .impurity_peaks
                    .iter()
                    .map(|x| x.peak.i_hkl as f32)
                    .sum();
            }
            ImpurityMax(dst) => {
                dst[pat_id] = self
                    .common
                    .impurity_peaks
                    .iter()
                    .map(|x| x.peak.i_hkl as f32)
                    .max_by(|a, b| a.partial_cmp(&b).expect("no NaNs in peak intensities"))
                    .unwrap_or(0.0);
            }
            BinghamODFParams { orientations, ks } => {
                for (i, bingham_odf) in (0..n_phases)
                    .filter_map(|i| {
                        let flat_idx = self.common.idx(i);
                        self.common.sim_res.all_preferred_orientations[flat_idx].as_ref()
                    })
                    .enumerate()
                {
                    orientations[(pat_id, i, 0)] = bingham_odf.orientation.w as f32;
                    orientations[(pat_id, i, 1)] = bingham_odf.orientation.x as f32;
                    orientations[(pat_id, i, 2)] = bingham_odf.orientation.y as f32;
                    orientations[(pat_id, i, 3)] = bingham_odf.orientation.z as f32;

                    let phase_ks = &bingham_odf.ks;

                    ks[(pat_id, i, 0)] = phase_ks.w as f32;
                    ks[(pat_id, i, 1)] = phase_ks.x as f32;
                    ks[(pat_id, i, 2)] = phase_ks.y as f32;
                    ks[(pat_id, i, 3)] = phase_ks.z as f32;
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
            BackgroundParameters(dst) => {
                match &self.meta.background {
                    Background::None => unreachable!("all patterns must have the same background type. Background::None does not initialize the background output."),
                    Background::Chebyshev { coef, scale } => {
                        dst[(pat_id, 0)] = *scale;
                        for (coef_idx, c) in coef.iter().enumerate() {
                            dst[(pat_id, coef_idx + 1)] = *c;
                        }
                    },
                    Background::Exponential { slope, scale } => {
                        dst[(pat_id, 0)] = *scale;
                        dst[(pat_id, 1)] = *slope;
                    },
                }
            }
            Mustrains(dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.mustrain[i] as f32;
                }
            },
            MustrainEtas(dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.mustrain_eta[i] as f32;
                }
            },
        }
    }

    fn init_meta_data(n_samples: usize, p: &JobParams) -> Vec<PatternMeta> {
        use ndarray::{Array1, Array2, Array3};
        use PatternMeta::*;
        let mut v = vec![
            Strains(Array3::<f32>::zeros((n_samples, p.n_phases, 6))),
            InstrumentParameters(Array2::<f32>::zeros((n_samples, 6))),
            MeanDsNm(Array2::<f32>::zeros((n_samples, p.n_phases))),
            DsEtas(Array2::<f32>::zeros((n_samples, p.n_phases))),
            Mustrains(Array2::<f32>::zeros((n_samples, p.n_phases))),
            MustrainEtas(Array2::<f32>::zeros((n_samples, p.n_phases))),
            VolumeFractions(Array2::<f32>::zeros((n_samples, p.n_phases))),
            ImpuritySum(Array1::<f32>::zeros(n_samples)),
            SampleDisplacementMuM(Array1::<f32>::zeros(n_samples)),
            ImpurityMax(Array1::<f32>::zeros(n_samples)),
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

    fn next(&mut self) -> Option<DiscretizeSample<Self::Item>> {
        if self.n >= self.sim_params.n_patterns {
            return None;
        }

        let mut job = self.discretize_info.generate_adxrd_job(
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
            self.discretize_info
                .sample_parameters
                .structures
                .iter()
                .map(|x| x.preferred_orientation.as_ref().map(|_| 1).unwrap_or(0))
                .sum()
        });

        JobParams {
            n_phases: self.discretize_info.structures.len(),
            abstol: self.sim_params.abstol,
            has_weight_fracs: self
                .discretize_info
                .structures
                .iter()
                .all(|s| s.density.is_some()),
            textured_phases,
            texture_measurement: self.sim_params.texture_measurement,
            bkg_params: self.cfg.background.as_ref().map(|x| x.n_coefs()),
        }
    }
}
