use itertools::Itertools;
use log::error;
use ordered_float::NotNan;
use rand::distr::uniform::SampleUniform;
use rand::distr::{Distribution, Uniform};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::background::Background;
use crate::pattern::adxrd::{ADXRDMeta, DiscretizeAngleDisperse, EmissionLine};
use crate::pattern::edxrd::{Beamline, DiscretizeEnergyDispersive, EDXRDMeta};
use crate::pattern::Peaks;
use crate::preferred_orientation::{MarchDollase, MarchDollaseCfg};
use crate::structure::{Strain, Structure};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum Parameter<T> {
    Fixed(T),
    Range(T, T),
    // Choice(Vec<T>),
    // ChoiceWithWeights(Vec<T>, Vec<f32>)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BackgroundSpec {
    None,
    Chebyshev {
        coefs: Vec<Parameter<f32>>,
        scale: Parameter<f32>,
    },
    Exponential {
        slope: Parameter<f32>,
        scale: Parameter<f32>,
    },
}

impl<T> Parameter<T>
where
    T: SampleUniform + PartialOrd + Copy,
{
    pub fn generate(&self, rng: &mut impl Rng) -> T {
        match self {
            Parameter::Fixed(v) => *v,
            Parameter::Range(lo, hi) => rng.random_range(*lo..=*hi),
        }
    }

    pub fn sampler(&self) -> Result<Uniform<T>, T> {
        match self {
            Parameter::Fixed(v) => Err(*v),
            Parameter::Range(lo, hi) => Ok(Uniform::try_from(*lo..=*hi).unwrap_or_else(|err| {
                error!("Could not sample mean domain size: {err}");
                std::process::exit(1);
            })),
        }
    }
}

impl BackgroundSpec {
    fn generate_bkg(&self, rng: &mut impl Rng) -> Background {
        match self {
            BackgroundSpec::None => Background::None,
            BackgroundSpec::Chebyshev { ref coefs, scale } => {
                let mut cheby_coefs = Vec::with_capacity(coefs.len());
                for param in coefs.iter() {
                    cheby_coefs.push(param.generate(rng));
                }
                let scale = scale.generate(rng);
                Background::chebyshev_polynomial(&cheby_coefs, scale)
            }
            BackgroundSpec::Exponential { slope, scale } => Background::Exponential {
                slope: slope.generate(rng),
                scale: scale.generate(rng),
            },
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Caglioti {
    pub u: Parameter<f64>,
    pub v: Parameter<f64>,
    pub w: Parameter<f64>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum Noise {
    None,
    Gaussian { sigma_min: f64, sigma_max: f64 },
    // Uniform // TODO
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AngleDisperse {
    pub emission_lines: Box<[EmissionLine]>,

    pub n_steps: usize,
    pub two_theta_range: (f64, f64),

    pub noise: Noise,
    pub caglioti: Caglioti,
    pub background: BackgroundSpec,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EnergyDisperse {
    pub n_steps: usize,
    pub energy_range_kev: (f64, f64),
    pub theta_deg: f64,
    pub beamline: Beamline,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SimulationParameters {
    pub normalize: bool,
    pub seed: Option<u64>,
    pub n_patterns: usize,

    pub abstol: f32,
}

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(untagged)]
pub enum StrainCfg {
    Maximum(f64),
    ParameterizedOrtho([Parameter<f64>; 3]),
    ParameterizedFull([Parameter<f64>; 6]),
}

pub fn apply_strain_cfg(
    cfg: &Option<StrainCfg>,
    s: &Structure,
    rng: &mut impl Rng,
) -> Option<(Structure, Strain)> {
    use StrainCfg::*;
    match cfg {
        Some(Maximum(max_strain)) => Some(s.permute(*max_strain, rng)),
        Some(ParameterizedOrtho(params)) => {
            let strain = Strain::from_diag(
                params[0].generate(rng),
                params[1].generate(rng),
                params[2].generate(rng),
            );
            Some((s.apply_strain(&strain), strain))
        }
        Some(ParameterizedFull(params)) => {
            let strain = Strain::new_verified([
                params[0].generate(rng),
                params[1].generate(rng),
                params[2].generate(rng),
                params[3].generate(rng),
                params[4].generate(rng),
                params[5].generate(rng),
            ])?;
            Some((s.apply_strain(&strain), strain))
        }
        None => Some((s.clone(), Strain::none())),
    }
}

#[derive(PartialEq, Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct StructureDef {
    pub path: String,
    pub preferred_orientation: Option<MarchDollaseCfg>,
    pub strain: Option<StrainCfg>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct SampleParameters {
    pub structures_po: Vec<StructureDef>,
    pub mean_ds_nm: Parameter<f64>,
    pub sample_displacement_mu_m: Parameter<f64>,

    pub eta: Parameter<f64>,
    pub structure_permutations: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SimulationKind {
    AngleDisperse(AngleDisperse),
    EnergyDisperse(EnergyDisperse),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub kind: SimulationKind,
    pub sample_parameters: SampleParameters,
    pub simulation_parameters: SimulationParameters,
}

pub struct JobCfg<'a> {
    pub structures: &'a [Structure],
    pub sample_params: &'a SampleParameters,
    pub simulation_parameters: &'a SimulationParameters,
}

impl JobCfg<'_> {
    pub fn generate_adxrd_job<'a>(
        &self,
        all_simulated_peaks: &'a Vec<Vec<Peaks>>,
        all_strains: &'a Vec<Vec<Strain>>,
        all_preferred_orientations: &'a Vec<Vec<Option<MarchDollase>>>,
        concentration_buf: &mut [NotNan<f64>],
        angle_disperse: &'a AngleDisperse,
        mut rng: &mut impl Rng,
    ) -> DiscretizeAngleDisperse<'a> {
        let AngleDisperse {
            caglioti: Caglioti { u, v, w },
            // sample_displacement_range_mu_m,
            background,
            emission_lines,
            ..
        } = angle_disperse;

        let SampleParameters {
            mean_ds_nm,
            eta,
            structure_permutations,
            ..
        } = &self.sample_params;

        let n_phases = all_simulated_peaks.len();

        let eta = eta.generate(rng);
        let ds_sampler = mean_ds_nm.sampler();
        let mut mean_ds_nm: Vec<f64> = Vec::with_capacity(n_phases);
        match ds_sampler {
            Ok(sampler) => {
                mean_ds_nm.extend(sampler.sample_iter(&mut rng).take(n_phases));
            }
            Err(v) => mean_ds_nm.resize(n_phases, v),
        }
        let u = u.generate(rng);
        let v = v.generate(rng);
        let w = w.generate(rng);
        let background = background.generate_bkg(rng);

        concentration_buf[0] = NotNan::try_from(0.0).unwrap();
        concentration_buf[concentration_buf.len() - 1] = NotNan::try_from(1.0).unwrap();
        for i in 1..self.structures.len() {
            concentration_buf[i] = NotNan::try_from(rng.random_range(0.0..=1.0))
                .expect("numbers between 0 and 1 are not NaN")
        }
        concentration_buf.sort();
        for i in 0..concentration_buf.len() - 1 {
            concentration_buf[i] = concentration_buf[i + 1] - concentration_buf[i];
        }

        DiscretizeAngleDisperse {
            all_simulated_peaks,
            all_strains,
            all_preferred_orientations,
            indices: (0..self.structures.len())
                .map(|_| rng.random_range(0..*structure_permutations))
                .collect_vec(),
            emission_lines: &emission_lines,
            normalize: self.simulation_parameters.normalize,
            meta: ADXRDMeta {
                vol_fractions: concentration_buf[..concentration_buf.len() - 1]
                    .iter()
                    .map(|x| f64::from(*x))
                    .collect_vec()
                    .into(),
                eta,
                mean_ds_nm: mean_ds_nm.into_boxed_slice(),
                u,
                v,
                w,
                background,
            },
        }
    }

    pub fn generate_edxrd_job<'a>(
        &self,
        all_simulated_peaks: &'a Vec<Vec<Peaks>>,
        all_strains: &'a Vec<Vec<Strain>>,
        all_preferred_orientations: &'a Vec<Vec<Option<MarchDollase>>>,
        energy_disperse: &'a EnergyDisperse,
        concentration_buf: &mut [NotNan<f64>],
        mut rng: &mut impl Rng,
    ) -> DiscretizeEnergyDispersive<'a> {
        let SampleParameters {
            mean_ds_nm,
            eta,
            structure_permutations,
            ..
        } = &self.sample_params;

        let n_phases = all_simulated_peaks.len();

        let eta = eta.generate(rng);
        let ds_sampler = mean_ds_nm.sampler();
        let mut mean_ds_nm: Vec<f64> = Vec::with_capacity(n_phases);
        match ds_sampler {
            Ok(sampler) => {
                mean_ds_nm.extend(sampler.sample_iter(&mut rng).take(n_phases));
            }
            Err(v) => mean_ds_nm.resize(n_phases, v),
        }

        concentration_buf[0] = NotNan::try_from(0.0).unwrap();
        concentration_buf[concentration_buf.len() - 1] = NotNan::try_from(1.0).unwrap();
        for i in 1..self.structures.len() {
            concentration_buf[i] = NotNan::try_from(rng.random_range(0.0..=1.0))
                .expect("numbers between 0 and 1 are not NaN")
        }
        concentration_buf.sort();
        for i in 0..concentration_buf.len() - 1 {
            concentration_buf[i] = concentration_buf[i + 1] - concentration_buf[i];
        }

        DiscretizeEnergyDispersive {
            all_simulated_peaks,
            all_strains,
            all_preferred_orientations,
            beamline: &energy_disperse.beamline,
            indices: (0..self.structures.len())
                .map(|_| rng.random_range(0..*structure_permutations))
                .collect_vec(),
            normalize: self.simulation_parameters.normalize,
            meta: EDXRDMeta {
                vol_fractions: concentration_buf[..concentration_buf.len() - 1]
                    .iter()
                    .map(|x| f64::from(*x))
                    .collect_vec()
                    .into(),
                eta,
                mean_ds_nm: mean_ds_nm.into_boxed_slice(),
                theta_rad: energy_disperse.theta_deg.to_radians(),
            },
        }
    }
}
