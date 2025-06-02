use itertools::Itertools;
use log::error;
use ordered_float::NotNan;
use rand::distr::{Distribution, Uniform};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::background::Background;
use crate::pattern::adxrd::{ADXRDMeta, DiscretizeAngleDisperse, EmissionLine};
use crate::pattern::edxrd::{DiscretizeEnergyDispersive, EDXRDMeta};
use crate::pattern::Peaks;
use crate::preferred_orientation::{MarchDollase, MarchDollaseCfg};
use crate::structure::{Strain, Structure};

#[derive(Serialize, Deserialize, Debug, Clone)]
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
        coef_ranges: Vec<(f32, f32)>,
        scale_range: (f32, f32),
    },
    Exponential {
        slope_range: (f32, f32),
        scale_range: (f32, f32),
    },
}

impl BackgroundSpec {
    fn generate_bkg(&self, rng: &mut impl Rng) -> Background {
        match self {
            BackgroundSpec::None => Background::None,
            BackgroundSpec::Chebyshev {
                ref coef_ranges,
                scale_range,
            } => {
                let mut cheby_coefs = Vec::with_capacity(coef_ranges.len());
                for (lo, hi) in coef_ranges.iter() {
                    cheby_coefs.push(rng.random_range(*lo..=*hi));
                }
                Background::chebyshev_polynomial(
                    &cheby_coefs,
                    rng.random_range(scale_range.0..=scale_range.1),
                )
            }
            BackgroundSpec::Exponential {
                slope_range: (lo, hi),
                scale_range: (scale_lo, scale_hi),
            } => Background::Exponential {
                slope: rng.random_range(*lo..=*hi),
                scale: rng.random_range(*scale_lo..=*scale_hi),
            },
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Caglioti {
    pub u_range: (f64, f64),
    pub v_range: (f64, f64),
    pub w_range: (f64, f64),
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SimulationParameters {
    pub normalize: bool,
    pub seed: Option<u64>,
    pub n_patterns: usize,

    pub abstol: f32,
}

#[derive(PartialEq, Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct StructureDef {
    pub path: String,
    pub preferred_orientation: Option<MarchDollaseCfg>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SampleParameters {
    pub structures_po: Vec<StructureDef>,
    pub mean_ds_range_nm: (f64, f64),
    pub sample_displacement_range_mu_m: (f64, f64),
    pub max_strain: f64,

    pub eta_range: (f64, f64),
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EnergyDisperse {
    pub n_steps: usize,
    pub energy_range_kev: (f64, f64),
    pub theta_deg: f64,
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
            caglioti:
                Caglioti {
                    u_range,
                    v_range,
                    w_range,
                },
            // sample_displacement_range_mu_m,
            background,
            emission_lines,
            ..
        } = angle_disperse;

        let SampleParameters {
            mean_ds_range_nm,
            eta_range,
            structure_permutations,
            ..
        } = &self.sample_params;

        let eta = rng.random_range(eta_range.0..=eta_range.1);
        let ds_sampler =
            Uniform::try_from(mean_ds_range_nm.0..=mean_ds_range_nm.1).unwrap_or_else(|err| {
                error!("Could not sample mean domain size: {err}");
                std::process::exit(1);
            });
        let mut mean_ds_nm: Vec<f64> = Vec::with_capacity(concentration_buf.len());
        mean_ds_nm.extend(
            ds_sampler
                .sample_iter(&mut rng)
                .take(concentration_buf.len()),
        );
        let u = rng.random_range(u_range.0..=u_range.1);
        let v = rng.random_range(v_range.0..=v_range.1);
        let w = rng.random_range(w_range.0..=w_range.1);
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
            mean_ds_range_nm,
            eta_range,
            structure_permutations,
            ..
        } = &self.sample_params;

        let eta = rng.random_range(eta_range.0..=eta_range.1);
        let ds_sampler =
            Uniform::try_from(mean_ds_range_nm.0..=mean_ds_range_nm.1).unwrap_or_else(|err| {
                error!("Could not sample mean domain size: {err}");
                std::process::exit(1);
            });
        let mut mean_ds_nm: Vec<f64> = Vec::with_capacity(concentration_buf.len());
        mean_ds_nm.extend(
            ds_sampler
                .sample_iter(&mut rng)
                .take(concentration_buf.len()),
        );

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
