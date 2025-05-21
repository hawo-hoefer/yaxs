use std::io::{BufReader, Read};

use itertools::Itertools;
use ordered_float::NotNan;
use rand::distr::{Distribution, Uniform};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::background::Background;
use crate::cif::CifParser;
use crate::pattern::{DiscretizeAngleDisperse, EmissionLine, PatternMeta, Peaks};
use crate::structure::{MarchDollase, Strain, Structure};

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
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
    fn generate_bkg(&self, rng: &mut rand::rngs::StdRng) -> Background {
        use rand::prelude::*;
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SampleParameters {
    pub struct_cifs: Box<[String]>,
    pub preferred_orientation: Vec<Option<MarchDollase>>,
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

pub struct MetaGenerator {
    pub structures: Box<[Structure]>,
    pub angle_disperse: AngleDisperse,
    pub sample_params: SampleParameters,
    pub simulation_parameters: SimulationParameters,
}

impl TryFrom<Config> for MetaGenerator {
    type Error = String;

    fn try_from(cfg: Config) -> Result<Self, String> {
        let structures = cfg
            .sample_parameters
            .struct_cifs
            .iter()
            .map(|path| {
                // TODO: Errors
                let mut reader = BufReader::new(std::fs::File::open(path).unwrap());
                let mut cif = String::new();
                let _ = reader.read_to_string(&mut cif).unwrap();
                let mut p = CifParser::new(&cif);
                Structure::from(&p.parse())
            })
            .collect_vec()
            .into();

        match cfg.kind {
            SimulationKind::AngleDisperse(angle_disperse) => Ok(Self {
                angle_disperse,
                structures,
                sample_params: cfg.sample_parameters,
                simulation_parameters: cfg.simulation_parameters,
            }),
            SimulationKind::EnergyDisperse(_energy_disperse) => todo!(),
        }
    }
}

impl MetaGenerator {
    pub fn generate_job<'a>(
        &'a self,
        all_simulated_peaks: &'a Vec<Vec<Peaks>>,
        all_strains: &'a Vec<Vec<Strain>>,
        concentration_buf: &mut [NotNan<f64>],
        mut rng: &mut rand::rngs::StdRng,
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
        } = &self.angle_disperse;

        let SampleParameters {
            mean_ds_range_nm,
            eta_range,
            structure_permutations,
            ..
        } = &self.sample_params;

        let eta = rng.random_range(eta_range.0..=eta_range.1);
        let ds_sampler =
            Uniform::try_from(mean_ds_range_nm.0..=mean_ds_range_nm.1).unwrap_or_else(|err| {
                eprintln!("Could not sample mean domain size: {err}");
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
            indices: (0..self.structures.len())
                .map(|_| rng.random_range(0..*structure_permutations))
                .collect_vec(),
            emission_lines: &emission_lines,
            normalize: self.simulation_parameters.normalize,
            meta: PatternMeta {
                vol_fractions: concentration_buf
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
}
