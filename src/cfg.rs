use std::io::{BufReader, Read};

use itertools::Itertools;
use ordered_float::NotNan;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::background::Background;
use crate::cif::CifParser;
use crate::pattern::{DiscretizationJob, EmissionLine, PatternMeta, Peaks};
use crate::structure::Structure;

#[derive(serde::Deserialize, serde::Serialize)]
pub enum BackgroundSpec {
    None,
    Chebyshev {
        coef_ranges: Vec<(f32, f32)>,
        // height_range: (f32, f32), // TODO
    },
    Exponential {
        slope_range: (f32, f32),
        // height_range: (f32, f32), // TODO
    },
}

impl BackgroundSpec {
    fn generate_bkg(&self, rng: &mut rand::rngs::StdRng) -> Background {
        use rand::prelude::*;
        match self {
            BackgroundSpec::None => Background::None,
            BackgroundSpec::Chebyshev { ref coef_ranges } => Background::chebyshev_polynomial(
                &coef_ranges
                    .iter()
                    .map(|&(lo, hi)| rng.random_range(lo..=hi))
                    .collect_vec(),
            ),
            BackgroundSpec::Exponential {
                slope_range: (lo, hi),
            } => Background::Exponential(rng.random_range(*lo..=*hi)),
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct Caglioti {
    pub u_range: (f64, f64),
    pub v_range: (f64, f64),
    pub w_range: (f64, f64),
}

#[derive(Deserialize, Serialize)]
pub enum Noise {
    None,
    Gaussian { sigma_min: f64, sigma_max: f64 },
    // Uniform // TODO
}

#[derive(Deserialize, Serialize)]
pub struct Config {
    pub struct_cifs: Box<[String]>,
    pub emission_lines: Box<[EmissionLine]>,

    pub n_steps: usize,
    pub two_theta_range: (f64, f64),

    pub mean_ds_range_nm: (f64, f64),
    pub eta_range: (f64, f64),
    pub sample_displacement_range_mu_m: (f64, f64),
    pub max_strain: f64,

    pub noise: Noise,
    pub caglioti: Caglioti,
    pub background: BackgroundSpec,

    pub normalize: bool,
    pub seed: Option<u64>,
    pub n_patterns: usize,
    pub structure_permutations: usize,

    pub abstol: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            struct_cifs: Box::new([]),
            emission_lines: Box::new([]),
            n_steps: 2048,
            two_theta_range: (10.0, 70.0),
            eta_range: (0.1, 0.9),
            noise: Noise::Gaussian {
                sigma_min: 0.0,
                sigma_max: 100.0,
            },
            mean_ds_range_nm: (50.0, 50.0),
            caglioti: Caglioti {
                u_range: (0.0, 0.025),
                v_range: (-0.025, 0.0),
                w_range: (0.0, 0.025),
            },
            sample_displacement_range_mu_m: (-250.0, 250.0),
            background: BackgroundSpec::None,
            normalize: false,
            seed: Some(1234),
            n_patterns: 1,
            max_strain: 0.01,
            structure_permutations: 1,
            abstol: 1e-2,
        }
    }
}

pub struct MetaGenerator {
    pub structures: Box<[Structure]>,
    pub cfg: Config,
    pub i: usize,
}

impl From<Config> for MetaGenerator {
    fn from(cfg: Config) -> Self {
        let structures = cfg
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
            .collect_vec();
        Self {
            cfg,
            structures: structures.into_boxed_slice(),
            i: 0,
        }
    }
}

impl MetaGenerator {
    pub fn generate_job<'a>(
        &'a self,
        all_simulated_peaks: &'a Vec<Vec<Peaks>>,
        concentration_buf: &mut [NotNan<f64>],
        rng: &mut rand::rngs::StdRng,
    ) -> DiscretizationJob<'a> {
        let Config {
            eta_range,
            mean_ds_range_nm,
            caglioti:
                Caglioti {
                    u_range,
                    v_range,
                    w_range,
                },
            // sample_displacement_range_mu_m,
            background,
            emission_lines,
            normalize,
            structure_permutations,
            ..
        } = &self.cfg;

        let eta = rng.random_range(eta_range.0..=eta_range.1);
        let mean_ds_nm = rng.random_range(mean_ds_range_nm.0..=mean_ds_range_nm.1);
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

        DiscretizationJob {
            all_simulated_peaks,
            indices: (0..self.structures.len())
                .map(|_| rng.random_range(0..*structure_permutations))
                .collect_vec(),
            emission_lines: &emission_lines,
            normalize: *normalize,
            meta: PatternMeta {
                vol_fractions: concentration_buf
                    .iter()
                    .map(|x| f64::from(*x))
                    .collect_vec()
                    .into(),
                eta,
                mean_ds_nm,
                u,
                v,
                w,
                background,
            },
        }
    }
}
