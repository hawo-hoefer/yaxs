use std::io::{BufReader, Read};
use std::iter::Map;
use std::ops::Range;

use itertools::Itertools;
use rand::{Rng, SeedableRng};

use crate::background::Background;
use crate::cif::CifParser;
use crate::pattern::{EmissionLine, SimulationJob};
use crate::structure::Structure;

// class AugmentationCfg:
//     n_steps: int = 512
//     dst_two_theta_range: tuple[float, float] = (10.0, 70.0)

//     eta_range: tuple[float, float] = (0.1, 0.9)
//     crystallite_sz_range_nm: tuple[float, float] = (10, 100)

//     noise_scale_range: tuple[float, float] = (0.03, 0.07)
//     bkg_spec: BackgroundSpec = BackgroundSpec()

//     cag_u_range: tuple[float, float] = (0.00, 0.1)
//     cag_v_range: tuple[float, float] = (-0.1, 0.00)
//     cag_w_range: tuple[float, float] = (0.00, 0.1)

//     sample_displacement_range_mu_m: tuple[float, float] = (-100.0, 100.0)
//     diffractometer_radius_mm: float = 280.0

//     dst_wavelengths: list[float] = field(default_factory=lambda: [1.54])
//     intensity_ratios: NDArray[np.float32] = field(default=None, metadata=config(decoder=np.asarray))  # type: ignore

//     normalize: bool = True

pub enum BackgroundSpec {
    None,
    Chebyshev {
        coef_ranges: Vec<(f64, f64)>,
        // height_range: (f64, f64), // TODO
    },
    Exponential {
        slope_range: (f64, f64),
        // height_range: (f64, f64), // TODO
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

// TODO: derive from yaml
pub struct Config {
    pub struct_cifs: Box<[String]>,

    pub n_steps: u32,
    pub dst_two_theta_range: (f64, f64),

    pub eta_range: (f64, f64),
    pub noise_scale_range: (f64, f64),
    pub mean_ds_range: (f64, f64),

    pub cag_u_range: (f64, f64),
    pub cag_v_range: (f64, f64),
    pub cag_w_range: (f64, f64),

    pub sample_displacement_range_mu_m: (f64, f64),
    pub background_spec: BackgroundSpec,

    pub emission_lines: Box<[EmissionLine]>,

    pub normalize: bool,
    pub seed: Option<u64>,

    pub n_simulations: u32,
}

pub struct MetaGenerator {
    pub structures: Box<[Structure]>,
    pub config: Config,
    pub rng: rand::rngs::StdRng,
    pub i: usize,
}

impl From<Config> for MetaGenerator {
    fn from(config: Config) -> Self {
        let structures = config
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
        let rng = rand::rngs::StdRng::seed_from_u64(config.seed.unwrap_or(0));
        Self {
            config,
            structures: structures.into_boxed_slice(),
            rng,
            i: 0,
        }
    }
}

impl MetaGenerator {
    pub fn generate_job(&mut self) -> SimulationJob {
        let Config {
            n_steps,
            dst_two_theta_range,
            eta_range,
            noise_scale_range,
            mean_ds_range,
            cag_u_range,
            cag_v_range,
            cag_w_range,
            sample_displacement_range_mu_m,
            background_spec,
            emission_lines,
            normalize,
            ..
        } = &self.config;
        let eta = self.rng.random_range(eta_range.0..=eta_range.1);
        let mean_ds = self.rng.random_range(mean_ds_range.0..=mean_ds_range.1);
        let u = self.rng.random_range(cag_u_range.0..=cag_u_range.1);
        let v = self.rng.random_range(cag_v_range.0..=cag_v_range.1);
        let w = self.rng.random_range(cag_w_range.0..=cag_w_range.1);
        let background = background_spec.generate_bkg(&mut self.rng);

        SimulationJob {
            structures: &self.structures,
            emission_lines: &self.config.emission_lines,
            n_steps: self.config.n_steps,
            two_theta_range: self.config.dst_two_theta_range,
            eta,
            u,
            v,
            w,
            mean_ds,
            background,
            normalize: *normalize,
        }
    }
}
