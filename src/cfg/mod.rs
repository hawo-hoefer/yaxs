mod background;
mod impurity;
mod noise;
mod parameter;
mod preferred_orientation;
mod probability;
mod structure;
mod volume_fraction;

use std::sync::Arc;

use background::BackgroundSpec;
use impurity::{generate_impurities, ImpuritySpec};
use log::{debug, info};
use parameter::Parameter;
use probability::Probability;

pub use noise::NoiseSpec;
pub use preferred_orientation::MarchDollaseCfg;
pub use structure::{apply_strain_cfg, StrainCfg, StructureDef};
pub use volume_fraction::VolumeFraction;

use itertools::Itertools;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::math::e_kev_to_lambda_ams;
use crate::pattern::adxrd::{ADXRDMeta, DiscretizeAngleDispersive, EmissionLine};
use crate::pattern::edxrd::{Beamline, DiscretizeEnergyDispersive, EDXRDMeta};
use crate::pattern::{ImpurityPeak, Peaks, RenderCommon, VFGenerator};
use crate::preferred_orientation::MarchDollase;
use crate::structure::{Strain, Structure};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub kind: SimulationKind,
    pub sample_parameters: SampleParameters,
    pub simulation_parameters: SimulationParameters,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SimulationKind {
    AngleDispersive(AngleDispersive),
    EnergyDispersive(EnergyDispersive),
}

impl SimulationKind {
    pub fn get_r_range(&self) -> (f64, f64) {
        match self {
            SimulationKind::AngleDispersive(AngleDispersive {
                emission_lines,
                two_theta_range,
                ..
            }) => {
                let get_recip_r = |e: &EmissionLine| -> (f64, f64) {
                    let wavelength_ams = e.wavelength_ams;
                    (
                        (two_theta_range.0 / 2.0).to_radians().sin() / wavelength_ams * 2.0,
                        (two_theta_range.1 / 2.0).to_radians().sin() / wavelength_ams * 2.0,
                    )
                };
                let (min_r, max_r) = emission_lines.iter().map(get_recip_r).fold(
                    (std::f64::MAX, std::f64::MIN),
                    |(mut min, mut max), (new_min, new_max)| {
                        if new_min < min {
                            min = new_min;
                        }

                        if new_max > max {
                            max = new_max
                        }

                        (min, max)
                    },
                );

                info!(
                    "Simulating angle dispersive XRD 2-theta range [{t0}, {t1}] degrees",
                    t0 = two_theta_range.0,
                    t1 = two_theta_range.1
                );

                (min_r, max_r)
            }
            SimulationKind::EnergyDispersive(EnergyDispersive {
                energy_range_kev,
                theta_deg,
                ..
            }) => {
                let lambda_0 = e_kev_to_lambda_ams(energy_range_kev.1);
                let lambda_1 = e_kev_to_lambda_ams(energy_range_kev.0);

                let theta_rad = theta_deg.to_radians();

                info!(
                    "Simulating energy dispersive XRD with theta {theta_deg:.2} in energy range [{e0}, {e1}] keV",
                    e0 = energy_range_kev.0,
                    e1 = energy_range_kev.1
                );

                let min_r = theta_rad.sin() / lambda_1 * 2.0;
                let max_r = theta_rad.sin() / lambda_0 * 2.0;

                (min_r, max_r)
            }
        }
    }

    pub fn simulate_peaks(
        &self,
        structures: Box<[Structure]>,
        pref_o: Box<[Option<MarchDollaseCfg>]>,
        strain_cfgs: Box<[Option<StrainCfg>]>,
        structure_paths: Box<[String]>,
        sample_parameters: SampleParameters,
        rng: &mut impl Rng,
    ) -> Result<ToDiscretize, String> {
        let (min_r, max_r) = self.get_r_range();
        debug!("d-spacing range: [{},{}]", min_r, max_r);
        crate::structure::simulate_peaks(
            (min_r, max_r),
            sample_parameters,
            structures,
            pref_o,
            strain_cfgs,
            structure_paths,
            rng,
        )
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AngleDispersive {
    pub emission_lines: Box<[EmissionLine]>,

    pub n_steps: usize,
    pub two_theta_range: (f64, f64),
    pub goniometer_radius_mm: f64,

    pub sample_displacement_mu_m: Option<Parameter<f64>>,
    pub caglioti: Caglioti,
    pub background: BackgroundSpec,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Caglioti {
    pub u: Parameter<f64>,
    pub v: Parameter<f64>,
    pub w: Parameter<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EnergyDispersive {
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
    pub noise: Option<NoiseSpec>,

    pub abstol: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct SampleParameters {
    pub structures: Vec<StructureDef>,
    pub impurities: Option<Vec<ImpuritySpec>>,

    pub eta: Parameter<f64>,
    pub structure_permutations: usize,
}

pub struct Sample {
    eta: f64,
    mean_ds_nm: Box<[f64]>,
    impurity_peaks: Box<[ImpurityPeak]>,
    struct_ids: Box<[usize]>,
}

impl SampleParameters {
    pub fn generate(&self, rng: &mut impl Rng) -> Sample {
        let eta = self.eta.generate(rng);

        let mean_ds_nm = self
            .structures
            .iter()
            .map(|s| s.mean_ds_nm.generate(rng))
            .collect_vec()
            .into_boxed_slice();

        let impurity_peaks = self
            .impurities
            .as_ref()
            .map(|impurities| generate_impurities(impurities, rng))
            .unwrap_or(Box::new([]));

        let struct_ids = (0..self.structures.len())
            .map(|_| rng.random_range(0..self.structure_permutations))
            .collect_vec()
            .into();

        Sample {
            eta,
            mean_ds_nm,
            impurity_peaks,
            struct_ids,
        }
    }
}

pub struct CompactSimResults {
    pub all_simulated_peaks: Box<[Peaks]>,
    pub all_strains: Box<[Strain]>,
    pub all_preferred_orientations: Box<[Option<MarchDollase>]>,
    pub n_permutations: usize,
}

impl CompactSimResults {
    pub fn idx(&self, struct_idx: usize, perm_idx: usize) -> usize {
        struct_idx * self.n_permutations + perm_idx
    }
}

pub struct ToDiscretize {
    pub structures: Arc<[Structure]>,
    pub sample_parameters: SampleParameters,
    pub sim_res: Arc<CompactSimResults>,
    // pub simulation_parameters: SimulationParameters,
}

impl ToDiscretize {
    pub fn generate_adxrd_job(
        &self,
        vf_generator: &VFGenerator,
        angle_dispersive: &AngleDispersive,
        simulation_parameters: &SimulationParameters,
        rng: &mut impl Rng,
    ) -> DiscretizeAngleDispersive {
        let AngleDispersive {
            caglioti: Caglioti { u, v, w },
            background,
            emission_lines,
            n_steps: _,
            two_theta_range: _,
            goniometer_radius_mm,
            sample_displacement_mu_m,
        } = angle_dispersive;

        let Sample {
            eta,
            mean_ds_nm,
            impurity_peaks,
            struct_ids,
        } = self.sample_parameters.generate(rng);

        let u = u.generate(rng);
        let v = v.generate(rng);
        let w = w.generate(rng);
        let background = background.generate_bkg(rng);
        let sample_displacement_mu_m = (*sample_displacement_mu_m).map_or(0.0, |s| s.generate(rng));

        DiscretizeAngleDispersive {
            common: RenderCommon {
                sim_res: Arc::clone(&self.sim_res),
                impurity_peaks,
                indices: struct_ids,
                random_seed: rng.random(),
                noise: simulation_parameters
                    .noise
                    .as_ref()
                    .map(|x| x.generate(rng)),
            },
            emission_lines: emission_lines.clone(),
            goniometer_radius_mm: *goniometer_radius_mm,
            normalize: simulation_parameters.normalize,
            meta: ADXRDMeta {
                vol_fractions: vf_generator.generate(rng),
                eta,
                mean_ds_nm,
                u,
                v,
                w,
                background,
                sample_displacement_mu_m,
            },
        }
    }

    pub fn generate_edxrd_job(
        &self,
        vf_generator: &VFGenerator,
        energy_dispersive: &EnergyDispersive,
        simulation_parameters: &SimulationParameters,
        rng: &mut impl Rng,
    ) -> DiscretizeEnergyDispersive {
        let Sample {
            eta,
            mean_ds_nm,
            impurity_peaks,
            struct_ids,
        } = self.sample_parameters.generate(rng);

        DiscretizeEnergyDispersive {
            common: RenderCommon {
                sim_res: Arc::clone(&self.sim_res),
                indices: struct_ids,
                impurity_peaks,
                noise: simulation_parameters
                    .noise
                    .as_ref()
                    .map(|x| x.generate(rng)),
                random_seed: rng.random(),
            },
            beamline: energy_dispersive.beamline.clone(),
            normalize: simulation_parameters.normalize,
            meta: EDXRDMeta {
                vol_fractions: vf_generator.generate(rng),
                eta,
                mean_ds_nm,
                theta_rad: energy_dispersive.theta_deg.to_radians(),
            },
        }
    }
}
