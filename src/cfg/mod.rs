mod background;
mod impurity;
mod noise;
mod parameter;
mod preferred_orientation;
mod probability;
mod structure;
mod volume_fraction;

use background::BackgroundSpec;
use impurity::{generate_impurities, ImpuritySpec};
use log::info;
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
use crate::pattern::adxrd::{ADXRDMeta, DiscretizeAngleDisperse, EmissionLine};
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
    AngleDisperse(AngleDisperse),
    EnergyDisperse(EnergyDisperse),
}

impl SimulationKind {
    pub fn simulate_peaks(
        &self,
        structures: Box<[Structure]>,
        pref_o: Box<[&Option<MarchDollaseCfg>]>,
        strain_cfgs: Box<[&Option<StrainCfg>]>,
        structure_paths: Box<[&String]>,
        sample_parameters: SampleParameters,
        rng: &mut impl Rng,
    ) -> ToDiscretize {
        let (min_r, max_r) = match self {
            SimulationKind::AngleDisperse(angle_disperse) => {
                let min_line = &angle_disperse
                    .emission_lines
                    .iter()
                    .min_by(|a, b| {
                        a.wavelength_ams
                            .partial_cmp(&b.wavelength_ams)
                            .expect("no NaNs in wavelengths")
                    })
                    .expect("at least one emission line");

                let (two_theta_range, wavelength_ams) =
                    (angle_disperse.two_theta_range, min_line.wavelength_ams);

                let min_r = (two_theta_range.0 / 2.0).to_radians().sin() / wavelength_ams * 2.0;
                let max_r = (two_theta_range.1 / 2.0).to_radians().sin() / wavelength_ams * 2.0;

                info!("Simulating {two_theta_range:?} {wavelength_ams:.2}");
                (min_r, max_r)
            }
            SimulationKind::EnergyDisperse(EnergyDisperse {
                n_steps: _,
                energy_range_kev,
                theta_deg,
                beamline: _,
            }) => {
                let lambda_0 = e_kev_to_lambda_ams(energy_range_kev.1);
                let lambda_1 = e_kev_to_lambda_ams(energy_range_kev.0);

                let theta_rad = theta_deg.to_radians();

                let min_r = theta_rad.sin() / lambda_1 * 2.0;
                let max_r = theta_rad.sin() / lambda_0 * 2.0;

                (min_r, max_r)
            }
        };

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
pub struct AngleDisperse {
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

impl SampleParameters {
    pub fn generate(
        &self,
        rng: &mut impl Rng,
    ) -> (f64, Box<[f64]>, Box<[ImpurityPeak]>, Box<[usize]>) {
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

        let indices = (0..self.structures.len())
            .map(|_| rng.random_range(0..self.structure_permutations))
            .collect_vec()
            .into();

        (eta, mean_ds_nm, impurity_peaks, indices)
    }
}

pub struct ToDiscretize {
    pub structures: Box<[Structure]>,
    pub sample_parameters: SampleParameters,
    // pub simulation_parameters: SimulationParameters,
    pub all_simulated_peaks: Box<[Box<[Peaks]>]>,
    pub all_strains: Box<[Box<[Strain]>]>,
    pub all_preferred_orientations: Box<[Box<[Option<MarchDollase>]>]>,
}

impl ToDiscretize {
    pub fn generate_adxrd_job<'a>(
        &'a self,
        vf_generator: &VFGenerator<'a>,
        angle_disperse: &'a AngleDisperse,
        simulation_parameters: &'a SimulationParameters,
        rng: &mut impl Rng,
    ) -> DiscretizeAngleDisperse<'a> {
        let AngleDisperse {
            caglioti: Caglioti { u, v, w },
            background,
            emission_lines,
            n_steps: _,
            two_theta_range: _,
            goniometer_radius_mm,
            sample_displacement_mu_m,
        } = angle_disperse;

        let (eta, mean_ds_nm, impurity_peaks, indices) = self.sample_parameters.generate(rng);

        let u = u.generate(rng);
        let v = v.generate(rng);
        let w = w.generate(rng);
        let background = background.generate_bkg(rng);
        let sample_displacement_mu_m = (*sample_displacement_mu_m).map_or(0.0, |s| s.generate(rng));

        DiscretizeAngleDisperse {
            common: RenderCommon {
                all_simulated_peaks: &self.all_simulated_peaks,
                all_strains: &self.all_strains,
                all_preferred_orientations: &self.all_preferred_orientations,
                impurity_peaks,
                indices,
                noise: simulation_parameters.noise.clone(),
            },
            emission_lines: &emission_lines,
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

    pub fn generate_edxrd_job<'a>(
        &'a self,
        energy_disperse: &'a EnergyDisperse,
        vf_generator: &VFGenerator<'a>,
        simulation_parameters: &'a SimulationParameters,
        rng: &mut impl Rng,
    ) -> DiscretizeEnergyDispersive<'a> {
        let (eta, mean_ds_nm, impurity_peaks, indices) = self.sample_parameters.generate(rng);

        DiscretizeEnergyDispersive {
            common: RenderCommon {
                all_simulated_peaks: &self.all_simulated_peaks,
                all_strains: &self.all_strains,
                all_preferred_orientations: &self.all_preferred_orientations,
                indices,
                impurity_peaks,
                noise: simulation_parameters.noise.clone(),
            },
            beamline: &energy_disperse.beamline,
            normalize: simulation_parameters.normalize,
            meta: EDXRDMeta {
                vol_fractions: vf_generator.generate(rng),
                eta,
                mean_ds_nm,
                theta_rad: energy_disperse.theta_deg.to_radians(),
            },
        }
    }
}
