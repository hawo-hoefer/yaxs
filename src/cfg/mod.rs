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
use noise::Noise;
use parameter::Parameter;
use probability::Probability;

pub use preferred_orientation::MarchDollaseCfg;
pub use structure::{apply_strain_cfg, StrainCfg, StructureDef};
pub use volume_fraction::VolumeFraction;

use itertools::Itertools;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::pattern::adxrd::{ADXRDMeta, DiscretizeAngleDisperse, EmissionLine};
use crate::pattern::edxrd::{Beamline, DiscretizeEnergyDispersive, EDXRDMeta};
use crate::pattern::{Peaks, RenderCommon, VFGenerator};
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

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct AngleDisperse {
    pub emission_lines: Box<[EmissionLine]>,

    pub n_steps: usize,
    pub two_theta_range: (f64, f64),
    pub goniometer_radius_mm: f64,

    pub sample_displacement_mu_m: Option<Parameter<f64>>,
    pub noise: Option<Noise>,
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
        vf_generator: &VFGenerator<'a>,
        angle_disperse: &'a AngleDisperse,
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
            noise: _,
        } = angle_disperse;

        let SampleParameters {
            structures,
            eta,
            structure_permutations,
            impurities,
        } = &self.sample_params;

        let eta = eta.generate(rng);
        let mean_ds_nm = structures
            .iter()
            .map(|s| s.mean_ds_nm.generate(rng))
            .collect_vec()
            .into_boxed_slice();

        let u = u.generate(rng);
        let v = v.generate(rng);
        let w = w.generate(rng);
        let background = background.generate_bkg(rng);
        let sample_displacement_mu_m = (*sample_displacement_mu_m).map_or(0.0, |s| s.generate(rng));

        let impurity_peaks = impurities
            .as_ref()
            .map(|impurities| generate_impurities(impurities, rng))
            .unwrap_or(Box::new([]));

        DiscretizeAngleDisperse {
            common: RenderCommon {
                all_simulated_peaks,
                all_strains,
                all_preferred_orientations,
                impurity_peaks,
                indices: (0..self.structures.len())
                    .map(|_| rng.random_range(0..*structure_permutations))
                    .collect_vec(),
            },
            emission_lines: &emission_lines,
            normalize: self.simulation_parameters.normalize,
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
            goniometer_radius_mm: *goniometer_radius_mm,
        }
    }

    pub fn generate_edxrd_job<'a>(
        &self,
        all_simulated_peaks: &'a Vec<Vec<Peaks>>,
        all_strains: &'a Vec<Vec<Strain>>,
        all_preferred_orientations: &'a Vec<Vec<Option<MarchDollase>>>,
        energy_disperse: &'a EnergyDisperse,
        vf_generator: &VFGenerator<'a>,
        rng: &mut impl Rng,
    ) -> DiscretizeEnergyDispersive<'a> {
        let SampleParameters {
            eta,
            structure_permutations,
            structures,
            impurities,
        } = &self.sample_params;

        let eta = eta.generate(rng);

        let mean_ds_nm = structures
            .iter()
            .map(|s| s.mean_ds_nm.generate(rng))
            .collect_vec()
            .into_boxed_slice();

        let impurity_peaks = impurities
            .as_ref()
            .map(|impurities| generate_impurities(impurities, rng))
            .unwrap_or(Box::new([]));

        DiscretizeEnergyDispersive {
            common: RenderCommon {
                all_simulated_peaks,
                all_strains,
                all_preferred_orientations,
                indices: (0..self.structures.len())
                    .map(|_| rng.random_range(0..*structure_permutations))
                    .collect_vec(),
                impurity_peaks,
            },
            beamline: &energy_disperse.beamline,
            normalize: self.simulation_parameters.normalize,
            meta: EDXRDMeta {
                vol_fractions: vf_generator.generate(rng),
                eta,
                mean_ds_nm,
                theta_rad: energy_disperse.theta_deg.to_radians(),
            },
        }
    }
}
