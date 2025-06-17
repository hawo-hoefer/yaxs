use itertools::Itertools;
use rand::distr::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::background::Background;
use crate::parameter::Parameter;
use crate::pattern::adxrd::{ADXRDMeta, DiscretizeAngleDisperse, EmissionLine};
use crate::pattern::edxrd::{Beamline, DiscretizeEnergyDispersive, EDXRDMeta};
use crate::pattern::{ImpurityPeak, Peak, Peaks, RenderCommon, VFGenerator};
use crate::preferred_orientation::{MarchDollase, MarchDollaseCfg};
use crate::structure::{Strain, Structure};

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
    pub goniometer_radius_mm: f64,

    pub sample_displacement_mu_m: Option<Parameter<f64>>,
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
pub enum StrainCfg {
    Maximum(f64),
    Ortho([Parameter<f64>; 3]),
    Full([Parameter<f64>; 6]),
}

pub fn apply_strain_cfg(
    cfg: &Option<StrainCfg>,
    s: &Structure,
    rng: &mut impl Rng,
) -> Option<(Structure, Strain)> {
    use StrainCfg::*;
    match cfg {
        Some(Maximum(max_strain)) => Some(s.permute(*max_strain, rng)),
        Some(Ortho(params)) => {
            let strain = Strain::from_diag(
                params[0].generate(rng),
                params[1].generate(rng),
                params[2].generate(rng),
            );
            Some((s.apply_strain(&strain), strain))
        }
        Some(Full(params)) => {
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

#[derive(Debug, Serialize, PartialEq, Clone, Copy)]
#[serde(transparent)]
pub struct VolumeFraction(pub f64);

impl<'de> Deserialize<'de> for VolumeFraction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let v = f64::deserialize(deserializer)?;
        if v < 0.0 || v > 1.0 {
            return Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(v),
                &"Volume fraction needs to be in [0.0, 1.0]",
            ));
        }

        Ok(VolumeFraction(v))
    }
}

#[derive(PartialEq, Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct StructureDef {
    pub path: String,
    pub preferred_orientation: Option<MarchDollaseCfg>,
    pub strain: Option<StrainCfg>,
    pub volume_fraction: Option<VolumeFraction>,
}

#[derive(Serialize, Debug, Clone)]
pub struct ImpuritySpec {
    d_hkl_ams: Parameter<f64>,
    intensity: Parameter<f64>,
    eta: Parameter<f64>,
    mean_ds_nm: Parameter<f64>,
    probability: Option<f64>,
    n_peaks: Option<usize>,
}

impl ImpuritySpec {
    pub fn new(
        d_hkl_ams: Parameter<f64>,
        intensity: Parameter<f64>,
        eta: Parameter<f64>,
        mean_ds_nm: Parameter<f64>,
        probability: Option<Probability>,
        n_peaks: Option<usize>,
    ) -> Self {
        Self {
            d_hkl_ams,
            intensity,
            eta,
            mean_ds_nm,
            probability: probability.map(|p| p.0),
            n_peaks,
        }
    }
}

#[derive(Serialize, Debug, Clone, Copy)]
pub struct Probability(f64);

impl Probability {
    pub fn new(p: f64) -> Option<Self> {
        if p < 0.0 || p > 1.0 {
            return None;
        }

        Some(Self(p))
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ProbVisitor;
        impl<'de> serde::de::Visitor<'de> for ProbVisitor {
            type Value = Probability;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(formatter, "struct Probability")
            }

            fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let p = Probability::new(v);
                p.ok_or(serde::de::Error::invalid_value(
                    serde::de::Unexpected::Float(v),
                    &"value in the range [0, 1]",
                ))
            }
        }

        deserializer.deserialize_f64(ProbVisitor)
    }
}

impl<'de> Deserialize<'de> for ImpuritySpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct _ImpuritySpec {
            d_hkl_ams: Parameter<f64>,
            intensity: Parameter<f64>,
            eta: Parameter<f64>,
            mean_ds_nm: Parameter<f64>,
            probability: Option<Probability>,
            n_peaks: Option<usize>,
        }

        let _ImpuritySpec {
            d_hkl_ams,
            intensity,
            eta,
            mean_ds_nm,
            probability,
            n_peaks,
        } = _ImpuritySpec::deserialize(deserializer)?;

        Ok(ImpuritySpec::new(
            d_hkl_ams,
            intensity,
            eta,
            mean_ds_nm,
            probability,
            n_peaks,
        ))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct SampleParameters {
    pub structures: Vec<StructureDef>,
    pub mean_ds_nm: Parameter<f64>,
    pub impurities: Option<Vec<ImpuritySpec>>,

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

fn generate_impurities(impurity_specs: &[ImpuritySpec], rng: &mut impl Rng) -> Box<[ImpurityPeak]> {
    let mut impurity_peaks = Vec::new();
    for spec in impurity_specs.iter() {
        for _ in 0..spec.n_peaks.unwrap_or(1) {
            if !spec.probability.map(|p| rng.random_bool(p)).unwrap_or(true) {
                continue;
            }
            let d_hkl = spec.d_hkl_ams.generate(rng);
            let i_hkl = spec.intensity.generate(rng);
            impurity_peaks.push(ImpurityPeak {
                peak: Peak {
                    d_hkl,
                    i_hkl,
                    hkls: Vec::new(),
                },
                eta: spec.eta.generate(rng),
                mean_ds_nm: spec.mean_ds_nm.generate(rng),
            })
        }
    }
    impurity_peaks.into()
}

impl JobCfg<'_> {
    pub fn generate_adxrd_job<'a>(
        &self,
        all_simulated_peaks: &'a Vec<Vec<Peaks>>,
        all_strains: &'a Vec<Vec<Strain>>,
        all_preferred_orientations: &'a Vec<Vec<Option<MarchDollase>>>,
        vf_generator: &VFGenerator<'a>,
        angle_disperse: &'a AngleDisperse,
        mut rng: &mut impl Rng,
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
            structures: _,
            mean_ds_nm,
            eta,
            structure_permutations,
            impurities,
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
                mean_ds_nm: mean_ds_nm.into_boxed_slice(),
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
        mut rng: &mut impl Rng,
    ) -> DiscretizeEnergyDispersive<'a> {
        let SampleParameters {
            mean_ds_nm,
            eta,
            structure_permutations,
            structures: _,
            impurities,
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
                mean_ds_nm: mean_ds_nm.into_boxed_slice(),
                theta_rad: energy_disperse.theta_deg.to_radians(),
            },
        }
    }
}
