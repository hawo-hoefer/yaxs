mod background;
mod composition;
mod impurity;
mod noise;
mod parameter;
mod preferred_orientation;
mod probability;
mod structure;

use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

use crate::absorption::{compute_mixture_attenuation_coef, MACGenerator};
use crate::cif::CifParser;
use crate::util::{
    deserialize_angle_rad_to_deg, deserialize_nonzero_float, deserialize_nonzero_usize,
    deserialize_range,
};
use background::BackgroundSpec;
use impurity::{generate_impurities, ImpuritySpec};
use log::{debug, info};
pub use parameter::Parameter;
use probability::Probability;

pub use composition::CompositionPart;
pub use noise::NoiseSpec;
pub use preferred_orientation::{KDEApprox, POCfg, POGenerator};
pub use structure::{apply_strain_cfg, StrainCfg, StructureDef};

use itertools::Itertools;
use rand::Rng;
use serde::{Deserialize, Serialize};
mod texture;

use crate::math::{e_kev_to_lambda_ams, funcs};
use crate::pattern::adxrd::{
    ADXRDMeta, DiscretizeAngleDispersive, EmissionLine, InstrumentParameters, PrecomputedLACs,
};
use crate::pattern::edxrd::{Beamline, DiscretizeEnergyDispersive, EDXRDMeta};
use crate::pattern::{
    get_volume_fractions, get_weight_fractions, CompositionGenerator, CompositionSubset,
    ImpurityPeak, Peaks, RenderCommon,
};
use crate::preferred_orientation::BinghamParams;
use crate::strain::Strain;
use crate::structure::Structure;

pub use self::texture::TextureMeasurement;
pub use pyo3::pyclass;

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

pub struct PreparedSimData {
    pub structures: Box<[Structure]>,
    pub pref_o: Box<[Option<POGenerator>]>,
    pub strain_cfgs: Box<[Option<StrainCfg>]>,
    pub structure_paths: Box<[String]>,
    pub composition_constraints: Box<[Option<CompositionPart>]>,
    pub b_iso_ranges: Box<[Option<Parameter<f64>>]>,
}

pub fn prepare_peak_simulation(
    cfg: &mut Config,
    root_path: impl AsRef<Path>,
    rng: &mut impl Rng,
) -> Result<PreparedSimData, String> {
    if let Some(ref mut imp) = cfg.sample_parameters.impurities {
        // get upper and lower bound for d_hkl
        let (lb, ub) = {
            let (r_min, r_max) = cfg.kind.get_r_range();
            (1.0 / r_max, 1.0 / r_min)
        };
        for spec in imp.iter_mut() {
            spec.validate_d_hkl_or_adjust(lb, ub);
        }
    }

    let mut structures = Vec::new();
    let mut pref_o = Vec::new();
    let mut strain_cfgs = Vec::new();
    let mut structure_paths = Vec::new();
    let mut composition_constraints = Vec::new();
    let mut b_iso_params = Vec::new();

    for StructureDef {
        path,
        preferred_orientation: po,
        strain,
        composition,
        mean_ds_nm,
        ds_eta: _,
        mustrain: _,
        b_iso,
    } in cfg.sample_parameters.structures.iter()
    {
        let mut struct_path = root_path.as_ref().to_path_buf();
        struct_path.push(path.clone());

        let mut reader = std::fs::File::open(&struct_path)
            .map(BufReader::new)
            .map_err(|err| format!("Could not load cif at '{struct_path}': {err}", struct_path=struct_path.display()))?;

        let mut cif = String::new();
        let _ = reader
            .read_to_string(&mut cif)
            .map_err(|err| format!("Invalid UTF8 in cif {}: {}", struct_path.display(), err,))?;
        let mut p = CifParser::new(&cif).with_file(struct_path.display().to_string());

        let structure = p
            .parse()
            .map_err(|err| format!("Invalid CIF Syntax for '{path}': {err}"))
            .and_then(|x| {
                Structure::try_from(&x)
                    .map_err(|err| format!("Invalid contents for CIF '{path}': {err}"))
            })?;

        if mean_ds_nm.upper_bound() > 200.0 {
            return Err(format!("Specified a mean domain size with an upper bound of {hi} nm. The scherrer Formula is only valid up until 200 nm. Larger domain sizes are not supported for now. Quitting...", hi=mean_ds_nm.upper_bound()));
        }

        structure_paths.push(struct_path.to_str().expect("valid path").to_owned());
        composition_constraints.push(composition.clone());
        strain_cfgs.push(strain.clone());
        b_iso_params.push(b_iso.clone());
        let po_gen = po
            .as_ref()
            .map(|x| {
                x.try_into_generator(rng).map_err(|x| {
                    format!(
                        "Could not get preferred orientation generator for {p}: {x}",
                        p = struct_path.display()
                    )
                })
            })
            .transpose()?;
        pref_o.push(po_gen);
        structures.push(structure);
    }

    Ok(PreparedSimData {
        structures: structures.into(),
        pref_o: pref_o.into(),
        strain_cfgs: strain_cfgs.into(),
        structure_paths: structure_paths.into(),
        composition_constraints: composition_constraints.into(),
        b_iso_ranges: b_iso_params.into(),
    })
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

                let min_r = theta_rad.sin() / lambda_1 * 2.0;
                let max_r = theta_rad.sin() / lambda_0 * 2.0;

                (min_r, max_r)
            }
        }
    }

    pub fn simulate_peaks(
        &self,
        // structures: Box<[Structure]>,
        // pref_o: Box<[Option<POGenerator>]>,
        // strain_cfgs: Box<[Option<StrainCfg>]>,
        // structure_paths: Box<[String]>,
        // b_iso_ranges: Box<[Option<Parameter<f64>>]>,
        psd: PreparedSimData,
        sample_parameters: SampleParameters,
        texture_measurement: Option<TextureMeasurement>,
        rng: &mut impl Rng,
    ) -> Result<ToDiscretize, String> {
        let (min_r, max_r) = self.get_r_range();
        match self {
            SimulationKind::AngleDispersive(AngleDispersive {
                two_theta_range,
                emission_lines,
                ..
            }) => {
                let mut lines = String::new();
                use std::fmt::Write;
                for (i, line) in emission_lines.iter().enumerate() {
                    write!(&mut lines, "{}", line.wavelength_ams).expect("enough memory");
                    if i != emission_lines.len() - 1 {
                        write!(&mut lines, ", ").expect("enough memory");
                    }
                }
                info!(
                    "Simulating angle dispersive XRD with emission lines [{lines}] in 2-theta range [{t0}, {t1}] degrees",
                    t0 = two_theta_range.0,
                    t1 = two_theta_range.1
                );
            }
            SimulationKind::EnergyDispersive(EnergyDispersive {
                energy_range_kev,
                theta_deg,
                ..
            }) => {
                info!(
                "Simulating energy dispersive XRD with theta {theta_deg:.2} in energy range [{e0}, {e1}] keV",
                e0 = energy_range_kev.0,
                e1 = energy_range_kev.1
            );
            }
        }

        debug!("d-spacing range: [{},{}]", min_r, max_r);
        crate::peak_sim::simulate_peaks(
            (min_r, max_r),
            sample_parameters,
            psd,
            texture_measurement,
            rng,
        )
    }
}

pub fn precompute_absorption_factors(
    wavelengths: impl Iterator<Item = f64>,
    structures: &[Structure],
) -> Result<PrecomputedLACs, String> {
    let mut ret = Vec::new();
    for w in wavelengths {
        let energy_kev = funcs::e_kev_to_lambda_ams(w);

        let mut structure_absorption_factors = Vec::with_capacity(structures.len());
        for s in structures.iter() {
            let mac = s.wt_composition.get_mac_at_energy(energy_kev)?;
            let lac = mac * s.density_g_cm3;
            structure_absorption_factors.push(1.0 / (2.0 * lac));
        }
        ret.push(structure_absorption_factors.into());
    }

    Ok(PrecomputedLACs(ret.into()))
}

fn default_monochromator_angle() -> f64 {
    0.0
}

#[pyclass(from_py_object)]
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct AngleDispersive {
    pub emission_lines: Box<[EmissionLine]>,

    #[serde(deserialize_with = "deserialize_nonzero_usize")]
    pub n_steps: usize,
    #[serde(deserialize_with = "deserialize_range")]
    pub two_theta_range: (f64, f64),
    #[serde(deserialize_with = "deserialize_nonzero_float")]
    pub goniometer_radius_mm: f64,

    #[serde(
        rename = "monochromator_angle_deg",
        default = "default_monochromator_angle",
        deserialize_with = "deserialize_angle_rad_to_deg"
    )]
    pub monochromator_angle: f64,

    pub sample_displacement_mu_m: Option<Parameter<f64>>,
    pub instrument_parameters: Option<InstrumentParameterCfg>,
    pub background: Option<BackgroundSpec>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstprmKind {
    Raw,
    GSAS,
}

impl Default for InstprmKind {
    fn default() -> Self {
        InstprmKind::Raw
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct InstrumentParameterCfg {
    #[serde(default)]
    pub kind: InstprmKind,
    #[serde(default)]
    pub u: Parameter<f64>,
    #[serde(default)]
    pub v: Parameter<f64>,
    #[serde(default)]
    pub w: Parameter<f64>,
    #[serde(default)]
    pub x: Parameter<f64>,
    #[serde(default)]
    pub y: Parameter<f64>,
    #[serde(default)]
    pub z: Parameter<f64>,
}

impl InstrumentParameterCfg {
    pub fn mean(&self) -> InstrumentParameters {
        InstrumentParameters::new(
            self.u.mean(),
            self.v.mean(),
            self.w.mean(),
            self.x.mean(),
            self.y.mean(),
            self.z.mean(),
        )
    }

    pub fn generate(&self, rng: &mut impl Rng) -> InstrumentParameters {
        let mut u = self.u.generate(rng);
        let mut v = self.v.generate(rng);
        let mut w = self.w.generate(rng);
        let mut x = self.x.generate(rng);
        let mut y = self.y.generate(rng);
        let mut z = self.z.generate(rng);

        match self.kind {
            InstprmKind::Raw => {
                // leave parameters as-is
            }
            InstprmKind::GSAS => {
                // GSAS computes FWHM in centidegrees
                // Gaussian instrument parameters therefore are FWHM^2 coefficients in
                // centidegrees squared. therefore scale them by 10000
                u /= 10000.0;
                v /= 10000.0;
                w /= 10000.0;
                // Lorentzian parameters are coefficients for FWHM in centidegrees,
                // therefore they should be scaled by 100 for our purposes
                x /= 100.0;
                y /= 100.0;
                z /= 100.0;
            }
        }

        InstrumentParameters::new(u, v, w, x, y, z)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnergyDispersive {
    #[serde(deserialize_with = "deserialize_nonzero_usize")]
    pub n_steps: usize,
    #[serde(deserialize_with = "deserialize_range")]
    pub energy_range_kev: (f64, f64),
    pub theta_deg: f64,
    pub beamline: Beamline,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SimulationParameters {
    pub normalize: bool,
    pub seed: Option<u64>,
    #[serde(deserialize_with = "deserialize_nonzero_usize")]
    pub n_patterns: usize,
    pub noise: Option<NoiseSpec>,
    pub texture_measurement: Option<TextureMeasurement>,
    pub randomly_scale_peaks: Option<RandomlyScalePeaks>,

    pub abstol: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RandomlyScalePeaks {
    pub scale: Parameter<f64>,
    pub probability: Probability,
}

impl RandomlyScalePeaks {
    pub fn scale_peak(&self, peak_weight: f64, rng: &mut impl Rng) -> f64 {
        if self.probability.generate_bool(rng) {
            peak_weight * self.scale.generate(rng)
        } else {
            peak_weight
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum CompositionKind {
    ByMass,
    ByVolume,
}

impl Default for CompositionKind {
    fn default() -> Self {
        CompositionKind::ByVolume
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct SampleParameters {
    #[serde(default)]
    pub composition_kind: CompositionKind,
    pub structures: Vec<StructureDef>,
    pub concentration_subset: Option<CompositionSubset>,
    pub impurities: Option<Vec<ImpuritySpec>>,

    #[serde(deserialize_with = "deserialize_nonzero_usize")]
    pub structure_permutations: usize,
}

pub struct Sample {
    ds_eta: Box<[f64]>,
    mean_ds_nm: Box<[f64]>,
    mustrain: Box<[f64]>,
    mustrain_eta: Box<[f64]>,
    impurity_peaks: Box<[ImpurityPeak]>,
    permutation_ids: Box<[usize]>,
}

impl SampleParameters {
    pub fn generate(&self, rng: &mut impl Rng) -> Sample {
        let mean_ds_nm = self
            .structures
            .iter()
            .map(|s| s.mean_ds_nm.generate(rng))
            .collect_vec()
            .into_boxed_slice();

        let ds_eta = self
            .structures
            .iter()
            .map(|x| x.ds_eta.generate(rng))
            .collect_vec()
            .into_boxed_slice();

        let mustrain = self
            .structures
            .iter()
            .map(|x| {
                x.mustrain
                    .as_ref()
                    .map(|x| x.amplitude.generate(rng))
                    .unwrap_or(0.0)
            })
            .collect_vec()
            .into_boxed_slice();

        let mustrain_eta = self
            .structures
            .iter()
            .map(|x| {
                x.mustrain
                    .as_ref()
                    .map(|x| x.eta.generate(rng))
                    .unwrap_or(0.0)
            })
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
            ds_eta,
            mean_ds_nm,
            impurity_peaks,
            permutation_ids: struct_ids,
            mustrain,
            mustrain_eta,
        }
    }
}

/// A structure of arrays containing simulated peak positions and corresponding simulation
/// parameters      
///
/// * `all_simulated_peaks`:
/// * `all_strains`:
/// * `all_preferred_orientations`:
/// * `n_permutations`:
pub struct CompactSimResults {
    pub all_simulated_peaks: Box<[Peaks]>,
    pub all_strains: Box<[Strain]>,
    pub random_b_isos: Option<Box<[f64]>>,
    pub all_preferred_orientations: Box<[Option<BinghamParams>]>,
    pub n_permutations: usize,
    pub texture_measurement: Option<TextureMeasurement>,
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
}

impl ToDiscretize {
    pub fn generate_adxrd_job(
        &self,
        composition_generator: &CompositionGenerator,
        angle_dispersive: &AngleDispersive,
        simulation_parameters: &SimulationParameters,
        precomputed_lacs: PrecomputedLACs,
        rng: &mut impl Rng,
    ) -> DiscretizeAngleDispersive {
        let AngleDispersive {
            instrument_parameters,
            background,
            emission_lines,
            n_steps: _,
            two_theta_range: _,
            goniometer_radius_mm,
            sample_displacement_mu_m,
            monochromator_angle,
        } = angle_dispersive;

        let Sample {
            mean_ds_nm,
            impurity_peaks,
            permutation_ids,
            ds_eta,
            mustrain,
            mustrain_eta,
        } = self.sample_parameters.generate(rng);

        let background = background
            .as_ref()
            .map(|x| x.generate_bkg(rng))
            .unwrap_or(crate::background::Background::None);

        let composition = composition_generator.generate(rng);
        let (vol_fractions, weight_fractions) = match self.sample_parameters.composition_kind {
            CompositionKind::ByMass => (
                get_volume_fractions(&composition, &self.structures),
                composition,
            ),
            CompositionKind::ByVolume => {
                let wt_fractions = get_weight_fractions(&composition, &self.structures);
                (composition, wt_fractions)
            }
        };

        let sample_displacement_mu_m = (*sample_displacement_mu_m).map_or(0.0, |s| s.generate(rng));

        let attenuation_coefs = compute_mixture_attenuation_coef(&vol_fractions, &precomputed_lacs);

        let random_b_iso = self.get_random_b_iso(&permutation_ids);

        DiscretizeAngleDispersive {
            common: RenderCommon {
                sim_res: Arc::clone(&self.sim_res),
                impurity_peaks,
                indices: permutation_ids,
                random_seed: rng.random(),
                noise: simulation_parameters
                    .noise
                    .as_ref()
                    .map(|x| x.generate(rng)),
            },
            emission_lines: emission_lines.clone(),
            goniometer_radius_mm: *goniometer_radius_mm,
            normalize: simulation_parameters.normalize,
            attenuation_coefs,
            meta: ADXRDMeta {
                vol_fractions,
                weight_fractions,

                mean_ds_nm,
                ds_eta,
                mustrain,
                mustrain_eta,

                sample_displacement_mu_m,

                instrument_parameters: instrument_parameters
                    .as_ref()
                    .map(|x| x.generate(rng))
                    .unwrap_or(InstrumentParameters::zero()),
                background,
                random_b_iso,
            },
            monochromator_angle_rad: *monochromator_angle,
        }
    }

    pub fn generate_edxrd_job(
        &self,
        composition_generator: &CompositionGenerator,
        mac_generator: &MACGenerator,
        energy_dispersive: &EnergyDispersive,
        simulation_parameters: &SimulationParameters,
        rng: &mut impl Rng,
    ) -> DiscretizeEnergyDispersive {
        let Sample {
            mean_ds_nm,
            impurity_peaks,
            permutation_ids,
            ds_eta,
            mustrain,
            mustrain_eta,
        } = self.sample_parameters.generate(rng);

        let composition = composition_generator.generate(rng);
        let (vol_fractions, weight_fractions) = match self.sample_parameters.composition_kind {
            CompositionKind::ByMass => (
                get_volume_fractions(&composition, &self.structures),
                composition,
            ),
            CompositionKind::ByVolume => {
                let wt_fractions = get_weight_fractions(&composition, &self.structures);
                (composition, wt_fractions)
            }
        };

        let random_b_iso = self.get_random_b_iso(&permutation_ids);

        let mixture_mac = mac_generator.get_mixture(
            self.structures
                .iter()
                .zip(weight_fractions.iter())
                .map(|(s, wf)| (&s.wt_composition, *wf)),
        );

        DiscretizeEnergyDispersive {
            common: RenderCommon {
                sim_res: Arc::clone(&self.sim_res),
                indices: permutation_ids,
                impurity_peaks,
                noise: simulation_parameters
                    .noise
                    .as_ref()
                    .map(|x| x.generate(rng)),
                random_seed: rng.random(),
            },
            beamline: energy_dispersive.beamline.clone(),
            normalize: simulation_parameters.normalize,
            mixture_mac,
            meta: EDXRDMeta {
                vol_fractions,
                weight_fractions,
                mean_ds_nm,
                theta_rad: energy_dispersive.theta_deg.to_radians(),
                ds_eta,
                mustrain,
                mustrain_eta,
                random_b_iso,
            },
        }
    }

    fn get_random_b_iso(&self, permutation_ids: &[usize]) -> Option<Box<[f64]>> {
        let n_permutations = self.sample_parameters.structure_permutations;
        self.sim_res.random_b_isos.as_ref().map(|bisos| {
            permutation_ids
                .iter()
                .enumerate()
                .map(|(struct_id, pid)| bisos[struct_id * n_permutations + *pid])
                .collect_vec()
                .into_boxed_slice()
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn deser_instrument_param() {
        let _err = serde_yaml::from_str::<InstrumentParameterCfg>(
            "
u: [0.0, -0.25]
v: [-0.25, 0.0]
w: [0.0, -0.25]
",
        )
        .expect_err("invalid range parameters");
    }
}
