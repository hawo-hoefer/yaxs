use itertools::Itertools;
use rand::Rng;
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Serialize};

use crate::background::Background;
use crate::cfg::{EnergyDispersive, SimulationParameters, ToDiscretize};
use crate::io::PatternMeta;
use crate::math::{C_M_S, ELECTRON_MASS_KG, EV_TO_JOULE, H_EV_S};
use crate::noise::Noise;
use crate::pattern::lorentz_polarization_factor;

use super::{
    DiscretizeJobGenerator, Discretizer, Peak, PeakRenderParams, RenderCommon, VFGenerator,
};

/// Wiggler Beamline Parameters
///
/// * `e_crit_kev`: critical electron energy in keV
/// * `storage_ring_electron_energy_gev`: storage ring electron energy in GeV
/// * `storage_ring_current_a`: storage ring current in ampere
/// * `n_wiggler_magnets`: number of wiggler magnets
/// * `distance_from_device_m`: distance of sample from device in m
#[derive(Serialize, Debug, Clone)]
pub struct Beamline {
    #[serde(skip_serializing)]
    e_crit_kev: f64,
    #[serde(skip_serializing)]
    electron_lorentz_factor: f64,

    storage_ring_electron_energy_gev: f64,
    storage_ring_current_a: f64,
    n_wiggler_magnets: f64,
    distance_from_device_m: f64,
}

impl<'de> Deserialize<'de> for Beamline {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            StorageRingElectronEnergyGev,
            StorageRingCurrentA,
            NWigglerMagnets,
            DistanceFromDeviceM,
        }

        struct BeamlineVisitor;
        impl<'de> Visitor<'de> for BeamlineVisitor {
            type Value = Beamline;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Beamline")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Beamline, V::Error>
            where
                V: MapAccess<'de>,
            {
                use serde::de;
                let mut storage_ring_electron_energy_gev = None;
                let mut storage_ring_current_a = None;
                let mut n_wiggler_magnets = None;
                let mut distance_from_device_m = None;

                while let Some(key) = map.next_key()? {
                    use Field::*;
                    match key {
                        StorageRingElectronEnergyGev => {
                            if storage_ring_electron_energy_gev.is_some() {
                                return Err(de::Error::duplicate_field(
                                    "storage_ring_electron_energy_gev",
                                ));
                            }
                            storage_ring_electron_energy_gev = Some(map.next_value()?);
                        }
                        StorageRingCurrentA => {
                            if storage_ring_current_a.is_some() {
                                return Err(de::Error::duplicate_field("storage_ring_current_a"));
                            }
                            storage_ring_current_a = Some(map.next_value()?);
                        }
                        NWigglerMagnets => {
                            if n_wiggler_magnets.is_some() {
                                return Err(de::Error::duplicate_field("n_wiggler_magnets"));
                            }
                            n_wiggler_magnets = Some(map.next_value()?);
                        }
                        DistanceFromDeviceM => {
                            if distance_from_device_m.is_some() {
                                return Err(de::Error::duplicate_field("distance_from_device_m"));
                            }
                            distance_from_device_m = Some(map.next_value()?);
                        }
                    }
                }

                let storage_ring_electron_energy_gev = storage_ring_electron_energy_gev
                    .ok_or_else(|| de::Error::missing_field("storage_ring_electron_energy_gev"))?;
                let storage_ring_current_a = storage_ring_current_a
                    .ok_or_else(|| de::Error::missing_field("storage_ring_current_a"))?;
                let n_wiggler_magnets = n_wiggler_magnets
                    .ok_or_else(|| de::Error::missing_field("n_wiggler_magnets"))?;
                let distance_from_device_m = distance_from_device_m
                    .ok_or_else(|| de::Error::missing_field("distance_from_device_m"))?;

                Ok(Beamline::new(
                    storage_ring_electron_energy_gev,
                    storage_ring_current_a,
                    n_wiggler_magnets,
                    distance_from_device_m,
                ))
            }
        }

        const FIELDS: &[&str] = &[
            "storage_ring_electron_energy_gev",
            "storage_ring_current_a",
            "n_wiggler_magnets",
            "distance_from_device_m",
        ];
        deserializer.deserialize_struct("Beamline", FIELDS, BeamlineVisitor)
    }
}

impl Beamline {
    /// Compute relativistic lorentz factor of an electron with given energy.
    ///
    /// The lorentz factor is defined as:
    ///     gamma = 1 / sqrt(1 - (u^2) / c^2).
    ///
    /// Alternatively, from kinetic energy:
    ///     gamma = e_kin / (c * m_e).
    ///
    /// * `e_gev`: electron kinetic energy in giga electron volts
    fn electron_lorentz_factor(e_gev: f64) -> f64 {
        e_gev * 1e9 * EV_TO_JOULE / (C_M_S.powi(2) * ELECTRON_MASS_KG)
    }

    /// Compute number of photons at energy e_kev for beamline
    ///
    /// Taken from [here](http://pirate.shu.edu/~sahineme/synchrotron/Synchrotron%20Radiation%20Introduction.pdf)
    /// page 17, section 1.7. We translate photon fluxe per solid angle to area using the opening
    /// angle and circular area approximation. This seems to be good enough for now, but we may
    /// need to change this in the future.
    ///
    /// * `e_kev`: photon energy in kev
    pub fn get_intensity(&self, e_kev: f64) -> f64 {
        use crate::math::acm757::synch_2;
        use std::f64::consts::PI;
        let y = e_kev / self.e_crit_kev;
        let photon_flux_per_solid_angle = 1.327e13
            * self.storage_ring_electron_energy_gev.powi(2)
            * self.storage_ring_current_a
            * synch_2(y).expect("energy is larger than 0"); // TODO: is this ok?

        let opening_angle = 1.0 / self.electron_lorentz_factor;
        let r = self.distance_from_device_m * opening_angle.tan();
        let a = PI * r.powi(2) * 1e6; // Area in mm^2

        photon_flux_per_solid_angle / a * self.n_wiggler_magnets
    }

    fn new(
        storage_ring_electron_energy_gev: f64,
        storage_ring_current_a: f64,
        n_wiggler_magnets: f64,
        distance_from_device_m: f64,
    ) -> Self {
        let electron_lorentz_factor =
            Self::electron_lorentz_factor(storage_ring_electron_energy_gev);
        let e_crit_kev = 3.0 * H_EV_S * C_M_S * electron_lorentz_factor * 1e3;

        Self {
            e_crit_kev,
            electron_lorentz_factor,
            storage_ring_electron_energy_gev,
            storage_ring_current_a,
            n_wiggler_magnets,
            distance_from_device_m,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EDXRDMeta {
    pub vol_fractions: Box<[f64]>,
    pub weight_fractions: Option<Box<[f64]>>,
    pub mean_ds_nm: Box<[f64]>,
    pub eta: f64,
    pub theta_rad: f64,
}

pub struct DiscretizeEnergyDispersive {
    pub common: RenderCommon,
    pub beamline: Beamline,
    pub normalize: bool,
    pub meta: EDXRDMeta,
}

impl Discretizer for DiscretizeEnergyDispersive {
    fn peak_info_iterator(&self) -> impl Iterator<Item = PeakRenderParams> {
        let f_lorentz = lorentz_polarization_factor(self.meta.theta_rad);

        let EDXRDMeta {
            vol_fractions,
            mean_ds_nm,
            eta,
            theta_rad,
            weight_fractions: _,
        } = &self.meta;

        itertools::izip!(0..self.common.n_phases(), vol_fractions, mean_ds_nm,)
            .map(move |(phase_idx, vf, phase_mean_ds_nm)| {
                let flat_idx = self.common.idx(phase_idx);
                self.common.sim_res.all_simulated_peaks[flat_idx]
                    .iter()
                    .map(move |peak: &Peak| {
                        let (e_hkl_kev, peak_weight, fwhm) = peak.get_edxrd_render_params(
                            *theta_rad,
                            f_lorentz,
                            *phase_mean_ds_nm,
                            *vf,
                            &self.beamline,
                        );
                        PeakRenderParams {
                            pos: e_hkl_kev,
                            intensity: peak_weight,
                            fwhm,
                            eta: *eta as f32,
                        }
                    })
            })
            .flatten()
            .chain(self.common.impurity_peaks.iter().map(move |ip| {
                let (e_hkl_kev, _, fwhm) = ip.peak.get_edxrd_render_params(
                    *theta_rad,
                    f_lorentz,
                    ip.mean_ds_nm,
                    1.0,
                    &self.beamline,
                );
                let peak_weight = ip.peak.i_hkl;
                PeakRenderParams {
                    pos: e_hkl_kev,
                    intensity: peak_weight as f32,
                    fwhm,
                    eta: ip.eta as f32,
                }
            }))
    }

    fn n_peaks_tot(&self) -> usize {
        (0..self.common.n_phases())
            .map(|i| self.common.sim_res.all_simulated_peaks[self.common.idx(i)].len())
            .sum::<usize>()
            + self.common.impurity_peaks.len()
    }

    fn bkg(&self) -> &Background {
        &Background::None
    }

    fn noise(&self) -> &Option<Noise> {
        &self.common.noise
    }

    fn normalize(&self) -> bool {
        self.normalize
    }

    fn write_meta_data(&self, data: &mut PatternMeta, pat_id: usize) {
        use PatternMeta::*;
        let n_phases = self.common.indices.len();
        match data {
            VolumeFractions(ref mut dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.vol_fractions[i] as f32;
                }
            }
            Strains(ref mut dst) => {
                for i in 0..n_phases {
                    let flat_idx = self.common.idx(i);
                    let strain = &self.common.sim_res.all_strains[flat_idx];

                    for j in 0..6 {
                        dst[(pat_id, i, j)] = strain.0[j] as f32;
                    }
                }
            }
            Etas(dst) => {
                dst[pat_id] = self.meta.eta as f32;
            }
            MeanDsNm(dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.mean_ds_nm[i] as f32;
                }
            }
            ImpuritySum(dst) => {
                dst[pat_id] = self
                    .common
                    .impurity_peaks
                    .iter()
                    .map(|x| x.peak.i_hkl as f32)
                    .sum();
            }
            ImpurityMax(dst) => {
                dst[pat_id] = self
                    .common
                    .impurity_peaks
                    .iter()
                    .map(|x| x.peak.i_hkl as f32)
                    .max_by(|a, b| a.partial_cmp(&b).expect("no NaNs in peak intensities"))
                    .unwrap_or(0.0);
            }
            CagliotiParams(_) => unreachable!("No Caglioti parameters in EDXRD"),
            SampleDisplacementMuM(_) => unreachable!("No sample displacement in EDXRD"),
            MarchParameter(dst) => {
                for i in 0..n_phases {
                    let flat_idx = self.common.idx(i);
                    let po = &self.common.sim_res.all_preferred_orientations[flat_idx];
                    dst[(pat_id, i)] = po.as_ref().map_or(1.0, |x| x.r) as f32;
                }
            }
            WeightFractions(dst) => {
                let Some(ref wfs) = self.meta.weight_fractions else {
                    panic!("Can only call this if weight fractions were computed before.");
                };
                for i in 0..n_phases {
                    dst[(pat_id, i)] = wfs[i] as f32;
                }
            }
        }
    }

    fn init_meta_data(
        n_patterns: usize,
        n_phases: usize,
        with_weight_fractions: bool,
    ) -> Vec<PatternMeta> {
        use ndarray::{Array1, Array2, Array3};
        use PatternMeta::*;
        let mut v = vec![
            Strains(Array3::<f32>::zeros((n_patterns, n_phases, 6))),
            Etas(Array1::<f32>::zeros(n_patterns)),
            MeanDsNm(Array2::<f32>::zeros((n_patterns, n_phases))),
            VolumeFractions(Array2::<f32>::zeros((n_patterns, n_phases))),
            MarchParameter(Array2::<f32>::zeros((n_patterns, n_phases))),
            ImpuritySum(Array1::<f32>::zeros(n_patterns)),
            ImpurityMax(Array1::<f32>::zeros(n_patterns)),
        ];
        if with_weight_fractions {
            v.push(WeightFractions(Array2::<f32>::zeros((
                n_patterns, n_phases,
            ))))
        }
        v
    }

    fn seed(&self) -> u64 {
        self.common.random_seed
    }
}

pub struct JobGen<T> {
    cfg: EnergyDispersive,
    discretize_info: ToDiscretize,
    sim_params: SimulationParameters,
    vf_generator: VFGenerator,
    energies: Vec<f32>,
    n: usize,
    rng: T,
}

impl<T> JobGen<T> {
    pub fn new(
        cfg: EnergyDispersive,
        discretize_info: ToDiscretize,
        sim_params: SimulationParameters,
        vf_generator: VFGenerator,
        rng: T,
    ) -> Self
    where
        T: Rng,
    {
        let (e0, e1) = cfg.energy_range_kev;
        let energies = (0..cfg.n_steps)
            .map(|x| x as f32 / (cfg.n_steps - 1) as f32 * (e1 - e0) as f32 + e0 as f32)
            .collect_vec();

        Self {
            vf_generator,
            cfg,
            discretize_info,
            sim_params,
            rng,
            energies,
            n: 0,
        }
    }
}

impl<T> DiscretizeJobGenerator for JobGen<T>
where
    T: Rng,
{
    type Item = DiscretizeEnergyDispersive;

    fn next(&mut self) -> Option<Self::Item> {
        if self.n >= self.sim_params.n_patterns {
            return None;
        }

        let job = self.discretize_info.generate_edxrd_job(
            &self.vf_generator,
            &self.cfg,
            &self.sim_params,
            &mut self.rng,
        );

        self.n += 1;

        Some(job)
    }

    fn remaining(&self) -> usize {
        self.sim_params.n_patterns - self.n
    }

    fn xs(&self) -> &[f32] {
        &self.energies
    }

    fn n_phases(&self) -> usize {
        self.discretize_info.structures.len()
    }

    fn abstol(&self) -> f32 {
        self.sim_params.abstol
    }

    fn with_weight_fractions(&self) -> bool {
        self.discretize_info
            .structures
            .iter()
            .all(|s| s.density.is_some())
    }
}
