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
    DiscretizeJobGenerator, DiscretizeSample, Discretizer, JobParams, Peak, PeakRenderParams,
    RenderCommon, VFGenerator,
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
    pub ds_eta: Box<[f64]>,
    pub mustrain: Box<[f64]>,
    pub mustrain_eta: Box<[f64]>,
    pub theta_rad: f64,
}

#[derive(Clone)]
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
            ds_eta,
            theta_rad,
            weight_fractions: _,
            mustrain,
            mustrain_eta,
        } = &self.meta;

        itertools::izip!(
            0..self.common.n_phases(),
            vol_fractions,
            mean_ds_nm,
            ds_eta,
            mustrain,
            mustrain_eta
        )
        .map(
            move |(phase_idx, vf, phase_mean_ds_nm, phase_ds_eta, mus_phase, mus_eta_phase)| {
                let flat_idx = self.common.idx(phase_idx);
                self.common.sim_res.all_simulated_peaks[flat_idx]
                    .iter()
                    .map(move |peak: &Peak| {
                        peak.get_edxrd_render_params(
                            *theta_rad,
                            f_lorentz,
                            *phase_mean_ds_nm,
                            *phase_ds_eta,
                            *mus_phase,
                            *mus_eta_phase,
                            *vf,
                            &self.beamline,
                        )
                    })
            },
        )
        .flatten()
        .chain(self.common.impurity_peaks.iter().map(move |ip| {
            ip.peak.get_edxrd_render_params(
                *theta_rad,
                f_lorentz,
                ip.mean_ds_nm,
                ip.eta,
                0.0,
                0.0,
                1.0,
                &self.beamline,
            )
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
            DsEtas(dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.ds_eta[i] as f32;
                }
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
            InstrumentParameters(_) => unreachable!("No Caglioti parameters in EDXRD"),
            SampleDisplacementMuM(_) => unreachable!("No sample displacement in EDXRD"),
            WeightFractions(dst) => {
                let Some(ref wfs) = self.meta.weight_fractions else {
                    panic!("Can only call this if weight fractions were computed before.");
                };
                for i in 0..n_phases {
                    dst[(pat_id, i)] = wfs[i] as f32;
                }
            }
            BackgroundParameters(_) => {
                unreachable!("EDXRD measurements are currently implemented without background")
            }
            Mustrains(dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.mustrain[i] as f32;
                }
            }
            MustrainEtas(dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.mustrain_eta[i] as f32;
                }
            }
            BinghamODFParams { orientations, ks } => {
                for (i, bingham_odf) in (0..n_phases)
                    .filter_map(|i| {
                        let flat_idx = self.common.idx(i);
                        self.common.sim_res.all_preferred_orientations[flat_idx].as_ref()
                    })
                    .enumerate()
                {
                    orientations[(pat_id, i, 0)] = bingham_odf.orientation[0] as f32;
                    orientations[(pat_id, i, 1)] = bingham_odf.orientation[1] as f32;
                    orientations[(pat_id, i, 2)] = bingham_odf.orientation[2] as f32;
                    orientations[(pat_id, i, 3)] = bingham_odf.orientation[3] as f32;

                    let phase_ks = &bingham_odf.ks;

                    ks[(pat_id, i, 0)] = phase_ks[0] as f32;
                    ks[(pat_id, i, 1)] = phase_ks[1] as f32;
                    ks[(pat_id, i, 2)] = phase_ks[2] as f32;
                    ks[(pat_id, i, 3)] = phase_ks[3] as f32;
                }
            }
        }
    }

    fn init_meta_data(n_samples: usize, p: &JobParams) -> Vec<PatternMeta> {
        use ndarray::{Array1, Array2, Array3};
        use PatternMeta::*;
        let mut v = vec![
            Strains(Array3::<f32>::zeros((n_samples, p.n_phases, 6))),
            VolumeFractions(Array2::<f32>::zeros((n_samples, p.n_phases))),
            MeanDsNm(Array2::<f32>::zeros((n_samples, p.n_phases))),
            DsEtas(Array2::<f32>::zeros((n_samples, p.n_phases))),
            Mustrains(Array2::<f32>::zeros((n_samples, p.n_phases))),
            MustrainEtas(Array2::<f32>::zeros((n_samples, p.n_phases))),
            ImpuritySum(Array1::<f32>::zeros(n_samples)),
            ImpurityMax(Array1::<f32>::zeros(n_samples)),
        ];
        if p.has_weight_fracs {
            v.push(WeightFractions(Array2::<f32>::zeros((
                n_samples, p.n_phases,
            ))))
        }
        if let Some(n) = p.textured_phases {
            v.push(BinghamODFParams {
                orientations: Array3::zeros((n_samples, n, 4)),
                ks: Array3::zeros((n_samples, n, 4)),
            })
        }

        if let Some(_) = p.bkg_params {
            unreachable!("Backgrounds are currently not supported in EDXRD");
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

    fn next(&mut self) -> Option<DiscretizeSample<Self::Item>> {
        if self.n >= self.sim_params.n_patterns {
            return None;
        }

        let job = self.discretize_info.generate_edxrd_job(
            &self.vf_generator,
            &self.cfg,
            &self.sim_params,
            &mut self.rng,
        );

        let ret = match self.sim_params.texture_measurement {
            Some(t) => {
                let mut ret = Vec::new();
                for offset in 0..t.stride() {
                    let mut job = job.clone();
                    for idx in job.common.indices.iter_mut() {
                        *idx += offset;
                    }
                    ret.push(job.clone());
                }
                DiscretizeSample::TextureMeasurement(ret)
            }
            None => DiscretizeSample::Standard(job),
        };

        self.n += 1;

        Some(ret)
    }

    fn remaining(&self) -> usize {
        self.sim_params.n_patterns - self.n
    }

    fn xs(&self) -> &[f32] {
        &self.energies
    }

    fn get_job_params(&self) -> JobParams {
        let textured_phases = self.sim_params.texture_measurement.as_ref().map(|_| {
            self.discretize_info
                .sample_parameters
                .structures
                .iter()
                .map(|ref x| x.preferred_orientation.as_ref().map(|_| 1).unwrap_or(0))
                .sum()
        });

        JobParams {
            abstol: self.sim_params.abstol,
            n_phases: self.discretize_info.structures.len(),
            has_weight_fracs: self
                .discretize_info
                .structures
                .iter()
                .all(|s| s.density.is_some()),
            textured_phases,
            texture_measurement: self.sim_params.texture_measurement,
            bkg_params: None,
        }
    }
}
