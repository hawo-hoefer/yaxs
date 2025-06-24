use std::ops::DerefMut;

use super::{
    render_peak, DiscretizeJobGenerator, Discretizer, PeakRenderParams, RenderCommon, VFGenerator,
};
use crate::background::Background;
use crate::cfg::{AngleDisperse, SimulationParameters, ToDiscretize};
use crate::io::PatternMeta;
use crate::noise::Noise;
use itertools::Itertools;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct ADXRDMeta {
    pub vol_fractions: Box<[f64]>,
    pub mean_ds_nm: Box<[f64]>,
    pub eta: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub sample_displacement_mu_m: f64,
    pub background: Background,
}

#[derive(serde::Deserialize, serde::Serialize, PartialEq, Debug, Clone)]
#[repr(C)]
pub struct EmissionLine {
    // wavelength in amstrong
    pub wavelength_ams: f64,
    // wavelength relative weight
    pub weight: f64,
}

impl EmissionLine {
    /// create a new emission line from wavelength and weight
    ///
    /// * `wavelength`: wavelength in amstrong
    /// * `weight`: intensity of the emission line relative to other emission lines in the spectrum
    pub fn new(wavelength: f64, weight: f64) -> Self {
        Self {
            wavelength_ams: wavelength,
            weight,
        }
    }
}

pub struct DiscretizeAngleDisperse<'a> {
    pub common: RenderCommon<'a>,
    pub emission_lines: Box<[EmissionLine]>,
    pub normalize: bool,
    pub meta: ADXRDMeta,
    pub goniometer_radius_mm: f64,
}

impl<'a> Discretizer for DiscretizeAngleDisperse<'a> {
    fn peak_info_iterator(&self) -> impl Iterator<Item = PeakRenderParams> {
        let ADXRDMeta {
            vol_fractions,
            mean_ds_nm,
            eta,
            u,
            v,
            w,
            sample_displacement_mu_m,
            background: _,
        } = &self.meta;

        itertools::izip!(
            self.common.all_simulated_peaks,
            self.common.indices.clone(), // TODO: get rid of this clone
            vol_fractions,
            mean_ds_nm
        )
        .cartesian_product(&self.emission_lines)
        .map(
            move |((phase_peaks, idx, vf, phase_mean_ds_nm), emission_line)| {
                let wavelength_nm = emission_line.wavelength_ams / 10.0;
                phase_peaks[idx].iter().map(move |peak| {
                    let (two_theta_hkl_deg, peak_weight, fwhm) = peak.get_adxrd_render_params(
                        wavelength_nm,
                        *u,
                        *v,
                        *w,
                        *phase_mean_ds_nm,
                        vf * emission_line.weight,
                        *sample_displacement_mu_m,
                        self.goniometer_radius_mm,
                    );
                    PeakRenderParams {
                        pos: two_theta_hkl_deg,
                        intensity: peak_weight,
                        fwhm,
                        eta: *eta as f32,
                    }
                })
            },
        )
        .flatten()
        .chain(
            self.common
                .impurity_peaks
                .iter()
                .cartesian_product(&self.emission_lines)
                .map(move |(ip, emission_line)| {
                    let wavelength_nm = emission_line.wavelength_ams / 10.0;
                    let (two_theta_hkl_deg, peak_weight, fwhm) = ip.peak.get_adxrd_render_params(
                        wavelength_nm,
                        *u,
                        *v,
                        *w,
                        ip.mean_ds_nm,
                        1.0,
                        0.0,
                        self.goniometer_radius_mm,
                    );
                    PeakRenderParams {
                        pos: two_theta_hkl_deg,
                        intensity: peak_weight,
                        fwhm,
                        eta: ip.eta as f32,
                    }
                }),
        )
    }

    fn n_peaks_tot(&self) -> usize {
        (self
            .common
            .all_simulated_peaks
            .iter()
            .zip(&self.common.indices)
            .map(|(phase_peaks, idx)| phase_peaks[*idx].len())
            .sum::<usize>()
            + self.common.impurity_peaks.len())
            * self.emission_lines.len()
    }

    fn bkg(&self) -> &Background {
        &self.meta.background
    }

    fn normalize(&self) -> bool {
        self.normalize
    }

    fn write_meta_data(&self, data: &mut PatternMeta, pat_id: usize) {
        use PatternMeta::*;
        let n_phases = self.common.all_simulated_peaks.len();
        match data {
            VolumeFractions(ref mut dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.vol_fractions[i] as f32;
                }
            }
            Strains(ref mut dst) => {
                for i in 0..n_phases {
                    let strain = &self.common.all_strains[i][self.common.indices[i]];

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
            CagliotiParams(dst) => {
                dst[(pat_id, 0)] = self.meta.u as f32;
                dst[(pat_id, 1)] = self.meta.v as f32;
                dst[(pat_id, 2)] = self.meta.w as f32;
            }
            MarchParameter(dst) => {
                for i in 0..n_phases {
                    let po = &self.common.all_preferred_orientations[i][self.common.indices[i]];
                    dst[(pat_id, i)] = po.as_ref().map_or(1.0, |x| x.r) as f32;
                }
            }
        }
    }

    fn init_meta_data(n_patterns: usize, n_phases: usize) -> Vec<PatternMeta> {
        use ndarray::{Array1, Array2, Array3};
        use PatternMeta::*;
        vec![
            Strains(Array3::<f32>::zeros((n_patterns, n_phases, 6))),
            Etas(Array1::<f32>::zeros(n_patterns)),
            CagliotiParams(Array2::<f32>::zeros((n_patterns, 3))),
            MeanDsNm(Array2::<f32>::zeros((n_patterns, n_phases))),
            VolumeFractions(Array2::<f32>::zeros((n_patterns, n_phases))),
            MarchParameter(Array2::<f32>::zeros((n_patterns, n_phases))),
        ]
    }

    fn seed(&self) -> u64 {
        self.common.random_seed
    }

    fn noise(&self) -> &Option<Noise> {
        &self.common.noise
    }

    fn discretize_into(&self, intensities: &mut [f32], positions: &[f32], abstol: f32) {
        for PeakRenderParams {
            pos,
            intensity,
            fwhm,
            eta,
        } in self.peak_info_iterator()
        {
            render_peak(pos, intensity, fwhm, eta, abstol, positions, intensities)
        }

        self.bkg().render(intensities, positions);
        if let Some(noise) = self.noise() {
            noise.apply(intensities, self.seed());
        }

        if self.normalize() {
            // TODO: check for NaNs and normalization
            let f = *intensities.first().unwrap();
            let vmin = intensities.iter().fold(f, |a, b| f32::min(a, *b));
            let vmax = intensities.iter().fold(f, |a, b| f32::max(a, *b));
            intensities.iter_mut().for_each(|x| {
                *x = (*x - vmin) / (vmax - vmin);
            });
        }
    }
}

impl<'a> DiscretizeAngleDisperse<'a> {
    pub fn discretize_into(&self, pat: &mut [f32], two_thetas: &[f32], abstol: f32) {
        let ADXRDMeta {
            vol_fractions,
            eta,
            mean_ds_nm,
            u,
            v,
            w,
            background,
            sample_displacement_mu_m,
        } = &self.meta;

        for (phase_peaks, idx, vf, phase_mean_ds_nm) in itertools::izip!(
            self.common.all_simulated_peaks.iter(),
            self.common.indices.iter(),
            vol_fractions,
            mean_ds_nm,
        ) {
            let peaks = &phase_peaks[*idx];
            // * `pat`: target pattern
            // * `two_thetas`: two theta values of pattern's intensities in degrees
            // * `wavelength`: wavelength of the x-rays in nanometers
            // * `mean_ds`: mean domain size used for scherrer broadening
            // * `u`: caglioti parameter u
            // * `v`: caglioti parameter v
            // * `w`: caglioti parameter w
            // $$\Delta 2\theta = 2 \Delta_\text{R} / R \cos\theta$$
            for emission_line in &self.emission_lines {
                let wavelength_nm = emission_line.wavelength_ams / 10.0;
                for peak in peaks.iter() {
                    let (two_theta_hkl_deg, peak_weight, fwhm) = peak.get_adxrd_render_params(
                        wavelength_nm,
                        *u,
                        *v,
                        *w,
                        *phase_mean_ds_nm,
                        vf * emission_line.weight,
                        *sample_displacement_mu_m,
                        self.goniometer_radius_mm,
                    );
                    render_peak(
                        two_theta_hkl_deg,
                        peak_weight,
                        fwhm,
                        *eta as f32,
                        abstol,
                        two_thetas,
                        pat,
                    )
                }
            }
        }
        background.render(pat, two_thetas);

        if self.normalize {
            // TODO: check for NaNs and normalization
            let f = *pat.first().unwrap();
            let vmin = pat.iter().fold(f, |a, b| f32::min(a, *b));
            let vmax = pat.iter().fold(f, |a, b| f32::max(a, *b));
            pat.iter_mut().for_each(|x| {
                *x = (*x - vmin) / (vmax - vmin);
            });
        }
    }
}

pub struct JobGen<'a, T> {
    cfg: AngleDisperse,
    discretize_info: &'a ToDiscretize,
    sim_params: SimulationParameters,
    vf_generator: VFGenerator,
    two_thetas: Vec<f32>,
    n: usize,
    rng: T,
}

impl<'a, T> JobGen<'a, T> {
    pub fn new(
        cfg: AngleDisperse,
        discretize_info: &'a ToDiscretize,
        sim_params: SimulationParameters,
        vf_generator: VFGenerator,
        rng: T,
    ) -> Self {
        let mut two_thetas = Vec::with_capacity(cfg.n_steps);
        two_thetas.resize(two_thetas.capacity(), 0.0f32);
        for (i, t) in two_thetas.iter_mut().enumerate() {
            let r = cfg.two_theta_range;
            *t = (r.0 + (r.1 - r.0) * (i as f64 / (cfg.n_steps as f64 - 1.0))) as f32;
        }

        Self {
            cfg,
            discretize_info,
            sim_params,
            vf_generator,
            two_thetas,
            n: 0,
            rng,
        }
    }
}

impl<'a, T> DiscretizeJobGenerator for JobGen<'a, T>
where
    T: Rng,
{
    fn next(&mut self) -> Option<Self::Item> {
        if self.n >= self.sim_params.n_patterns {
            return None;
        }

        let job = self.discretize_info.generate_adxrd_job(
            &self.vf_generator,
            &self.cfg,
            &self.sim_params,
            &mut self.rng,
        );

        self.n += 1;

        Some(job)
    }

    fn xs(&self) -> &[f32] {
        &self.two_thetas
    }

    fn n_phases(&self) -> usize {
        self.discretize_info.structures.len()
    }

    fn remaining(&self) -> usize {
        self.sim_params.n_patterns - self.n
    }

    type Item = DiscretizeAngleDisperse<'a>;

    fn abstol(&self) -> f32 {
        self.sim_params.abstol
    }
}
