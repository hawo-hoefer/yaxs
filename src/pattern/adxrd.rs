use super::{render_peak, Discretizer, PeakRenderParams, Peaks};
use crate::background::Background;
use crate::cfg::{AngleDisperse, JobCfg, SampleParameters, SimulationParameters};
use crate::io::PatternMeta;
use crate::preferred_orientation::MarchDollase;
use crate::structure::{Strain, Structure};
use itertools::Itertools;
use ordered_float::NotNan;
use rand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct ADXRDMeta {
    pub vol_fractions: Box<[f64]>,
    pub mean_ds_nm: Box<[f64]>,
    pub eta: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
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
    // all simulated peaks for all phases in order [structure, structure permutations]
    pub all_simulated_peaks: &'a Vec<Vec<Peaks>>,
    pub all_preferred_orientations: &'a Vec<Vec<Option<MarchDollase>>>,
    pub all_strains: &'a Vec<Vec<Strain>>,
    // indices to select from simulated peaks, length is number of structures
    pub indices: Vec<usize>,
    pub emission_lines: &'a [EmissionLine],
    pub normalize: bool,
    pub meta: ADXRDMeta,
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
            ..
        } = &self.meta;

        itertools::izip!(
            self.all_simulated_peaks,
            self.indices.clone(), // TODO: get rid of this clone
            vol_fractions,
            mean_ds_nm
        )
        .cartesian_product(self.emission_lines)
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
    }

    fn n_peaks_tot(&self) -> usize {
        self.all_simulated_peaks
            .iter()
            .zip(&self.indices)
            .map(|(phase_peaks, idx)| phase_peaks[*idx].len())
            .sum::<usize>()
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
        let n_phases = self.all_simulated_peaks.len();
        match data {
            VolumeFractions(ref mut dst) => {
                for i in 0..n_phases {
                    dst[(pat_id, i)] = self.meta.vol_fractions[i] as f32;
                }
            }
            Strains(ref mut dst) => {
                for i in 0..n_phases {
                    let strain = &self.all_strains[i][self.indices[i]];

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
                    let po = &self.all_preferred_orientations[i][self.indices[i]];
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
        } = &self.meta;
        for (((phase_peaks, idx), vf), phase_mean_ds_nm) in self
            .all_simulated_peaks
            .iter()
            .zip(&self.indices)
            .zip(vol_fractions)
            .zip(mean_ds_nm)
        {
            let peaks = &phase_peaks[*idx];
            // * `pat`: target pattern
            // * `two_thetas`: two theta values of pattern's intensities in degrees
            // * `wavelength`: wavelength of the x-rays in nanometers
            // * `mean_ds`: mean domain size used for scherrer broadening
            // * `u`: caglioti parameter u
            // * `v`: caglioti parameter v
            // * `w`: caglioti parameter w
            for emission_line in self.emission_lines {
                let wavelength_nm = emission_line.wavelength_ams / 10.0;
                for peak in peaks.iter() {
                    let (two_theta_hkl_deg, peak_weight, fwhm) = peak.get_adxrd_render_params(
                        wavelength_nm,
                        *u,
                        *v,
                        *w,
                        *phase_mean_ds_nm,
                        vf * emission_line.weight,
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
pub fn generate_adxrd_jobs<'a>(
    angle_disperse: &'a AngleDisperse,
    sample_params: &'a SampleParameters,
    simulation_parameters: &'a SimulationParameters,
    structures: &'a [Structure],
    all_simulated_peaks: &'a Vec<Vec<Peaks>>,
    all_strains: &'a Vec<Vec<Strain>>,
    all_preferred_orientations: &'a Vec<Vec<Option<MarchDollase>>>,
    rng: &mut impl Rng,
) -> (Vec<DiscretizeAngleDisperse<'a>>, Vec<f32>, JobCfg<'a>) {
    let job_cfg = JobCfg {
        structures,
        sample_params,
        simulation_parameters,
    };
    // Prepare rendering / generation (two_thetas buffer, concentrations)
    let mut two_thetas = Vec::with_capacity(angle_disperse.n_steps);
    two_thetas.resize(two_thetas.capacity(), 0.0f32);
    for (i, t) in two_thetas.iter_mut().enumerate() {
        let r = angle_disperse.two_theta_range;
        *t = (r.0 + (r.1 - r.0) * (i as f64 / (angle_disperse.n_steps as f64 - 1.0))) as f32;
    }

    // initialize concentration buffer for metadata generator
    let mut concentration_buf = Vec::with_capacity(job_cfg.sample_params.structures_po.len() + 1);
    concentration_buf.resize(
        concentration_buf.capacity(),
        NotNan::new(0.0).expect("0.0 is not NaN"),
    );

    // create rendering jobs
    let mut jobs = Vec::with_capacity(job_cfg.simulation_parameters.n_patterns);
    for _ in 0..job_cfg.simulation_parameters.n_patterns {
        let job = job_cfg.generate_adxrd_job(
            all_simulated_peaks,
            all_strains,
            all_preferred_orientations,
            &mut concentration_buf,
            &angle_disperse,
            rng,
        );
        jobs.push(job);
    }
    (jobs, two_thetas, job_cfg)
}
