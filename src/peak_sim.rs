use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::fmt::Write;
use std::mem::MaybeUninit;
use std::ops::RangeInclusive;
use std::sync::Arc;

use itertools::Itertools;
use log::{debug, error, info};
use ordered_float::NotNan;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::cfg::{
    apply_strain_cfg, CompactSimResults, POGenerator, Parameter, SampleParameters, StrainCfg,
    TextureMeasurement, ToDiscretize,
};

use crate::math::linalg::Vec3;
use crate::pattern::Peaks;
use crate::preferred_orientation::{BinghamParams, KDEBinghamODF};
use crate::scatter::Scatter;
use crate::species::Atom;
use crate::strain::Strain;
use crate::structure::Structure;
use crate::uninit_vec;

enum PossiblyTextureMeasurementPeaks {
    NoTexture(Peaks),
    Texture(Vec<Peaks>),
}

pub enum Alignment<'a> {
    Raw {
        po: &'a KDEBinghamODF,
        phi: f64,
        chi: f64,
    },
    Precomputed {
        po: &'a KDEBinghamODF,
    },
}

impl<'a> Alignment<'a> {
    pub fn weight(&self, pos: &Vec3<f64>) -> NotNan<f64> {
        match self {
            Alignment::Raw { po, phi, chi } => {
                NotNan::new(po.weight(&pos, *chi, *phi)).expect("weight is not nan")
            }
            Alignment::Precomputed { po } => {
                NotNan::new(po.weight_aligned(&pos)).expect("weight is not NaN")
            }
        }
    }
}

struct PeakSimResult {
    strain: Strain,
    po: Option<BinghamParams>,
    random_b_iso: Option<f64>,
    peaks: PossiblyTextureMeasurementPeaks,
    struct_id: usize,
    permutation_id: usize,
}

struct WriteCtx {
    inner: UnsafeCell<WriteInner>,
}

const UNSET_BISO_SENTINEL_VALUE: f64 = -1.0;

struct WriteInner {
    strain: Vec<Strain>,
    b_iso: Option<Vec<f64>>,
    pos: Vec<Option<BinghamParams>>,
    peaks: Vec<MaybeUninit<Peaks>>,
    ok: Vec<bool>,
    n_measurements: usize,
    n_permutations: usize,
}

unsafe impl Sync for WriteCtx {}

impl WriteCtx {
    pub fn new(
        n_structs: usize,
        n_measurements: usize,
        n_permutations: usize,
        random_b_iso: bool,
    ) -> Self {
        let n_peak_sets = n_structs * n_permutations * n_measurements;
        let n_simulations = n_structs * n_permutations;

        Self {
            inner: UnsafeCell::new(WriteInner {
                peaks: unsafe { uninit_vec(n_peak_sets) },
                strain: unsafe { uninit_vec(n_simulations) },
                pos: unsafe { uninit_vec(n_simulations) },
                ok: unsafe { uninit_vec(n_simulations) },
                b_iso: if random_b_iso {
                    Some(unsafe { uninit_vec(n_simulations) })
                } else {
                    None
                },
                n_permutations,
                n_measurements,
            }),
        }
    }

    pub unsafe fn add(&self, p: PeakSimResult) {
        let WriteInner {
            strain,
            pos,
            peaks,
            ok,
            n_permutations,
            n_measurements,
            b_iso,
        } = unsafe { &mut *(self.inner.get()) };

        let sample_idx = p.struct_id * *n_permutations + p.permutation_id;
        match p.peaks {
            PossiblyTextureMeasurementPeaks::NoTexture(res) => {
                assert_eq!(*n_measurements, 1, "n_measurements needs to be 1 if no texture measurement is done. this is likely a bug in yaxs.");

                peaks[sample_idx] = MaybeUninit::new(res);
            }
            PossiblyTextureMeasurementPeaks::Texture(mut texture_measurement_peaks) => {
                assert_eq!(texture_measurement_peaks.len(), *n_measurements, "number of peak sets must match number of simulated peaks. this is likely a bug in yaxs.");
                let sample_idx = p.struct_id * *n_permutations + p.permutation_id;
                // index is [structure_id, permutation_id, texture_measurment_id]
                for (measurement_id, res) in texture_measurement_peaks.drain(..).enumerate() {
                    let idx = sample_idx * *n_measurements + measurement_id;

                    peaks[idx] = MaybeUninit::new(res);
                }
            }
        }

        pos[sample_idx] = p.po;
        strain[sample_idx] = p.strain;
        ok[sample_idx] = true;

        if let Some(b_iso) = b_iso {
            b_iso[sample_idx] = p.random_b_iso.unwrap_or(UNSET_BISO_SENTINEL_VALUE);
        }
    }

    pub fn make_to_discretize(
        self,
        structures: Arc<[Structure]>,
        sample_parameters: SampleParameters,
        texture_measurement: Option<TextureMeasurement>,
    ) -> ToDiscretize {
        let WriteInner {
            strain,
            pos,
            mut peaks,
            ok,
            n_permutations,
            n_measurements: _,
            b_iso,
        } = self.inner.into_inner();

        // TODO: make this return a result instead so we can error gracefully
        if !ok.iter().all(|x| *x) {
            let n_ok: u32 = ok.iter().map(|x| if *x { 1 } else { 0 }).sum();
            let mut err = format!("Error in peak computation. Not all peak simulations ({n_ok}/{n}) terminated successfully. Outputting below. x=err, o=ok\n", n=ok.len());
            for (i, v) in ok.iter().enumerate() {
                let _ = write!(&mut err, "{}", if *v { "o" } else { "x" });
                if i % 16 == 15 {
                    let _ = write!(&mut err, " ");
                }
                if i % 128 == 127 {
                    let _ = write!(&mut err, "\n");
                }
            }
            error!("{}", err);
            std::process::exit(1);
        }

        ToDiscretize {
            structures,
            sample_parameters,
            sim_res: Arc::new(CompactSimResults {
                all_simulated_peaks: peaks
                    .drain(..)
                    .map(|x| unsafe { x.assume_init() })
                    .collect(),
                all_strains: strain.into(),
                all_preferred_orientations: pos.into(),
                n_permutations,
                texture_measurement,
                random_b_isos: b_iso.map(|x| x.into_boxed_slice()),
            }),
        }
    }
}

struct PeakSim {
    structure: usize,
    permutation: usize,
    seed: u64,
    min_r: f64,
    max_r: f64,
    t: Option<TextureMeasurement>,
    randomize_b_iso: Option<Parameter<f64>>,
}

#[derive(Clone)]
struct RunCtx {
    structs: Arc<[Structure]>,
    po_gens: Box<[Option<POGenerator>]>,
    strain_cfgs: Box<[Option<StrainCfg>]>,
    structure_files: Box<[String]>,
}

impl RunCtx {
    fn run(
        &mut self,
        job: PeakSim,
        scattering_parameters: &HashMap<Atom, Scatter>,
    ) -> Result<PeakSimResult, String> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(job.seed);
        let Some((mut perm_s, strain)) = apply_strain_cfg(
            &self.strain_cfgs[job.structure],
            &self.structs[job.structure],
            &mut rng,
        ) else {
            return Err(format!("Could not apply strain to structure '{file}'. Strain matrix is not invertible. Please check the strain configuration.", file=self.structure_files[job.structure]));
        };
        let po = self.po_gens[job.structure]
            .as_mut()
            .map(|x| x.sample(&mut rng));

        let b_iso = perm_s.randomize_b_iso(&job.randomize_b_iso, &mut rng);

        let peaks = if let Some(t) = job.t {
            let hkls_intensities_spacings = perm_s
                .get_hkl_intensities_spacings(job.min_r, job.max_r, scattering_parameters, None)
                .1
                .into_boxed_slice();

            let mut peaks = Vec::new();
            for (_, (chi, phi)) in t
                .chi
                .into_iter()
                .cartesian_product(t.phi.into_iter())
                .enumerate()
            {
                let transformed_po = po.as_ref().map(|x| x.with_orientation(chi, phi));
                let p = perm_s.apply_alignment_to_hkls_intensities(
                    &hkls_intensities_spacings,
                    transformed_po
                        .as_ref()
                        .map(|x| Alignment::Precomputed { po: x }),
                );
                peaks.push(p.into_boxed_slice());
            }
            PossiblyTextureMeasurementPeaks::Texture(peaks)
        } else {
            let peaks = perm_s
                .get_d_spacings_intensities(job.min_r, job.max_r, None, scattering_parameters)
                .into_boxed_slice();
            PossiblyTextureMeasurementPeaks::NoTexture(peaks)
        };

        Ok(PeakSimResult {
            strain,
            po: po.map(|x| x.params),
            peaks,
            struct_id: job.structure,
            permutation_id: job.permutation,
            random_b_iso: b_iso,
        })
    }
}

/// Generate Peaks for the input structures and their physical parameters.
///
/// * `sample_params`: physical parameter ranges for the structures
/// * `structures`: structures to simulate peaks for
/// * `rng`: random number generator to use
///
/// * `sample_params`: the user specified sample parameters
/// * `structures`: the structures to simulate
/// * `structure_po_configs`: preferred orientation configuration for all structures
/// * `structure_strain_configs`: strain configurations for each of the structures
/// * `structure_files`: file paths to the structure's cifs
/// * `rng`: rng to use
pub fn simulate_peaks(
    (min_r, max_r): (f64, f64),
    sample_parameters: SampleParameters,
    structures: Box<[Structure]>,
    structure_po_configs: Box<[Option<POGenerator>]>,
    structure_strain_configs: Box<[Option<StrainCfg>]>,
    structure_files: Box<[String]>,
    b_iso_ranges: Box<[Option<Parameter<f64>>]>,
    texture_measurement: Option<TextureMeasurement>,
    rng: &mut impl Rng,
) -> Result<ToDiscretize, String> {
    let n_structs = structures.len();
    let n_permutations = sample_parameters.structure_permutations;
    let n_texture_measurements = texture_measurement.map(|t| t.stride()).unwrap_or(1);

    let mut n_threads: usize = std::thread::available_parallelism()
        .map(|x| x.into())
        .unwrap_or(1);

    if n_structs * n_permutations < 50 {
        info!("Small number of simulations. Using single-threaded mode.");
        n_threads = 1;
    }

    let mut scattering_parameters = HashMap::new();
    for s in structures.iter() {
        s.gather_scattering_params(&mut scattering_parameters);
    }

    let ctx = RunCtx {
        structs: structures.into(),
        po_gens: structure_po_configs,
        strain_cfgs: structure_strain_configs,
        structure_files,
    };

    let results = WriteCtx::new(
        n_structs,
        n_texture_measurements,
        n_permutations,
        b_iso_ranges.iter().any(|x| x.is_some()),
    );

    cfg_if::cfg_if! {
        if #[cfg(feature = "use-gpu")] {
            if let Some(texture_measurement) = texture_measurement {
                cuda::peak_sim_gpu((min_r, max_r), texture_measurement, scattering_parameters, n_structs, n_permutations, sample_parameters, b_iso_ranges, ctx, results, rng)
            } else {
                peak_sim_cpu((min_r, max_r), texture_measurement, scattering_parameters, b_iso_ranges, n_structs, n_permutations, sample_parameters, ctx, results, n_threads, rng)
            }
        } else {
            peak_sim_cpu(
                (min_r, max_r),
                texture_measurement,
                scattering_parameters,
                n_structs,
                n_permutations,
                sample_parameters,
                ctx,
                results,
                n_threads,
                rng,
            )
        }
    }
}

#[cfg(feature = "use-gpu")]
mod cuda {
    use std::collections::HashMap;
    use std::sync::Arc;

    use ahash::HashMapExt;
    use itertools::Itertools;
    use log::debug;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    use crate::cfg::{
        apply_strain_cfg, KDEApprox, POGenerator, Parameter, SampleParameters, StrainCfg,
        TextureMeasurement, ToDiscretize,
    };
    use crate::cuda_common::CUDA_DEVICE_INFO;
    use crate::math::linalg::Vec3;
    use crate::math::quaternion::Quaternion;
    use crate::peak_sim::PeakSimResult;
    use crate::preferred_orientation::BinghamParams;
    use crate::scatter::Scatter;
    use crate::species::Atom;
    use crate::strain::Strain;

    use crate::peak_sim_cuda::single_phase_weight_hkls;
    use crate::structure::{ReflectionPart, Structure};

    use super::{
        PeakSim, PossiblyTextureMeasurementPeaks, RunCtx, WriteCtx, UNSET_BISO_SENTINEL_VALUE,
    };

    struct CudaBatch {
        reflection_parts: Vec<ReflectionPart>,

        orientation_samples: Vec<Quaternion>,
        bingham_alignments: Vec<Quaternion>,
        chis: Vec<f32>,
        phis: Vec<f32>,

        permuted_structures: Vec<Structure>,
        n_hkls: Vec<usize>,
        strains: Vec<Strain>,
        random_b_isos: Option<Vec<f64>>,
        bingham_params: Vec<BinghamParams>,
        weights: Vec<f32>,

        texture_measurement: TextureMeasurement,

        permutation_start: usize,
        n_hkls_batch: usize,
        struct_id: usize,
        sampling_parameters: KDEApprox,
    }

    impl CudaBatch {
        fn new(
            random_b_isos: bool,
            texture_measurement: TextureMeasurement,
            n_permutations: usize,
            chis: Vec<f32>,
            phis: Vec<f32>,
        ) -> Self {
            Self {
                reflection_parts: Vec::new(),
                orientation_samples: Vec::with_capacity(
                    texture_measurement.stride() * n_permutations / 10,
                ),
                permuted_structures: Vec::with_capacity(n_permutations),
                bingham_alignments: Vec::new(),

                n_hkls: Vec::new(),
                strains: Vec::new(),
                bingham_params: Vec::new(),
                weights: Vec::new(),
                texture_measurement,
                random_b_isos: if random_b_isos {
                    Some(Vec::new())
                } else {
                    None
                },

                chis,
                phis,

                n_hkls_batch: 0,
                permutation_start: 0,
                struct_id: 0,
                sampling_parameters: KDEApprox { n: 0, kappa: 0.0 },
            }
        }

        fn init_struct(&mut self, sampling_parameters: KDEApprox, struct_id: usize) {
            self.reset(0);
            self.struct_id = struct_id;
            self.sampling_parameters = sampling_parameters;
        }

        fn memory_stats(&self) -> (usize, usize) {
            // TODO: if we switch to f64, change
            let n_allocated_bytes_host = std::mem::size_of_val(&*self.permuted_structures)
                + std::mem::size_of_val(&*self.n_hkls)
                + std::mem::size_of_val(&*self.strains)
                + std::mem::size_of_val(&*self.bingham_params)
                + std::mem::size_of_val(&*self.chis)
                + std::mem::size_of_val(&*self.phis)
                + std::mem::size_of_val(&*self.orientation_samples)
                + std::mem::size_of_val(&*self.reflection_parts)
                + std::mem::size_of_val(&*self.weights);

            let n_transformed_quaternions =
                self.chis.len() * self.phis.len() * self.orientation_samples.len();
            let n_weights = self.n_hkls_batch * self.texture_measurement.stride();

            let n_required_bytes_cuda = std::mem::size_of::<Vec3<f32>>() * self.n_hkls_batch
                + std::mem::size_of::<Quaternion>() * n_transformed_quaternions
                + std::mem::size_of_val(&*self.chis)
                + std::mem::size_of_val(&*self.phis)
                // + std::mem::size_of::<f32>() * n_partial_weights
                + std::mem::size_of::<f32>() * n_weights;

            (n_allocated_bytes_host, n_required_bytes_cuda)
        }

        fn reset(&mut self, permutation_id: usize) {
            self.permutation_start = permutation_id;
            self.n_hkls_batch = 0;

            self.reflection_parts.clear();
            self.bingham_alignments.clear();
            self.orientation_samples.clear();
            self.permuted_structures.clear();
            self.n_hkls.clear();
            self.strains.clear();
            self.bingham_params.clear();
            self.weights.clear();
        }

        fn update(
            &'_ mut self,
            min_r: f64,
            max_r: f64,
            structure: &Structure,
            struct_file: &str,
            strain_cfg: &Option<StrainCfg>,
            b_iso_range: &Option<Parameter<f64>>,
            scattering_parameters: &HashMap<Atom, Scatter>,
            po: &mut POGenerator,
            seed: u64,
        ) -> Result<(), String> {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let Some((mut perm_s, strain)) = apply_strain_cfg(strain_cfg, structure, &mut rng)
            else {
                return Err(format!("Could not apply strain to structure '{file}'. Strain matrix is not invertible. Please check the strain configuration.", file=struct_file));
            };

            let b_iso = perm_s.randomize_b_iso(b_iso_range, &mut rng);

            let bingham_params = po.sample_into(&mut rng, &mut self.orientation_samples);

            let mut v = Vec::new();
            std::mem::swap(&mut self.reflection_parts, &mut v);

            let n_hkl;
            (n_hkl, self.reflection_parts) =
                perm_s.get_hkl_intensities_spacings(min_r, max_r, scattering_parameters, Some(v));

            self.n_hkls_batch += n_hkl;
            self.n_hkls.push(n_hkl);
            self.strains.push(strain);
            if let Some(b_isos) = self.random_b_isos.as_mut() {
                b_isos.push(b_iso.unwrap_or(UNSET_BISO_SENTINEL_VALUE));
            }
            self.permuted_structures.push(perm_s);
            // TODO: is there a better way?
            self.bingham_alignments
                .push(bingham_params.orientation.clone());
            self.bingham_params.push(bingham_params);

            Ok(())
        }

        fn computations_left(&self) -> bool {
            self.n_hkls.len() > 0
        }

        fn compute_chunk(&mut self, results: Arc<WriteCtx>, struct_file: &str) {
            let (n_allocated_bytes_host, n_required_bytes_cuda) = self.memory_stats();
            debug!(
            "Computing texture weights for permutations {permutation_start}-{permutation_end} of structure {struct_file}. Requires {mib_cuda:.2} MiB of memory for processing. Current chunk allocates {mib_host:.2} MiB of memory.",
                permutation_start=self.permutation_start,
                permutation_end=self.permutation_start+self.n_hkls.len(),
                mib_cuda = n_required_bytes_cuda as f64 / 1e6,
                mib_host = n_allocated_bytes_host as f64 / 1e6
            );
            single_phase_weight_hkls(
                &self.reflection_parts,
                &self.orientation_samples,
                &self.bingham_alignments,
                &self.phis,
                &self.chis,
                &self.n_hkls,
                self.sampling_parameters.normalization_constant(),
                self.sampling_parameters.kappa,
                self.sampling_parameters.n,
                self.texture_measurement.stride(),
                &mut self.weights,
            );

            let batch_size_permutations = self.n_hkls.len();

            assert!(self.permuted_structures.len() == batch_size_permutations);
            assert!(self.n_hkls.len() == batch_size_permutations);
            assert!(self.strains.len() == batch_size_permutations);
            assert!(self.bingham_params.len() == batch_size_permutations);

            // TODO: multithread this maybe
            let mut map = ahash::HashMap::new();

            let mut hkl_pos = 0;
            for (local_perm_id, (perm_s, n_hkl, strain, bingham_params)) in itertools::izip!(
                self.permuted_structures.drain(..),
                self.n_hkls.drain(..),
                self.strains.drain(..),
                self.bingham_params.drain(..),
            )
            .enumerate()
            {
                let mut peaks = Vec::new();

                // weights is of size chi.steps * phi.steps * n_hkl
                // and indexed by
                // [i_chi, i_phi, n_hkl]
                for _ in 0..self.texture_measurement.stride() {
                    let p = perm_s.apply_precomputed_weights_to_hkls_intensities(
                        &self.reflection_parts[hkl_pos..hkl_pos + n_hkl],
                        &self.weights[hkl_pos..hkl_pos + n_hkl],
                        &mut map,
                    );
                    peaks.push(p.into_boxed_slice());
                }

                let b_iso = self
                    .random_b_isos
                    .as_ref()
                    .map(|b_isos| b_isos[local_perm_id]);

                hkl_pos += n_hkl;
                unsafe {
                    results.add(PeakSimResult {
                        strain,
                        po: Some(bingham_params),
                        peaks: PossiblyTextureMeasurementPeaks::Texture(peaks),
                        struct_id: self.struct_id,
                        permutation_id: local_perm_id + self.permutation_start,
                        random_b_iso: b_iso,
                    })
                }
            }

            self.reset(self.permutation_start + batch_size_permutations);
        }
    }

    pub fn peak_sim_gpu(
        (min_r, max_r): (f64, f64),
        texture_measurement: TextureMeasurement,
        scattering_parameters: HashMap<Atom, Scatter>,
        n_structs: usize,
        n_permutations: usize,
        sample_parameters: SampleParameters,
        random_b_iso_ranges: Box<[Option<Parameter<f64>>]>,
        mut ctx: RunCtx,
        results: WriteCtx,
        rng: &mut impl Rng,
    ) -> Result<ToDiscretize, String> {
        let chis = texture_measurement
            .chi
            .into_iter()
            .map(|x| x as f32)
            .collect_vec();
        let phis = texture_measurement
            .phi
            .into_iter()
            .map(|x| x as f32)
            .collect_vec();
        let mut batch = CudaBatch::new(
            random_b_iso_ranges.iter().any(|x| x.is_some()),
            texture_measurement,
            n_permutations,
            chis,
            phis,
        );

        let results = Arc::new(results);

        for struct_id in 0..n_structs {
            let Some(ref mut po) = ctx.po_gens[struct_id] else {
                for permutation_id in 0..n_permutations {
                    let res = ctx.run(
                        PeakSim {
                            structure: struct_id,
                            permutation: permutation_id,
                            seed: rng.random(),
                            min_r,
                            max_r,
                            randomize_b_iso: random_b_iso_ranges[struct_id].clone(),
                            t: None,
                        },
                        &scattering_parameters,
                    )?;
                    unsafe { results.add(res) };
                }
                continue;
            };

            let sp = po.sampling_parameters();

            batch.init_struct(sp, struct_id);

            for _ in 0..n_permutations {
                let seed = rng.random();
                batch.update(
                    min_r,
                    max_r,
                    &ctx.structs[struct_id],
                    &ctx.structure_files[struct_id],
                    &ctx.strain_cfgs[struct_id],
                    &random_b_iso_ranges[struct_id],
                    &scattering_parameters,
                    po,
                    seed,
                )?;
                let (_, n_required_bytes_cuda) = batch.memory_stats();

                if n_required_bytes_cuda >= CUDA_DEVICE_INFO.init_free_memory_bytes * 9 / 10 {
                    batch.compute_chunk(Arc::clone(&results), &ctx.structure_files[struct_id]);
                }
            }

            // dispatch remaining chunk
            if batch.computations_left() {
                batch.compute_chunk(Arc::clone(&results), &ctx.structure_files[struct_id]);
            }
        }

        let results = Arc::into_inner(results).expect("no more references to write context");
        Ok(results.make_to_discretize(ctx.structs, sample_parameters, Some(texture_measurement)))
    }
}

mod cpu {
    use itertools::Itertools;
    use log::{debug, info};
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::cfg::{Parameter, SampleParameters, TextureMeasurement};
    use crate::peak_sim::PeakSim;
    use crate::scatter::Scatter;
    use crate::species::Atom;

    use super::{RunCtx, WriteCtx};

    enum Task {
        Job(PeakSim),
        Stop,
    }

    pub fn simulate_single_threaded(
        n_structs: usize,
        n_permutations: usize,
        rng: &mut impl Rng,
        (min_r, max_r): (f64, f64),
        texture_measurement: Option<TextureMeasurement>,
        b_iso_ranges: Box<[Option<Parameter<f64>>]>,
        scattering_parameters: HashMap<Atom, Scatter>,
        ctx: &mut RunCtx,
        results: WriteCtx,
    ) -> Result<WriteCtx, String> {
        info!("Running single-threaded peak simulation");

        for (struct_id, permutation_id) in (0..n_structs).cartesian_product(0..n_permutations) {
            let job = PeakSim {
                structure: struct_id,
                permutation: permutation_id,
                seed: rng.random(),
                min_r,
                max_r,
                t: texture_measurement,
                randomize_b_iso: b_iso_ranges[struct_id],
            };
            let p = ctx.run(job, &scattering_parameters)?;
            unsafe { results.add(p) };
        }

        Ok(results)
    }

    pub fn simulate_multi_threaded(
        n_structs: usize,
        n_permutations: usize,
        rng: &mut impl Rng,
        (min_r, max_r): (f64, f64),
        texture_measurement: Option<TextureMeasurement>,
        b_iso_ranges: Box<[Option<Parameter<f64>>]>,
        scattering_parameters: HashMap<Atom, Scatter>,
        n_threads: usize,
        ctx: &mut RunCtx,
        results: WriteCtx,
    ) -> Result<WriteCtx, String> {
        let (job_sender, job_receiver) = crossbeam_channel::unbounded();
        info!("Launching {n_threads} threads for peak simulation");

        let results = Arc::new(results);
        let handles = (0..n_threads)
            .map(|i| {
                let results = Arc::clone(&results);
                let job_receiver = job_receiver.clone();

                let mut ctx = ctx.clone();
                let scattering_parameters = scattering_parameters.clone();

                for po_gen in ctx.po_gens.iter_mut().filter_map(|x| x.as_mut()) {
                    // just to pull n_thinning samples out of the po_gen internal Hit and Run sampler
                    // this is hopefully enough to make the samplers of every thread independent
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng.random());
                    _ = po_gen.sample(&mut rng);
                }

                std::thread::spawn(move || -> Result<(), String> {
                    loop {
                        let job: PeakSim = match job_receiver.recv() {
                            Ok(Task::Stop) => break,
                            Ok(Task::Job(v)) => v,
                            Err(_) => break,
                        };

                        let p = ctx
                            .run(job, &scattering_parameters)
                            .map_err(|err| format!("Peak simulation thread {i}: {err}"))?;
                        unsafe {
                            results.add(p);
                        }
                    }
                    debug!("Peak simulation thread {i} finished.");
                    Ok(())
                })
            })
            .collect_vec();

        for (struct_id, permutation_id) in (0..n_structs).cartesian_product(0..n_permutations) {
            let job = PeakSim {
                structure: struct_id,
                permutation: permutation_id,
                seed: rng.random(),
                min_r,
                max_r,
                t: texture_measurement,
                randomize_b_iso: b_iso_ranges[struct_id],
            };
            let _ = job_sender.send(Task::Job(job));
        }

        debug!("Sending stop signal to peak simulation threads");
        for _ in 0..n_threads {
            job_sender
                .send(Task::Stop)
                .map_err(|err| format!("Could not send stop signal for peak simulation: '{err}'"))?
        }

        for handle in handles {
            handle.join().map_err(|err| {
                format!("Could not join peak simulation thread: '{err:?}'. Exiting...")
            })??;
        }

        debug!("All simulation threads joined");

        Ok(Arc::into_inner(results)
            .expect("all threads are joined, therefore no more shared references alive"))
    }
}

fn peak_sim_cpu(
    (min_r, max_r): (f64, f64),
    texture_measurement: Option<TextureMeasurement>,
    scattering_parameters: HashMap<Atom, Scatter>,
    b_iso_ranges: Box<[Option<Parameter<f64>>]>,
    n_structs: usize,
    n_permutations: usize,
    sample_parameters: SampleParameters,
    mut ctx: RunCtx,
    results: WriteCtx,
    n_threads: usize,
    rng: &mut impl Rng,
) -> Result<ToDiscretize, String> {
    let results = if n_threads == 1 {
        cpu::simulate_single_threaded(
            n_structs,
            n_permutations,
            rng,
            (min_r, max_r),
            texture_measurement,
            b_iso_ranges,
            scattering_parameters,
            &mut ctx,
            results,
        )?
    } else {
        cpu::simulate_multi_threaded(
            n_structs,
            n_permutations,
            rng,
            (min_r, max_r),
            texture_measurement,
            b_iso_ranges,
            scattering_parameters,
            n_threads,
            &mut ctx,
            results,
        )?
    };

    Ok(results.make_to_discretize(ctx.structs, sample_parameters, texture_measurement))
}
