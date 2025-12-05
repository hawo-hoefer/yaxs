use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::Arc;

use itertools::Itertools;
use log::{debug, info};
use ordered_float::NotNan;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::cfg::{
    apply_strain_cfg, CompactSimResults, POGenerator, SampleParameters, StrainCfg,
    TextureMeasurement, ToDiscretize,
};

use crate::math::linalg::Vec3;
use crate::math::quaternion::Quaternion;
use crate::pattern::Peaks;
use crate::preferred_orientation::{BinghamParams, KDEBinghamODF};
use crate::scatter::Scatter;
use crate::species::Atom;
use crate::strain::Strain;
use crate::structure::{ReflectionPart, Structure};
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
    peaks: PossiblyTextureMeasurementPeaks,
    struct_id: usize,
    permutation_id: usize,
}

struct WriteCtx {
    inner: UnsafeCell<WriteInner>,
}

struct WriteInner {
    strain: Vec<Strain>,
    pos: Vec<Option<BinghamParams>>,
    peaks: Vec<MaybeUninit<Peaks>>,
    ok: Vec<bool>,
    n_measurements: usize,
    n_permutations: usize,
}

unsafe impl Sync for WriteCtx {}

impl WriteCtx {
    pub fn new(n_structs: usize, n_measurements: usize, n_permutations: usize) -> Self {
        let n_peak_sets = n_structs * n_permutations * n_measurements;
        let n_simulations = n_structs * n_permutations;

        Self {
            inner: UnsafeCell::new(WriteInner {
                peaks: unsafe { uninit_vec(n_peak_sets) },
                strain: unsafe { uninit_vec(n_simulations) },
                pos: unsafe { uninit_vec(n_simulations) },
                ok: unsafe { uninit_vec(n_simulations) },
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
        } = unsafe { &mut *(self.inner.get()) };

        let sample_idx = p.struct_id * *n_permutations + p.permutation_id;
        match p.peaks {
            PossiblyTextureMeasurementPeaks::NoTexture(res) => {
                assert_eq!(*n_measurements, 1, "n_measurements needs to be 1 if no texture measurement is done. this is likely a bug in yaxs.");

                pos[sample_idx] = p.po;
                strain[sample_idx] = p.strain;
                ok[sample_idx] = true;

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

                pos[sample_idx] = p.po;
                strain[sample_idx] = p.strain.clone();
                ok[sample_idx] = true;
            }
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
        } = self.inner.into_inner();

        // TODO: make this return a result instead so we can error gracefully
        assert!(ok.iter().all(|x| *x));

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
        let Some((perm_s, strain)) = apply_strain_cfg(
            &self.strain_cfgs[job.structure],
            &self.structs[job.structure],
            &mut rng,
        ) else {
            return Err(format!("Could not apply strain to structure '{file}'. Strain matrix is not invertible. Please check the strain configuration.", file=self.structure_files[job.structure]));
        };
        let po = self.po_gens[job.structure]
            .as_mut()
            .map(|x| x.sample(&mut rng));

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
        })
    }
}

enum Task {
    Job(PeakSim),
    Stop,
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

    let results = Arc::new(WriteCtx::new(
        n_structs,
        n_texture_measurements,
        n_permutations,
    ));

    cfg_if::cfg_if! {
        if #[cfg(feature = "use-gpu")] {
            if let Some(texture_measurement) = texture_measurement {
                cuda::peak_sim_gpu((min_r, max_r), texture_measurement, scattering_parameters, n_structs, n_permutations, sample_parameters, ctx, results, n_threads, rng)
            } else {
                peak_sim_cpu((min_r, max_r), texture_measurement, scattering_parameters, n_structs, n_permutations, sample_parameters, ctx, results, n_threads, rng)
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
    use std::cell::UnsafeCell;
    use std::collections::HashMap;
    use std::mem::MaybeUninit;
    use std::sync::Arc;

    use itertools::Itertools;
    use log::{debug, info};
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    use crate::cfg::{apply_strain_cfg, SampleParameters, TextureMeasurement, ToDiscretize};
    use crate::cuda_common::CUDA_DEVICE_INFO;
    use crate::math::quaternion::Quaternion;
    use crate::pattern::Peaks;
    use crate::peak_sim::PeakSimResult;
    use crate::preferred_orientation::BinghamParams;
    use crate::scatter::Scatter;
    use crate::species::Atom;
    use crate::strain::Strain;

    use crate::peak_sim_cuda::single_phase_weight_hkls;
    use crate::structure::{ReflectionPart, Structure};

    use super::{PeakSim, PossiblyTextureMeasurementPeaks, RunCtx, WriteCtx};

    struct TextureCudaCtx {
        inner: UnsafeCell<TextureCudaCtxInner>,
    }

    struct TextureCudaCtxInner {
        strain: Vec<Strain>,
        pos: Vec<Option<BinghamParams>>,
        peaks: Vec<MaybeUninit<Peaks>>,
        ok: Vec<bool>,
        n_measurements: usize,
        n_permutations: usize,
    }

    pub fn peak_sim_gpu(
        (min_r, max_r): (f64, f64),
        texture_measurement: TextureMeasurement,
        scattering_parameters: HashMap<Atom, Scatter>,
        n_structs: usize,
        n_permutations: usize,
        sample_parameters: SampleParameters,
        mut ctx: RunCtx,
        results: Arc<WriteCtx>,
        n_threads: usize,
        rng: &mut impl Rng,
    ) -> Result<ToDiscretize, String> {
        let mut reflection_parts = Vec::new();
        let mut precomputed_alignments = Vec::new();
        let mut permuted_structures = Vec::new();
        let mut n_hkls = Vec::new();
        let mut strains = Vec::new();
        let mut bingham_params = Vec::new();
        let mut weights = Vec::new();

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
                            t: None,
                        },
                        &scattering_parameters,
                    )?;
                    unsafe { results.add(res) };
                }
                continue;
            };

            let sp = po.sampling_parameters();

            let mut permutation_start = 0;
            let mut n_hkls_batch = 0;

            for permutation_id in 0..n_permutations {
                let seed = rng.random();
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                let Some((perm_s, strain)) = apply_strain_cfg(
                    &ctx.strain_cfgs[struct_id],
                    &ctx.structs[struct_id],
                    &mut rng,
                ) else {
                    return Err(format!("Could not apply strain to structure '{file}'. Strain matrix is not invertible. Please check the strain configuration.", file=ctx.structure_files[struct_id]));
                };

                let bingham_samples = po.sample(&mut rng);

                let n_hkl;
                (n_hkl, reflection_parts) = perm_s.get_hkl_intensities_spacings(
                    min_r,
                    max_r,
                    &scattering_parameters,
                    Some(reflection_parts),
                );

                n_hkls_batch += n_hkl;
                n_hkls.push(n_hkl);
                strains.push(strain);
                permuted_structures.push(perm_s);
                bingham_params.push(bingham_samples.params.clone());

                for (_, (chi, phi)) in texture_measurement
                    .chi
                    .into_iter()
                    .cartesian_product(texture_measurement.phi.into_iter())
                    .enumerate()
                {
                    bingham_samples.push_transformed_samples_into(
                        chi,
                        phi,
                        &mut precomputed_alignments,
                    );
                }

                // TODO: if we switch to f64, change
                let n_allocated_bytes_host = std::mem::size_of_val(&*permuted_structures)
                    + std::mem::size_of_val(&*n_hkls)
                    + std::mem::size_of_val(&*strains)
                    + std::mem::size_of_val(&*bingham_params)
                    + std::mem::size_of_val(&*precomputed_alignments)
                    + std::mem::size_of_val(&*reflection_parts)
                    + std::mem::size_of_val(&*weights);

                let n_required_bytes_cuda = 3 * std::mem::size_of::<f32>() * n_hkls_batch
                    + std::mem::size_of::<Quaternion>() * precomputed_alignments.len()
                    + std::mem::size_of::<f32>()
                        * n_hkls_batch
                        * texture_measurement.stride()
                        * sp.n
                    + std::mem::size_of::<f32>() * n_hkls_batch * texture_measurement.stride();

                if n_required_bytes_cuda >= CUDA_DEVICE_INFO.init_free_memory_bytes * 9 / 10 {
                    debug!(
                        "Prepared permutation {permutation_start}-{permutation_id} of structure {s}. Currently allocated {mib:.2}
MiB. Current chunk requires {mib_cuda:.2} MiB of memory for cuda processing.",
                        s = ctx.structure_files[struct_id],
                        mib = n_allocated_bytes_host as f64 / 1e6,
                        mib_cuda = n_required_bytes_cuda as f64 / 1e6
                    );

                    compute_chunk(
                        &mut permuted_structures,
                        &mut n_hkls,
                        &mut strains,
                        &mut bingham_params,
                        &mut precomputed_alignments,
                        &mut reflection_parts,
                        &mut weights,
                        Arc::clone(&results),
                        sp,
                        texture_measurement,
                        permutation_start,
                        permutation_id - permutation_start,
                        struct_id,
                        &ctx.structure_files[struct_id],
                    );

                    permutation_start = permutation_id;
                    n_hkls_batch = 0;
                }
            }

            // dispatch remaining chunk
            let permutations_last_chunk = n_permutations - permutation_start - 1; // aaah off by one error
            if permutations_last_chunk > 0 {
                compute_chunk(
                    &mut permuted_structures,
                    &mut n_hkls,
                    &mut strains,
                    &mut bingham_params,
                    &mut precomputed_alignments,
                    &mut reflection_parts,
                    &mut weights,
                    Arc::clone(&results),
                    sp,
                    texture_measurement,
                    permutation_start,
                    permutations_last_chunk,
                    struct_id,
                    &ctx.structure_files[struct_id],
                );
            }
        }

        let results = Arc::into_inner(results).expect("no more references to write context");
        Ok(results.make_to_discretize(ctx.structs, sample_parameters, Some(texture_measurement)))
    }

    fn compute_chunk(
        permuted_structures: &mut Vec<Structure>,
        n_hkls: &mut Vec<usize>,
        strains: &mut Vec<Strain>,
        bingham_params: &mut Vec<BinghamParams>,
        precomputed_alignments: &mut Vec<Quaternion>,
        reflection_parts: &mut Vec<ReflectionPart>,
        weights: &mut Vec<f32>,

        results: Arc<WriteCtx>,

        sampling_parameters: crate::cfg::KDEApprox,

        texture_measurement: TextureMeasurement,
        permutation_start: usize,
        n_permutations: usize,
        struct_id: usize,
        struct_file: &str,
    ) {
        info!(
        "Computing texture weights for permutations {permutation_start}-{permutation_end} of structure {struct_file}",
            permutation_end=permutation_start+n_permutations,
    );
        single_phase_weight_hkls(
            &reflection_parts,
            &precomputed_alignments,
            &n_hkls,
            sampling_parameters.normalization_constant(),
            sampling_parameters.kappa,
            sampling_parameters.n,
            texture_measurement.chi.steps * texture_measurement.phi.steps,
            weights,
        );

        let mut hkl_pos = 0;
        for local_perm_id in 0..n_permutations {
            let mut peaks = Vec::new();
            let perm_s = permuted_structures[local_perm_id].clone();
            let n_hkl = n_hkls[local_perm_id];
            let strain = strains[local_perm_id].clone();

            // weights is of size chi.steps * phi.steps * n_hkl
            // and indexed by
            // [i_chi, i_phi, n_hkl]
            for _ in 0..texture_measurement.stride() {
                let p = perm_s.apply_precomputed_weights_to_hkls_intensities(
                    &reflection_parts[hkl_pos..hkl_pos + n_hkl],
                    &weights[hkl_pos..hkl_pos + n_hkl],
                );
                peaks.push(p.into_boxed_slice());
            }

            hkl_pos += n_hkl;
            unsafe {
                results.add(PeakSimResult {
                    strain,
                    po: Some(bingham_params[local_perm_id].clone()),
                    peaks: PossiblyTextureMeasurementPeaks::Texture(peaks),
                    struct_id,
                    permutation_id: local_perm_id + permutation_start,
                })
            }
        }

        weights.clear();
        reflection_parts.clear();
        precomputed_alignments.clear();
        permuted_structures.clear();
        n_hkls.clear();
        strains.clear();
        bingham_params.clear();
    }
}

fn peak_sim_cpu(
    (min_r, max_r): (f64, f64),
    texture_measurement: Option<TextureMeasurement>,
    scattering_parameters: HashMap<Atom, Scatter>,
    n_structs: usize,
    n_permutations: usize,
    sample_parameters: SampleParameters,
    mut ctx: RunCtx,
    results: Arc<WriteCtx>,
    n_threads: usize,
    rng: &mut impl Rng,
) -> Result<ToDiscretize, String> {
    if n_threads == 1 {
        info!("Running single-threaded peak simulation");

        for (struct_id, permutation_id) in (0..n_structs).cartesian_product(0..n_permutations) {
            let job = PeakSim {
                structure: struct_id,
                permutation: permutation_id,
                seed: rng.random(),
                min_r,
                max_r,
                t: texture_measurement,
            };
            let p = ctx.run(job, &scattering_parameters)?;
            unsafe { results.add(p) };
        }
    } else {
        let (job_sender, job_receiver) = crossbeam_channel::unbounded();
        info!("Launching {n_threads} threads for peak simulation");

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

        for (struct_id, permutation_id) in
            (0..n_structs).cartesian_product(0..sample_parameters.structure_permutations)
        {
            let job = PeakSim {
                structure: struct_id,
                permutation: permutation_id,
                seed: rng.random(),
                min_r,
                max_r,
                t: texture_measurement,
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

        debug!("All simulation threads joined")
    }

    let results = Arc::into_inner(results).expect("no more references to write context");
    Ok(results.make_to_discretize(ctx.structs, sample_parameters, texture_measurement))
}
