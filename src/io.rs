use std::io::BufReader;
use std::io::BufWriter;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;

use cfg_if::cfg_if;
use chrono::Utc;
use clap::Args;
use log::warn;
use log::{error, info};
use ndarray::ArrayBase;
use ndarray::Data;
use ndarray::Dimension;
use ndarray::RawData;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::{NpzWriter, WritableElement};
use serde::Deserialize;
use serde::Serialize;

use crate::cfg::SimulationKind;
use crate::cfg::TextureMeasurement;
use crate::pattern::DiscretizeJobGenerator;
use crate::pattern::Discretizer;
use crate::pattern::Intensities;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HKLDisplayMode {
    Standard { normalized: bool },
    Structure { normalized: bool },
}

impl FromStr for HKLDisplayMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" => Ok(HKLDisplayMode::Standard { normalized: false }),
            "structure" => Ok(HKLDisplayMode::Structure { normalized: false }),
            "standard-n" => Ok(HKLDisplayMode::Standard { normalized: true }),
            "structure-n" => Ok(HKLDisplayMode::Structure { normalized: true }),
            _ => Err(format!(
                "Invalid Mode: {s}. Expected ('standard'|'structure')[-n]."
            )),
        }
    }
}

#[derive(Args, Clone)]
pub struct Opts {
    #[arg(long, short, default_value=None, help="Chunk size (in number of patterns) for computation and saving. Set to the entire dataset if not specified.")]
    pub chunk_size: Option<usize>,

    #[arg(long, default_value_t = false, help = "Overwrite existing data.")]
    pub overwrite: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Use with '--overwrite'. Forces Re-Simulation of data even though config hashes match."
    )]
    pub re_simulate: bool,

    // TODO: this should be in the configuration file to integrate bettern with follow-up scripts
    // and make generation of patterns and subsequent training more reproducible
    #[arg(
        long,
        short,
        default_value = "out",
        help = "Name of the output to wite. If --chunk-size is specified, this is the output directory name. If not, '.npz' will be appended as a file name."
    )]
    pub output_path: PathBuf,

    #[arg(long, default_value_t = false, help = "Write to compressed numpy .npz")]
    pub compress: bool,

    #[arg(
        long,
        default_value = None,
        help = "show simulated intensities and hkls for each phase and exit. Either 'standard' or 'structure'. If 'standard-n' or 'structure-n' is specified, intensities will be normalized."
    )]
    pub display_hkls: Option<HKLDisplayMode>,

    #[arg(
        short,
        long,
        help = "don't output ascii art :(",
        default_value_t = false
    )]
    pub quiet: bool,
}

#[derive(PartialEq, Clone)]
pub enum PatternMeta {
    Strains(Array3<f32>),
    VolumeFractions(Array2<f32>),
    WeightFractions(Array2<f32>),
    RandomBIsos(Array2<f32>),
    DsEtas(Array2<f32>),
    Mustrains(Array2<f32>),
    MustrainEtas(Array2<f32>),
    MeanDsNm(Array2<f32>),
    InstrumentParameters(Array2<f32>),
    ImpuritySum(Array1<f32>),
    ImpurityMax(Array1<f32>),
    SampleDisplacementMuM(Array1<f32>),
    BackgroundParameters(Array2<f32>),
    BinghamODFParams {
        // patterns, active phases, quaternion
        orientations: Array3<f32>,
        // patterns, active phases, 4 ks
        ks: Array3<f32>,
    },
}

impl PatternMeta {
    fn push_arr<T, S, D>(
        w: &mut NpzWriter<T>,
        a: &ArrayBase<S, D>,
        name: &'static str,
        meta_names: &mut Vec<String>,
    ) -> Result<(), ndarray_npy::WriteNpzError>
    where
        S: Data,
        <S as RawData>::Elem: WritableElement,
        D: Dimension,
        T: std::io::Seek + std::io::Write,
    {
        meta_names.push(name.to_string());
        w.add_array(name, a)
    }

    pub fn add<T: std::io::Seek + std::io::Write>(
        &self,
        w: &mut NpzWriter<T>,
        meta_names: &mut Vec<String>,
    ) -> Result<(), ndarray_npy::WriteNpzError> {
        use PatternMeta::*;
        match self {
            VolumeFractions(x) => Self::push_arr(w, x, "volume_fractions", meta_names),
            WeightFractions(x) => Self::push_arr(w, x, "weight_fractions", meta_names),
            Strains(x) => Self::push_arr(w, x, "strains", meta_names),
            ImpuritySum(x) => Self::push_arr(w, x, "impurity_sum", meta_names),
            ImpurityMax(x) => Self::push_arr(w, x, "impurity_max", meta_names),
            SampleDisplacementMuM(x) => {
                Self::push_arr(w, x, "sample_displacement_mu_m", meta_names)
            }
            MeanDsNm(x) => Self::push_arr(w, x, "mean_ds_nm", meta_names),
            DsEtas(x) => Self::push_arr(w, x, "ds_etas", meta_names),
            InstrumentParameters(x) => Self::push_arr(w, x, "instrument_parameters", meta_names),
            BackgroundParameters(x) => Self::push_arr(w, x, "background_parameters", meta_names),
            Mustrains(x) => Self::push_arr(w, x, "mustrain", meta_names),
            MustrainEtas(x) => Self::push_arr(w, x, "mustrain_etas", meta_names),
            BinghamODFParams { orientations, ks } => {
                Self::push_arr(w, orientations, "bingham_odf_orientations", meta_names)?;
                Self::push_arr(w, ks, "bingham_odf_ks", meta_names)
            }
            RandomBIsos(x) => Self::push_arr(w, x, "random_b_isos", meta_names),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Extra {
    pub cfg: SimulationKind,
    pub max_phases: usize,
    pub texture: Option<TextureMeasurement>,
    pub encoding: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct SimulationMetadata {
    pub timestamp_started: chrono::DateTime<Utc>,
    pub timestamp_finished: chrono::DateTime<Utc>,
    pub yaxs_version: String,
    pub cfg_hash: String,
    pub datafiles: Option<Vec<String>>,
    pub chunked: bool,
    pub input_names: Vec<String>,
    pub target_names: Vec<String>,
    pub extra: Extra,
}

pub enum WriteJob<T>
where
    T: AsRef<Path>,
{
    Write {
        intensities: Intensities,
        meta: Vec<PatternMeta>,
        path: T,
    },
    Done,
}

pub fn write_to_npz(
    path: impl AsRef<Path>,
    intensities: &Intensities,
    meta: &[PatternMeta],
    compress: bool,
    progress: usize,
    total: usize,
) -> Result<(Vec<String>, Vec<String>), String> {
    info!(
        "(Chunk {progress} / {total}) path: {path}",
        path = path.as_ref().display()
    );
    let w = BufWriter::new(std::fs::File::create_new(&path).map_err(|err| {
        format!(
            "Could not create file '{p}' for writing: {e}",
            p = path.as_ref().display(),
            e = err.to_string()
        )
    })?);

    let mut opts = zip::write::SimpleFileOptions::default().large_file(true);
    if compress {
        opts = opts.compression_method(zip::CompressionMethod::Deflated);
    } else {
        opts = opts.compression_method(zip::CompressionMethod::Stored);
    }
    let mut w = NpzWriter::new_with_options(w, opts);

    let mut meta_names = Vec::new();
    for m in meta.iter() {
        m.add(&mut w, &mut meta_names)
            .map_err(|err| err.to_string())?;
    }

    match intensities {
        Intensities::Standard(intensities) => {
            w.add_array("intensities", intensities)
                .map_err(|err| err.to_string())?;
        }
        Intensities::TextureMeasurement(intensities) => {
            w.add_array("intensities", intensities)
                .map_err(|err| err.to_string())?;
        }
    }
    let data_names = vec!["intensities".to_string()];
    Ok((data_names, meta_names))
}

pub enum CheckedOutput {
    ResimulationNotNeeded,
    ContinueNormally,
}

/// prepare an output directory for saving generated XRD patterns
///
/// * `opts`: IO options for writing the data
pub fn prepare_output_directory(opts: &Opts, cfg_hash: &str) -> Result<CheckedOutput, String> {
    match std::fs::DirBuilder::new().create(&opts.output_path) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && opts.overwrite => {

            let metadata_path = opts.output_path.join("meta.json");
            let skip_sim = std::fs::File::open(&metadata_path).map(BufReader::new)
                .map(serde_json::from_reader::<_, SimulationMetadata>)
                .map(|config| config.map(|x| {
                    (x.cfg_hash == cfg_hash) && (x.yaxs_version == env!("YAXS_VERSION"))
                }).unwrap_or(false))
                .unwrap_or_else(|err|{
                    warn!("Could not open previous simulation's metadata at {p}: {err}
Cannot avoid re-simulation.", p=metadata_path.display());
                    false
                });
            if skip_sim && !opts.re_simulate {
                return Ok(CheckedOutput::ResimulationNotNeeded);
            }


            info!(
                "Removing '{out_dir}' according to user input...",
                out_dir = opts.output_path.display()
            );
            std::fs::remove_dir_all(&opts.output_path).map_err(|err| {
                format!(
                    "Could not remove output directory '{out_dir}': {err}",
                    out_dir = opts.output_path.display()
                )
            })?;
            std::fs::create_dir(&opts.output_path).map_err(|err| {
                format!(
                    "Could not (re)create output directory '{out_dir}': {err}",
                    out_dir = opts.output_path.display()
                )
            })?;
            Ok(CheckedOutput::ContinueNormally)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && !opts.overwrite => {
            Err(format!("Output directory '{}' already exists. Use '--overwrite' to overwrite existing files and directories.", opts.output_path.display()))
        }
        Err(e) => {
            Err(format!(
                "Could not create output directory {out_dir}: {e:?}",
                out_dir = opts.output_path.display()
            ))
        }
        Ok(()) => Ok(CheckedOutput::ContinueNormally) // all good,
    }
}

pub struct OutputNames {
    pub chunk_names: Option<Vec<String>>,
    pub data_slot_names: Vec<String>,
    pub metadata_slot_names: Vec<String>,
}

fn io_thread_fn(
    rx: std::sync::mpsc::Receiver<WriteJob<PathBuf>>,
    compress: bool,
    n_chunks: usize,
) -> Option<(Vec<String>, Vec<String>)> {
    let mut chunks_done = 0;
    let mut names = None;
    loop {
        match rx.recv() {
            Ok(v) => match v {
                WriteJob::Done => return names,
                WriteJob::Write {
                    ref intensities,
                    ref path,
                    ref meta,
                } => match crate::io::write_to_npz(
                    path,
                    intensities,
                    meta,
                    compress,
                    chunks_done + 1,
                    n_chunks,
                ) {
                    Err(err) => {
                        error!(
                            "Could not write data to file '{path}': {err}. IO thread quitting.",
                            path = path.display()
                        );
                        error!("IO thread quitting...");
                        return names;
                    }
                    Ok(n) => {
                        chunks_done += 1;
                        names = Some(n)
                    }
                },
            },
            Err(err) => {
                error!("IO thread: Could not receive from channel: {err}. Quitting...");
                return names;
            }
        }
    }
}

#[cfg(feature = "use-gpu")]
pub mod cuda {
    use std::path::PathBuf;
    use std::sync::Arc;

    use log::info;

    use crate::discretize_cuda::PreparedCudaBatch;
    use crate::io::io_thread_fn;
    use crate::pattern::{
        DiscretizeJobGenerator, DiscretizeSample, Discretizer, Intensities, JobParams,
    };
    use crate::threading::ExecuteSender;

    use super::{OutputNames, PatternMeta, WriteJob};

    type Batch<T> = Arc<Vec<DiscretizeSample<T>>>;

    pub struct CudaRenderCommand {
        batch: PreparedCudaBatch,
        meta: Vec<PatternMeta>,
        params: JobParams,
        file_dst: PathBuf,
        n_peak_sets: usize,
        n_steps: usize,
        n_samples: usize,
        chunk_idx: usize,
        n_chunks: usize,
    }

    pub fn cuda_dispatcher<'a>(
        cmd: Arc<CudaRenderCommand>,
        io_tx: &std::sync::mpsc::Sender<WriteJob<PathBuf>>,
    ) -> Result<(), String> {
        let CudaRenderCommand {
            batch,
            meta,
            params,
            file_dst,
            n_peak_sets,
            n_steps,
            n_samples,
            chunk_idx,
            n_chunks,
        } = Arc::into_inner(cmd).expect("only one reference");

        let intensities = crate::discretize_cuda::render_with_cuda(batch, chunk_idx, n_chunks)
            .map(|i| {
                ndarray::Array2::from_shape_vec((n_peak_sets, n_steps), i)
                    .expect("sizes must match")
            })?;

        let intensities = if let Some(t) = params.texture_measurement {
            Intensities::TextureMeasurement(
                intensities
                    .into_shape_with_order((n_samples, t.chi.steps, t.phi.steps, n_steps))
                    .expect("shapes match"),
            )
        } else {
            Intensities::Standard(intensities)
        };

        io_tx
            .send(WriteJob::Write {
                intensities,
                meta,
                path: file_dst,
            })
            .map_err(|err| {
                format!(
                    "(Chunk {} / {}) Could not queue write job: {}",
                    chunk_idx + 1,
                    n_chunks,
                    err.to_string()
                )
            })
    }

    pub fn cuda_prep_thread<T>(
        jobs: Batch<T>,
        file_dst: PathBuf,
        chunk_idx: usize,
        job_params: JobParams,
        xs: Vec<f32>,
        n_chunks: usize,
        cuda_tx: &ExecuteSender<Arc<CudaRenderCommand>>,
    ) -> Result<(), String>
    where
        T: Discretizer + Send + Sync + 'static,
    {
        let jobs = Arc::<Vec<DiscretizeSample<T>>>::into_inner(jobs)
            .expect("the arc does not exist on the other side");

        let n_samples = jobs.len();
        let n_peak_sets = jobs.iter().map(|x| x.n_patterns()).sum();
        let n_steps = xs.len();

        let mut metadata = T::init_meta_data(n_samples, &job_params);
        info!(
            "(Chunk {} / {n_chunks}) Initialized metadata for {n_samples} sample(s).",
            chunk_idx + 1
        );

        for (i, job) in jobs.iter().enumerate() {
            for m in metadata.iter_mut() {
                let job = match job {
                    DiscretizeSample::Standard(job) => job,
                    DiscretizeSample::TextureMeasurement(items) => items
                        .first()
                        .expect("at least one pattern in texture measurement"),
                };
                job.write_meta_data(m, i)
            }
        }

        // actual rendering of the patterns
        let batch = crate::discretize_cuda::prepare_cuda_discretize(
            jobs,
            xs.to_vec(),
            chunk_idx,
            n_chunks,
        )?;
        cuda_tx
            .queue(Arc::new(CudaRenderCommand {
                batch,
                meta: metadata,
                params: job_params,
                file_dst,
                n_peak_sets,
                n_steps,
                n_samples,
                chunk_idx,
                n_chunks,
            }))
            .map_err(|err| format!("Could not send chunk {chunk_idx} / {n_chunks} to cuda controller thread: {err}"))?;

        Ok(())
    }

    pub fn render_write_chunked<T>(
        mut gen: impl DiscretizeJobGenerator<Item = T>,
        io_opts: &crate::io::Opts,
    ) -> Result<OutputNames, String>
    where
        T: Discretizer + Send + Sync + 'static,
    {
        use crate::threading::ExecuteSender;

        let samples = gen.remaining();
        let chunk_size = io_opts.chunk_size.unwrap_or(samples);
        let n_chunks = samples / chunk_size + (samples % chunk_size > 0) as usize;
        info!("Rendering {n_chunks} chunks of {chunk_size} patterns each");
        let pad_width = if n_chunks > 1 {
            1 + (n_chunks - 1).ilog10()
        } else {
            1
        };
        let compress = io_opts.compress;

        let (io_tx, io_rx) = std::sync::mpsc::channel::<WriteJob<PathBuf>>();
        let io_thread_handle = std::thread::spawn(move || io_thread_fn(io_rx, compress, n_chunks));

        let (cuda_tx, cuda_controller_thread) = {
            let io_tx = io_tx.clone();
            ExecuteSender::create(move |cmd| cuda_dispatcher(cmd, &io_tx))
        };

        let job_params = gen.get_job_params();
        let xs = gen.xs().to_vec();

        let (prep_tx, prep_thread) = {
            let cuda_tx = cuda_tx.clone();
            ExecuteSender::create(
                move |(jobs, file_dst, chunk_idx): (Batch<T>, PathBuf, usize)| {
                    cuda_prep_thread(
                        jobs,
                        file_dst,
                        chunk_idx,
                        job_params.clone(),
                        xs.clone(),
                        n_chunks,
                        &cuda_tx,
                    )
                },
            )
        };

        let mut i = 0;
        let mut datafiles = Vec::new();
        while i < samples {
            let chunk_idx = i / chunk_size;
            let chunk_file_name =
                format!("data_{:0width$}.npz", chunk_idx, width = pad_width as usize);
            let mut chunk = Vec::with_capacity(chunk_size.min(gen.remaining()));
            for _ in 0..chunk_size {
                let Some(job) = gen.next() else {
                    break;
                };

                chunk.push(job);
            }

            let actual_chunk_size = chunk.len();

            let mut chunk_path = std::path::PathBuf::new();
            chunk_path.push(&io_opts.output_path);
            chunk_path.push(&chunk_file_name);
            datafiles.push(chunk_file_name);

            prep_tx
                .queue((Arc::new(chunk), chunk_path.clone(), chunk_idx))
                .map_err(|err| format!("Could not queue chunk in prep thread: {err}"))?;

            i += actual_chunk_size;
        }

        prep_tx
            .finish()
            .map_err(|err| format!("Could not send finish signal to cuda prep thread: {err}"))?;
        prep_thread
            .join()
            .map_err(|err| format!("Could not join cuda prep thread: {err:?}"))?
            .map_err(|err| format!("Error in cuda prep thread: {err}"))?;

        cuda_tx.finish().map_err(|err| {
            format!("Could not send finish signal to cuda controller thread: {err}")
        })?;
        cuda_controller_thread
            .join()
            .map_err(|err| format!("Could not join cuda controller thread: {err:?}"))?
            .map_err(|err| format!("Error in cuda controller thread: {err}"))?;

        io_tx
            .send(WriteJob::Done)
            .map_err(|err| format!("Could not send stop signal to io thread: {err}"))?;

        let Some((data_slot_names, metadata_slot_names)) = io_thread_handle
            .join()
            .map_err(|err| format!("Could not join io thread: {err:?}"))?
        else {
            return Err(format!("Unspecified error in io thread."));
        };

        Ok(OutputNames {
            chunk_names: Some(datafiles),
            data_slot_names,
            metadata_slot_names,
        })
    }
}

#[cfg(not(feature = "use-gpu"))]
mod cpu {
    use std::path::PathBuf;

    use log::info;
    use ndarray::Array2;

    use crate::io::{io_thread_fn, PatternMeta, WriteJob};
    use crate::pattern::{
        DiscretizeJobGenerator, DiscretizeSample, Discretizer, Intensities, JobParams,
    };

    use super::OutputNames;

    pub fn render_write_chunked<T>(
        mut gen: impl DiscretizeJobGenerator<Item = T>,
        io_opts: &crate::io::Opts,
    ) -> Result<OutputNames, String>
    where
        T: Discretizer + Send + Sync + 'static,
    {
        let samples = gen.remaining();
        let chunk_size = io_opts.chunk_size.unwrap_or(samples);
        let n_chunks = samples / chunk_size + (samples % chunk_size > 0) as usize;
        info!("Rendering {n_chunks} chunks of {chunk_size} patterns each");
        let pad_width = if n_chunks > 1 {
            1 + (n_chunks - 1).ilog10()
        } else {
            1
        };
        let compress = io_opts.compress;

        let (io_tx, io_rx) = std::sync::mpsc::channel::<WriteJob<PathBuf>>();
        let io_thread_handle = std::thread::spawn(move || io_thread_fn(io_rx, compress, n_chunks));

        let job_params = gen.get_job_params();
        let xs = gen.xs().to_vec();

        let mut i = 0;
        let mut datafiles = Vec::new();
        while i < samples {
            let chunk_file_name = format!(
                "data_{:0width$}.npz",
                i / chunk_size,
                width = pad_width as usize
            );
            let mut chunk = Vec::with_capacity(chunk_size.min(gen.remaining()));
            for _ in 0..chunk_size {
                let Some(job) = gen.next() else {
                    break;
                };

                chunk.push(job);
            }

            let actual_chunk_size = chunk.len();

            let mut chunk_path = std::path::PathBuf::new();
            chunk_path.push(&io_opts.output_path);
            chunk_path.push(&chunk_file_name);
            datafiles.push(chunk_file_name);

            let (intensities, meta) = render_chunk(chunk, &job_params, &xs);
            io_tx
                .send(WriteJob::Write {
                    intensities,
                    meta,
                    path: chunk_path,
                })
                .map_err(|err| format!("Could not queue chunk for writing: {err:?}"))?;

            i += actual_chunk_size;
        }

        io_tx
            .send(WriteJob::Done)
            .map_err(|err| format!("Could not send stop signal to io thread: {err}"))?;

        let Some((data_slot_names, metadata_slot_names)) = io_thread_handle
            .join()
            .map_err(|err| format!("Could not join io thread: {err:?}"))?
        else {
            return Err(format!("Unspecified error in io thread."));
        };

        Ok(OutputNames {
            chunk_names: Some(datafiles),
            data_slot_names,
            metadata_slot_names,
        })
    }

    fn render_chunk<T>(
        mut jobs: Vec<DiscretizeSample<T>>,
        params: &JobParams,
        xs: &[f32],
    ) -> (Intensities, Vec<PatternMeta>)
    where
        T: Discretizer,
    {
        let n_peak_sets = jobs.iter().map(|x| x.n_patterns()).sum();
        let mut peak_set = 0;

        let n_samples = jobs.len();
        let n_steps = xs.len();

        let mut metadata = T::init_meta_data(n_samples, params);
        info!("Initialized metadata for {n_samples} sample(s).");

        for (i, job) in jobs.iter().enumerate() {
            for m in metadata.iter_mut() {
                let job = match job {
                    DiscretizeSample::Standard(job) => job,
                    DiscretizeSample::TextureMeasurement(items) => items
                        .first()
                        .expect("at least one pattern in texture measurement"),
                };
                job.write_meta_data(m, i)
            }
        }

        let mut intensities = Array2::<f32>::zeros((n_peak_sets, xs.len()));

        for job in jobs.drain(..) {
            // TODO: somehow encode that all samples have the same simulation type
            // in the type system
            match job {
                DiscretizeSample::Standard(job) => {
                    job.discretize_into(
                        intensities.row_mut(peak_set).as_slice_mut().unwrap(),
                        &xs,
                        params.abstol,
                    );
                    peak_set += 1;
                }
                DiscretizeSample::TextureMeasurement(items) => {
                    for job in items.iter() {
                        job.discretize_into(
                            intensities.row_mut(peak_set).as_slice_mut().unwrap(),
                            &xs,
                            params.abstol,
                        );
                        peak_set += 1;
                    }
                }
            }
        }

        let intensities = if let Some(t) = params.texture_measurement {
            Intensities::TextureMeasurement(
                intensities
                    .into_shape_with_order((n_samples, t.chi.steps, t.phi.steps, n_steps))
                    .expect("shapes match"),
            )
        } else {
            Intensities::Standard(intensities)
        };

        (intensities, metadata)
    }
}

pub fn render_write_chunked<T>(
    gen: impl DiscretizeJobGenerator<Item = T>,
    opts: &Opts,
) -> Result<OutputNames, String>
where
    T: Discretizer + Sync + Send + 'static,
{
    cfg_if! {
        if #[cfg(feature = "use-gpu")] {
            cuda::render_write_chunked(gen, opts)
        } else {
            cpu::render_write_chunked(gen, opts)
        }
    }
}
