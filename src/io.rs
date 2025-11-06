use std::io::BufWriter;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use chrono::Utc;
use clap::Args;
use log::{error, info};
use ndarray::ArrayBase;
use ndarray::Data;
use ndarray::Dimension;
use ndarray::RawData;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzWriter;
use ndarray_npy::WritableElement;
use serde::Serialize;

use crate::cfg::SimulationKind;
use crate::math::linalg::Vec3;
use crate::pattern::render_jobs;
use crate::pattern::DiscretizeJobGenerator;
use crate::pattern::Discretizer;

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
            "structure" => Ok(HKLDisplayMode::Structure {normalized: false}),
            "standard-n" => Ok(HKLDisplayMode::Standard { normalized: true }),
            "structure-n" => Ok(HKLDisplayMode::Structure {normalized: true}),
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
}

#[derive(PartialEq, Clone)]
pub enum PatternMeta {
    Strains(Array3<f32>),
    VolumeFractions(Array2<f32>),
    WeightFractions(Array2<f32>),
    Etas(Array1<f32>),
    MeanDsNm(Array2<f32>),
    CagliotiParams(Array2<f32>),
    ImpuritySum(Array1<f32>),
    ImpurityMax(Array1<f32>),
    SampleDisplacementMuM(Array1<f32>),
    BackgroundParameters(Array2<f32>),
    MarchParameter(Array2<f32>), // for now, we're only going to allow one march parameter (and orientation) per phase
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
            Etas(x) => Self::push_arr(w, x, "etas", meta_names),
            ImpuritySum(x) => Self::push_arr(w, x, "impurity_sum", meta_names),
            ImpurityMax(x) => Self::push_arr(w, x, "impurity_max", meta_names),
            SampleDisplacementMuM(x) => {
                Self::push_arr(w, x, "sample_displacement_mu_m", meta_names)
            }
            MeanDsNm(x) => Self::push_arr(w, x, "mean_ds_nm", meta_names),
            CagliotiParams(x) => Self::push_arr(w, x, "caglioti_params", meta_names),
            MarchParameter(x) => Self::push_arr(w, x, "march_param_r", meta_names),
            BackgroundParameters(x) => Self::push_arr(w, x, "background_parameters", meta_names),
        }
    }
}

#[derive(Serialize)]
pub struct Extra {
    pub cfg: SimulationKind,
    pub max_phases: usize,
    pub encoding: Vec<String>,
    pub preferred_orientation_hkl: Vec<Option<Vec3<f64>>>,
}

#[derive(Serialize)]
pub struct SimulationMetadata<'a> {
    pub timestamp_started: chrono::DateTime<Utc>,
    pub timestamp_finished: chrono::DateTime<Utc>,
    pub yaxs_version: String,
    pub datafiles: Option<Vec<String>>,
    pub chunked: bool,
    pub input_names: &'a [String],
    pub target_names: &'a [String],
    pub extra: Extra,
}

pub enum WriteJob<T>
where
    T: AsRef<Path>,
{
    Write {
        intensities: Array2<f32>,
        meta: Vec<PatternMeta>,
        path: T,
    },
    Done,
}

pub fn write_to_npz(
    path: impl AsRef<Path>,
    intensities: &Array2<f32>,
    meta: &[PatternMeta],
    compress: bool,
    progress: usize,
    total: usize,
) -> Result<(Vec<String>, Vec<String>), String> {
    info!(
        "Writing ({progress}/{total}) path: {path}",
        path = path.as_ref().display()
    );
    let w = BufWriter::new(std::fs::File::create_new(&path).map_err(|err| {
        format!(
            "Could not create file '{p}' for writing: {e}",
            p = path.as_ref().display(),
            e = err.to_string()
        )
    })?);

    let mut w = if compress {
        NpzWriter::new_compressed(w)
    } else {
        NpzWriter::new(w)
    };

    let mut meta_names = Vec::new();
    for m in meta.iter() {
        m.add(&mut w, &mut meta_names)
            .map_err(|err| err.to_string())?;
    }

    w.add_array("intensities", intensities)
        .map_err(|err| err.to_string())?;
    let data_names = vec!["intensities".to_string()];
    Ok((data_names, meta_names))
}

/// render a chunk of angle dispersive XRD patterns
///
/// # Errors
///
/// This function will return an error if the writing thread has died and cannot be sent to
pub fn render_chunk_and_queue_write_in_thread<D, T>(
    jobs: Vec<D>,
    two_thetas: &[f32],
    path: T,
    send: Sender<Arc<WriteJob<T>>>,
    abstol: f32,
    n_structs: usize,
    with_weight_fractions: bool,
) -> Result<(), String>
where
    T: AsRef<Path> + Send + Sync,
    D: Discretizer + Send + Sync + 'static,
{
    let (intensities, meta) =
        render_jobs(jobs, two_thetas, abstol, n_structs, with_weight_fractions)?;
    send.send(Arc::new(WriteJob::Write {
        intensities,
        meta,
        path,
    }))
    .map_err(|err| err.to_string())
}

/// prepare an output directory for saving generated XRD patterns
///
/// * `opts`: IO options for writing the data
pub fn prepare_output_directory(opts: &Opts) -> Result<(), String> {
    match std::fs::DirBuilder::new().create(&opts.output_path) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && opts.overwrite => {
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
            Ok(())
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
        Ok(()) => {Ok(())} // all good,
    }
}

pub struct OutputNames {
    pub chunk_names: Option<Vec<String>>,
    pub data_slot_names: Vec<String>,
    pub metadata_slot_names: Vec<String>,
}

pub fn render_write_chunked<T>(
    mut gen: impl DiscretizeJobGenerator<Item = T>,
    io_opts: &crate::io::Opts,
) -> Result<OutputNames, String>
where
    T: Discretizer + Send + Sync + 'static,
{
    let l = gen.remaining();
    let chunk_size = io_opts.chunk_size.unwrap_or(l);
    let n_chunks = l / chunk_size + (l % chunk_size > 0) as usize;
    info!("Rendering {n_chunks} chunks of {chunk_size} patterns each");
    let pad_width = if n_chunks > 1 {
        1 + (n_chunks - 1).ilog10()
    } else {
        1
    };

    let (tx, rx) = std::sync::mpsc::channel::<Arc<WriteJob<PathBuf>>>();
    let compress = io_opts.compress;
    let io_thread_handle = std::thread::spawn(move || {
        let mut chunks_done = 0;
        let mut names = None;
        loop {
            match rx.recv() {
                Ok(v) => match *v {
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
    });

    let mut i = 0;

    let mut datafiles = Vec::new();
    while i < l {
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

        render_chunk_and_queue_write_in_thread(
            chunk,
            gen.xs(),
            chunk_path,
            tx.clone(),
            gen.abstol(),
            gen.n_phases(),
            gen.with_weight_fractions(),
        )
        .map_err(|err| format!("Could not queue write job: {err}."))?;

        i += actual_chunk_size;
    }

    let _ = tx.send(Arc::new(WriteJob::Done));
    let Some((data_slot_names, metadata_slot_names)) = io_thread_handle
        .join()
        .map_err(|err| format!("Could not join io thread: {err:?}"))?
    else {
        return Err("Did not write any chunks.".to_string());
    };

    Ok(OutputNames {
        chunk_names: Some(datafiles),
        data_slot_names,
        metadata_slot_names,
    })
}
