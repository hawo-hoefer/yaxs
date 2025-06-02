use std::io::BufWriter;
use std::path::Path;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use chrono::Utc;
use clap::Args;
use log::{error, info};
use nalgebra::Vector3;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzWriter;
use serde::Serialize;

use crate::cfg::AngleDisperse;
use crate::pattern::render_jobs;
use crate::pattern::Discretizer;

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
    pub output_name: String,

    #[arg(long, default_value_t = false, help = "Write to compressed numpy .npz")]
    pub compress: bool,
}

#[derive(PartialEq, Clone)]
pub enum PatternMeta {
    Strains(Array3<f32>),
    VolumeFractions(Array2<f32>),
    Etas(Array1<f32>),
    MeanDsNm(Array2<f32>),
    CagliotiParams(Array2<f32>),
    MarchParameter(Array2<f32>), // for now, we're only going to allow one march parameter (and orientation) per phase
}

impl PatternMeta {
    pub fn name(&self) -> &'static str {
        use PatternMeta::*;
        match self {
            VolumeFractions(_) => "volume_fractions",
            Strains(_) => "strains",
            Etas(_) => "eta",
            MeanDsNm(_) => "mean_ds_nm",
            CagliotiParams(_) => "caglioti_params",
            MarchParameter(_) => "march_param_r",
        }
    }
}

// const N_PATTERN_META: usize = 5; // CHANGE THIS IF NUMBER OF FIELDS IN PatternMetaData CHANGES
// pub struct PatternMetaData {
//     pub volume_fractions: Array2<f32>,
//     pub strains: Array3<f32>,
//     pub etas: Array1<f32>,
//     pub mean_ds_nm: Array2<f32>,
//     pub caglioti_params: Array2<f32>,
// }

#[derive(Serialize)]
pub struct Extra {
    pub cfg: AngleDisperse,
    pub max_phases: usize,
    pub encoding: Vec<String>,
    pub preferred_orientation_hkl: Vec<Option<Vector3<f64>>>,
}

#[derive(Serialize)]
pub struct SimulationMetadata<'a> {
    pub timestamp_started: chrono::DateTime<Utc>,
    pub timestamp_finished: chrono::DateTime<Utc>,
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
) -> Result<(Vec<String>, Vec<String>), ()> {
    info!("Writing {path}", path = path.as_ref().display());
    let w =
        BufWriter::new(std::fs::File::create_new(&path).expect("We deleted the directory before"));

    let mut w = if compress {
        NpzWriter::new_compressed(w)
    } else {
        NpzWriter::new(w)
    };

    let mut meta_names = Vec::new();
    for m in meta.iter() {
        use PatternMeta::*;
        meta_names.push(m.name().to_string());
        let succ = match m {
            Etas(array_base) => w.add_array(m.name(), array_base),
            VolumeFractions(array_base) => w.add_array(m.name(), array_base),
            Strains(array_base) => w.add_array(m.name(), array_base),
            MeanDsNm(array_base) => w.add_array(m.name(), array_base),
            CagliotiParams(array_base) => w.add_array(m.name(), array_base),
            MarchParameter(array_base) => w.add_array(m.name(), array_base),
        };
        succ.unwrap_or_else(|err| {
            error!(
                "Could not write data to file '{path}': {err}",
                path = path.as_ref().display()
            );
            std::process::exit(1);
        });
    }

    w.add_array("intensities", &intensities).map_err(|err| {
        error!(
            "Could not write data file '{path}': {err}",
            path = path.as_ref().display()
        )
    })?;
    let data_names = vec!["intensities".to_string()];
    Ok((data_names, meta_names))
}

/// render a chunk of angle dispersive XRD patterns
///
/// # Errors
///
/// This function will return an error if the writing thread has died and cannot be sent to
pub fn render_chunk_and_queue_write_in_thread<D, T>(
    jobs: &[D],
    two_thetas: &[f32],
    path: T,
    send: Sender<Arc<WriteJob<T>>>,
    abstol: f32,
    n_structs: usize,
) -> Result<(), ()>
where
    T: AsRef<Path> + Send + Sync,
    D: Discretizer,
{
    let (intensities, meta) = render_jobs(jobs, two_thetas, abstol, n_structs);
    send.send(Arc::new(WriteJob::Write {
        intensities,
        meta,
        path,
    }))
    .map_err(|err| {
        error!("Could not queue write job: {err}.");
        ()
    })
}

/// prepare an output directory for saving generated XRD patterns
///
/// * `opts`: IO options for writing the data
pub fn prepare_output_directory(opts: &Opts) {
    match std::fs::DirBuilder::new().create(&opts.output_name) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && opts.overwrite => {
            info!(
                "Removing '{out_dir}' according to user input...",
                out_dir = &opts.output_name
            );
            std::fs::remove_dir_all(&opts.output_name).unwrap_or_else(|err| {
                error!(
                    "Could not remove output directory '{out_dir}': {err}",
                    out_dir = &opts.output_name
                );
                std::process::exit(1);
            });
            std::fs::create_dir(&opts.output_name).unwrap_or_else(|err| {
                error!(
                    "Could not (re)create output directory '{out_dir}': {err}",
                    out_dir = opts.output_name
                );
                std::process::exit(1);
            });
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && !opts.overwrite => {
            error!("Could not create output directory '{}': Already exists. Use '--overwrite' to overwrite existing files and directories.", &opts.output_name);
            std::process::exit(1);
        }
        Err(e) => {
            error!(
                "Could not create output directory {out_dir}: {e:?}",
                out_dir = &opts.output_name
            );
            std::process::exit(1);
        }
        Ok(_) => {} // all good,
    }
}

pub struct OutputNames {
    pub chunk_names: Option<Vec<String>>,
    pub data_slot_names: Vec<String>,
    pub metadata_slot_names: Vec<String>,
}

pub fn render_write_chunked<T>(
    jobs: &[T],
    two_thetas: &[f32],
    abstol: f32,
    n_phases: usize,
    io_opts: &crate::io::Opts,
) -> OutputNames
where
    T: Discretizer,
{
    let (tx, rx) = std::sync::mpsc::channel::<Arc<WriteJob<_>>>();
    let compress = io_opts.compress;
    let io_thread_handle = std::thread::spawn(move || {
        let mut names = None;
        loop {
            match rx.recv() {
                Ok(v) => match *v {
                    WriteJob::Done => return names,
                    WriteJob::Write {
                        ref intensities,
                        ref path,
                        ref meta,
                    } => match crate::io::write_to_npz(path, intensities, meta, compress) {
                        Err(()) => {
                            error!("IO thread quitting...");
                            return names;
                        }
                        Ok(n) => names = Some(n),
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
    let l = jobs.len();
    let chunk_size = io_opts.chunk_size.unwrap_or(l);
    let n_chunks = l / chunk_size + (l % chunk_size > 0) as usize;
    let pad_width = if n_chunks > 1 {
        1 + (n_chunks - 1).ilog10()
    } else {
        1
    };

    let mut datafiles = Vec::new();
    while i < l {
        let chunk_file_name = format!(
            "data_{:0width$}.npz",
            i / chunk_size,
            width = pad_width as usize
        );
        let chunk: &[T] = &jobs[i..(i + chunk_size).min(l)];

        let mut chunk_path = std::path::PathBuf::new();
        chunk_path.push(&io_opts.output_name);
        chunk_path.push(&chunk_file_name);
        datafiles.push(chunk_file_name);

        let _ = render_chunk_and_queue_write_in_thread(
            chunk,
            &two_thetas,
            chunk_path,
            tx.clone(),
            abstol,
            n_phases,
        )
        .map_err(|_| {
            // Error is logged in thread
            std::process::exit(1);
        });
        i += chunk_size;
    }
    let _ = tx.send(Arc::new(WriteJob::Done));
    let Some((data_slot_names, metadata_slot_names)) =
        io_thread_handle.join().unwrap_or_else(|err| {
            error!("Could not join io thread: {err:?}");
            std::process::exit(1)
        })
    else {
        error!("Did not write any chunks.");
        std::process::exit(1);
    };
    OutputNames {
        chunk_names: Some(datafiles),
        data_slot_names,
        metadata_slot_names,
    }
}
