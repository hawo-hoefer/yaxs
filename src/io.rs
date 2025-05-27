use std::io::BufWriter;
use std::path::Path;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use chrono::Utc;
use clap::Args;
use log::{error, info};
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzWriter;
use serde::Serialize;

use crate::cfg::AngleDisperse;
use crate::pattern::{render_jobs, DiscretizeAngleDisperse};

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

const N_PATTERN_META: usize = 5; // CHANGE THIS IF NUMBER OF FIELDS IN PatternMetaData CHANGES
pub struct PatternMetaData {
    pub volume_fractions: Array2<f32>,
    pub strains: Array3<f32>,
    pub etas: Array1<f32>,
    pub mean_ds_nm: Array2<f32>,
    pub caglioti_params: Array2<f32>,
}

#[derive(Serialize)]
pub struct Extra {
    pub cfg: AngleDisperse,
    pub max_phases: usize,
    pub encoding: Vec<String>,
}

#[derive(Serialize)]
pub struct SimulationMetadata<'a> {
    pub timestamp_started: chrono::DateTime<Utc>,
    pub timestamp_finished: chrono::DateTime<Utc>,
    pub datafiles: Option<Vec<String>>,
    pub chunked: bool,
    pub input_names: &'a [&'static str],
    pub target_names: &'a [&'static str],
    pub extra: Extra,
}

pub enum WriteJob<T>
where
    T: AsRef<Path>,
{
    Write {
        intensities: Array2<f32>,
        meta: PatternMetaData,
        path: T,
    },
    Done,
}

pub fn write_to_npz(
    path: impl AsRef<Path>,
    intensities: &Array2<f32>,
    meta: &PatternMetaData,
    compress: bool,
) -> Result<(), ()> {
    info!("Writing {path}", path = path.as_ref().display());
    let w =
        BufWriter::new(std::fs::File::create_new(&path).expect("We deleted the directory before"));

    let mut w = if compress {
        NpzWriter::new_compressed(w)
    } else {
        NpzWriter::new(w)
    };
    w.add_array("intensities", &intensities).map_err(|err| {
        error!(
            "Could not write data file '{path}': {err}",
            path = path.as_ref().display()
        )
    })?;
    w.add_array("strain", &meta.strains).map_err(|err| {
        error!(
            "Could not write data file '{path}': {err}",
            path = path.as_ref().display()
        )
    })?;

    w.add_array("volume_fractions", &meta.volume_fractions)
        .map_err(|err| {
            error!(
                "Could not write data file '{path}': {err}",
                path = path.as_ref().display()
            )
        })?;

    w.add_array("etas", &meta.etas).map_err(|err| {
        error!(
            "Could not write data file '{path}': {err}",
            path = path.as_ref().display()
        )
    })?;

    w.add_array("mean_ds_nm", &meta.mean_ds_nm).map_err(|err| {
        error!(
            "Could not write data file '{path}': {err}",
            path = path.as_ref().display()
        )
    })?;

    w.add_array("caglioti_params", &meta.caglioti_params)
        .map_err(|err| {
            error!(
                "Could not write data file '{path}': {err}",
                path = path.as_ref().display()
            )
        })?;

    const _: () = assert!(N_PATTERN_META == 5);
    Ok(())
}

pub const TARGET_NAMES: [&'static str; N_PATTERN_META] = [
    "volume_fractions",
    "strains",
    "etas",
    "mean_ds_nm",
    "caglioti_params",
];

pub const INPUT_NAMES: [&'static str; 1] = ["intensities"];

/// render a chunk of angle dispersive XRD patterns
///
/// # Errors
///
/// This function will return an error if the writing thread has died and cannot be sent to
pub fn render_ad_chunk_and_queue_write_in_thread<T>(
    jobs: &[DiscretizeAngleDisperse],
    two_thetas: &[f32],
    path: T,
    send: Sender<Arc<WriteJob<T>>>,
    abstol: f32,
    n_structs: usize,
) -> Result<(), ()>
where
    T: AsRef<Path> + Send + Sync,
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

pub fn render_write_chunked(
    jobs: &[DiscretizeAngleDisperse],
    two_thetas: &[f32],
    abstol: f32,
    n_phases: usize,
    io_opts: &crate::io::Opts,
) -> Vec<String> {
    let (tx, rx) = std::sync::mpsc::channel::<Arc<WriteJob<_>>>();
    let compress = io_opts.compress;
    let io_thread_handle = std::thread::spawn(move || loop {
        match rx.recv() {
            Ok(v) => match *v {
                WriteJob::Done => return,
                WriteJob::Write {
                    ref intensities,
                    ref path,
                    ref meta,
                } => match crate::io::write_to_npz(path, intensities, meta, compress) {
                    Err(()) => {
                        error!("IO thread quitting...");
                        return;
                    }
                    Ok(()) => (),
                },
            },
            Err(err) => {
                error!("IO thread: Could not receive from channel: {err}. Quitting...");
                return;
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
        let chunk: &[DiscretizeAngleDisperse] = &jobs[i..(i + chunk_size).min(l)];

        let mut chunk_path = std::path::PathBuf::new();
        chunk_path.push(&io_opts.output_name);
        chunk_path.push(&chunk_file_name);
        datafiles.push(chunk_file_name);

        let _ = render_ad_chunk_and_queue_write_in_thread(
            &chunk,
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
    io_thread_handle.join().unwrap_or_else(|err| {
        error!("Could not join io thread: {err:?}");
        std::process::exit(1)
    });
    datafiles
}
