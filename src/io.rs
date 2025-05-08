use std::io::BufWriter;
use std::path::Path;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::SystemTime;

use chrono::Utc;
use clap::Args;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzWriter;
use serde::Serialize;

use crate::cfg::Config;
use crate::pattern::{render_jobs, DiscretizationJob};

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
    pub cfg: Config,
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
    eprintln!("Writing {path}", path = path.as_ref().display());
    let w =
        BufWriter::new(std::fs::File::create_new(&path).expect("We deleted the directory before"));

    let mut w = if compress {
        NpzWriter::new_compressed(w)
    } else {
        NpzWriter::new(w)
    };
    w.add_array("intensities", &intensities).map_err(|err| {
        eprintln!(
            "Error writing data file '{path}': {err}",
            path = path.as_ref().display()
        )
    })?;
    w.add_array("strain", &meta.strains).map_err(|err| {
        eprintln!(
            "Error writing data file '{path}': {err}",
            path = path.as_ref().display()
        )
    })?;

    w.add_array("volume_fractions", &meta.volume_fractions)
        .map_err(|err| {
            eprintln!(
                "Error writing data file '{path}': {err}",
                path = path.as_ref().display()
            )
        })?;

    w.add_array("etas", &meta.etas).map_err(|err| {
        eprintln!(
            "Error writing data file '{path}': {err}",
            path = path.as_ref().display()
        )
    })?;

    w.add_array("mean_ds_nm", &meta.mean_ds_nm).map_err(|err| {
        eprintln!(
            "Error writing data file '{path}': {err}",
            path = path.as_ref().display()
        )
    })?;

    w.add_array("caglioti_params", &meta.caglioti_params)
        .map_err(|err| {
            eprintln!(
                "Error writing data file '{path}': {err}",
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

pub fn render_and_queue_write_in_thread<T>(
    jobs: &[DiscretizationJob],
    two_thetas: &[f32],
    path: T,
    send: Sender<Arc<WriteJob<T>>>,
    cfg: &Config,
) -> Result<(), ()>
where
    T: AsRef<Path> + Send + Sync,
{
    let (intensities, meta) = render_jobs(jobs, two_thetas, cfg.abstol, cfg.struct_cifs.len());
    send.send(Arc::new(WriteJob::Write {
        intensities,
        meta,
        path,
    }))
    .map_err(|err| {
        eprintln!("Could not queue write job: {err}.");
        ()
    })
}

pub fn prepare_output_directory(opts: &Opts) {
    // prepare output directory
    match std::fs::DirBuilder::new().create(&opts.output_name) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && opts.overwrite => {
            eprintln!(
                "Removing '{out_dir}' according to user input...",
                out_dir = &opts.output_name
            );
            std::fs::remove_dir_all(&opts.output_name).unwrap_or_else(|err| {
                eprintln!(
                    "Could not remove output directory '{out_dir}': {err}",
                    out_dir = &opts.output_name
                );
                std::process::exit(1);
            });
            std::fs::create_dir(&opts.output_name).unwrap_or_else(|err| {
                eprintln!(
                    "Could not (re)create output directory '{out_dir}': {err}",
                    out_dir = opts.output_name
                );
                std::process::exit(1);
            });
        }
        Err(e) => {
            eprintln!(
                "Error creating output directory {out_dir}: {e:?}",
                out_dir = &opts.output_name
            );
            std::process::exit(1);
        }
        Ok(_) => {} // all good,
    }
}
