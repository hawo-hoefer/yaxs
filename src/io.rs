use std::io::BufWriter;
use std::path::Path;
use std::sync::mpsc::Sender;
use std::sync::Arc;

use clap::Args;
use ndarray::Array2;
use ndarray_npy::NpzWriter;

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

pub enum WriteJob<T>
where
    T: AsRef<Path>,
{
    Write { intensities: Array2<f32>, path: T },
    Done,
}

pub fn write_to_npz(
    path: impl AsRef<Path>,
    intensities: &Array2<f32>,
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

    Ok(())
}

pub fn render_jobs_to_npz<T>(
    jobs: &[DiscretizationJob],
    two_thetas: &[f32],
    path: T,
    send: Sender<Arc<WriteJob<T>>>,
    cfg: &Config,
) -> Result<(), ()>
where
    T: AsRef<Path> + Send + Sync,
{
    let intensities = render_jobs(jobs, two_thetas, cfg.abstol);
    send.send(Arc::new(WriteJob::Write { intensities, path }))
        .map_err(|err| {
            eprintln!("Could not queue write job: {err}.");
            ()
        })
}

