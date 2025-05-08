use ndarray::Array2;
use ndarray_npy::NpzWriter;
use ordered_float::NotNan;
use rand::SeedableRng;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Instant;
use yaxs::cfg::{Config, MetaGenerator};
use yaxs::discretize_cuda::discretize_peaks_cuda;
use yaxs::pattern::{DiscretizationJob, Peaks};

use clap::Parser;

const H_EV_S: f64 = 4.135_667_696e-15f64;
const C_M_S: f64 = 299_792_485.0f64;

enum WriteJob<T>
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

pub fn render_jobs(jobs: &[DiscretizationJob], two_thetas: &[f32]) -> Array2<f32> {
    let intensities = discretize_peaks_cuda(&jobs, &two_thetas);
    ndarray::Array2::from_shape_vec((jobs.len(), two_thetas.len()), intensities)
        .expect("sizes must match")
}

fn render_jobs_to_npz<T>(
    jobs: &[DiscretizationJob],
    two_thetas: &[f32],
    path: T,
    send: Sender<Arc<WriteJob<T>>>,
) -> Result<(), ()>
where
    T: AsRef<Path> + Send + Sync,
{
    let intensities = render_jobs(jobs, two_thetas);
    send.send(Arc::new(WriteJob::Write { intensities, path }))
        .map_err(|err| {
            eprintln!("Could not queue write job: {err}.");
            ()
        })
}

fn output_exists(path: &str, chunked: bool) -> (bool, String) {
    let path = if chunked {
        path.to_string()
    } else {
        format!("{path}.npz")
    };

    match std::fs::exists(&path) {
        Ok(exists) => (exists, path),
        Err(e) => {
            eprintln!("Could not check whether output file/directory {path} exists: {e}");
            std::process::exit(1)
        }
    }
}

fn process_chunked(args: &Args, jobs: &[DiscretizationJob], two_thetas: &[f32]) {
    let chunk_size = args.chunk_size.unwrap();
    let (tx, rx) = std::sync::mpsc::channel::<Arc<WriteJob<_>>>();
    let compress = args.compress;
    let io_thread_handle = std::thread::spawn(move || loop {
        match rx.recv() {
            Ok(v) => match *v {
                WriteJob::Done => return,
                WriteJob::Write {
                    ref intensities,
                    ref path,
                } => match write_to_npz(path, intensities, compress) {
                    Err(()) => {
                        eprintln!("IO thread quitting...");
                        return;
                    }
                    Ok(()) => (),
                },
            },
            Err(err) => {
                eprintln!("IO thread: Error receiving from channel: {err}. Quitting...");
                return;
            }
        }
    });

    // prepare output directory
    match std::fs::DirBuilder::new().create(&args.output_name) {
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && args.overwrite => {
            eprintln!(
                "Removing '{out_dir}' according to user input...",
                out_dir = &args.output_name
            );
            std::fs::remove_dir_all(&args.output_name).unwrap_or_else(|err| {
                eprintln!(
                    "Could not remove output directory '{out_dir}': {err}",
                    out_dir = &args.output_name
                );
                std::process::exit(1);
            });
            std::fs::create_dir(&args.output_name).unwrap_or_else(|err| {
                eprintln!(
                    "Could not (re)create output directory '{out_dir}': {err}",
                    out_dir = args.output_name
                );
                std::process::exit(1);
            });
        }
        Err(e) => {
            eprintln!(
                "Error creating output directory {out_dir}: {e:?}",
                out_dir = &args.output_name
            );
            std::process::exit(1);
        }
        Ok(_) => {} // all good,
    }

    let mut i = 0;
    let l = jobs.len();
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
        let chunk: &[DiscretizationJob] = &jobs[i..(i + chunk_size).min(l)];

        let mut chunk_path = std::path::PathBuf::new();
        chunk_path.push(&args.output_name);
        chunk_path.push(&chunk_file_name);
        datafiles.push(chunk_file_name);

        let _ = render_jobs_to_npz(&chunk, &two_thetas, chunk_path, tx.clone()).map_err(|_| {
            std::process::exit(1);
        });
        i += chunk_size;
    }
    let _ = tx.send(Arc::new(WriteJob::Done));
    io_thread_handle.join().unwrap_or_else(|err| {
        eprintln!("Error joining io thread: {err:?}");
        std::process::exit(1)
    });
}

pub fn e_kev_to_lambda_ams(e_kev: f64) -> f64 {
    // e = h * c / lambda
    // lambda = h * c / e
    // m      = ev * s * m / ev
    H_EV_S * C_M_S / e_kev * 1e7
}

#[derive(Parser)]
#[command(
    version,
    about = "Simulate a dataset of XRD patterns.",
    long_about = "XRD datasets are simulated for input sets in the cif"
)]
struct Args {
    #[arg(value_name = "FILE", help = "Configuration yaml file.")]
    cfg: PathBuf,

    #[arg(long, short, default_value=None, help="Chunk size (in number of patterns) for computation and saving. Set to the entire dataset if not specified.")]
    chunk_size: Option<usize>,

    #[arg(long, default_value_t = false, help = "Overwrite existing data.")]
    overwrite: bool,

    #[arg(
        long,
        default_value_t = false,
        help = "Don't use GPU for pattern rendering."
    )]
    no_gpu: bool,

    #[arg(long, default_value_t = false, help = "Write to compressed numpy .npz")]
    compress: bool,

    // TODO: this should be in the configuration file to integrate bettern with follow-up scripts
    // and make generation of patterns and subsequent training more reproducible
    #[arg(
        long,
        short,
        default_value = "out",
        help = "Name of the output to wite. If --chunk-size is specified, this is the output directory name. If not, '.npz' will be appended as a file name."
    )]
    output_name: String,
}

fn main() {
    let args = Args::parse();

    let f = match std::fs::File::open(&args.cfg) {
        Ok(f) => f,
        Err(e) => {
            eprintln!(
                "Error: Could not open File '{}': {}",
                args.cfg.to_str().unwrap(),
                e
            );
            std::process::exit(1);
        }
    };

    let (gen, mut rng) = {
        let cfg: Config = match serde_yaml::from_reader(BufReader::new(f)) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!(
                    "Could not parse config: '{x}': {e}",
                    x = args.cfg.to_str().unwrap()
                );
                std::process::exit(1);
            }
        };
        eprintln!("struct_cifs: {:?}", cfg.struct_cifs);
        let rng = rand::rngs::StdRng::seed_from_u64(cfg.seed.unwrap_or(0));
        (MetaGenerator::from(cfg), rng)
    };

    let (output_path_exists, chunk_dependent_output_path) =
        output_exists(&args.output_name, args.chunk_size.is_some());
    if output_path_exists && !args.overwrite {
        eprintln!("Output path '{chunk_dependent_output_path}' already exists.");
        std::process::exit(1);
    }

    let begin = Instant::now();
    let mut two_thetas = Vec::with_capacity(gen.cfg.n_steps);
    two_thetas.resize(two_thetas.capacity(), 0.0f32);
    for (i, t) in two_thetas.iter_mut().enumerate() {
        let r = gen.cfg.two_theta_range;
        *t = (r.0 + (r.1 - r.0) * (i as f64 / (gen.cfg.n_steps as f64 - 1.0))) as f32;
    }

    let min_line = &gen
        .cfg
        .emission_lines
        .iter()
        .min_by(|a, b| {
            a.wavelength_ams
                .partial_cmp(&b.wavelength_ams)
                .expect("no NaNs in wavelengths")
        })
        .expect("at least one emission line");

    let mut all_simulated_peaks = Vec::with_capacity(gen.structures.len());
    for s in &gen.structures {
        let mut permuted_phase_peaks = Vec::with_capacity(gen.cfg.structure_permutations);
        for _ in 0..gen.cfg.structure_permutations {
            let peaks = Peaks {
                peaks: s
                    .permute(gen.cfg.max_strain, &mut rng)
                    .get_pattern(min_line.wavelength_ams, &gen.cfg.two_theta_range)
                    .into(),
                wavelength_nm: min_line.wavelength_ams / 10.0,
            };
            permuted_phase_peaks.push(peaks);
        }
        all_simulated_peaks.push(permuted_phase_peaks);
    }
    let elapsed = begin.elapsed().as_secs_f64();
    eprintln!("Simulating Peak Positions took {elapsed:.2}s");

    let begin = Instant::now();

    let mut concentration_buf = Vec::with_capacity(gen.cfg.struct_cifs.len());
    concentration_buf.resize(
        concentration_buf.capacity(),
        NotNan::new(0.0).expect("0.0 is not NaN"),
    );

    let mut jobs = Vec::with_capacity(gen.cfg.n_patterns);
    for _ in 0..gen.cfg.n_patterns {
        let job = gen.generate_job(&all_simulated_peaks, &mut concentration_buf, &mut rng);
        jobs.push(job);
    }

    if let Some(_) = args.chunk_size {
        process_chunked(&args, &jobs, &two_thetas);
    } else {
        let intensities = render_jobs(&jobs, &two_thetas);
        if output_path_exists {
            std::fs::remove_file(&chunk_dependent_output_path).unwrap_or_else(|e| {
                eprintln!("Error removing output path '{chunk_dependent_output_path}': {e}");
                std::process::exit(1);
            });
        }
        let _ = write_to_npz(chunk_dependent_output_path, &intensities, args.compress)
            .unwrap_or_else(|_| std::process::exit(1));
    }

    let elapsed = begin.elapsed().as_secs_f64();
    eprintln!("Done rendering patterns. Took {elapsed:.2}s");

    // TODO: select rendering engine if no cuda support
    // for (i, mut pattern) in data.outer_iter_mut().enumerate() {
    //     if i % 100 == 0 {
    //         println!("Processing Job {i}");
    //     }
    //     let abstol = gen.cfg.abstol;
    //     let job = gen.generate_job(&all_simulated_peaks);
    //  td   job.discretize_into(pattern.as_slice_mut().unwrap(), &two_thetas, abstol);
    // }
}
