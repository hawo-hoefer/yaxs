use chrono::Utc;
use clap::Parser;
use itertools::Itertools;
use ordered_float::NotNan;
use rand::{Rng, SeedableRng};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime};
use yaxs::cif::CifParser;
use yaxs::preferred_orientation::MarchDollase;
use yaxs::structure::{
    simulate_peaks_angle_disperse, simulate_peaks_energy_disperse, Strain, Structure,
};

use log::{error, info};

use yaxs::cfg::{
    AngleDisperse, Config, EnergyDisperse, JobCfg, SampleParameters, SimulationKind,
    SimulationParameters, StructureDef,
};
use yaxs::io::{
    self, prepare_output_directory, render_write_chunked, write_to_npz, OutputNames,
    SimulationMetadata,
};
use yaxs::pattern::{render_jobs, DiscretizeAngleDisperse, Discretizer, Peaks};

#[derive(Parser)]
#[command(
    version,
    about = "Simulate a dataset of XRD patterns from YAML config.",
    long_about = if cfg!(feature="cpu-only") { 
        "YaXS simulates XRD patterns from CIF
CPU-only build: This may be much slower than GPU with support."
    } else {
        "GPU-accelerated build: CUDA-based peak rendering."
    }
)]
struct Cli {
    #[arg(value_name = "FILE", help = "Configuration yaml file.")]
    cfg: PathBuf,

    #[command(flatten)]
    io: io::Opts,
}

fn generate_edxrd_jobs<'a>(
    energy_disperse: &'a EnergyDisperse,
    sample_params: &'a SampleParameters,
    simulation_parameters: &'a SimulationParameters,
    structures: &'a [Structure],
    all_simulated_peaks: &'a Vec<Vec<Peaks>>,
    all_strains: &'a Vec<Vec<Strain>>,
    all_preferred_orientations: &'a Vec<Vec<Option<MarchDollase>>>,
    rng: &mut impl Rng,
) -> (
    Vec<yaxs::pattern::edxrd::DiscretizeEnergyDispersive<'a>>,
    Vec<f32>,
    JobCfg<'a>,
) {
    let (e0, e1) = energy_disperse.energy_range_kev;
    let energies = (0..energy_disperse.n_steps)
        .map(|x| x as f32 / (energy_disperse.n_steps - 1) as f32 * (e1 - e0) as f32 + e0 as f32)
        .collect_vec();
    let mut intensities = Vec::new();
    intensities.resize(energy_disperse.n_steps, 0.0f32);

    let mut concentration_buf = Vec::with_capacity(sample_params.structures_po.len() + 1);
    concentration_buf.resize(
        concentration_buf.capacity(),
        NotNan::new(0.0).expect("0.0 is not NaN"),
    );

    let job_cfg = JobCfg {
        structures,
        sample_params,
        simulation_parameters,
    };

    // create rendering jobs
    let mut jobs = Vec::with_capacity(job_cfg.simulation_parameters.n_patterns);
    for _ in 0..job_cfg.simulation_parameters.n_patterns {
        let job = job_cfg.generate_edxrd_job(
            all_simulated_peaks,
            all_strains,
            all_preferred_orientations,
            &energy_disperse,
            &mut concentration_buf,
            rng,
        );
        jobs.push(job);
    }

    (jobs, energies, job_cfg)
}

fn render_and_write_jobs<T>(
    cfg: JobCfg,
    args: Cli,
    jobs: &[T],
    xs: &[f32],
    timestamp_started: chrono::DateTime<Utc>,
    kind: SimulationKind,
) where
    T: Discretizer,
{
    let begin_render = Instant::now();
    let output_names = if let Some(_) = args.io.chunk_size {
        render_write_chunked(
            &jobs,
            &xs,
            cfg.simulation_parameters.abstol,
            cfg.sample_params.structures_po.len(),
            &args.io,
        )
    } else {
        let (intensities, pattern_metadata) = render_jobs(
            &jobs,
            &xs,
            cfg.simulation_parameters.abstol,
            cfg.sample_params.structures_po.len(),
        );
        let mut data_path = std::path::PathBuf::new();
        data_path.push(&args.io.output_name);
        data_path.push("data.npz");
        let (data_slot_names, metadata_slot_names) =
            write_to_npz(data_path, &intensities, &pattern_metadata, args.io.compress)
                .unwrap_or_else(|_| std::process::exit(1));
        OutputNames {
            chunk_names: None,
            data_slot_names,
            metadata_slot_names,
        }
    };

    let elapsed = begin_render.elapsed().as_secs_f64();

    let timestamp_finished: chrono::DateTime<Utc> = SystemTime::now().into();

    let meta = serde_json::to_string(&SimulationMetadata {
        timestamp_started,
        timestamp_finished,
        chunked: args.io.chunk_size.is_some(),
        datafiles: output_names.chunk_names,
        input_names: &output_names.data_slot_names,
        target_names: &output_names.metadata_slot_names,
        extra: io::Extra {
            encoding: cfg
                .sample_params
                .structures_po
                .iter()
                .map(|StructureDef { path, .. }| path.to_string())
                .collect_vec(),
            max_phases: cfg.sample_params.structures_po.len(),
            cfg: kind.clone(),
            preferred_orientation_hkl: cfg
                .sample_params
                .structures_po
                .iter()
                .map(|x| x.preferred_orientation.as_ref().map(|po| po.hkl))
                .collect_vec(),
        },
    })
    .expect("SimulationMetadata is serializable");

    let mut path = std::path::PathBuf::new();
    path.push(args.io.output_name);
    path.push("meta.json");
    info!("Writing {}", path.display());
    let f = std::fs::File::create_new(&path).unwrap_or_else(|err| {
        if err.kind() == ErrorKind::AlreadyExists {
            // TODO: time of check / time of use issue?
            error!("Could not write meta.json. Since check at start of simulation, a file was written at '{}'. Printing contents to stderr just to be sure.", path.display());
            error!("{}", meta);
            std::process::exit(1);
        } else {
            // TODO: time of check / time of use issue?
            error!("Could not create meta.json (at '{}'): {err}. Printing contents to stderr just to be sure.", path.display());
            error!("{}", meta);
            std::process::exit(1);
        }
    });
    BufWriter::new(f).write_all(meta.as_bytes()).unwrap_or_else(|err| {
        // TODO: time of check / time of use issue?
        error!("Could not write meta.json (at '{}'): {err}. Printing contents to stderr just to be sure.", path.display());
        error!("{}", meta);
        std::process::exit(1);
    });

    info!("Done rendering patterns. Took {elapsed:.2}s");
}

fn generate_adxrd_jobs<'a>(
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

fn main() {
    colog::init();
    let args = Cli::parse();

    let f = match std::fs::File::open(&args.cfg) {
        Ok(f) => f,
        Err(e) => {
            error!(
                "Could not open File '{}': {}",
                args.cfg.to_str().unwrap(),
                e
            );
            std::process::exit(1);
        }
    };

    let cfg: Config = match serde_yaml::from_reader(BufReader::new(f)) {
        Ok(cfg) => cfg,
        Err(e) => {
            error!(
                "Could not parse config: '{x}': {e}",
                x = args.cfg.to_str().unwrap()
            );
            std::process::exit(1);
        }
    };

    prepare_output_directory(&args.io);

    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.simulation_parameters.seed.unwrap_or(0));
    let timestamp_started: chrono::DateTime<Utc> = SystemTime::now().into();

    let mut structures = Vec::new();
    let mut pref_o = Vec::new();

    for StructureDef {
        path,
        preferred_orientation: po,
    } in cfg.sample_parameters.structures_po.iter()
    {
        let mut reader = BufReader::new(
            std::fs::File::open(path)
                .map_err(|err| {
                    error!("Could not load cif at '{path}': {err}");
                    std::process::exit(1);
                })
                .expect("we exit if error"),
        );
        let mut cif = String::new();
        let _ = reader.read_to_string(&mut cif).unwrap();
        let mut p = CifParser::new(&cif);
        structures.push(Structure::from(&p.parse()));
        pref_o.push(po);
    }

    let begin = Instant::now();
    let (all_simulated_peaks, all_strains, all_preferred_orientations) = match &cfg.kind {
        SimulationKind::AngleDisperse(angle_disperse) => {
            let min_line = &angle_disperse
                .emission_lines
                .iter()
                .min_by(|a, b| {
                    a.wavelength_ams
                        .partial_cmp(&b.wavelength_ams)
                        .expect("no NaNs in wavelengths")
                })
                .expect("at least one emission line");

            let (two_theta_range, wavelength_ams) =
                (angle_disperse.two_theta_range, min_line.wavelength_ams);

            info!("Simulating {two_theta_range:?} {wavelength_ams:.2}");
            simulate_peaks_angle_disperse(
                &cfg.sample_parameters,
                &structures,
                &pref_o,
                two_theta_range,
                wavelength_ams,
                &mut rng,
            )
        }
        SimulationKind::EnergyDisperse(energy_disperse) => simulate_peaks_energy_disperse(
            &cfg.sample_parameters,
            &structures,
            &pref_o,
            energy_disperse.energy_range_kev,
            energy_disperse.theta_deg,
            &mut rng,
        ),
    };

    let elapsed = begin.elapsed().as_secs_f64();

    info!("Simulating Peak Positions took {elapsed:.2}s");

    match &cfg.kind {
        SimulationKind::AngleDisperse(angle_disperse) => {
            let (jobs, xs, job_cfg) = generate_adxrd_jobs(
                &angle_disperse,
                &cfg.sample_parameters,
                &cfg.simulation_parameters,
                &structures,
                &all_simulated_peaks,
                &all_strains,
                &all_preferred_orientations,
                &mut rng,
            );
            render_and_write_jobs(
                job_cfg,
                args,
                &jobs,
                &xs,
                timestamp_started,
                cfg.kind.clone(),
            )
        }
        SimulationKind::EnergyDisperse(energy_disperse) => {
            let (jobs, xs, job_cfg) = generate_edxrd_jobs(
                &energy_disperse,
                &cfg.sample_parameters,
                &cfg.simulation_parameters,
                &structures,
                &all_simulated_peaks,
                &all_strains,
                &all_preferred_orientations,
                &mut rng,
            );
            render_and_write_jobs(
                job_cfg,
                args,
                &jobs,
                &xs,
                timestamp_started,
                cfg.kind.clone(),
            )
        }
    }
}
