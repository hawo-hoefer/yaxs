use chrono::Utc;
use clap::Parser;
use itertools::Itertools;
use ordered_float::NotNan;
use rand::SeedableRng;
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime};
use yaxs::cif::CifParser;
use yaxs::math::{e_kev_to_lambda_ams, pseudo_voigt, scherrer_broadening, C_M_S, H_EV_S};
use yaxs::structure::{simulate_peaks, Strain, Structure};

use yaxs::cfg::{
    AngleDisperse, Config, EnergyDisperse, MetaGenerator, SampleParameters, SimulationKind,
    SimulationParameters,
};
use yaxs::io::{self, prepare_output_directory, write_to_npz, SimulationMetadata};
use yaxs::pattern::{render_jobs, render_write_chunked, Peaks};

fn output_exists(path: &str) -> bool {
    std::fs::exists(&path).unwrap_or_else(|err| {
        eprintln!("Could not check whether output file/directory {path} exists: {err}");
        std::process::exit(1);
    })
}

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

fn render_energy_disperse(
    kind: EnergyDisperse,
    sample_params: SampleParameters,
    simulation_parameters: SimulationParameters,
    args: Cli,
    all_simulated_peaks: &Vec<Vec<Peaks>>,
    all_strains: &Vec<Vec<Strain>>,
    timestamp_started: chrono::DateTime<Utc>,
    rng: &mut rand::rngs::StdRng,
) {
    let begin_render = Instant::now();

    let (e0, e1) = kind.energy_range_kev;
    let energies = (0..kind.n_steps)
        .map(|x| x as f32 / (kind.n_steps - 1) as f32 * (e1 - e0) as f32 + e0 as f32)
        .collect_vec();
    let mut intensities = Vec::new();
    intensities.resize(kind.n_steps, 0.0f32);
    let vfs = [1.0, 1.0];
    // for (structure, _vf) in all_simulated_peaks.iter().zip(vfs) {
    //     for i in 0..kind.n_steps {
    //         for peak in structure[0].peaks.iter() {
    //             // E = H C / Lambda
    //             // lambda = H C / E
    //             let wavelength_ams = e_kev_to_lambda_ams(energies[i] as f64);
    //             let pc = peak.convert(structure[0].wavelength_nm, wavelength_ams / 10.0);

    //             let dx = kind.theta_deg * 2.0 - pc.pos;
    //             let mean_ds_nm = 50.0;
    //             let fwhm = scherrer_broadening(
    //                 wavelength_ams / 10.0,
    //                 pc.pos.to_radians() / 2.0,
    //                 mean_ds_nm,
    //             );
    //             let pv = pseudo_voigt(dx as f32, 0.5f32, fwhm as f32);
    //             // eprintln!("{:.2} {:.2} {:.4} | {}", pc.pos, dx, pv, fwhm);
    //             intensities[i] += pv * peak.intensity as f32;
    //         }
    //     }
    // }

    for (structure, vf) in all_simulated_peaks.iter().zip(vfs) {
        for peak in structure[0].peaks.iter() {
            eprintln!("{:?}", peak);
            peak.render(
                &mut intensities,
                &energies,
                1.0,
                vf,
                100.0,
                0.5,
                0.0,
                0.0,
                0.0,
                0.00001,
            )
        }
    }
    for (e, i) in energies.iter().zip(intensities) {
        println!("{} {}", e, i);
    }
}

fn render_angle_disperse(
    angle_disperse: AngleDisperse,
    sample_params: SampleParameters,
    simulation_parameters: SimulationParameters,
    args: Cli,
    structures: Box<[Structure]>,
    all_simulated_peaks: &Vec<Vec<Peaks>>,
    all_strains: &Vec<Vec<Strain>>,
    timestamp_started: chrono::DateTime<Utc>,
    rng: &mut rand::rngs::StdRng,
) {
    let begin_render = Instant::now();
    let gen = MetaGenerator {
        angle_disperse,
        structures,
        sample_params,
        simulation_parameters,
    };
    // Prepare rendering / generation (two_thetas buffer, concentrations)
    let mut two_thetas = Vec::with_capacity(gen.angle_disperse.n_steps);
    two_thetas.resize(two_thetas.capacity(), 0.0f32);
    for (i, t) in two_thetas.iter_mut().enumerate() {
        let r = gen.angle_disperse.two_theta_range;
        *t = (r.0 + (r.1 - r.0) * (i as f64 / (gen.angle_disperse.n_steps as f64 - 1.0))) as f32;
    }

    let mut concentration_buf = Vec::with_capacity(gen.sample_params.struct_cifs.len());
    concentration_buf.resize(
        concentration_buf.capacity(),
        NotNan::new(0.0).expect("0.0 is not NaN"),
    );

    // create rendering jobs
    let mut jobs = Vec::with_capacity(gen.simulation_parameters.n_patterns);
    for _ in 0..gen.simulation_parameters.n_patterns {
        let job = gen.generate_job(
            all_simulated_peaks,
            all_strains,
            &mut concentration_buf,
            rng,
        );
        jobs.push(job);
    }

    // simulate and write to file(s)
    let datafiles = if let Some(_) = args.io.chunk_size {
        Some(render_write_chunked(
            &jobs,
            &two_thetas,
            gen.simulation_parameters.abstol,
            gen.sample_params.struct_cifs.len(),
            &args.io,
        ))
    } else {
        let (intensities, pattern_metadata) = render_jobs(
            &jobs,
            &two_thetas,
            gen.simulation_parameters.abstol,
            gen.sample_params.struct_cifs.len(),
        );
        let mut data_path = std::path::PathBuf::new();
        data_path.push(&args.io.output_name);
        data_path.push("data.npz");
        let _ = write_to_npz(data_path, &intensities, &pattern_metadata, args.io.compress)
            .unwrap_or_else(|_| std::process::exit(1));
        None
    };

    let elapsed = begin_render.elapsed().as_secs_f64();

    let timestamp_finished: chrono::DateTime<Utc> = SystemTime::now().into();

    let meta = serde_json::to_string(&SimulationMetadata {
        timestamp_started,
        timestamp_finished,
        chunked: datafiles.is_some(),
        datafiles,
        input_names: &io::INPUT_NAMES,
        target_names: &io::TARGET_NAMES,
        extra: io::Extra {
            encoding: gen.sample_params.struct_cifs.clone().to_vec(),
            max_phases: gen.sample_params.struct_cifs.len(),
            cfg: gen.angle_disperse,
        },
    })
    .expect("SimulationMetadata is serializable");

    let mut path = std::path::PathBuf::new();
    path.push(args.io.output_name);
    path.push("meta.json");
    eprintln!("Writing {}", path.display());
    let f = std::fs::File::create_new(&path).unwrap_or_else(|err| {
        if err.kind() == ErrorKind::AlreadyExists {
            // TODO: time of check / time of use issue?
            eprintln!("Could not write meta.json. Since check at start of simulation, a file was written at '{}'. Printing contents to stderr just to be sure.", path.display());
            eprintln!("{}", meta);
            std::process::exit(1);
        } else {
            // TODO: time of check / time of use issue?
            eprintln!("Could not create meta.json (at '{}'): {err}. Printing contents to stderr just to be sure.", path.display());
            eprintln!("{}", meta);
            std::process::exit(1);
        }
    });
    BufWriter::new(f).write_all(meta.as_bytes()).unwrap_or_else(|err| {
        // TODO: time of check / time of use issue?
        eprintln!("Could not write meta.json (at '{}'): {err}. Printing contents to stderr just to be sure.", path.display());
        eprintln!("{}", meta);
        std::process::exit(1);
    });

    eprintln!("Done rendering patterns. Took {elapsed:.2}s");
}

fn main() {
    let args = Cli::parse();

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

    // let output_path_exists = output_exists(&args.io.output_name);
    // if output_path_exists {
    //     if args.io.overwrite {
    //         std::fs::remove_dir_all(&args.io.output_name).unwrap_or_else(|err| {
    //             eprintln!(
    //                 "Could not delete output directory '{}': {}",
    //                 &args.io.output_name, err
    //             );
    //             std::process::exit(1);
    //         });
    //     } else {
    //         eprintln!("Output path '{}' already exists.", args.io.output_name);
    //         std::process::exit(1);
    //     }
    // }
    prepare_output_directory(&args.io);

    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.simulation_parameters.seed.unwrap_or(0));
    let timestamp_started: chrono::DateTime<Utc> = SystemTime::now().into();

    let structs = cfg
        .sample_parameters
        .struct_cifs
        .iter()
        .map(|path| {
            // TODO: Errors
            let mut reader = BufReader::new(std::fs::File::open(path).unwrap());
            let mut cif = String::new();
            let _ = reader.read_to_string(&mut cif).unwrap();
            let mut p = CifParser::new(&cif);
            Structure::from(&p.parse())
        })
        .collect_vec();

    let begin = Instant::now();
    let (all_simulated_peaks, all_strains) = match &cfg.kind {
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

            eprintln!("Simulating {two_theta_range:?} {wavelength_ams:.2}");
            simulate_peaks(
                &cfg.sample_parameters,
                &structs,
                two_theta_range,
                wavelength_ams,
                &mut rng,
            )
        }
        SimulationKind::EnergyDisperse(energy_disperse) => {
            // let sim_wav_ams = e_kev_to_lambda_ams(energy_disperse.energy_range_kev.1);

            // let (two_theta_range, wavelength_ams) = ((0.5, 5.0), sim_wav_ams);

            // eprintln!("Simulating {two_theta_range:?} {wavelength_ams:.2}");
            // simulate_peaks(
            //     &cfg.sample_parameters,
            //     &structs,
            //     two_theta_range,
            //     wavelength_ams,
            //     &mut rng,
            // )

            let mut all_simulated_peaks = Vec::new();
            for structure in structs.iter() {
                // let (_, mut strain) = structure.permute(0.01, &mut rng);
                // for s in strain.0.iter_mut() {
                //     *s *= 1.0 + 5.9e-6 * 900.0;
                // }
                // let s2 = structure.apply_strain(strain);
                // eprintln!("{} {}", structure.lat.mat, s2.lat.mat);
                let p = Peaks {
                    peaks: structure
                        .get_pattern_edxrd(
                            energy_disperse.theta_deg,
                            &energy_disperse.energy_range_kev,
                        )
                        .into_boxed_slice(),
                    wavelength_nm: 0.0,
                };
                all_simulated_peaks.push(vec![p]);
            }
            // // TODO: check if peaks match up when using Gnanavel's and Michaels Data
            let all_strains: Vec<Vec<Strain>> = Vec::new();
            (all_simulated_peaks, all_strains)
        }
    };

    let elapsed = begin.elapsed().as_secs_f64();

    eprintln!("Simulating Peak Positions took {elapsed:.2}s");

    match cfg.kind {
        SimulationKind::AngleDisperse(angle_disperse) => {
            render_angle_disperse(
                angle_disperse,
                cfg.sample_parameters,
                cfg.simulation_parameters,
                args,
                structs.into(),
                &all_simulated_peaks,
                &all_strains,
                timestamp_started,
                &mut rng,
            );
        }
        SimulationKind::EnergyDisperse(energy_disperse) => render_energy_disperse(
            energy_disperse,
            cfg.sample_parameters,
            cfg.simulation_parameters,
            args,
            &all_simulated_peaks,
            &all_strains,
            timestamp_started,
            &mut rng,
        ),
    }
}
