use std::io::{BufWriter, ErrorKind, Write};
use std::time::{Instant, SystemTime};

use chrono::Utc;
use log::info;

use self::io::SimulationMetadata;
use self::pattern::{DiscretizeJobGenerator, Discretizer};

pub mod background;
pub mod cfg;
pub mod cif;
pub mod util;

pub mod absorption;
pub mod composition;
pub mod element;
pub mod io;
pub mod lattice;
pub mod math;
pub mod noise;
pub mod pattern;
pub mod peak_sim;
pub mod preferred_orientation;
pub mod scatter;
pub mod site;
pub mod strain;
pub mod structure;
pub mod symop;

pub mod threading;

pub mod pylib;

#[cfg(feature = "use-gpu")]
pub mod discretize_cuda;

#[cfg(feature = "use-gpu")]
pub mod peak_sim_cuda;

#[cfg(feature = "use-gpu")]
pub mod cuda_common;

#[allow(clippy::uninit_vec)]
pub(crate) unsafe fn uninit_vec<T>(len: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(len);
    unsafe { v.set_len(len) };
    v
}

pub fn init_gpu_if_applicable() {
    cfg_if::cfg_if! {
        if #[cfg(feature = "use-gpu")] {
            use cuda_common::CUDA_DEVICE_INFO;
            use log::info;
            info!("Enabled CUDA-Based Simulation. Devices:");
            for d in CUDA_DEVICE_INFO.iter() {
                info!("Device name: {}
Available memory:    {:.3} GiB
Initial free memory: {:.3} GiB
Memory usage limit:  {:.3} GiB
API version:         {}
Runtime version:     {}
Device ID:           {}", 
                d.device_name,
                d.available_memory_bytes as f32 / 1e9,
                d.init_free_memory_bytes as f32 / 1e9,
                d.mem_limit_bytes as f32 / 1e9,
                d.api_version,
                d.runtime_version,
                d.device_id
            );
            }
        }
    }
}

pub fn render_and_write_jobs<T, G>(
    gen: G,
    io_opts: io::Opts,
    timestamp_started: chrono::DateTime<Utc>,
    extra: io::Extra,
    cfg_hash: String,
) -> Result<(), String>
where
    T: Discretizer + Send + Sync + 'static,
    G: DiscretizeJobGenerator<Item = T>,
{
    let begin_render = Instant::now();

    cfg_if::cfg_if! {
        if #[cfg(feature = "use-gpu")] {
            let output_names = io::cuda::render_write_chunked(gen, &io_opts, io::io_thread_fn);
        } else {
            let output_names = io::cpu::render_write_chunked(gen, &io_opts, io::io_thread_fn);
        }
    }
    let output_names = output_names.map_err(|err| format!("Error writing data to disk: {err}"))?;

    let elapsed = begin_render.elapsed().as_secs_f64();

    let timestamp_finished: chrono::DateTime<Utc> = SystemTime::now().into();

    let meta = serde_json::to_string(&SimulationMetadata {
        timestamp_started,
        timestamp_finished,
        yaxs_version: env!("YAXS_VERSION").to_string(),
        chunked: io_opts.chunk_size.is_some(),
        datafiles: output_names.chunk_names,
        input_names: output_names.data_slot_names,
        target_names: output_names.metadata_slot_names,
        extra,
        cfg_hash,
    })
    .expect("SimulationMetadata is serializable");

    let mut path = std::path::PathBuf::new();
    path.push(io_opts.output_path);
    path.push("meta.json");
    info!("Writing {}", path.display());
    let f = std::fs::File::create_new(&path).map_err(|err|{
            if err.kind() == ErrorKind::AlreadyExists {
                // TODO: time of check / time of use issue?
                format!("Could not write meta.json. Since check at start of simulation, a file was written at '{}'. Printing contents to stderr just to be sure:\n {}", path.display(), meta)
            } else {
                format!("Could not create meta.json (at '{}'): {err}. Printing contents to stderr just to be sure:\n {}", path.display(), meta)
            }
        }
    )?;

    BufWriter::new(f).write_all(meta.as_bytes()).map_err(|err| {
        // TODO: time of check / time of use issue?
        format!("Could not write meta.json (at '{}'): {err}. Printing contents to stderr just to be sure:\n{}", path.display(), meta)
    })?;

    info!("Done rendering patterns. Took {elapsed:.2}s");
    Ok(())
}
