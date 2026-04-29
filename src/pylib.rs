use pyo3::pymodule;

#[pymodule]
pub mod yaxs {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::mpsc::Receiver;
    use std::sync::{Arc, Mutex};

    use crate::cfg::domain_size::DomainSize;
    use crate::cfg::{
        prepare_peak_simulation, AngleDispersive, Config, InstrumentParameterCfg, Parameter,
        SampleParameters, SimulationKind, SimulationParameters, StrainCfg, StructureDef,
    };
    use crate::io::{PatternMeta, WriteJob};
    use crate::pattern::adxrd::{self, EmissionLine, PrecomputedLACs};
    use crate::pattern::CompositionGenerator;
    use crate::{init_gpu_if_applicable, io};
    use itertools::Itertools;
    use log::{debug, error};
    use ndarray::ArrayD;
    use numpy::IntoPyArray;
    use ordered_float::NotNan;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::types::{IntoPyDict, PyDict};
    use pyo3::{pyclass, pyfunction, pymethods, Bound, PyResult, Python};
    use rand::SeedableRng;

    #[pyclass(from_py_object, name = "Structure")]
    #[derive(Clone, Debug)]
    pub struct PyStructure {
        #[pyo3(get, set)]
        pub path: String,
        #[pyo3(get, set)]
        pub max_strain: f64,
        #[pyo3(get, set)]
        pub domain_size_nm: (f64, f64),
        #[pyo3(get, set)]
        pub domain_size_eta: (f64, f64),
    }

    #[pymethods]
    impl PyStructure {
        #[new]
        fn new(
            path: String,
            max_strain: f64,
            domain_size_nm: (f64, f64),
            domain_size_eta: (f64, f64),
        ) -> Self {
            Self {
                path,
                max_strain,
                domain_size_nm,
                domain_size_eta,
            }
        }

        fn __repr__(&self) -> String {
            format!("{:?}", self)
        }
    }

    #[pyclass(from_py_object, name = "Sample")]
    #[derive(Clone, Debug)]
    pub struct PySample {
        #[pyo3(get, set)]
        displacement_mu_m: (f64, f64),
    }

    #[pymethods]
    impl PySample {
        #[new]
        pub fn new(displacement_mu_m: (f64, f64)) -> Self {
            Self { displacement_mu_m }
        }

        pub fn __repr__(&self) -> String {
            format!("{self:?}")
        }
    }

    #[pyclass(from_py_object)]
    #[derive(Clone, Debug)]
    pub struct InstrumentParams {
        #[pyo3(get, set)]
        pub u: PyParameter,
        #[pyo3(get, set)]
        pub v: PyParameter,
        #[pyo3(get, set)]
        pub w: PyParameter,
        #[pyo3(get, set)]
        pub x: PyParameter,
        #[pyo3(get, set)]
        pub y: PyParameter,
        #[pyo3(get, set)]
        pub z: PyParameter,
    }

    #[pyclass(from_py_object, name = "EmissionLine")]
    #[derive(Clone, Debug)]
    pub struct PyEmissionLine {
        #[pyo3(get, set)]
        wavelength: f64,
        #[pyo3(get, set)]
        weight: f64,
    }

    #[pymethods]
    impl PyEmissionLine {
        #[new]
        pub fn new(wavelength: f64, weight: f64) -> Self {
            Self { wavelength, weight }
        }

        #[staticmethod]
        pub fn mo_ka() -> (Self, Self) {
            (Self::new(0.7093, 0.6533), Self::new(0.713574, 0.3467))
        }

        #[staticmethod]
        pub fn cu_ka() -> (Self, Self) {
            (Self::new(1.5406, 1.0), Self::new(1.5445, 0.5206))
        }
    }

    #[pymethods]
    impl InstrumentParams {
        #[new]
        pub fn new(
            u: PyParameter,
            v: PyParameter,
            w: PyParameter,
            x: PyParameter,
            y: PyParameter,
            z: PyParameter,
        ) -> Self {
            Self { u, v, w, x, y, z }
        }

        #[staticmethod]
        pub fn zero() -> Self {
            Self {
                u: PyParameter::Fixed(0.0),
                v: PyParameter::Fixed(0.0),
                w: PyParameter::Fixed(0.0),
                x: PyParameter::Fixed(0.0),
                y: PyParameter::Fixed(0.0),
                z: PyParameter::Fixed(0.0),
            }
        }
    }

    #[pyclass(from_py_object, name = "Parameter")]
    #[derive(Clone, Debug)]
    pub enum PyParameter {
        Fixed(f64),
        Range(f64, f64),
    }

    #[pymethods]
    impl PyParameter {
        #[staticmethod]
        pub fn fixed(v: f64) -> PyParameter {
            return PyParameter::Fixed(v);
        }

        #[staticmethod]
        pub fn range(lo: f64, hi: f64) -> PyResult<PyParameter> {
            if lo >= hi {
                return Err(PyValueError::new_err(format!(
                    "lower bound must be smaller than upper bound. got {lo} >= {hi}"
                )));
            }

            return Ok(PyParameter::Range(lo, hi));
        }
    }

    impl Into<Parameter<f64>> for PyParameter {
        fn into(self) -> Parameter<f64> {
            match self {
                PyParameter::Fixed(f) => Parameter::Fixed(f),
                PyParameter::Range(lo, hi) => Parameter::Range(lo, hi),
            }
        }
    }

    #[pyclass(from_py_object, name = "Instrument")]
    #[derive(Clone, Debug)]
    pub struct PyInstrument {
        instprms: InstrumentParams,          // GSAS-II u, v, w, x, y, z
        emission_lines: Vec<PyEmissionLine>, //
        goniometer_radius_mm: f64,
        two_theta_range_deg: (f64, f64),
        n_steps: usize,
        monochromator_angle_deg: f64,
    }

    #[pymethods]
    impl PyInstrument {
        #[new]
        fn new(
            instprms: InstrumentParams,
            monochromator_angle_deg: f64,
            emission_lines: Vec<PyEmissionLine>,
            goniometer_radius_mm: f64,
            two_theta_range_deg: (f64, f64),
            n_steps: usize,
        ) -> PyResult<Self> {
            if n_steps == 0 {
                return Err(PyValueError::new_err(format!(
                    "number of steps must be larger than 0. got {n_steps}"
                )));
            }

            if goniometer_radius_mm < 0.0 {
                return Err(PyValueError::new_err(format!(
                    "goniometer radius must be larger than 0. got {goniometer_radius_mm}"
                )));
            }

            Ok(Self {
                instprms,
                emission_lines,
                goniometer_radius_mm,
                two_theta_range_deg,
                n_steps,
                monochromator_angle_deg,
            })
        }

        pub fn __repr__(&self) -> String {
            format!("{self:?}")
        }
    }

    fn save_to_mem<'py>(
        rx: Receiver<WriteJob<PathBuf>>,
        out: Arc<Mutex<HashMap<String, Vec<ArrayD<f32>>>>>,
    ) -> Option<(Vec<String>, Vec<String>)> {
        let mut meta_names = Vec::new();
        let input_names = vec![String::from("intensities")];

        let mut push_pyarr = |key: &'static str, arr: ArrayD<f32>| {
            let mut out = out.lock().expect(
                "we should not look at the output in the main thread until rendering is finished",
            );
            out.entry(String::from(key)).or_default().push(arr);
            meta_names.push(String::from(key))
        };

        loop {
            match rx.recv() {
                Ok(v) => match v {
                    WriteJob::Done => {
                        debug!("WRITE_THREAD: exiting");
                        return Some((input_names, meta_names));
                    }
                    WriteJob::Write {
                        intensities,
                        path: _,
                        mut meta,
                    } => {
                        use crate::pattern::Intensities::*;
                        let pyi = match intensities {
                            Standard(intens) => intens.into_dyn(),
                            TextureMeasurement(intens) => intens.into_dyn(),
                        };
                        {
                            let mut out = out.lock().expect("main thread does not have this");
                            out.entry(String::from("intensities"))
                                .or_default()
                                .push(pyi);
                        }

                        for m in meta.drain(..) {
                            use PatternMeta::*;
                            match m {
                                VolumeFractions(x) => push_pyarr("volume_fractions", x.into_dyn()),
                                WeightFractions(x) => push_pyarr("weight_fractions", x.into_dyn()),
                                Strains(x) => push_pyarr("strains", x.into_dyn()),
                                ImpuritySum(x) => push_pyarr("impurity_sum", x.into_dyn()),
                                ImpurityMax(x) => push_pyarr("impurity_max", x.into_dyn()),
                                SampleDisplacementMuM(x) => {
                                    push_pyarr("sample_displacement_mu_m", x.into_dyn())
                                }
                                MeanDsNm(x) => push_pyarr("mean_ds_nm", x.into_dyn()),
                                DsEtas(x) => push_pyarr("ds_etas", x.into_dyn()),
                                InstrumentParameters(x) => {
                                    push_pyarr("instrument_parameters", x.into_dyn())
                                }
                                BackgroundParameters(x) => {
                                    push_pyarr("background_parameters", x.into_dyn())
                                }
                                Mustrains(x) => push_pyarr("mustrain", x.into_dyn()),
                                MustrainEtas(x) => push_pyarr("mustrain_etas", x.into_dyn()),
                                BinghamODFParams { orientations, ks } => {
                                    push_pyarr("bingham_odf_orientations", orientations.into_dyn());
                                    push_pyarr("bingham_odf_ks", ks.into_dyn());
                                }
                                RandomBIsos(x) => push_pyarr("random_b_isos", x.into_dyn()),
                            }
                        }
                        debug!("WRITE_THREAD: processed input");
                    }
                },
                Err(err) => {
                    error!("Could not receive from channel: {err}");
                    return None;
                }
            }
        }
    }

    #[pyfunction]
    #[pyo3(name = "simulate_adxrd")]
    pub fn simulate_adxrd<'py>(
        py: Python<'py>,
        structures: Vec<PyStructure>,
        instrument: PyInstrument,
        sample: PySample,
        seed: usize,
        n_patterns: usize,
        structure_permutations: usize,
        cif_root: String,
    ) -> PyResult<Bound<'py, PyDict>> {
        io::init_logging();

        if n_patterns == 0 {
            return Err(PyValueError::new_err(format!(
                "number of patterns must be a positive integer. got {n_patterns}"
            )));
        }

        let PyInstrument {
            instprms: _,
            emission_lines,
            two_theta_range_deg,
            goniometer_radius_mm,
            n_steps,
            monochromator_angle_deg,
        } = instrument;

        let mut s_defs = Vec::new();
        for s in structures {
            if s.domain_size_eta.0 < 0.0 || s.domain_size_eta.0 > 1.0 {
                return Err(PyValueError::new_err(format!("domain size eta must be between 0 and 1, got {} for lower bound of structure {}.", s.domain_size_eta.0, s.path)));
            }
            if s.domain_size_eta.1 < 0.0 || s.domain_size_eta.1 > 1.0 {
                return Err(PyValueError::new_err(format!("domain size eta must be between 0 and 1, got {} for upper bound of structure {}.", s.domain_size_eta.1, s.path)));
            }

            s_defs.push(StructureDef {
                path: s.path.clone(),
                preferred_orientation: None,
                composition: None,
                mustrain: None,
                strain: Some(StrainCfg::Maximum(s.max_strain)),
                domain_size: DomainSize::Uniform(Parameter::Range(
                    s.domain_size_nm.0,
                    s.domain_size_nm.1,
                )),
                ds_eta: Parameter::range_checked(s.domain_size_eta.0, s.domain_size_eta.1)
                    .map_err(|err| {
                        PyValueError::new_err(format!(
                            "Invalid range of domain size eta for structure {}: {}",
                            s.path, err
                        ))
                    })?,
                b_iso: None,
            });
        }

        let mut cfg = Config {
            kind: SimulationKind::AngleDispersive(AngleDispersive {
                emission_lines: emission_lines
                    .iter()
                    .map(|PyEmissionLine { wavelength, weight }| EmissionLine {
                        wavelength_ams: *wavelength,
                        weight: *weight,
                    })
                    .collect(),
                n_steps: n_steps,
                two_theta_range: two_theta_range_deg,
                goniometer_radius_mm,
                monochromator_angle: monochromator_angle_deg.to_radians(),
                sample_displacement_mu_m: Some(Parameter::Range(
                    sample.displacement_mu_m.0,
                    sample.displacement_mu_m.1,
                )),
                instrument_parameters: Some(InstrumentParameterCfg {
                    kind: crate::cfg::InstprmKind::GSAS,
                    u: instrument.instprms.u.into(),
                    v: instrument.instprms.v.into(),
                    w: instrument.instprms.w.into(),
                    x: instrument.instprms.x.into(),
                    y: instrument.instprms.y.into(),
                    z: instrument.instprms.z.into(),
                }),
                background: None, // TODO: for now
            }),
            sample_parameters: SampleParameters {
                composition_kind: crate::cfg::CompositionKind::ByMass,
                structures: s_defs,
                concentration_subset: None, // TODO
                impurities: None,
                structure_permutations,
            },
            simulation_parameters: SimulationParameters {
                normalize: false,
                seed: Some(seed.try_into().expect("usize fits into u64")),
                n_patterns,
                noise: None,
                texture_measurement: None,
                randomly_scale_peaks: None,
                abstol: 1e-3,
            },
        };

        init_gpu_if_applicable();

        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(
            cfg.simulation_parameters.seed.unwrap_or(0),
        );

        let mut psd = prepare_peak_simulation(&mut cfg, &cif_root, &mut rng)
            .map_err(|err| PyValueError::new_err(format!("Invalid configuration: {err}")))?;

        let structures = psd.structures.clone();
        let structure_paths = psd.structure_paths.clone();

        let vf_generator = CompositionGenerator::try_new(
            &mut psd.composition_constraints,
            cfg.sample_parameters.concentration_subset.clone(),
        )
        .map_err(|err| {
            PyRuntimeError::new_err(format!(
                "Error: Could not generate volume fractions: '{err}'"
            ))
        })?;

        let mut to_discretize = cfg
            .kind
            .simulate_peaks(
                psd,
                cfg.sample_parameters.clone(),
                cfg.simulation_parameters.texture_measurement,
                &mut rng,
            )
            .map_err(|err| PyRuntimeError::new_err(format!("Could not simulate peaks: {err}")))?;

        if let Some(ref rand_scale) = cfg.simulation_parameters.randomly_scale_peaks {
            let v = std::sync::Arc::get_mut(&mut to_discretize.sim_res)
                .expect("no other references to sim_res should exist at this point");
            for phase_peaks in v.all_simulated_peaks.iter_mut() {
                for peak in phase_peaks.iter_peaks_mut() {
                    peak.i_hkl = NotNan::try_from(rand_scale.scale_peak(*peak.i_hkl, &mut rng))
                        .expect("peak scaling should not be nan");
                }
            }
        }

        // let params = cfg.simulation_parameters;
        // let extra = io::Extra {
        //     max_phases: cfg.sample_parameters.structures.len(),
        //     texture: params.texture_measurement,
        //     encoding: cfg
        //         .sample_parameters
        //         .structures
        //         .iter()
        //         .map(|StructureDef { path, .. }| path.to_string())
        //         .collect_vec(),
        //     cfg: cfg.kind.clone(),
        //     n_patterns,
        // };

        if let SimulationKind::AngleDispersive(angle_dispersive) = cfg.kind {
            let absorption_factors = PrecomputedLACs::try_new(
                angle_dispersive
                    .emission_lines
                    .iter()
                    .map(|line| line.wavelength_ams),
                &structures,
                &structure_paths,
            )
            .map_err(|err| {
                PyValueError::new_err(format!("Could not precompute absorption factors: {err}."))
            })?;

            let gen = adxrd::JobGen::new(
                angle_dispersive,
                to_discretize,
                cfg.simulation_parameters,
                vf_generator,
                absorption_factors,
                rng,
            );
            let io_opts = io::Opts {
                chunk_size: Some(n_patterns),
                overwrite: false,
                re_simulate: false,
                output_path: "./out".into(),
                compress: false,
                display_hkls: None,
                quiet: true,
            };

            let out = Arc::new(Mutex::new(HashMap::new()));

            {
                let out = Arc::clone(&out);
                cfg_if::cfg_if! {
                    if #[cfg(feature = "use-gpu")] {
                        let output_names = crate::io::cuda::render_write_chunked(gen, &io_opts, move |rx, _compress, _n_chunks| save_to_mem(rx, Arc::clone(&out)));
                    } else {
                        let output_names = crate::io::cpu::render_write_chunked(gen, &io_opts, io::io_thread_fn);
                    }
                }
                let _ = output_names.map_err(|err| {
                    PyRuntimeError::new_err(format!("Error writing data to disk: {err}"))
                })?;
            }

            let out: HashMap<String, _> = out
                .lock()
                .expect("all rendering threads have stopped")
                .drain()
                .map(|(k, mut v)| (k, v.drain(..).map(|x| x.into_pyarray(py)).collect_vec()))
                .collect();

            return out.into_py_dict(py);
        } else {
            todo!("wtf")
        };
    }
}
