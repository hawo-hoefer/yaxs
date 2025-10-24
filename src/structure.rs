use itertools::Itertools;
use log::{debug, info};
use num_complex::Complex;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::Arc;

use ordered_float::NotNan;
use rand::{Rng, SeedableRng};

use crate::cfg::{
    apply_strain_cfg, CompactSimResults, POGenerator, SampleParameters, StrainCfg,
    TextureMeasurement, ToDiscretize,
};
use crate::cif::CIFContents;
use crate::math::e_kev_to_lambda_ams;
use crate::math::linalg::{Mat, Mat3, Vec3};
use crate::pattern::{Peak, Peaks};
use crate::preferred_orientation::{BinghamODF, BinghamParams};
use crate::site::Site;
use crate::uninit_vec;

const D_SPACING_ABSTOL_AMS: f64 = 1e-5;
const SCALED_INTENSITY_TOL: f64 = 1e-5;

#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    pub mat: Mat3<f64>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Strain(pub [f64; 6]);

impl std::fmt::Display for Strain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Strain(")?;
        for (i, a) in self.0.iter().enumerate() {
            if i != self.0.len() - 1 {
                write!(f, "{}, ", a)?;
            } else {
                write!(f, "{}", a)?;
            }
        }
        write!(f, ")")
    }
}
impl Strain {
    pub fn from_diag(a: f64, b: f64, c: f64) -> Self {
        Self([a, 0.0, b, 0.0, 0.0, c])
    }

    pub fn new_verified(data: [f64; 6]) -> Option<Self> {
        // TODO: find a better way to do this. we may not actually need to try calculating the inverse

        // strain is ok if we can take the inverse of the strain matrix
        // use this to verify user input
        let v = Self(data);
        if v.to_mat3().try_inverse().is_some() {
            return Some(v);
        }
        None
    }

    pub fn from_mat3(mat: &Mat3<f64>) -> Self {
        Self([
            mat[(0, 0)],
            mat[(1, 0)],
            mat[(1, 1)],
            mat[(2, 0)],
            mat[(2, 1)],
            mat[(2, 2)],
        ])
    }

    pub fn none() -> Self {
        Self([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    }

    pub fn to_mat3(&self) -> Mat3<f64> {
        let mut ret = Mat3::zeros();
        ret[(0, 0)] = self.0[0];

        ret[(1, 0)] = self.0[1];
        ret[(0, 1)] = self.0[1];

        ret[(1, 1)] = self.0[2];

        ret[(2, 0)] = self.0[3];
        ret[(0, 2)] = self.0[3];

        ret[(2, 1)] = self.0[4];
        ret[(1, 2)] = self.0[4];

        ret[(2, 2)] = self.0[5];

        ret
    }
}

impl Lattice {
    fn recip_lattice_crystallographic(&self) -> Lattice {
        Self {
            mat: self.mat.try_inverse().unwrap().transpose(),
        }
    }

    fn recip_lattice(&self) -> Lattice {
        Self {
            mat: self
                .mat
                .try_inverse()
                .unwrap()
                .transpose()
                .scale(2.0 * std::f64::consts::PI),
        }
    }

    /// Returns the volume of this [`Lattice`] in amstrong cubed.
    pub fn volume(&self) -> f64 {
        self.mat
            .row(0)
            .cross(&self.mat.row(1))
            .dot(&self.mat.row(2))
            .abs()
    }

    fn abc(&self) -> Vec3<f64> {
        let mut values = [0.0; 3];
        for i in 0..self.mat.rows() {
            values[i] = self.mat.row(i).magnitude();
        }
        Vec3::new(values[0], values[1], values[2])
    }
}

impl std::fmt::Display for Lattice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Lattice(")?;
        for ri in 0..self.mat.rows() {
            writeln!(
                f,
                "  {:5.2}, {:5.2}, {:5.2}",
                self.mat[(ri, 0)],
                self.mat[(ri, 1)],
                self.mat[(ri, 2)]
            )?;
        }
        writeln!(f, ")")
    }
}

#[derive(Debug, Clone, PartialEq)]
/// A phase's crystallographic structure
///
/// * `lat`: lattice
/// * `sites`: sites in the structure
/// * `sg_no`: space group number
/// * `sg_class`: space group class
/// * `density`: density of the phase in g/cm3, if present in the cif
pub struct Structure {
    pub lat: Lattice,
    pub sites: Vec<Site>,
    pub sg_no: u8, // there are 230 space groups, so u8 should be enough
    pub sg_class: SGClass,
    pub density: Option<f64>,
}

impl TryFrom<&CIFContents> for Structure {
    type Error = String;
    fn try_from(value: &CIFContents) -> Result<Self, Self::Error> {
        let (sg_no, sg_class) = value.get_sg_no_and_class()?;
        Ok(Structure {
            sites: value.get_sites()?,
            lat: value.get_lattice(),
            density: value.get_density()?,
            sg_no,
            sg_class,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SGClass {
    // SG class for determining possible structure permutation without affecting symmetry
    Cubic,
    Orthorombic,
    Monoclinic,
    Triclinic,
    LowSymHexagonalOrTetragonal,
    HighSymHexagonalOrTetragonal,
}

impl TryFrom<u8> for SGClass {
    type Error = String;
    fn try_from(value: u8) -> Result<SGClass, Self::Error> {
        use SGClass::*;
        // TODO: Figure out exactly what Jan meant by this
        // What is Low/HighSymHexagonalOrTetragonal
        match value {
            ..1 | 231.. => Err(format!("SG-Number should be in [1, 230], got {value}.")),
            1..3 => Ok(Triclinic),
            3..16 => Ok(Monoclinic),
            16..75 => Ok(Orthorombic),
            75..83 | 133..149 | 168..175 => Ok(LowSymHexagonalOrTetragonal),
            83..133 | 149..168 | 175..195 => Ok(HighSymHexagonalOrTetragonal),
            195..=230 => Ok(Cubic),
        }
    }
}

impl Structure {
    pub fn permute(&self, max_strain: f64, rng: &mut impl Rng) -> (Structure, Strain) {
        if max_strain == 0.0 {
            return (self.clone(), Strain::none());
        }

        let tensile_range = 1.0 - max_strain..=1.0 + max_strain;
        let shear_range = -max_strain..=max_strain;

        let strain_tensor: Mat3<f64> = match self.sg_class {
            SGClass::Cubic => Mat3::identity().scale(rng.random_range(tensile_range)),
            SGClass::Orthorombic => {
                // all directions independent for orthorombic
                let mut m = Mat3::zeros();
                m[(0, 0)] = rng.random_range(tensile_range.clone());
                m[(1, 1)] = rng.random_range(tensile_range.clone());
                m[(2, 2)] = rng.random_range(tensile_range.clone());
                m
            }
            SGClass::Monoclinic => {
                // one plane can be sheared
                let mut m = Mat3::zeros();
                m[(0, 0)] = rng.random_range(tensile_range.clone());
                m[(1, 1)] = rng.random_range(tensile_range.clone());
                m[(2, 2)] = rng.random_range(tensile_range.clone());

                m[(1, 2)] = rng.random_range(shear_range.clone());
                m[(2, 1)] = m[(1, 2)];

                m
            }
            SGClass::Triclinic => {
                // symmetric tensor with all values free
                let mut m = Mat3::zeros();
                m[(0, 0)] = rng.random_range(tensile_range.clone());
                m[(1, 1)] = rng.random_range(tensile_range.clone());
                m[(2, 2)] = rng.random_range(tensile_range.clone());

                m[(1, 0)] = rng.random_range(shear_range.clone());
                m[(0, 1)] = m[(1, 0)];

                m[(0, 2)] = rng.random_range(shear_range.clone());
                m[(2, 0)] = m[(0, 2)];

                m[(1, 2)] = rng.random_range(shear_range.clone());
                m[(2, 1)] = m[(1, 2)];

                m
            }
            SGClass::LowSymHexagonalOrTetragonal => {
                let mut m = Mat3::zeros();

                m[(0, 0)] = rng.random_range(tensile_range.clone());
                m[(1, 1)] = rng.random_range(tensile_range.clone());
                m[(2, 2)] = rng.random_range(tensile_range.clone());

                // no clue what this is
                m[(0, 1)] = rng.random_range(shear_range.clone());
                m[(1, 0)] = -m[(0, 1)];

                m
            }
            SGClass::HighSymHexagonalOrTetragonal => {
                // X and Y stretched the same, z differently
                let mut m = Mat3::zeros();

                m[(0, 0)] = rng.random_range(tensile_range.clone());
                m[(1, 1)] = m[(0, 0)];
                m[(2, 2)] = rng.random_range(tensile_range.clone());

                m
            }
        };

        let mut r = self.clone();
        r.lat.mat = r.lat.mat.matmul(&strain_tensor);

        (r, Strain::from_mat3(&strain_tensor))
    }

    pub fn apply_strain(&self, strain: &Strain) -> Structure {
        let mut ret = self.clone();

        ret.lat.mat = ret.lat.mat.matmul(&strain.to_mat3());

        ret
    }

    pub fn get_hkl_intensities_spacings(
        &self,
        min_r: f64,
        max_r: f64,
    ) -> Vec<(Vec3<f64>, NotNan<f64>, NotNan<f64>)> {
        let mut agg = Vec::new();
        for (hkl, g_hkl) in self.lat.iter_hkls(min_r, max_r) {
            let s = g_hkl / 2.0;
            let s2 = s.powi(2);

            let mut f_hkl = Complex::new(0.0, 0.0);
            // TODO: Debye-Waller Correction
            // (we ignore it for now, in the test data we don't have DW-factors)
            // dw_correction = np.exp(-dw_factors * s2)
            let dw_correction = 1.0;

            for site in &self.sites {
                // g_dot_r = np.dot(frac_coords, np.transpose([hkl])).T[0]
                let g_dot_r: f64 = site.coords.dot(&hkl);
                for species in &site.species {
                    let fs = species.el.scattering_factor(s2);

                    // f_hkl = np.sum(fs * occus * np.exp(2j * np.pi * g_dot_r) * dw_correction)
                    let f_part =
                        fs * site.occu * Complex::new(0.0, std::f64::consts::TAU * g_dot_r).exp();
                    f_hkl += f_part;
                }
            }
            f_hkl *= dw_correction;

            // # Intensity for hkl is modulus square of structure factor
            let i_hkl = NotNan::new((f_hkl * f_hkl.conj()).re).expect("not nan");

            let d_hkl = 1.0 / g_hkl;
            let d_spacing = NotNan::new(d_hkl).expect("not nan");
            agg.push((hkl, i_hkl, d_spacing));
        }

        agg
    }

    pub fn apply_alignment_to_hkls_intensities<'a>(
        &self,
        input: &[(Vec3<f64>, NotNan<f64>, NotNan<f64>)],
        alignment: Option<Alignment<'a>>,
    ) -> Vec<Peak> {
        let mut agg = HashMap::<NotNan<f64>, (NotNan<f64>, Vec<Vec3<i16>>)>::new();

        for (hkl, i_hkl, d_hkl) in input {
            let mut i_hkl = *i_hkl;

            if let Some(Alignment { po, phi, chi }) = alignment {
                let w =
                    NotNan::new(po.weight(&hkl, &self.lat, chi, phi)).expect("weight is not nan");
                i_hkl = i_hkl * w;
            }

            let (ref mut i_hkl_map, ref mut hkls_map) = agg
                .entry(*d_hkl)
                .or_insert((NotNan::new(0.0).expect("valid float"), Vec::new()));
            *i_hkl_map += NotNan::try_from(i_hkl).expect("i_hkl may not be nan");
            hkls_map.push(hkl.map(|x| *x as i16))
        }

        self.compress_aggregated_hkls(agg)
    }

    pub fn compress_aggregated_hkls(
        &self,
        agg: HashMap<NotNan<f64>, (NotNan<f64>, Vec<Vec3<i16>>)>,
    ) -> Vec<Peak> {
        let Some((_, (vmax, _))) = agg.iter().max_by_key(|&(_, (b, _))| b) else {
            return Vec::new();
        };

        let mut agg = agg
            .iter()
            .sorted_unstable_by_key(|&(a, _)| -a)
            .map(|(d_hkl, (i_hkl, hkls))| (f64::from(*d_hkl), f64::from(*i_hkl), hkls))
            .filter(|&(_, b, _)| b / f64::from(*vmax) >= SCALED_INTENSITY_TOL)
            .collect_vec();

        let mut compressed: Vec<Peak> = Vec::with_capacity(agg.len() / 2 * 3);
        for (d_hkl, i_hkl, hkls) in agg.drain(..) {
            match compressed.last_mut() {
                Some(Peak {
                    d_hkl: last_d_hkl,
                    i_hkl: last_i_hkl,
                    hkls: last_hkls,
                }) if ((d_hkl - *last_d_hkl).abs() < D_SPACING_ABSTOL_AMS) => {
                    *last_i_hkl += i_hkl;
                    last_hkls.extend(hkls.clone())
                }
                None | Some(&mut Peak { .. }) => compressed.push(Peak {
                    d_hkl,
                    i_hkl,
                    hkls: hkls.clone(),
                }),
            }
        }
        let volume = self.lat.volume();
        for peak in compressed.iter_mut() {
            peak.i_hkl /= volume.powi(2);
        }
        compressed
    }

    /// scan lattice for crystallographic planes with given d-spacings
    /// compute peak intensities and d-spacings corresponding to the lattice planes
    /// miller indices are **not** returned
    ///
    /// * `min_r`: minimum d-spacing to consider
    /// * `max_r`: maximum d-spacing to consider
    /// * `alignment`: optional preferred orientation as Bingham Orientation distribution function
    ///         relative to the beam direction
    pub fn get_d_spacings_intensities<'a>(
        &self,
        min_r: f64,
        max_r: f64,
        alignment: Option<Alignment<'a>>,
    ) -> Vec<Peak> {
        let mut agg = HashMap::<NotNan<f64>, (NotNan<f64>, Vec<Vec3<i16>>)>::new();

        for (hkl, g_hkl) in self.lat.iter_hkls(min_r, max_r) {
            let hkl = hkl.map(|x| *x as f64);
            let s = g_hkl / 2.0;
            let s2 = s.powi(2);

            let mut f_hkl = Complex::new(0.0, 0.0);
            // TODO: Debye-Waller Correction
            // (we ignore it for now, in the test data we don't have DW-factors)
            // dw_correction = np.exp(-dw_factors * s2)
            let dw_correction = 1.0;

            for site in &self.sites {
                // g_dot_r = np.dot(frac_coords, np.transpose([hkl])).T[0]
                let g_dot_r: f64 = site.coords.dot(&hkl);
                for species in &site.species {
                    let fs = species.el.scattering_factor(s2);

                    // f_hkl = np.sum(fs * occus * np.exp(2j * np.pi * g_dot_r) * dw_correction)
                    let f_part =
                        fs * site.occu * Complex::new(0.0, std::f64::consts::TAU * g_dot_r).exp();
                    f_hkl += f_part;
                }
            }
            f_hkl *= dw_correction;

            // # Intensity for hkl is modulus square of structure factor
            let mut i_hkl = (f_hkl * f_hkl.conj()).re;

            if let Some(Alignment { po, phi, chi }) = alignment {
                let w = po.weight(&hkl, &self.lat, chi, phi);
                i_hkl *= w;
            }
            let d_hkl = 1.0 / g_hkl;
            let d_spacing = NotNan::new(d_hkl).expect("not nan");
            let (ref mut i_hkl_map, ref mut hkls_map) = agg
                .entry(d_spacing)
                .or_insert((NotNan::new(0.0).expect("valid float"), Vec::new()));
            *i_hkl_map += NotNan::try_from(i_hkl).expect("i_hkl may not be nan");
            hkls_map.push(hkl.map(|x| *x as i16))
        }

        self.compress_aggregated_hkls(agg)
    }

    /// compute peak positions and intensities for angle dispersive XRD
    ///
    /// * `wavelength_ams`: wavelength for peaks in Amstrong
    /// * `two_theta_range`: two theta range to search for peaks in degrees
    /// * `po`: preferred orientation march-dollase parameters, if desired
    /// * `goniometer_pos`: goniometer angles phi and chi in radians
    ///    chi is the angle of rotation around the beam, and
    ///    phi is the angle of rotation around the axis vertical to the beam
    pub fn get_adxrd_peaks<'a>(
        &self,
        wavelength_ams: f64,
        two_theta_range: &(f64, f64),
        alignment: Option<Alignment<'a>>,
    ) -> Vec<Peak> {
        let min_r = (two_theta_range.0 / 2.0).to_radians().sin() / wavelength_ams * 2.0;
        let max_r = (two_theta_range.1 / 2.0).to_radians().sin() / wavelength_ams * 2.0;

        self.get_d_spacings_intensities(min_r, max_r, alignment)
    }

    /// compute peak positions and intensities for energy dispersive XRD
    ///
    /// * `theta_deg`: fixed angle of sample to beam
    /// * `energy_kev_range`: energy range in keV to consider for d-spacings
    /// * `po`: preferred orientation march-dollase parameters, if desired
    pub fn get_edxrd_peaks<'a>(
        &self,
        theta_deg: f64,
        energy_kev_range: &(f64, f64),
        alignment: Option<Alignment<'a>>,
    ) -> Vec<Peak> {
        let lambda_0 = e_kev_to_lambda_ams(energy_kev_range.1);
        let lambda_1 = e_kev_to_lambda_ams(energy_kev_range.0);

        let theta_rad = theta_deg.to_radians();

        let min_r = theta_rad.sin() / lambda_1 * 2.0;
        let max_r = theta_rad.sin() / lambda_0 * 2.0;

        self.get_d_spacings_intensities(min_r, max_r, alignment)
    }
}

impl<'a> Lattice {
    pub fn iter_hkls(
        &'a self,
        min_r: f64,
        max_r: f64,
    ) -> impl Iterator<Item = (Vec3<f64>, f64)> + use<'a> {
        const RADIUS_TOL: f64 = 1e-8;
        let recip_lat = self.recip_lattice_crystallographic();
        let recp_len = recip_lat.recip_lattice().abc();

        let r_cells = max_r + 1e-8;
        let r_max = (recp_len.scale((r_cells + 0.15) / (2.0 * std::f64::consts::PI)))
            .map(|x| x.ceil() as i32);
        let global_min = -max_r - RADIUS_TOL;
        let global_max = max_r + RADIUS_TOL;

        let n_min = -r_max.clone();
        let n_max = r_max;
        (n_min[0]..n_max[0])
            .cartesian_product(n_min[1]..n_max[1])
            .cartesian_product(n_min[2]..n_max[2])
            .filter_map(move |((a, b), c)| -> Option<_> {
                let hkl = Vec3::new(a as f64, b as f64, c as f64);
                let pos = recip_lat.mat.matmul(&hkl);
                let g_hkl = pos.magnitude();

                // currently, we produce XRD patterns like pymatgen
                // Neighbor mapping from pymatgen.core.lattice.get_points_in_spheres
                // does not seem to have any effect if center_coords is the 0-vector
                // As far as I can tell, it only applies when center_coords are something
                // other than the 0-vector so we will ignore it for now.
                // i tested this using a modification of their code and random cifs from
                // the COD-database
                if (g_hkl < max_r + RADIUS_TOL && g_hkl > min_r - RADIUS_TOL)
                    && g_hkl > 0.0
                    && pos
                        .iter_values()
                        .map(|&x| (x > global_min) && (x < global_max))
                        .all(|x| x)
                {
                    Some((hkl, g_hkl))
                } else {
                    None
                }
            })
    }
}

enum PossiblyTextureMeasurementPeaks {
    NoTexture(Peaks),
    Texture(Vec<Peaks>),
}

struct PeakSimResult {
    strain: Strain,
    po: Option<BinghamODF>,
    peaks: PossiblyTextureMeasurementPeaks,
    struct_id: usize,
    permutation_id: usize,
}

struct WriteCtx {
    inner: UnsafeCell<Inner>,
}

struct Inner {
    strain: Vec<Strain>,
    pos: Vec<Option<BinghamParams>>,
    peaks: Vec<MaybeUninit<Peaks>>,
    ok: Vec<bool>,
    n_measurements: usize,
    n_permutations: usize,
}

unsafe impl Sync for WriteCtx {}

impl WriteCtx {
    pub fn new(n_structs: usize, n_measurements: usize, n_permutations: usize) -> Self {
        let n_peak_sets = n_structs * n_permutations * n_measurements;
        let n_simulations = n_structs * n_permutations;

        Self {
            inner: UnsafeCell::new(Inner {
                peaks: unsafe { uninit_vec(n_peak_sets) },
                strain: unsafe { uninit_vec(n_simulations) },
                pos: unsafe { uninit_vec(n_simulations) },
                ok: unsafe { uninit_vec(n_simulations) },
                n_permutations,
                n_measurements,
            }),
        }
    }

    pub unsafe fn add(&self, p: PeakSimResult) {
        let Inner {
            strain,
            pos,
            peaks,
            ok,
            n_permutations,
            n_measurements,
        } = unsafe { &mut *(self.inner.get()) };

        let sample_idx = p.struct_id * *n_permutations + p.permutation_id;
        match p.peaks {
            PossiblyTextureMeasurementPeaks::NoTexture(res) => {
                assert_eq!(*n_measurements, 1, "n_measurements needs to be 1 if no texture measurement is done. this is likely a bug in yaxs.");

                pos[sample_idx] = p.po.map(|x| x.params);
                strain[sample_idx] = p.strain;
                ok[sample_idx] = true;

                peaks[sample_idx] = MaybeUninit::new(res);
            }
            PossiblyTextureMeasurementPeaks::Texture(mut texture_measurement_peaks) => {
                assert_eq!(texture_measurement_peaks.len(), *n_measurements, "number of peak sets must match number of simulated peaks. this is likely a bug in yaxs.");
                let sample_idx = p.struct_id * *n_permutations + p.permutation_id;
                // index is [structure_id, permutation_id, texture_measurment_id]
                for (measurement_id, res) in texture_measurement_peaks.drain(..).enumerate() {
                    let idx = sample_idx * *n_measurements + measurement_id;

                    peaks[idx] = MaybeUninit::new(res);
                }

                pos[sample_idx] = p.po.as_ref().map(|x| x.params.clone());
                strain[sample_idx] = p.strain.clone();
                ok[sample_idx] = true;
            }
        }
    }

    pub fn make_to_discretize(
        self,
        structures: Arc<[Structure]>,
        sample_parameters: SampleParameters,
        texture_measurement: Option<TextureMeasurement>,
    ) -> ToDiscretize {
        let Inner {
            strain,
            pos,
            mut peaks,
            ok,
            n_permutations,
            n_measurements: _,
        } = self.inner.into_inner();

        // TODO: make this return a result instead so we can error gracefully
        assert!(ok.iter().all(|x| *x));

        ToDiscretize {
            structures,
            sample_parameters,
            sim_res: Arc::new(CompactSimResults {
                all_simulated_peaks: peaks
                    .drain(..)
                    .map(|x| unsafe { x.assume_init() })
                    .collect(),
                all_strains: strain.into(),
                all_preferred_orientations: pos.into(),
                n_permutations,
                texture_measurement,
            }),
        }
    }
}

pub struct Alignment<'a> {
    pub po: &'a BinghamODF,
    pub phi: f64,
    pub chi: f64,
}

/// Generate Peaks for the input structures and their physical parameters.
///
/// * `sample_params`: physical parameter ranges for the structures
/// * `structures`: structures to simulate peaks for
/// * `rng`: random number generator to use
///
/// * `sample_params`: the user specified sample parameters
/// * `structures`: the structures to simulate
/// * `structure_po_configs`: preferred orientation configuration for all structures
/// * `structure_strain_configs`: strain configurations for each of the structures
/// * `structure_files`: file paths to the structure's cifs
/// * `rng`: rng to use
pub fn simulate_peaks(
    (min_r, max_r): (f64, f64),
    sample_parameters: SampleParameters,
    structures: Box<[Structure]>,
    structure_po_configs: Box<[Option<POGenerator>]>,
    structure_strain_configs: Box<[Option<StrainCfg>]>,
    structure_files: Box<[String]>,
    texture_measurement: Option<TextureMeasurement>,
    rng: &mut impl Rng,
) -> Result<ToDiscretize, String> {
    struct PeakSim {
        structure: usize,
        permutation: usize,
        seed: u64,
        min_r: f64,
        max_r: f64,
        t: Option<TextureMeasurement>,
    }

    #[derive(Clone)]
    struct RunCtx {
        structs: Box<[Structure]>,
        po_gens: Box<[Option<POGenerator>]>,
        strain_cfgs: Box<[Option<StrainCfg>]>,
        structure_files: Box<[String]>,
    }

    impl RunCtx {
        fn run(&mut self, job: PeakSim) -> Result<PeakSimResult, String> {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(job.seed);
            let Some((perm_s, strain)) = apply_strain_cfg(
                &self.strain_cfgs[job.structure],
                &self.structs[job.structure],
                &mut rng,
            ) else {
                return Err(format!("Could not apply strain to structure '{file}'. Strain matrix is not invertible. Please check the strain configuration.", file=self.structure_files[job.structure]));
            };
            let po = self.po_gens[job.structure]
                .as_mut()
                .map(|x| x.sample(&mut rng));

            let peaks = if let Some(t) = job.t {
                let hkls_intensities_spacings = perm_s
                    .get_hkl_intensities_spacings(job.min_r, job.max_r)
                    .into_boxed_slice();
                let mut peaks = Vec::new();
                for (_, (chi, phi)) in t
                    .chi
                    .into_iter()
                    .cartesian_product(t.phi.into_iter())
                    .enumerate()
                {
                    let p = perm_s.apply_alignment_to_hkls_intensities(
                        &hkls_intensities_spacings,
                        po.as_ref().map(|x| Alignment { po: x, chi, phi }),
                    );
                    peaks.push(p.into_boxed_slice());
                }
                PossiblyTextureMeasurementPeaks::Texture(peaks)
            } else {
                let peaks = perm_s
                    .get_d_spacings_intensities(job.min_r, job.max_r, None)
                    .into_boxed_slice();
                PossiblyTextureMeasurementPeaks::NoTexture(peaks)
            };

            Ok(PeakSimResult {
                strain,
                po: po.clone(),
                peaks,
                struct_id: job.structure,
                permutation_id: job.permutation,
            })
        }
    }

    let n_structs = structures.len();
    let n_permutations = sample_parameters.structure_permutations;
    let n_texture_measurements = texture_measurement.map(|t| t.stride()).unwrap_or(1);

    enum Task {
        Job(PeakSim),
        Stop,
    }

    let mut n_threads: usize = std::thread::available_parallelism()
        .map(|x| x.into())
        .unwrap_or(1);

    if n_structs * n_permutations < 50 {
        info!("Small number of simulations. Using single-threaded mode.");
        n_threads = 1;
    }

    let mut ctx = RunCtx {
        structs: structures.into(),
        po_gens: structure_po_configs,
        strain_cfgs: structure_strain_configs,
        structure_files,
    };

    let results = Arc::new(WriteCtx::new(
        n_structs,
        n_texture_measurements,
        n_permutations,
    ));

    if n_threads == 1 {
        info!("Running single-threaded peak simulation");

        for (struct_id, permutation_id) in (0..n_structs).cartesian_product(0..n_permutations) {
            let job = PeakSim {
                structure: struct_id,
                permutation: permutation_id,
                seed: rng.random(),
                min_r,
                max_r,
                t: texture_measurement,
            };
            let p = ctx.run(job)?;
            unsafe { results.add(p) };
        }
    } else {
        let (job_sender, job_receiver) = crossbeam_channel::unbounded();
        info!("Launching {n_threads} threads for peak simulation");

        let handles = (0..n_threads)
            .map(|i| {
                let results = Arc::clone(&results);
                let job_receiver = job_receiver.clone();

                let mut ctx = ctx.clone();
                for po_gen in ctx.po_gens.iter_mut().filter_map(|x| x.as_mut()) {
                    // just to pull n_thinning samples out of the po_gen internal Hit and Run sampler
                    // this is hopefully enough to make the samplers of every thread independent
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(rng.random());
                    _ = po_gen.sample(&mut rng);
                }

                std::thread::spawn(move || -> Result<(), String> {
                    loop {
                        let job: PeakSim = match job_receiver.recv() {
                            Ok(Task::Stop) => break,
                            Ok(Task::Job(v)) => v,
                            Err(_) => break,
                        };

                        let p = ctx
                            .run(job)
                            .map_err(|err| format!("Peak simulation thread {i}: {err}"))?;
                        unsafe {
                            results.add(p);
                        }
                    }
                    debug!("Peak simulation thread {i} finished.");
                    Ok(())
                })
            })
            .collect_vec();

        for (struct_id, permutation_id) in (0..n_structs).cartesian_product(0..n_permutations) {
            let job = PeakSim {
                structure: struct_id,
                permutation: permutation_id,
                seed: rng.random(),
                min_r,
                max_r,
                t: texture_measurement,
            };
            let _ = job_sender.send(Task::Job(job));
        }

        debug!("Sending stop signal to peak simulation threads");
        for _ in 0..n_threads {
            job_sender
                .send(Task::Stop)
                .map_err(|err| format!("Could not send stop signal for peak simulation: '{err}'"))?
        }

        for handle in handles {
            handle.join().map_err(|err| {
                format!("Could not join peak simulation thread: '{err:?}'. Exiting...")
            })??;
        }

        debug!("All simulation threads joined")
    }

    let results = Arc::into_inner(results).expect("no more references to write context");
    Ok(results.make_to_discretize(ctx.structs.into(), sample_parameters, texture_measurement))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::cif::CifParser;

    #[test]
    #[rustfmt::skip]
    fn iter_hkls() {
        let mat = Mat3::from_rows([
            [8.08528000e+00, 0.00000000e+00, 4.95080614e-16],
            [1.30021219e-15, 8.08528000e+00, 4.95080614e-16],
            [0.00000000e+00, 0.00000000e+00, 8.08528000e+00],
        ]);

        let lat = Lattice { mat }.recip_lattice_crystallographic();

        let hkls = lat.iter_hkls(0.0, 15.0).count();
        assert_eq!(hkls, 26);

        let mut iter = lat.iter_hkls(0.0, 15.0)
            .sorted_by_key(|(hkl, g_hkl)| (
                NotNan::new(*g_hkl).unwrap(),
                NotNan::new(hkl[0]).unwrap(),
                NotNan::new(hkl[1]).unwrap(),
                NotNan::new(hkl[2]).unwrap(),
            ));

        assert_eq!(iter.next(), Some((Vec3::new(-1.,  0.,  0.), 8.08528)));
        assert_eq!(iter.next(), Some((Vec3::new( 0., -1.,  0.),  8.08528)));
        assert_eq!(iter.next(), Some((Vec3::new( 0.,  0., -1.),  8.08528)));
        assert_eq!(iter.next(), Some((Vec3::new(0., 0., 1.),  8.08528)));
        assert_eq!(iter.next(), Some((Vec3::new(0., 1., 0.),  8.08528)));
        assert_eq!(iter.next(), Some((Vec3::new(1., 0., 0.),  8.08528)));
        assert_eq!(iter.next(), Some((Vec3::new(-1.,  1.,  0.),  11.434312631583936)));
        assert_eq!(iter.next(), Some((Vec3::new( 1., -1.,  0.),  11.434312631583936)));
        assert_eq!(iter.next(), Some((Vec3::new(-1., -1.,  0.),  11.434312631583937)));
        assert_eq!(iter.next(), Some((Vec3::new(-1.,  0., -1.),  11.434312631583937)));
        assert_eq!(iter.next(), Some((Vec3::new(-1.,  0.,  1.),  11.434312631583937)));
        assert_eq!(iter.next(), Some((Vec3::new( 0., -1., -1.),  11.434312631583937)));
        assert_eq!(iter.next(), Some((Vec3::new( 0., -1.,  1.),  11.434312631583937)));
        assert_eq!(iter.next(), Some((Vec3::new( 0.,  1., -1.),  11.434312631583937)));
        assert_eq!(iter.next(), Some((Vec3::new(0., 1., 1.),  11.434312631583937)));
        assert_eq!(iter.next(), Some((Vec3::new( 1.,  0., -1.),  11.434312631583937)));
        assert_eq!(iter.next(), Some((Vec3::new(1., 0., 1.),  11.434312631583937)));
        assert_eq!(iter.next(), Some((Vec3::new(1., 1., 0.),  11.434312631583937)));
        assert_eq!(iter.next(), Some((Vec3::new(-1.,  1., -1.),  14.00411575342049)));
        assert_eq!(iter.next(), Some((Vec3::new(-1.,  1.,  1.),  14.00411575342049)));
        assert_eq!(iter.next(), Some((Vec3::new( 1., -1., -1.),  14.00411575342049)));
        assert_eq!(iter.next(), Some((Vec3::new( 1., -1.,  1.),  14.00411575342049)));
        assert_eq!(iter.next(), Some((Vec3::new(-1., -1., -1.),  14.004115753420491)));
        assert_eq!(iter.next(), Some((Vec3::new(-1., -1.,  1.),  14.004115753420491)));
        assert_eq!(iter.next(), Some((Vec3::new( 1.,  1., -1.),  14.004115753420491)));
        assert_eq!(iter.next(), Some((Vec3::new(1., 1., 1.),  14.004115753420491)));
        assert_eq!(iter.next(), None);
    }

    const ATOL: f32 = 1e-3;
    const FM3M_CIF_DATA: &'static str = "# generated using pymatgen
data_test
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.59420000
_cell_length_b   3.59420000
_cell_length_c   3.59420000
_cell_angle_alpha   90.00000000
_cell_angle_beta    90.00000000
_cell_angle_gamma   90.00000000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Cu
_chemical_formula_sum   Cu4
_cell_volume   46.43085912
_cell_formula_units_Z   4
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_type_symbol
 _atom_type_oxidation_number
  Cu0+  0.0
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Cu0+  Cu1  1  0.00000000  0.00000000  0.00000000  1.0
  Cu0+  Cu1  1  0.00000000  0.50000000  0.50000000  1.0
  Cu0+  Cu1  1  0.50000000  0.00000000  0.50000000  1.0
  Cu0+  Cu1  1  0.50000000  0.50000000  0.00000000  1.0";

    const FM3M_EXPECTED: [(f32, f32); 4] = [
        (19.70066419257163, 1.0),
        (22.786339661733468, 0.4864883706070105),
        (32.444550987327624, 0.302926682750686),
        (38.244378463227896, 0.33030941984707324),
    ];

    #[test]
    fn fm3m_simulation_positions() {
        let d = CifParser::new(&FM3M_CIF_DATA)
            .parse()
            .expect("valid cif contents");
        let s = Structure::try_from(&d).expect("valid cif contents");
        let peaks = s.get_adxrd_peaks(0.71, &(5.0, 40.0), None);
        let peaks = peaks
            .iter()
            .map(|peak| {
                let (pos, intens, _) =
                    peak.get_adxrd_render_params(0.071, 0.0, 0.0, 0.0, 100.0, 1.0, 0.0, 180.0);
                (pos, intens)
            })
            .collect_vec();
        for ((s_pos, _), (a_pos, _)) in peaks.iter().zip(FM3M_EXPECTED) {
            let diff = (s_pos - a_pos).abs();
            assert!(diff < ATOL, "Simulated and actual positions difference exceeds tolerance. Simulated: {s_pos}, actual: {a_pos}. diff: {diff}");
        }
    }

    #[test]
    fn fm3m_simulation_intensities() {
        let d = CifParser::new(&FM3M_CIF_DATA)
            .parse()
            .expect("valid cif contents");
        let s = Structure::try_from(&d).expect("valid cif contents");

        let peaks = s.get_adxrd_peaks(0.71, &(5.0, 40.0), None);
        let mut peaks = peaks
            .iter()
            .map(|peak| {
                let (pos, intens, _) =
                    peak.get_adxrd_render_params(0.071, 0.0, 0.0, 0.0, 100.0, 1.0, 0.0, 180.0);
                (pos, intens)
            })
            .collect_vec();
        let max_peak = peaks
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("more than one peak")
            .1;

        for (_, i) in peaks.iter_mut() {
            *i /= max_peak;
        }

        for ((s_pos, s_intens), (_, a_intens)) in peaks.iter().zip(FM3M_EXPECTED) {
            let diff = (s_intens - a_intens).abs();
            assert!(diff < ATOL, "Simulated and actual intensities difference exceeds tolerance (at position {s_pos}). Simulated: {s_intens}, actual: {a_intens}. diff: {diff}");
        }
    }
}
