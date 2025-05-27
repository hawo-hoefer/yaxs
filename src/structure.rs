use itertools::Itertools;
use std::collections::HashMap;

use nalgebra::{Complex, ComplexField, Matrix3, Vector3};
use ordered_float::NotNan;
use rand::Rng;

use crate::cfg::SampleParameters;
use crate::cif::CIFContents;
use crate::math::e_kev_to_lambda_ams;
use crate::pattern::{Peak, Peaks};
use crate::preferred_orientation::MarchDollase;
use crate::site::Site;

const D_SPACING_ABSTOL_AMS: f64 = 1e-5;
const SCALED_INTENSITY_TOL: f64 = 1e-5;

#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    pub mat: Matrix3<f64>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Strain(pub [f64; 6]);
impl Strain {
    pub fn from_diag(a: f64, b: f64, c: f64) -> Self {
        Self([a, 0.0, b, 0.0, 0.0, c])
    }

    pub fn from_mat3(mat: &Matrix3<f64>) -> Self {
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
        Self([0.0; 6])
    }

    pub fn to_mat3(&self) -> Matrix3<f64> {
        let mut ret = Matrix3::zeros();
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
            mat: self.mat.try_inverse().unwrap().transpose() * 2.0 * std::f64::consts::PI,
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

    fn abc(&self) -> Vector3<f64> {
        let mut values = [0.0; 3];
        for (i, v) in self
            .mat
            .row_iter()
            .into_iter()
            .map(|x| x.iter().map(|a| a.powi(2)).sum::<f64>().sqrt())
            .enumerate()
        {
            values[i] = v;
        }
        Vector3::from([values[0], values[1], values[2]])
    }
}

impl std::fmt::Display for Lattice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lattice(\n")?;
        for row in self.mat.row_iter() {
            write!(f, "  {:5.2}, {:5.2}, {:5.2}\n", row[0], row[1], row[2])?;
        }
        write!(f, ")\n")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Structure {
    pub lat: Lattice,
    pub sites: Vec<Site>,
    pub sg_no: u8, // there are 230 space groups, so u8 should be enough
    pub sg_class: SGClass,
}

impl From<&CIFContents> for Structure {
    fn from(value: &CIFContents) -> Self {
        let (sg_no, sg_class) = value.get_sg_no_and_class();
        Structure {
            sites: value.get_sites(),
            lat: value.get_lattice(),
            sg_no,
            sg_class,
        }
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
    pub fn permute(&self, max_strain: f64, rng: &mut rand::rngs::StdRng) -> (Structure, Strain) {
        if max_strain == 0.0 {
            return (self.clone(), Strain::none());
        }

        let tensile_range = 1.0 - max_strain..=1.0 + max_strain;
        let shear_range = -max_strain..=max_strain;

        let strain_tensor: Matrix3<f64> = match self.sg_class {
            SGClass::Cubic => Matrix3::identity() * rng.random_range(tensile_range),
            SGClass::Orthorombic => {
                // all directions independent for orthorombic
                let mut m = Matrix3::zeros();
                m[(0, 0)] = rng.random_range(tensile_range.clone());
                m[(1, 1)] = rng.random_range(tensile_range.clone());
                m[(2, 2)] = rng.random_range(tensile_range.clone());
                m
            }
            SGClass::Monoclinic => {
                // one plane can be sheared
                let mut m = Matrix3::zeros();
                m[(0, 0)] = rng.random_range(tensile_range.clone());
                m[(1, 1)] = rng.random_range(tensile_range.clone());
                m[(2, 2)] = rng.random_range(tensile_range.clone());

                m[(1, 2)] = rng.random_range(shear_range.clone());
                m[(2, 1)] = m[(1, 2)];

                m
            }
            SGClass::Triclinic => {
                // symmetric tensor with all values free
                let mut m = Matrix3::zeros();
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
                let mut m = Matrix3::zeros();

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
                let mut m = Matrix3::zeros();

                m[(0, 0)] = rng.random_range(tensile_range.clone());
                m[(1, 1)] = m[(0, 0)];
                m[(2, 2)] = rng.random_range(tensile_range.clone());

                m
            }
        };

        let mut r = self.clone();
        r.lat.mat = r.lat.mat * strain_tensor;

        (r, Strain::from_mat3(&strain_tensor))
    }

    pub fn apply_strain(&self, strain: Strain) -> Structure {
        let mut ret = self.clone();

        ret.lat.mat = ret.lat.mat * strain.to_mat3();

        ret
    }

    /// scan lattice for crystallographic planes with given d-spacings
    /// compute peak intensities and d-spacings corresponding to the lattice planes
    /// miller indices are **not** returned
    ///
    /// * `min_r`: minimum d-spacing to consider
    /// * `max_r`: maximum d-spacing to consider
    /// * `po`: preferred orientation march-dollase parameters, if desired
    pub fn get_d_spacings_intensities(
        &self,
        min_r: f64,
        max_r: f64,
        po: Option<&MarchDollase>,
    ) -> Vec<Peak> {
        let recip_lat = self.lat.recip_lattice_crystallographic();
        let recp_len = recip_lat.recip_lattice().abc();
        const RADIUS_TOL: f64 = 1e-8;
        let r_cells = max_r + 1e-8;
        let r_max =
            ((r_cells + 0.15) * recp_len / (2.0 * std::f64::consts::PI)).map(|x| x.ceil() as i32);

        let mut agg = HashMap::<NotNan<f64>, (NotNan<f64>, Vec<Vector3<i16>>)>::new();

        let global_min = -max_r - RADIUS_TOL;
        let global_max = max_r + RADIUS_TOL;

        let n_min = -r_max;
        let n_max = r_max;
        for (hkl, g_hkl) in (n_min[0]..n_max[0])
            .cartesian_product(n_min[1]..n_max[1])
            .cartesian_product(n_min[2]..n_max[2])
            .filter_map(|((a, b), c)| -> Option<(Vector3<f64>, f64)> {
                let hkl = Vector3::<f64>::new(a as f64, b as f64, c as f64);
                let pos = recip_lat.mat * hkl;
                let g_hkl = pos.magnitude();

                // currently, we produce XRD patterns like pymatgen
                // Neighbor mapping from pymatgen.core.lattice.get_points_in_spheres
                // does not seem to have any effect if center_coords is the 0-vector
                // As far as I can tell, it only applies when center_coords are something
                // other than the 0-vector so we will ignore it for now.
                // i tested this using a modification of their code and random cifs from
                // the COD-database
                if (g_hkl < max_r + RADIUS_TOL && g_hkl > min_r - RADIUS_TOL)
                    && pos
                        .iter()
                        .map(|&x| (x > global_min) && (x < global_max))
                        .all(|x| x)
                {
                    Some((hkl, g_hkl))
                } else {
                    None
                }
            })
        {
            if g_hkl == 0.0 {
                continue;
            }
            let s = g_hkl / 2.0;
            let s2 = s.powi(2);

            let mut f_hkl = Complex::new(0.0, 0.0);
            for site in &self.sites {
                // g_dot_r = np.dot(frac_coords, np.transpose([hkl])).T[0]
                let g_dot_r: f64 = site.coords.dot(&hkl);
                for species in &site.species {
                    // el = site.specie
                    // coeff = ATOMIC_SCATTERING_PARAMS[el.symbol]
                    // fs = el.Z - 41.78214 * s2 * sum(
                    //     [d[0] * exp(-d[1] * s2) for d in coeff])
                    let z = species.el.z() as f64;
                    // TODO: alternate (possibly more correct scattering parameters)
                    use crate::element::atomic_scattering_params;
                    let coef = atomic_scattering_params(species.el).unwrap();
                    let sum: f64 = coef.iter().map(|d| d[0] * (-d[1] * s2).exp()).sum();
                    let fs = z - 41.78213 * s2 * sum;

                    // TODO: Debye-Waller Correction
                    // (we ignore it for now, in the test data we don't have DW-factors)
                    // dw_correction = np.exp(-dw_factors * s2)
                    let dw_correction = 1.0;

                    // f_hkl = np.sum(fs * occus * np.exp(2j * np.pi * g_dot_r) * dw_correction)
                    let f_part = fs
                        * site.occu
                        * Complex::new(0.0, -2.0 * std::f64::consts::PI * g_dot_r).exp()
                        * dw_correction;
                    f_hkl += f_part;
                }
            }
            // # Intensity for hkl is modulus square of structure factor
            let mut i_hkl = (f_hkl * f_hkl.conjugate()).real();
            if let Some(po) = po {
                let w = po.weight(&hkl, &self.lat);
                i_hkl *= w;
            }
            let d_spacing = 1.0 / g_hkl;
            let d_spacing = NotNan::new(d_spacing).expect("not nan");
            let (ref mut i_hkl_map, ref mut hkls_map) = agg
                .entry(d_spacing)
                .or_insert((NotNan::new(0.0).expect("valid float"), Vec::new()));
            *i_hkl_map += NotNan::try_from(i_hkl).expect("i_hkl may not be nan");
            hkls_map.push(hkl.map(|x| x as i16))
        }

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

    /// compute peak positions and intensities for angle dispersive XRD
    ///
    /// * `wavelength_ams`: wavelength for peaks in Amstrong
    /// * `two_theta_range`: two theta range to search for peaks in degrees
    /// * `po`: preferred orientation march-dollase parameters, if desired
    pub fn get_adxrd_peaks(
        &self,
        wavelength_ams: f64,
        two_theta_range: &(f64, f64),
        po: Option<&MarchDollase>,
    ) -> Vec<Peak> {
        let min_r = (two_theta_range.0 / 2.0).to_radians().sin() / wavelength_ams * 2.0;
        let max_r = (two_theta_range.1 / 2.0).to_radians().sin() / wavelength_ams * 2.0;

        self.get_d_spacings_intensities(min_r, max_r, po)
    }

    /// compute peak positions and intensities for energy dispersive XRD
    ///
    /// * `theta_deg`: fixed angle of sample to beam
    /// * `energy_kev_range`: energy range in keV to consider for d-spacings
    /// * `po`: preferred orientation march-dollase parameters, if desired
    pub fn get_edxrd_peaks(
        &self,
        theta_deg: f64,
        energy_kev_range: &(f64, f64),
        po: Option<&MarchDollase>,
    ) -> Vec<Peak> {
        let lambda_0 = e_kev_to_lambda_ams(energy_kev_range.1);
        let lambda_1 = e_kev_to_lambda_ams(energy_kev_range.0);

        let theta_rad = theta_deg.to_radians();

        let min_r = theta_rad.sin() / lambda_1 * 2.0;
        let max_r = theta_rad.sin() / lambda_0 * 2.0;

        self.get_d_spacings_intensities(min_r, max_r, po)
    }
}

/// Generate ADXRD Peaks for the input structures and their physical parameters.
///
/// * `sample_params`: physical parameter ranges for the structures
/// * `structures`: structures to simulate peaks for
/// * `two_theta_range`: two-theta range to generate peaks for
/// * `wavelength_ams`: wavelength to consider for peak generation
/// * `rng`: random number generator to use
///
/// returns tuple of Vec of Vecs, shape: \[structures, permutations_per_structure\]
pub fn simulate_peaks_angle_disperse(
    sample_params: &SampleParameters,
    structures: &[Structure],
    pref_o: &[&Option<MarchDollase>],
    two_theta_range: (f64, f64),
    wavelength_ams: f64,
    rng: &mut rand::rngs::StdRng,
) -> (Vec<Vec<Peaks>>, Vec<Vec<Strain>>) {
    let mut all_simulated_peaks = Vec::with_capacity(structures.len());
    let mut all_strains = Vec::with_capacity(structures.len());
    for (s, po) in structures.iter().zip(pref_o) {
        let mut permuted_phase_peaks = Vec::with_capacity(sample_params.structure_permutations);
        let mut strains = Vec::with_capacity(sample_params.structure_permutations);
        for _ in 0..sample_params.structure_permutations {
            let (perm_s, strain) = s.permute(sample_params.max_strain, rng);
            let peaks = perm_s
                .get_adxrd_peaks(wavelength_ams, &two_theta_range, po.as_ref())
                .into_boxed_slice();
            permuted_phase_peaks.push(peaks);
            strains.push(strain);
        }
        all_simulated_peaks.push(permuted_phase_peaks);
        all_strains.push(strains);
    }
    (all_simulated_peaks, all_strains)
}

/// Generate EDXRD Peaks for the input structures and their physical parameters.
///
/// * `sample_params`: physical parameter ranges for the structures
/// * `structures`: structures to simulate peaks for
/// * `two_theta_range`: two-theta range to generate peaks for
/// * `wavelength_ams`: wavelength to consider for peak generation
/// * `rng`: random number generator to use
///
/// returns tuple of Vec of Vecs, shape: \[structures, permutations_per_structure\]
pub fn simulate_peaks_energy_disperse(
    sample_params: &SampleParameters,
    structures: &[Structure],
    energy_kev_range: (f64, f64),
    theta_deg: f64,
    rng: &mut rand::rngs::StdRng,
) -> (Vec<Vec<Peaks>>, Vec<Vec<Strain>>) {
    let mut all_simulated_peaks = Vec::with_capacity(structures.len());
    let mut all_strains = Vec::with_capacity(structures.len());
    for s in structures.iter() {
        let mut permuted_phase_peaks = Vec::with_capacity(sample_params.structure_permutations);
        let mut strains = Vec::with_capacity(sample_params.structure_permutations);
        for _ in 0..sample_params.structure_permutations {
            let (perm_s, strain) = s.permute(sample_params.max_strain, rng);
            let peaks = perm_s
                .get_edxrd_peaks(theta_deg, &energy_kev_range, None)
                .into();
            permuted_phase_peaks.push(peaks);
            strains.push(strain);
        }
        all_simulated_peaks.push(permuted_phase_peaks);
        all_strains.push(strains);
    }
    (all_simulated_peaks, all_strains)
}
