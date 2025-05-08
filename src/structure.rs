use itertools::Itertools;
use std::collections::HashMap;

use nalgebra::{Complex, ComplexField, Matrix3, Vector3};
use ordered_float::NotNan;
use rand::Rng;

use crate::cif::CIFContents;
use crate::element::atomic_scattering_params;
use crate::pattern::Peak;
use crate::site::Site;

const TWO_THETA_ABSTOL: f64 = 1e-5;
const SCALED_INTENSITY_TOL: f64 = 1e-5;

#[derive(Debug, Clone, PartialEq)]
pub struct Lattice {
    pub mat: Matrix3<f64>,
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
    pub fn permute(&self, max_strain: f64, rng: &mut rand::rngs::StdRng) -> Structure {
        if max_strain == 0.0 {
            return self.clone();
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

        r
    }

    pub fn get_pattern(&self, wavelength_ams: f64, two_theta_range: &(f64, f64)) -> Vec<Peak> {
        let min_r = (two_theta_range.0 / 2.0).to_radians().sin() / wavelength_ams * 2.0;
        let max_r = (two_theta_range.1 / 2.0).to_radians().sin() / wavelength_ams * 2.0;

        let recip_lat = self.lat.recip_lattice_crystallographic();
        let recp_len = recip_lat.recip_lattice().abc();
        const RADIUS_TOL: f64 = 1e-8;
        let r_cells = max_r + 1e-8;
        let r_max =
            ((r_cells + 0.15) * recp_len / (2.0 * std::f64::consts::PI)).map(|x| x.ceil() as i32);

        let mut agg = HashMap::new();

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
            // bragg condition
            let theta = (wavelength_ams * g_hkl / 2.0).asin();

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
            // Lorentz polarization correction for hkl
            // lorentz_factor = (1 + math.cos(2 * theta) ** 2) / (math.sin(theta) ** 2 * math.cos(theta))
            let lorentz_fact =
                (1.0 + (2.0 * theta).cos().powi(2)) / (theta.sin().powi(2) * theta.cos());

            // # Intensity for hkl is modulus square of structure factor
            // i_hkl = (f_hkl * f_hkl.conjugate()).real
            let i_hkl = (f_hkl * f_hkl.conjugate()).real();
            let two_theta = NotNan::new(theta.to_degrees() * 2.0).unwrap();
            *agg.entry(two_theta).or_insert(NotNan::new(0.0).unwrap()) +=
                NotNan::new(i_hkl * lorentz_fact).unwrap();
        }

        let Some((_, vmax)) = agg.iter().max_by_key(|&(_, b)| b) else {
            return Vec::new();
        };
        let vmax = f64::from(*vmax);
        let agg = agg
            .iter()
            .sorted_by_key(|&(a, _)| a)
            .map(|(a, b)| (f64::from(*a), f64::from(*b)))
            .filter(|&(_, b)| b / vmax >= SCALED_INTENSITY_TOL)
            .collect_vec();

        let mut compressed: Vec<Peak> = Vec::with_capacity(agg.len() / 2 * 3);
        for (two_theta, intens) in agg.iter() {
            match compressed.last_mut() {
                Some(Peak {
                    pos: lt,
                    intensity: li,
                }) if ((*two_theta - *lt) < TWO_THETA_ABSTOL) => {
                    *li += *intens;
                }
                None | Some(&mut Peak { .. }) => compressed.push(Peak {
                    pos: *two_theta,
                    intensity: *intens,
                }),
            }
        }

        let volume = self.lat.volume() as f64;
        for peak in compressed.iter_mut() {
            peak.intensity = peak.intensity / volume.powi(2) * wavelength_ams.powi(3);
        }
        compressed
    }
}
