use crate::math::linalg::{ColVec, Mat3, Vec3};
use crate::math::quaternion::Quaternion;
use crate::math::{C_M_S, H_EV_S};

#[derive(Debug, Clone, PartialEq)]
pub enum DomainSize {
    Isotropic(f64),
    Ellipsoidal {
        q_ori: Quaternion,
        orientation: Mat3<f64>,
        evals: Vec3<f64>,
        main_sizes: Vec3<f64>,
        /// flattened lower triangular part of spd matrix describing ellipsoid
        /// order: [a00, a10, a11, a20, a21, a22]
        mat: ColVec<f64, 6>,
    },
}

/// Scherrer broadening constant
//
/// like GSAS-II, we use the volume-weighted domain size, and therefore
/// can set K to 1
const K: f64 = 1.0;

/// calculate scherrer broadening in angle dispersive XRD
///
/// * `wavelength`: wavelength in nanometers
/// * `theta_rad`: theta in radians
/// * `mean_ds`: mean domain size in nanometers
pub fn scherrer_broadening(wavelength: f64, theta: f64, mean_ds: f64) -> f64 {
    // scherrer
    // tau = k * lambda / (fwhm * cos(theta))
    // fwhm = k * lambda / (tau * cos(theta))
    (K * wavelength / (theta.cos() * mean_ds)).to_degrees()
}

/// calculate scherrer broadening in energy dispersive XRD
///
/// The method is also described in
/// Gerward, Leif, S. Mo/rup, and H. Topso/e.
/// "Particle size and strain broadening in energy‐dispersive x‐ray powder patterns."
/// Journal of Applied Physics 47.3 (1976): 822-825.
///
/// DOI: <https://doi.org/10.1063/1.322714>
///
/// * `wavelength`: wavelength in nanometers
/// * `theta_rad`: theta in radians
/// * `mean_ds`: mean domain size in nanometers
fn scherrer_broadening_edxrd(theta_rad: f64, mean_ds: f64) -> f64 {
    // $$\begin{align}
    // \tau     &= \frac{K \lambda}{\beta \cos(\theta)} \\
    // \lambda  &= \frac{hc}{E} \\
    // \beta    &= \frac{\Delta E}{E} \tan(\theta) \\
    // \tau     &= \frac{K \lambda}{ (\frac{\Delta E}{E} \tan(\theta) \cos(\theta))} \\
    // \tau     &= \frac{K \lambda}{ (\frac{\Delta E}{E} \sin(\theta))} \\
    // \tau     &= \frac{K h c}{E (\frac{\Delta E}{E} \sin(\theta))} \\
    // \tau     &= \frac{K h c}{(\Delta E \sin(\theta))} \\
    // \Delta E &= \frac{K h c}{\tau\sin\theta}
    // \end{align}$$

    return K * C_M_S * H_EV_S * 1e6 / (mean_ds * theta_rad.sin());
}

/// calculate the extent of an ellipsoid in a given direction
///
/// the ellipsoid is given by it's eigenvalue decomposition as an orientation
/// and the ellipsoid's axis lengths.
///
/// we can define the ellipsoid as x in R^3, where x.T A x = 1
///
/// using the eigenvalue/eigenvector decomposition A = Q.T D Q with diagonal
/// eigenvalue matrix D = diag(lambda_1, ... lambda_n) (= axis_lengths)
///
/// x.T Q.T D Q x = 1
/// (Q x).T D Q x = 1
///
/// we can rewrite x as a product of direction r in R^3 and length s in R > 0
/// x = s r
///
/// (Q s r).T D Q s r = 1
/// s^2 (Q r).T D Q r = 1
/// s = [(Q r).T D Q r]^-1/2
///
/// * `orientation`: orientation of the ellipsoid
/// * `pos`: non-normalized direction
/// * `axis_lengths`: inverted squared semi-axis lengths of the ellipsoid
fn ellipse_radius_for_direction(
    orientation: &Mat3<f64>,
    pos: &Vec3<f64>,
    axis_lengths: &Vec3<f64>,
) -> f64 {
    let norm_pos = pos.normalize();

    let r = &orientation.matmul(&norm_pos).map(|x| x * x) * axis_lengths;
    let r = r.iter_values().sum::<f64>().sqrt();
    r.recip()
}

impl DomainSize {
    /// Compute the domain size broadening using the edxrd schererr broadening formula
    ///
    /// For ellipsoidal domain size, the domain size in the relevant hkl direction is first
    /// computed using the method used in GSAS for it's phenomenological ellipsoidal domain
    /// size model
    ///
    /// * `theta_rad`:
    /// * `pos`:
    pub fn edxrd_broadening(&self, theta_rad: f64, pos: &Vec3<f64>) -> f64 {
        match self {
            DomainSize::Isotropic(mean_ds_nm) => scherrer_broadening_edxrd(theta_rad, *mean_ds_nm),
            DomainSize::Ellipsoidal {
                orientation,
                evals,
                q_ori: _,
                main_sizes: _,
                mat: _,
            } => {
                let r = ellipse_radius_for_direction(orientation, pos, evals);
                scherrer_broadening_edxrd(theta_rad, r)
            }
        }
    }

    /// Compute the domain size broadening using the edxrd schererr broadening formula
    ///
    /// For ellipsoidal domain size, the domain size in the relevant hkl direction is first
    /// computed using the method used in GSAS for it's phenomenological ellipsoidal domain
    /// size model
    ///
    /// * `wavelength_nm`:
    /// * `theta_hkl_rad`:
    /// * `pos`:
    pub fn adxrd_size_gamma_broadening(
        &self,
        wavelength_nm: f64,
        theta_hkl_rad: f64,
        pos: &Vec3<f64>,
    ) -> f64 {
        match self {
            DomainSize::Isotropic(mean_ds_nm) => {
                scherrer_broadening(wavelength_nm, theta_hkl_rad, *mean_ds_nm)
            }
            DomainSize::Ellipsoidal {
                orientation,
                evals,
                q_ori: _,
                main_sizes: _,
                mat: _,
            } => {
                let r = ellipse_radius_for_direction(orientation, pos, evals);
                scherrer_broadening(wavelength_nm, theta_hkl_rad, r)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ellipse_radius() {
        let orientation = Mat3::identity();
        // print(f"let ellipse_size = {ellipseSize(np.array([1., 3., 7.]), [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])}")
        let r = ellipse_radius_for_direction(
            &orientation,
            &Vec3::new(1., 3., 7.),
            &Vec3::new(1., 2., 3.),
        );

        let r_ = 0.5961725310235183;
        assert!((r - r_).abs() < 1e-3, "expected: {r_}, actual: {r}");
    }
}
