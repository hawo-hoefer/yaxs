use crate::math::linalg::{Mat3, Vec3, Vec4};
use crate::math::quaternion::Quaternion;
use crate::math::{C_M_S, H_EV_S};

#[derive(Debug, Clone, PartialEq)]
pub enum DomainSize {
    Isotropic(f64),
    Ellipsoidal {
        q_ori: Quaternion,
        orientation: Mat3<f64>,
        main_sizes: Vec3<f64>,
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

fn ellipse_radius_for_direction(
    orientation: &Mat3<f64>,
    pos: &Vec3<f64>,
    strength: &Vec3<f64>,
) -> f64 {
    let norm_pos = pos.normalize();
    // we have an ellipse described by matrix M indicating the domain size
    // in all spatial directions in reciprocal space as
    //
    // pos.T @ M @ pos = 1
    //
    // we can use an affine transformation of the unit sphere representing
    // the ellipsoid to get the corresponding position on the ellipsoid's surface
    // the length of that vector is the domain size.
    //
    // R = M @ normpos
    //   = V.T @ E @ V @ normpos
    // since M is symmetric, V (and V.T) are orthogonal matrices with
    // column vectors of length 1
    //
    // V @ R = V @ V.T @ E @ V @ normpos
    //       = E @ V @ normpos
    //
    // and since the columns of V are orthonormal unit vectors
    // (V is rotation or mirroring)
    // mag(V @ R) = mag(R)
    // therefore mag(V @ R) = mag(E @ V @ normpos)
    let r = orientation.matmul(&norm_pos) * strength;
    let r = r.magnitude();
    r
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
    pub fn edxrd_broadening(&self, theta_rad: f64, hkl: &Vec3<f64>) -> f64 {
        match self {
            DomainSize::Isotropic(mean_ds_nm) => scherrer_broadening_edxrd(theta_rad, *mean_ds_nm),
            DomainSize::Ellipsoidal {
                orientation,
                main_sizes,
                q_ori: _,
            } => {
                let r = ellipse_radius_for_direction(orientation, hkl, main_sizes);
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
        hkl: &Vec3<f64>,
    ) -> f64 {
        match self {
            DomainSize::Isotropic(mean_ds_nm) => {
                scherrer_broadening(wavelength_nm, theta_hkl_rad, *mean_ds_nm)
            }
            DomainSize::Ellipsoidal {
                orientation,
                main_sizes,
                q_ori: _,
            } => {
                let r = ellipse_radius_for_direction(orientation, hkl, main_sizes);
                scherrer_broadening(wavelength_nm, theta_hkl_rad, r)
            }
        }
    }
}
