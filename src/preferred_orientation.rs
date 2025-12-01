use crate::math::linalg::{Vec3, Vec4};

use crate::lattice::Lattice;

#[derive(Clone, Debug)]
pub struct BinghamParams {
    pub orientation: Vec4<f64>,
    pub ks: Vec4<f64>,
}

/// orientation distribution for perferred orientation in sample coordinates
/// approximated via Kernel density estimation
///
/// * `orientation`: orientation of the distribution
/// * `axis_aligned_bingham_dist_samples`: axis aligned bingham distribution samples
/// * `kappa`: kernel width parmeter
#[derive(Clone, Debug)]
pub struct KDEBinghamODF {
    pub params: BinghamParams,
    pub axis_aligned_bingham_dist_samples: Vec<Vec4<f64>>,
    pub kappa: f64,
}

/// Compute the transformation from beam cordinates to sample coordinates
/// given a gonimeter phi and chi
///
/// * `chi`: goniometer chi in degrees
/// * `phi`: goniometer phi in degrees
///
/// - Beam z axis is chi rotation (beam direction)
/// - Beam y axis is phi rotation (up direction)
/// - Beam x axis is perpendicular to beam and up (duh!)
///
/// orientation distribution is relative to sample coordinates
/// therefore transform beam to sample coordinates using chi and phi
///
///```txt
/// detector
///  \
///    -
///     \ scattered   ^             ^
///       -     ray   | phi         | y
///         \         |             |
///           -       |             |
///      theta  \     |     z, chi  |
/// - - - - - - - [sample]<---------o x
/// ```
///
/// for the orientation transformation from beam to sample:
/// first, the rotate around beam z by chi
/// then, rotate around beam y by phi
fn get_beam_to_sample_tf(chi: f64, phi: f64) -> Vec4<f64> {
    let beam_chi = Vec4::quat_from_angle_axis(0.0, 0.0, 1.0, chi.to_radians());
    let beam_phi = Vec4::quat_from_angle_axis(0.0, 1.0, 0.0, phi.to_radians());

    // transform rotation of phi around global y to coordinates after chi rotation
    let chi_phi = beam_chi
        .quaternion_reciprocal()
        .quaternion_multiplication(&beam_phi);

    beam_chi.quaternion_multiplication(&chi_phi)
}

impl KDEBinghamODF {
    /// create a copy of self with rotated coordinates according to goniometer chi and phi
    ///
    /// this may be useful if weight needs to be called many times for different phi and chi
    pub fn with_orientation(&self, chi: f64, phi: f64) -> KDEBinghamODF {
        let mut ret = self.clone();

        // bingham distribution describes orientations of domains relative to sample
        //
        // transform bingham distribution's orientation from sample to beam coords

        let beam_to_sample = get_beam_to_sample_tf(chi, phi);
        let sample_to_bingham = &self.params.orientation;
        let beam_to_bingham = beam_to_sample.quaternion_multiplication(sample_to_bingham);

        for bingham_to_domain in ret.axis_aligned_bingham_dist_samples.iter_mut() {
            *bingham_to_domain = beam_to_bingham.quaternion_multiplication(bingham_to_domain);
        }

        ret.params.orientation = Vec4::new(1.0, 0.0, 0.0, 0.0);

        ret
    }

    pub fn weight_aligned(&self, hkl: &Vec3<f64>, lat: &Lattice) -> f64 {
        let norm_constant =
            self.kappa / (std::f64::consts::TAU * (self.kappa.exp() - (-self.kappa).exp()));

        let mut weight = 0.0;

        let hkl_in_domain_coords = lat.mat.matmul(&hkl);

        for bingham_to_domain in self.axis_aligned_bingham_dist_samples.iter() {
            let domain_to_beam = bingham_to_domain.unit_quaternion_recip_unchecked();
            let hkl_in_beam_coords =
                domain_to_beam.unit_quaternion_transform_unchecked(&hkl_in_domain_coords);

            let mag = hkl_in_beam_coords.magnitude();
            let dot_with_beam_z = hkl_in_beam_coords[2] / mag;

            // kernel density estimation using the von Mises-Fisher distribution
            // normalization is applied below
            weight += (self.kappa * dot_with_beam_z).exp();
        }

        weight *= norm_constant;
        weight /= self.axis_aligned_bingham_dist_samples.len() as f64;
        weight
    }

    /// compute the weight scaling of a hkl peak according to the domain orientation
    /// distribution described by this bingham ODF
    ///
    /// * `hkl`: hkl vector
    /// * `lat`: lattice for hkl vector
    /// * `chi`: goniometer chi in degrees
    /// * `phi`: goniometer phi in degrees
    /// * `rng`: random number generator
    pub fn weight(&self, hkl: &Vec3<f64>, lat: &Lattice, chi: f64, phi: f64) -> f64 {
        // bingham distribution describes orientations of domains relative to sample
        //
        // transform bingham distribution's orientation from sample to beam coords
        let beam_to_sample = get_beam_to_sample_tf(chi, phi);
        let sample_to_bingham = &self.params.orientation;
        let beam_to_bingham = beam_to_sample.quaternion_multiplication(sample_to_bingham);

        let hkl_in_domain_coords = lat.mat.matmul(&hkl);
        // now, we need to compute how well the distribution over physical hkl
        // directions aligns with the direction (beam z unit vector)
        //
        // sample many orientations (in beamline coordinates) and compute
        // the dot product of beam direction (beam coords z axis) with the hkl
        // vector in that orientation

        // Von Mises-Fisher distribution normalization constant
        let norm_constant =
            self.kappa / (std::f64::consts::TAU * (self.kappa.exp() - (-self.kappa).exp()));

        let mut weight = 0.0;

        for bingham_to_domain in self.axis_aligned_bingham_dist_samples.iter() {
            let beam_to_domain = beam_to_bingham.quaternion_multiplication(&bingham_to_domain);
            let domain_to_beam = beam_to_domain.unit_quaternion_recip_unchecked();
            let hkl_in_beam_coords = domain_to_beam.unit_quaternion_transform_unchecked(&hkl_in_domain_coords);

            let dot_with_beam_z = hkl_in_beam_coords.normalize()[2];

            // kernel density estimation using the von Mises-Fisher distribution
            // normalization is applied below
            weight += (self.kappa * dot_with_beam_z).exp();
        }
        weight *= norm_constant;
        weight /= self.axis_aligned_bingham_dist_samples.len() as f64;

        weight
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use crate::cfg::POCfg;
    use crate::lattice::Lattice;
    use crate::math::linalg::{Mat, Vec3, Vec4};

    const ATOL: f64 = 1e-5;

    #[test]
    fn test_transformed_ori() {
        let v = Vec3::new(1.0, 7.0, 3.0).normalize();
        let ori = Vec4::quat_from_angle_axis(v[0], v[1], v[2], 32.0f64.to_radians());
        let input = format!(
            "!DirectBingham
k: [1000, 0.5, 0.5, 1.0]
orientation: [{}, {}, {}, {}]
sampling: {{n: 30, kappa: 20}}
",
            ori[0], ori[1], ori[2], ori[3]
        );

        let pocfg: POCfg = serde_yaml::from_str(&input).expect("valid PO cfg");
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1128123);

        let bing = pocfg
            .try_into_generator(&mut rng)
            .expect("Valid parameters")
            .sample(&mut rng);

        let chi = 15.0f64;
        let phi = -20.0f64;
        let rotated = bing.with_orientation(chi, phi);

        let lattice = Lattice {
            mat: Mat::identity(),
        };
        let hkl = Vec3::new(1.0, 3.0, 2.0);
        let w0 = rotated.weight(&hkl, &lattice, 0.0, 0.0);
        let w1 = bing.weight(&hkl, &lattice, phi, chi);

        assert!((w0 - w1).abs() < ATOL);
    }

    #[test]
    fn test_transformed_ori_aligned() {
        let v = Vec3::new(1.0, 3.0, 3.0).normalize();
        let ori = Vec4::quat_from_angle_axis(v[0], v[1], v[2], 32.0f64.to_radians());
        let input = format!(
            "!DirectBingham
k: [1000, 0.5, 0.5, 1.0]
orientation: [{}, {}, {}, {}]
sampling: {{n: 1024, kappa: 20}}
",
            ori[0], ori[1], ori[2], ori[3]
        );
        let pocfg: POCfg = serde_yaml::from_str(&input).expect("valid PO cfg");
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1128123);

        let bing = pocfg
            .try_into_generator(&mut rng)
            .expect("Valid parameters")
            .sample(&mut rng);

        let chi = 15.0f64;
        let phi = -20.0f64;
        let rotated = bing.with_orientation(chi, phi);

        let lattice = Lattice {
            mat: Mat::identity(),
        };
        let hkl = Vec3::new(1.0, 3.0, 2.0);
        let w0 = rotated.weight_aligned(&hkl, &lattice);
        let w1 = bing.weight(&hkl, &lattice, chi, phi);

        assert!((w0 - w1).abs() < ATOL);
    }
}
