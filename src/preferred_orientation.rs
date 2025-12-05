use std::mem::MaybeUninit;

use itertools::Itertools;

use crate::math::linalg::Vec3;
use crate::math::quaternion::Quaternion;

use crate::lattice::Lattice;

#[derive(Clone, Debug)]
pub struct BinghamParams {
    pub orientation: Quaternion,
    pub ks: Quaternion,
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
    pub axis_aligned_bingham_dist_samples: Vec<Quaternion>,
    pub norm_const: f64,
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
fn get_beam_to_sample_tf(chi: f64, phi: f64) -> Quaternion {
    let beam_chi = Quaternion::from_angle_axis(0.0, 0.0, 1.0, chi.to_radians() as f32);
    let beam_phi = Quaternion::from_angle_axis(0.0, 1.0, 0.0, phi.to_radians() as f32);

    // transform rotation of phi around global y to coordinates after chi rotation
    let chi_phi = beam_chi.recip().hamilton_product(&beam_phi);

    beam_chi.hamilton_product(&chi_phi)
}

impl KDEBinghamODF {
    pub fn new(
        orientation: Quaternion,
        ks: Quaternion,
        bingham_samples: Vec<Quaternion>,
        kappa: f64,
        norm_const: f64,
    ) -> Self {
        Self {
            params: BinghamParams { orientation, ks },
            axis_aligned_bingham_dist_samples: bingham_samples,
            norm_const,
            kappa,
        }
    }

    pub fn push_transformed_samples_into(&self, chi: f64, phi: f64, dst: &mut Vec<Quaternion>) {
        // bingham distribution describes orientations of domains relative to sample
        //
        // transform bingham distribution's orientation from sample to beam coords
        let beam_to_sample = get_beam_to_sample_tf(chi, phi);
        let sample_to_bingham = &self.params.orientation;
        let beam_to_bingham = beam_to_sample.hamilton_product(sample_to_bingham);

        for s in self
            .axis_aligned_bingham_dist_samples
            .iter()
            .map(|bingham_to_domain| {
                beam_to_bingham
                    .hamilton_product(bingham_to_domain)
                    .unit_recip_unchecked()
            })
        {
            dst.push(s)
        }
    }

    /// create a copy of self with rotated coordinates according to goniometer chi and phi
    ///
    /// this may be useful if weight needs to be called many times for different phi and chi
    pub fn with_orientation(&self, chi: f64, phi: f64) -> KDEBinghamODF {
        let mut samples = Vec::with_capacity(self.axis_aligned_bingham_dist_samples.len());
        self.push_transformed_samples_into(chi, phi, &mut samples);

        KDEBinghamODF::new(
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            self.params.ks.clone(),
            samples,
            self.kappa,
            self.norm_const,
        )
    }

    pub fn weight_aligned(&self, hkl_in_domain_coords: &Vec3<f64>) -> f64 {
        let hkl_in_domain_coords = hkl_in_domain_coords.map(|x| *x as f32).normalize();
        let mut weight = 0.0;
        let kappa = self.kappa as f32;

        for domain_to_beam in self.axis_aligned_bingham_dist_samples.iter() {
            let hkl_in_beam_coords = domain_to_beam.unit_transform_unchecked(&hkl_in_domain_coords);
            let dot_with_beam_z = hkl_in_beam_coords[2];

            // kernel density estimation using the von Mises-Fisher distribution
            // normalization is applied below
            weight += (kappa * dot_with_beam_z).exp();
        }

        weight as f64 * self.norm_const
    }

    /// compute the weight scaling of a hkl peak according to the domain orientation
    /// distribution described by this bingham ODF
    ///
    /// * `hkl`: hkl vector
    /// * `lat`: lattice for hkl vector
    /// * `chi`: goniometer chi in degrees
    /// * `phi`: goniometer phi in degrees
    /// * `rng`: random number generator
    pub fn weight(&self, pos: &Vec3<f64>, chi: f64, phi: f64) -> f64 {
        // bingham distribution describes orientations of domains relative to sample
        //
        // transform bingham distribution's orientation from sample to beam coords
        let beam_to_sample = get_beam_to_sample_tf(chi, phi);
        let sample_to_bingham = &self.params.orientation;
        let beam_to_bingham = beam_to_sample.hamilton_product(sample_to_bingham);

        let hkl_in_domain_coords = pos.map(|x| *x as f32).normalize();
        // now, we need to compute how well the distribution over physical hkl
        // directions aligns with the direction (beam z unit vector)
        //
        // sample many orientations (in beamline coordinates) and compute
        // the dot product of beam direction (beam coords z axis) with the hkl
        // vector in that orientation

        // Von Mises-Fisher distribution normalization constant
        let mut weight = 0.0;

        for bingham_to_domain in self.axis_aligned_bingham_dist_samples.iter() {
            let beam_to_domain = beam_to_bingham.hamilton_product(&bingham_to_domain);
            let domain_to_beam = beam_to_domain.unit_recip_unchecked();
            let hkl_in_beam_coords = domain_to_beam.unit_transform_unchecked(&hkl_in_domain_coords);

            let dot_with_beam_z = hkl_in_beam_coords[2];

            // kernel density estimation using the von Mises-Fisher distribution
            // normalization is applied below
            weight += (self.kappa as f32 * dot_with_beam_z).exp();
        }

        weight as f64 * self.norm_const
    }
}

#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use crate::cfg::POCfg;
    use crate::math::linalg::Vec3;
    use crate::math::quaternion::Quaternion;

    const ATOL: f64 = 1e-5;

    #[test]
    fn test_transformed_ori() {
        let v = Vec3::new(1.0, 7.0, 3.0).normalize();
        let ori = Quaternion::from_angle_axis(v[0], v[1], v[2], 32.0f32.to_radians());
        let input = format!(
            "!DirectBingham
k: [1000, 0.5, 0.5, 1.0]
orientation: [{}, {}, {}, {}]
sampling: {{n: 30, kappa: 20}}
",
            ori.w, ori.x, ori.y, ori.z
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

        let pos = Vec3::new(1.0, 3.0, 2.0);
        let w0 = rotated.weight(&pos, 0.0, 0.0);
        let w1 = bing.weight(&pos, phi, chi);

        assert!((w0 - w1).abs() < ATOL);
    }

    #[test]
    fn test_transformed_ori_aligned() {
        let v = Vec3::new(1.0, 3.0, 3.0).normalize();
        let ori = Quaternion::from_angle_axis(v[0], v[1], v[2], 32.0f32.to_radians());
        let input = format!(
            "!DirectBingham
k: [1000, 0.5, 0.5, 1.0]
orientation: [{}, {}, {}, {}]
sampling: {{n: 1024, kappa: 20}}
",
            ori.w, ori.x, ori.y, ori.z
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

        let pos = Vec3::new(1.0, 3.0, 2.0);
        let w0 = rotated.weight_aligned(&pos);
        let w1 = bing.weight(&pos, chi, phi);

        assert!((w0 - w1).abs() < ATOL);
    }
}
