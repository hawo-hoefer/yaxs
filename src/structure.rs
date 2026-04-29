use log::warn;
use num_complex::{Complex, ComplexFloat};
use std::collections::HashMap;

use ordered_float::NotNan;
use rand::Rng;

use crate::cfg::Parameter;
use crate::cif::CIFContents;
use crate::composition::FractionalComposition;
use crate::lattice::Lattice;
use crate::math::linalg::{Mat3, Vec3};
use crate::math::{e_kev_to_lambda_ams, funcs};
use crate::peak_sim::Alignment;
use crate::scatter::Scatter;
use crate::site::{Atom, Site};
use crate::strain::Strain;

// const D_SPACING_ABSTOL_AMS: f64 = 1e-5;
const SCALED_INTENSITY_TOL: f64 = 1e-6;
const DENSITY_RTOL: f64 = 1e-3;
const VOLUME_RTOL: f64 = 1e-3;

#[derive(Debug, Clone, PartialEq)]
/// A phase's crystallographic structure
///
/// * `lat`: lattice
/// * `sites`: sites in the structure
/// * `sg_no`: space group number
/// * `sg_class`: space group class
/// * `density`: density of the phase in g/cm3, if present in the cif
/// * `wt_composition`: composition of the structure by atomic weights
pub struct Structure {
    pub lat: Lattice,
    pub sites: Vec<Site>,
    pub sg_no: u8, // there are 230 space groups, so u8 should be enough
    pub sg_class: SGClass,
    pub density_g_cm3: f64,
    pub wt_composition: FractionalComposition,
}

pub trait HasDensity {
    fn density(&self) -> f64;
}

impl HasDensity for Structure {
    fn density(&self) -> f64 {
        return self.density_g_cm3;
    }
}

impl<'a> TryFrom<&CIFContents<'a>> for Structure {
    type Error = String;
    fn try_from(value: &CIFContents) -> Result<Self, Self::Error> {
        let (sg_no, sg_class) = value.get_sg_no_and_class()?;
        let lattice = value.get_lattice()?;
        let sites = value.get_sites()?;
        let weight_dalton = sites.iter().map(|s| s.weight_contribution()).sum::<f64>();

        let path = value.file_path;

        let volume = lattice.volume();
        if let Some(given_volume) = value.get_volume()? {
            if (given_volume - volume).abs() / given_volume > VOLUME_RTOL {
                warn!("{}: Calculated and given volume do not match. Calculated: {} ang^3. Given: {} ang^3. Using calculated volume.",
                    path,
                    volume,
                    given_volume
                )
            }
        } else {
            warn!(
                "No volume in cif {}. Using calculated value {} ang^3.",
                volume, path,
            )
        }

        let density_dalton_per_amstrong_cubed = weight_dalton / volume;
        const ANGSTROM3_TO_CM3: f64 = 1e-24;
        let calc_density_g_cm3 =
            density_dalton_per_amstrong_cubed * funcs::AMU_TO_G / ANGSTROM3_TO_CM3;

        let given_density = value.get_density()?;

        if let Some(given_density) = given_density {
            let rel_diff = (given_density - calc_density_g_cm3).abs() / calc_density_g_cm3;
            if rel_diff > DENSITY_RTOL {
                let file_path = value.file_path;
                warn!("{file_path}: Given and calculated densities do not match. Given: {given_density} g/cm3, calculated: {calc_density_g_cm3:.5}. Using calculated density.");
            }
        }

        Ok(Structure {
            wt_composition: FractionalComposition::from_sites(&sites),
            sites,
            lat: lattice,
            density_g_cm3: calc_density_g_cm3,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Peak {
    /// hkl vector of peak
    pub hkl: Vec3<f64>,
    /// cartesian coordinates of peak in reciprocal space
    pub pos: Vec3<f64>,
    pub i_hkl: NotNan<f64>,
    pub d_hkl: NotNan<f64>,
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

    pub fn structure_factor(
        &self,
        hkl: &Vec3<f64>,
        pos: &Vec3<f64>,
        d_hkl: f64,
        scattering_parameters: &HashMap<Atom, Scatter>,
    ) -> Complex<f64> {
        let mut f_hkl = Complex::new(0.0, 0.0);

        // n lambda = 2 d sin(theta) | n = 1
        // lambda = 2 d sin(theta)
        // 1 / (2 d) = sin(theta) / lambda
        let sin_theta_over_lambda = 1.0 / (2.0 * d_hkl);

        for site in &self.sites {
            // g_dot_r = np.dot(frac_coords, np.transpose([hkl])).T[0]
            let g_dot_r: f64 = site.coords.dot(&hkl);
            let dw_factor = site
                .displacement
                .as_ref()
                .map(|x| x.debye_waller_factor(&pos, sin_theta_over_lambda))
                .unwrap_or(1.0);

            for atom in &site.site_label {
                let scatter = &scattering_parameters[atom];
                let fs = scatter.eval(sin_theta_over_lambda);

                // f_hkl = np.sum(fs * occus * np.exp(2j * np.pi * g_dot_r) * dw_correction)
                let f_part = fs
                    * site.occu
                    * dw_factor
                    * Complex::new(0.0, std::f64::consts::TAU * g_dot_r).exp();
                f_hkl += f_part;
            }
        }
        f_hkl
    }

    pub fn get_hkl_intensities_spacings(
        &self,
        min_r: f64,
        max_r: f64,
        scattering_parameters: &HashMap<Atom, Scatter>,
        agg: Option<Vec<Peak>>,
    ) -> (usize, Vec<Peak>) {
        let mut agg = agg.unwrap_or(Vec::new());
        let mut n = 0;
        for (hkl, pos, g_hkl) in self.lat.iter_hkls(min_r, max_r) {
            n += 1;
            let d_hkl = 1.0 / g_hkl;

            let f_hkl = self.structure_factor(&hkl, &pos, d_hkl, scattering_parameters);

            // # Intensity for hkl is modulus square of structure factor
            let i_hkl = NotNan::new((f_hkl * f_hkl.conj()).re).expect("not nan");

            let d_spacing = NotNan::new(d_hkl).expect("not nan");
            agg.push(Peak {
                hkl,
                pos,
                i_hkl,
                d_hkl: d_spacing,
            });
        }

        (n, agg)
    }

    pub fn apply_alignment_to_peaks<'a>(&self, input: &mut [Peak], alignment: Alignment<'a>) {
        for p in input.iter_mut() {
            let w = alignment.weight(&p.pos);

            p.i_hkl = p.i_hkl * w;
        }
    }

    #[cfg(feature = "use-gpu")]
    pub fn apply_precomputed_weights_to_hkls_intensities(
        &self,
        input: &[Peak],
        i_hkls: &[f32],
    ) -> Vec<Peak> {
        let mut agg = Vec::new();

        assert_eq!(input.len(), i_hkls.len());

        for (p, i_hkl) in input.iter().zip(i_hkls) {
            // TODO: maybe make this an f64 again
            let i_hkl =
                NotNan::new(*i_hkl as f64).expect("Error in CUDA processing. Should not be NaN");

            let mut peak = p.clone();
            peak.i_hkl = i_hkl;
            agg.push(peak);
        }

        self.finalize_peaks(&mut agg);
        agg
    }

    pub fn finalize_peaks(&self, agg: &mut Vec<Peak>) {
        let Some(vmax) = agg.iter().map(|p| p.i_hkl).max().map(|x| *x) else {
            return;
        };

        agg.sort_by_key(|p| -p.i_hkl); // negative because we want descending

        let number_of_kept_peaks = agg
            .iter()
            .rposition(|p| p.i_hkl / vmax > SCALED_INTENSITY_TOL) // find smallest kept peak index
            .map(|x| x + 1) // keep up until that peak
            .unwrap_or(agg.len());

        agg.truncate(number_of_kept_peaks);

        let v2 = self.lat.volume().powi(2);
        for peak in agg.iter_mut() {
            peak.i_hkl = (*peak.i_hkl / v2).try_into().expect("volume is not nan");
        }
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
        scattering_parameters: &HashMap<Atom, Scatter>,
    ) -> Vec<Peak> {
        let min_r = (two_theta_range.0 / 2.0).to_radians().sin() / wavelength_ams * 2.0;
        let max_r = (two_theta_range.1 / 2.0).to_radians().sin() / wavelength_ams * 2.0;

        let mut peaks = self
            .get_hkl_intensities_spacings(min_r, max_r, scattering_parameters, None)
            .1;

        if let Some(alignment) = alignment {
            self.apply_alignment_to_peaks(&mut peaks, alignment);
        }

        self.finalize_peaks(&mut peaks);
        peaks
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
        scattering_parameters: &mut HashMap<Atom, Scatter>,
    ) -> Vec<Peak> {
        let lambda_0 = e_kev_to_lambda_ams(energy_kev_range.1);
        let lambda_1 = e_kev_to_lambda_ams(energy_kev_range.0);

        let theta_rad = theta_deg.to_radians();

        let min_r = theta_rad.sin() / lambda_1 * 2.0;
        let max_r = theta_rad.sin() / lambda_0 * 2.0;

        let mut peaks = self
            .get_hkl_intensities_spacings(min_r, max_r, scattering_parameters, None)
            .1;
        if let Some(alignment) = alignment {
            self.apply_alignment_to_peaks(&mut peaks, alignment);
        }
        self.finalize_peaks(&mut peaks);
        peaks
    }

    pub fn gather_scattering_params(&self, scattering_parameters: &mut HashMap<Atom, Scatter>) {
        for site in self.sites.iter() {
            for atom in &site.site_label {
                if !scattering_parameters.contains_key(atom) {
                    let scatter = atom.scattering_params().expect(
                        format!("Could not find atomic scattering parameter for {atom}").as_str(),
                    );
                    scattering_parameters.insert(atom.clone(), scatter);
                }
            }
        }
    }

    pub fn randomize_b_iso(
        &mut self,
        randomize_b_iso: &Option<Parameter<f64>>,
        rng: &mut impl Rng,
    ) -> Option<f64> {
        if let Some(range) = randomize_b_iso {
            let b_iso = range.generate(rng);
            for s in self.sites.iter_mut() {
                s.displacement = Some(crate::site::AtomicDisplacement::Biso(b_iso));
            }
            Some(b_iso)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::*;
    use crate::cif::CifParser;
    use crate::domain_size::DomainSize;
    use crate::lattice::Lattice;
    use crate::pattern::adxrd::InstrumentParameters;
    use crate::pattern::{lorentz_polarization_factor, PeakRenderParams};

    // #[test]
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
            .map(|(hkl, _, g_hkl)| (hkl, g_hkl))
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
    const FM3M_CIF_DATA: &'static str = include_str!("./test-aux/basic_fm3m.cif");
    const FM3M_EXPECTED: [(f32, f32); 4] = [
        (19.70066419257163, 1.0),
        (22.786339661733468, 0.4831601),
        (32.444550987327624, 0.286404),
        (38.244378463227896, 0.2998862),
    ];

    // #[test]
    fn fm3m_simulation_positions() {
        let mut p = CifParser::new(&FM3M_CIF_DATA);
        let d = p.parse().expect("valid cif contents");
        let s = Structure::try_from(&d).expect("valid cif contents");
        let mut sp_c = HashMap::new();
        s.gather_scattering_params(&mut sp_c);

        let peaks = s.get_adxrd_peaks(0.71, &(5.0, 40.0), None, &sp_c);
        let peaks = peaks
            .iter()
            .map(|peak| {
                #[rustfmt::skip]
                let PeakRenderParams { pos, intensity, .. } =
                    peak.get_adxrd_render_params(0.071, &InstrumentParameters::zero(), 1.0, &DomainSize::Isotropic(100.0), 1.0, 0.0, 0.0, 1.0, 0.0, 180.0, 0.0);
                (pos, intensity)
            })
            .collect_vec();
        for ((s_pos, _), (a_pos, _)) in peaks.iter().zip(FM3M_EXPECTED) {
            let diff = (s_pos - a_pos).abs();
            assert!(diff < ATOL, "Simulated and actual positions difference exceeds tolerance. Simulated: {s_pos}, actual: {a_pos}. diff: {diff}");
        }
    }

    // #[test]
    fn fm3m_simulation_intensities() {
        let mut p = CifParser::new(&FM3M_CIF_DATA);
        let d = p.parse().expect("valid cif contents");
        let s = Structure::try_from(&d).expect("valid cif contents");
        let mut sp_c = HashMap::new();
        s.gather_scattering_params(&mut sp_c);

        let peaks = s.get_adxrd_peaks(0.71, &(5.0, 40.0), None, &sp_c);
        let mut peaks = peaks
            .iter()
            .map(|peak| {
                let PeakRenderParams { pos, intensity, .. } = peak.get_adxrd_render_params(
                    0.071,
                    &InstrumentParameters::zero(),
                    1.0,
                    &DomainSize::Isotropic(100.0),
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    180.0,
                    0.0,
                );
                (pos, intensity)
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

    #[test]
    fn full_cubic_sio2_parse() {
        let cif_str = include_str!("./test-aux/COD_1010921.cif");
        let mut p = CifParser::new(cif_str);
        let cifcontents = p.parse().expect("valid cif");
        let s = Structure::try_from(&cifcontents).expect("valid cif contents");
        let abc = s.lat.abc();
        assert_eq!([abc[0], abc[1], abc[2]], [7.16, 7.16, 7.16]);
        assert!((s.lat.volume() - 367.1).abs() < 0.05);
        let wavelength = 1.5401;

        let mut params = HashMap::new();
        s.gather_scattering_params(&mut params);
        let expected_peaks = [
            (17.49695102818386, 0.450287669588085),
            (21.471551729819332, 100.0),
            (24.842526090147402, 0.06726952883705128),
            (27.830524801819138, 1.9931107060773383),
            (30.548542198284455, 3.5450122462521554),
            (35.41938048545049, 19.38908150497193),
            (37.64611736418609, 2.185497541184719),
            (39.76586388123188, 2.2427227222025485),
            (41.79525192643519, 1.8768033635889552),
            (43.74720388492278, 5.206383487412792),
            (45.63200087422881, 2.9053344256188782),
            (47.45798594160765, 2.858229040763249),
            (50.959939303224445, 0.6843203138811946),
            (52.646559385725794, 1.0365070506975869),
            (54.29609469085744, 1.3601243047829465),
            (55.91217406602048, 11.444851441131044),
            (57.49796838077741, 0.03376721945109315),
            (59.05627109567773, 1.6389637562272117),
            (60.589561756641785, 0.21765241253119685),
            (63.58974971296071, 10.049368160229424),
            (65.06044580341133, 0.23942484069930534),
            (66.51378819866414, 0.9357135426188037),
            (67.95128135523139, 4.240802297658308),
            (70.78415528150111, 1.8577610917896739),
            (72.18200843196134, 0.3921604326599308),
            (74.94612295295615, 4.414030942224158),
            (76.31441560473135, 0.2940107246492448),
            (77.67479531030001, 0.8442944458103518),
            (79.02815236705985, 7.655130876888721),
            (80.3753387287996, 0.25872174042465246),
            (81.71717350973307, 0.5631974086823814),
            (83.05444797903999, 1.4818598343052476),
            (85.7183689527306, 4.639413043101438),
            (87.04649832759972, 0.3782835564178739),
            (88.37304085251397, 0.0993889241097576),
            (89.69871140659426, 0.8305757889325137),
        ];

        let calculated_peaks = s.get_adxrd_peaks(wavelength, &(5.0, 90.0), None, &params);
        let mut render_params = calculated_peaks
            .iter()
            .map(|peak| {
                let theta_hkl_rad = peak.get_adxrd_theta_rad(wavelength);

                (
                    (theta_hkl_rad * 2.0).to_degrees() as f32,
                    peak.i_hkl * lorentz_polarization_factor(theta_hkl_rad, 90.0),
                )
            })
            .collect_vec();

        render_params.sort_by(|a, b| a.partial_cmp(b).expect("not nan"));

        let mut compressed = Vec::new();
        for rp in render_params.drain(..) {
            match compressed.last_mut() {
                None => compressed.push((rp.0, rp.1)),
                Some((pos, intens)) => {
                    if (*pos - rp.0).abs() < 1e-5 {
                        *intens += rp.1
                    } else {
                        compressed.push((rp.0, rp.1));
                    }
                }
            }
        }

        let imax = *compressed
            .iter()
            .map(|p| NotNan::new(p.1).expect("peak i_hkl is not nan"))
            .max()
            .expect("at least one peak");

        for (calc, expected) in compressed.iter().zip(expected_peaks) {
            let i_percent = (calc.1 / imax) * 100.0;
            println!(
                "{:.2} {:.2} | {:8.4} {:8.4}",
                calc.0, expected.0, i_percent, expected.1
            );
            assert_eq!(calc.0, expected.0);
            // assert_eq!(i_percent, expected.1)
        }
        // panic!()
    }

    #[test]
    fn full_trigonal_fe2o3_parse() {
        let cif_str = include_str!("./test-aux/COD_1011240.cif");
        let mut p = CifParser::new(cif_str);
        let cifcontents = p.parse().expect("valid cif");
        let s = Structure::try_from(&cifcontents).expect("valid cif contents");
        let abc = s.lat.abc();
        assert_eq!([abc[0], abc[1], abc[2]], [5.43, 5.43, 5.43]);
        assert!((s.lat.volume() - 100.8).abs() < 0.05);
        let wavelength = 1.5401;

        let mut params = HashMap::new();
        s.gather_scattering_params(&mut params);
        let expected_peaks = [
            (24.1284073682313, 36.459535694752766),
            (33.1314272795658, 100.0),
            (35.599345836763305, 81.10960106584682),
            (39.25274744993037, 2.3474653387645557),
            (43.469639552269335, 1.5937356854649734),
            (49.41869384975806, 43.25763352098),
            (54.02234393993578, 50.435217884956856),
            (57.39485528663899, 2.230388545851912),
            (57.54963466011139, 9.842495223638267),
            (62.37815621162154, 36.55585347897849),
            (63.939355780213674, 32.27563520358085),
            (69.53248902231903, 3.0954838022030655),
            (71.88464872641427, 13.579218500709953),
            (75.37866817836144, 8.731178777769928),
            (77.66163725071665, 3.8427408155029577),
            (80.49920778298006, 2.3356633618555547),
            (80.6314965122716, 4.632027327854496),
            (82.87476636393215, 6.792561787053426),
            (84.84509831466976, 10.114754158014378),
            (88.46475162233695, 9.978539279338403),
        ];

        let calculated_peaks = s.get_adxrd_peaks(wavelength, &(5.0, 90.0), None, &params);
        let mut render_params = calculated_peaks
            .iter()
            .map(|peak| {
                let theta_hkl_rad = peak.get_adxrd_theta_rad(wavelength);

                (
                    (theta_hkl_rad * 2.0).to_degrees() as f32,
                    peak.i_hkl * lorentz_polarization_factor(theta_hkl_rad, 0.0f64.to_radians()),
                )
            })
            .collect_vec();

        render_params.sort_by(|a, b| a.partial_cmp(b).expect("not nan"));

        let mut compressed = Vec::new();
        for rp in render_params.drain(..) {
            match compressed.last_mut() {
                None => compressed.push((rp.0, rp.1)),
                Some((pos, intens)) => {
                    if (*pos - rp.0).abs() < 1e-5 {
                        *intens += rp.1
                    } else {
                        compressed.push((rp.0, rp.1));
                    }
                }
            }
        }

        let imax = *compressed
            .iter()
            .map(|p| NotNan::new(p.1).expect("peak i_hkl is not nan"))
            .max()
            .expect("at least one peak");

        for (calc, expected) in compressed.iter().filter(|x| x.1 > 0.1).zip(expected_peaks) {
            let i_percent = (calc.1 / imax) * 100.0;
            println!(
                "{:.2} {:.2} | {:8.4} {:8.4}",
                calc.0, expected.0, i_percent, expected.1
            );
            assert_eq!(calc.0, expected.0);
            // assert_eq!(i_percent, expected.1)
        }
        // panic!()
    }

    #[test]
    fn full_orthorombic_sio2_parse() {
        let cif_str = include_str!("./test-aux/COD_4002439.cif");
        let mut p = CifParser::new(cif_str);
        let cifcontents = p.parse().expect("valid cif");
        let s = Structure::try_from(&cifcontents).expect("valid cif contents");
        let abc = s.lat.abc();

        assert_eq!([abc[0], abc[1], abc[2]], [14.17, 12.73, 10.32]);
        assert!((s.lat.volume() - 1861.564).abs() < 0.05);
        let wavelength = 1.5401;

        let mut params = HashMap::new();
        s.gather_scattering_params(&mut params);

        let calculated_peaks = s.get_adxrd_peaks(wavelength, &(5.0, 45.0), None, &params);
        let mut render_params = calculated_peaks
            .iter()
            .map(|peak| {
                let theta_hkl_rad = peak.get_adxrd_theta_rad(wavelength);

                (
                    (theta_hkl_rad * 2.0).to_degrees() as f32,
                    peak.i_hkl * lorentz_polarization_factor(theta_hkl_rad, 0.0f64.to_radians()),
                )
            })
            .collect_vec();

        render_params.sort_by(|a, b| a.partial_cmp(b).expect("not nan"));

        let mut compressed = Vec::new();
        for rp in render_params.drain(..) {
            match compressed.last_mut() {
                None => compressed.push((rp.0, rp.1)),
                Some((pos, intens)) => {
                    if (*pos - rp.0).abs() < 1e-3 {
                        *intens += rp.1
                    } else {
                        compressed.push((rp.0, rp.1));
                    }
                }
            }
        }

        let expected_peaks = [
            (12.479307205417374, 4.621283601959673),
            (12.672553156971224, 100.0),
            (13.89755213100284, 13.77783268228492),
            (16.680481757879974, 21.130246726261472),
            (17.1651342115969, 26.796260176391982),
            (18.71954453558844, 1.5094226810715432),
            (21.27774656197368, 2.2785241279069393),
            (21.81509252677841, 6.407847765955759),
            (23.03318354263619, 9.088896512420128),
            (23.49523477266814, 3.794423957696523),
            (25.924119791934316, 4.398578738734656),
            (27.4960496030187, 5.742930268853698),
            (27.556383490155536, 1.9061492541469744),
            (28.0049565836142, 3.112456691998427),
            (28.809927702556692, 3.5736569902583266),
            (29.187719010714677, 3.8408646857746684),
            (29.55977874088992, 1.5188110592310895),
            (29.676275891014967, 5.123838830321796),
            (30.76404088360144, 2.5395734901969287),
            (33.64786616124172, 1.8318935662440632),
            (33.728160089485876, 2.6410756087496803),
            (35.87571733826256, 2.6781695780658636),
        ];

        let imax = *compressed
            .iter()
            .map(|p| NotNan::new(p.1).expect("peak i_hkl is not nan"))
            .max()
            .expect("at least one peak");

        for (calc, expected) in compressed
            .iter()
            .filter(|x| x.1 / imax * 100.0 > 1.1)
            .zip(expected_peaks)
        {
            let i_percent = (calc.1 / imax) * 100.0;
            println!(
                "{:.2} {:.2} | {:8.4} {:8.4}",
                calc.0, expected.0, i_percent, expected.1
            );
            assert_eq!(calc.0, expected.0);
            // assert_eq!(i_percent, expected.1)
        }
    }
}
