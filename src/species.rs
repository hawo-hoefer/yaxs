use std::str::FromStr;

use crate::element::Element;
#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Atom {
    pub el: Element,
    pub ionization: i16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Species(pub Vec<Atom>);

impl IntoIterator for Species {
    type Item = Atom;

    type IntoIter = <Vec<Atom> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Species {
    type Item = &'a Atom;

    type IntoIter = <&'a Vec<Atom> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl FromStr for Species {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut species = Vec::new();
        let mut val = value;
        loop {
            if val.len() == 0 {
                break;
            }
            let (el, v) = Species::try_parse_single_element(val)?;
            let (ionization, v) = Species::try_parse_ionization(v).unwrap_or_else(|| (0, v));
            val = v;
            species.push(Atom { el, ionization })
        }
        Ok(Self(species))
    }
}

impl Species {
    fn try_parse_single_element(val: &str) -> Result<(Element, &str), String> {
        let mut c = 0;
        if let Some((el, _)) = val.split_once(|x: char| {
            c += 1;
            if !x.is_alphabetic() {
                return true;
            }

            if c > 1 && x.is_ascii_uppercase() {
                return true;
            }
            false
        }) {
            let parsed = Element::try_from(el)?;
            Ok((
                parsed,
                std::str::from_utf8(&val.as_bytes()[el.len()..]).unwrap(),
            ))
        } else {
            Ok((Element::try_from(val)?, &val[..0]))
        }
    }

    fn try_parse_ionization(val: &str) -> Option<(i16, &str)> {
        // check for single character ionization like Fe+ or Cu-
        match val.chars().next() {
            Some('+') => return Some((1, std::str::from_utf8(&val.as_bytes()[1..]).unwrap())),
            Some('-') => return Some((-1, std::str::from_utf8(&val.as_bytes()[1..]).unwrap())),
            Some('*') => todo!("'*' - ionization specifier"),
            Some('(') => todo!("ionization starting with '(' - ionization string: '{val}'"),
            Some(_) => (),
            None => return None,
        }

        // check for multiple ionization specified with number
        if let Some((ion, _)) = val.split_once(|x: char| !x.is_numeric()) {
            let mut ionization: i16 = ion.parse().ok()?;
            let mut rest = std::str::from_utf8(&val.as_bytes()[ion.len()..]).unwrap();
            match rest.chars().next() {
                Some('+') => rest = std::str::from_utf8(&rest.as_bytes()[1..]).unwrap(),
                Some('-') => {
                    ionization = -ionization;
                    rest = std::str::from_utf8(&rest.as_bytes()[1..]).unwrap();
                }
                Some(_) => (),
                None => (),
            }
            return Some((ionization, rest));
        }
        None
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_single_element_ok() {
        let (el, rest) = Species::try_parse_single_element("Fe").unwrap();
        assert_eq!(el, Element::Fe);
        assert_eq!(rest, "");
    }

    #[test]
    fn parse_single_element_ion_ok() {
        let (el, rest) = Species::try_parse_single_element("Fe3+").unwrap();
        assert_eq!(el, Element::Fe);
        assert_eq!(rest, "3+");
    }

    #[test]
    fn parse_single_element_ion_rest_ok() {
        let (el, rest) = Species::try_parse_single_element("Fe3+C-").unwrap();
        assert_eq!(el, Element::Fe);
        assert_eq!(rest, "3+C-");
    }

    #[test]
    fn parse_single_element_ion_rest_wrong_symbol() {
        Species::try_parse_single_element("Fear+C-").expect_err("");
    }

    #[test]
    fn parse_element_two_non_ions() {
        let (el, rest) = Species::try_parse_single_element("NiFe").unwrap();
        assert_eq!(el, Element::Ni);
        assert_eq!(rest, "Fe");
    }

    #[test]
    fn parse_ionization_positive() {
        let (el, rest) = Species::try_parse_ionization("3+").unwrap();
        assert_eq!(el, 3);
        assert_eq!(rest, "");
    }

    #[test]
    fn parse_ionization_negative() {
        let (el, rest) = Species::try_parse_ionization("3-").unwrap();
        assert_eq!(el, -3);
        assert_eq!(rest, "");
    }

    #[test]
    fn parse_ionization_positive_no_number() {
        let (el, rest) = Species::try_parse_ionization("+").unwrap();
        assert_eq!(el, 1);
        assert_eq!(rest, "");
    }

    #[test]
    fn parse_site_no_charge() {
        let site: Species = "FeNi".parse().unwrap();
        assert_eq!(
            site.0,
            [
                Atom {
                    el: Element::Fe,
                    ionization: 0
                },
                Atom {
                    el: Element::Ni,
                    ionization: 0
                }
            ]
        )
    }

    #[test]
    fn parse_site() {
        let species: Species = "Fe3+Ni2-".parse().unwrap();
        assert_eq!(
            species.0,
            [
                Atom {
                    el: Element::Fe,
                    ionization: 3
                },
                Atom {
                    el: Element::Ni,
                    ionization: -2
                }
            ]
        )
    }
}

pub struct Scatter {
    a_s: [f64; 4],
    b_s: [f64; 4],
    c: f64,
}

impl Scatter {
    // TODO: revisit this: https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
    pub fn eval(&self, g_hkl: f64) -> f64 {
        use std::f64::consts::PI;
        self.a_s
            .iter()
            .zip(self.b_s.iter())
            .map(|(a, b)| (*a) * (-(*b) * (g_hkl / (4.0 * PI)).powi(2)))
            .sum::<f64>()
            + self.c
    }
}

pub const fn atomic_scattering_params(a: &Atom) -> Option<Scatter> {
    use Element::*;

    #[rustfmt::skip]
    // Source for values
    //
    // Intensity of diffracted intensities, 
    // P. J. Brown, A. G. Fox, E. N. Maslen, M. A. O'Keefe and B. T. M. Willis. 
    // International Tables for Crystallography (2006). Vol. C, ch. 6.1, pp. 554-595
    // doi:10.1107/97809553602060000600
    //
    // there are some entries with -val or -v appended to the elements
    // i suspect this means valence valence atoms or something, but have chosen to ignore 
    // them for now. We'll see if we need them later.
    //
    // Most of these approximations are valid within 0.0 | 2.0 < sin(\theta) / lambda < 6.0
    // or something similar. FOR EDXRD, these are probably waaaaay outside what is considered
    // a reasonable approximation, and we may get bad results 
    let ret = match a {
        Atom { el: H, ionization: 0, }     => Some(Scatter { a_s: [0.489918, 0.262003, 0.196767, 0.049879], b_s: [20.6593, 7.74039, 49.5519, 2.20159], c: 0.001305, }),
        Atom { el: H, ionization: -1, }    => Some(Scatter { a_s: [0.897661, 0.565616, 0.415815, 0.116973], b_s: [53.1368, 15.187, 186.576, 3.56709], c: 0.002389, }),
        Atom { el: He, ionization: 0, }    => Some(Scatter { a_s: [0.8734, 0.6309, 0.3112, 0.178], b_s: [9.1037, 3.3568, 22.9276, 0.9821], c: 0.0064, }),
        Atom { el: Li, ionization: 0, }    => Some(Scatter { a_s: [1.1282, 0.7508, 0.6175, 0.4653], b_s: [3.9546, 1.0524, 85.3905, 168.261], c: 0.0377, }),
        Atom { el: Li, ionization: 1, }    => Some(Scatter { a_s: [0.6968, 0.7888, 0.3414, 0.1563], b_s: [4.6237, 1.9557, 0.6316, 10.0953], c: 0.0167, }),
        Atom { el: Be, ionization: 0, }    => Some(Scatter { a_s: [1.5919, 1.1278, 0.5391, 0.7029], b_s: [43.6427, 1.8623, 103.483, 0.542], c: 0.0385, }),
        Atom { el: Be, ionization: 2, }    => Some(Scatter { a_s: [6.2603, 0.8849, 0.7993, 0.1647], b_s: [0.0027, 0.8313, 2.2758, 5.1146], c: -6.1092, }),
        Atom { el: B, ionization: 0, }     => Some(Scatter { a_s: [2.0545, 1.3326, 1.0979, 0.7068], b_s: [23.2185, 1.021, 60.3498, 0.1403], c: -0.1932, }),
        Atom { el: C, ionization: 0, }     => Some(Scatter { a_s: [2.31, 1.02, 1.5886, 0.865], b_s: [20.8439, 10.2075, 0.5687, 51.6512], c: 0.2156, }),
        // Atom { el: Cval, ionization: 0, }  => Some(Scatter { a_s: [2.26069, 1.56165, 1.05075, 0.839259], b_s: [22.6907, 0.656665, 9.75618, 55.5949], c: 0.286977, }),
        Atom { el: N, ionization: 0, }     => Some(Scatter { a_s: [12.2126, 3.1322, 2.0125, 1.1663], b_s: [0.0057, 9.8933, 28.9975, 0.5826], c: -11.529, }),
        Atom { el: O, ionization: 0, }     => Some(Scatter { a_s: [3.0485, 2.2868, 1.5463, 0.867], b_s: [13.2771, 5.7011, 0.3239, 32.9089], c: 0.2508, }),
        Atom { el: O, ionization: -1, }    => Some(Scatter { a_s: [4.1916, 1.63969, 1.52673, -20.307], b_s: [12.8573, 4.17236, 47.0179, -0.01404], c: 21.9412, }),
        /// Source for O2-: https://doi.org/10.1107/S0365110X65003729
        Atom { el: O, ionization: -2, }    => Some(Scatter { a_s: [4.758, 3.637, 0.0, 0.0], b_s: [0.0496, 0.1903, 0.0, 0.0], c: 1.594, }),
        Atom { el: F, ionization: 0, }     => Some(Scatter { a_s: [3.5392, 2.6412, 1.517, 1.0243], b_s: [10.2825, 4.2944, 0.2615, 26.1476], c: 0.2776, }),
        Atom { el: F, ionization: -1, }    => Some(Scatter { a_s: [3.6322, 3.51057, 1.26064, 0.940706], b_s: [5.27756, 14.7353, 0.442258, 47.3437], c: 0.653396, }),
        Atom { el: Ne, ionization: 0, }    => Some(Scatter { a_s: [3.9553, 3.1125, 1.4546, 1.1251], b_s: [8.4042, 3.4262, 0.2306, 21.7184], c: 0.3515, }),
        Atom { el: Na, ionization: 0, }    => Some(Scatter { a_s: [4.7626, 3.1736, 1.2674, 1.1128], b_s: [3.285, 8.8422, 0.3136, 129.424], c: 0.676, }),
        Atom { el: Na, ionization: 1, }    => Some(Scatter { a_s: [3.2565, 3.9362, 1.3998, 1.0032], b_s: [2.6671, 6.1153, 0.2001, 14.039], c: 0.404, }),
        Atom { el: Mg, ionization: 0, }    => Some(Scatter { a_s: [5.4204, 2.1735, 1.2269, 2.3073], b_s: [2.8275, 79.2611, 0.3808, 7.1937], c: 0.8584, }),
        Atom { el: Mg, ionization: 2, }    => Some(Scatter { a_s: [3.4988, 3.8378, 1.3284, 0.8497], b_s: [2.1676, 4.7542, 0.185, 10.1411], c: 0.4853, }),
        Atom { el: Al, ionization: 0, }    => Some(Scatter { a_s: [6.4202, 1.9002, 1.5936, 1.9646], b_s: [3.0387, 0.7426, 31.5472, 85.0886], c: 1.1151, }),
        Atom { el: Al, ionization: 3, }    => Some(Scatter { a_s: [4.17448, 3.3876, 1.20296, 0.528137], b_s: [1.93816, 4.14553, 0.228753, 8.28524], c: 0.706786, }),
        // Atom { el: Siv, ionization: 0, }   => Some(Scatter { a_s: [6.2915, 3.0353, 1.9891, 1.541], b_s: [2.4386, 32.3337, 0.6785, 81.6937], c: 1.1407, }),
        // Atom { el: Sival, ionization: 0, } => Some(Scatter { a_s: [5.66269, 3.07164, 2.62446, 1.3932], b_s: [2.6652, 38.6634, 0.916946, 93.5458], c: 1.24707, }),
        Atom { el: Si, ionization: 4, }    => Some(Scatter { a_s: [4.43918, 3.20345, 1.19453, 0.41653], b_s: [1.64167, 3.43757, 0.2149, 6.65365], c: 0.746297, }),
        Atom { el: P, ionization: 0, }     => Some(Scatter { a_s: [6.4345, 4.1791, 1.78, 1.4908], b_s: [1.9067, 27.157, 0.526, 68.1645], c: 1.1149, }),
        Atom { el: S, ionization: 0, }     => Some(Scatter { a_s: [6.9053, 5.2034, 1.4379, 1.5863], b_s: [1.4679, 22.2151, 0.2536, 56.172], c: 0.8669, }),
        Atom { el: Cl, ionization: 0, }    => Some(Scatter { a_s: [11.4604, 7.1964, 6.2556, 1.6455], b_s: [0.0104, 1.1662, 18.5194, 47.7784], c: -9.5574, }),
        Atom { el: Cl, ionization: -1, }   => Some(Scatter { a_s: [18.2915, 7.2084, 6.5337, 2.3386], b_s: [0.0066, 1.1717, 19.5424, 60.4486], c: -16.378, }),
        Atom { el: Ar, ionization: 0, }    => Some(Scatter { a_s: [7.4845, 6.7723, 0.6539, 1.6442], b_s: [0.9072, 14.8407, 43.8983, 33.3929], c: 1.4445, }),
        Atom { el: K, ionization: 0, }     => Some(Scatter { a_s: [8.2186, 7.4398, 1.0519, 0.8659], b_s: [12.7949, 0.7748, 213.187, 41.6841], c: 1.4228, }),
        Atom { el: K, ionization: 1, }     => Some(Scatter { a_s: [7.9578, 7.4917, 6.359, 1.1915], b_s: [12.6331, 0.7674, -0.002, 31.9128], c: -4.9978, }),
        Atom { el: Ca, ionization: 0, }    => Some(Scatter { a_s: [8.6266, 7.3873, 1.5899, 1.0211], b_s: [10.4421, 0.6599, 85.7484, 178.437], c: 1.3751, }),
        Atom { el: Ca, ionization: 2, }    => Some(Scatter { a_s: [15.6348, 7.9518, 8.4372, 0.8537], b_s: [-0.0074, 0.6089, 10.3116, 25.9905], c: -14.875, }),
        Atom { el: Sc, ionization: 0, }    => Some(Scatter { a_s: [9.189, 7.3679, 1.6409, 1.468], b_s: [9.0213, 0.5729, 136.108, 51.3531], c: 1.3329, }),
        Atom { el: Sc, ionization: 3, }    => Some(Scatter { a_s: [13.4008, 8.0273, 1.65943, 1.57936], b_s: [0.29854, 7.9629, -0.28604, 16.0662], c: -6.6667, }),
        Atom { el: Ti, ionization: 0, }    => Some(Scatter { a_s: [9.7595, 7.3558, 1.6991, 1.9021], b_s: [7.8508, 0.5, 35.6338, 116.105], c: 1.2807, }),
        Atom { el: Ti, ionization: 2, }    => Some(Scatter { a_s: [9.11423, 7.62174, 2.2793, 0.087899], b_s: [7.5243, 0.457585, 19.5361, 61.6558], c: 0.897155, }),
        Atom { el: Ti, ionization: 3, }    => Some(Scatter { a_s: [17.7344, 8.73816, 5.25691, 1.92134], b_s: [0.22061, 7.04716, -0.15762, 15.9768], c: -14.652, }),
        Atom { el: Ti, ionization: 4, }    => Some(Scatter { a_s: [19.5114, 8.23473, 2.01341, 1.5208], b_s: [0.178847, 6.67018, -0.29263, 12.9464], c: -13.28, }),
        Atom { el: V, ionization: 0, }     => Some(Scatter { a_s: [10.2971, 7.3511, 2.0703, 2.0571], b_s: [6.8657, 0.4385, 26.8938, 102.478], c: 1.2199, }),
        Atom { el: V, ionization: 2, }     => Some(Scatter { a_s: [10.106, 7.3541, 2.2884, 0.0223], b_s: [6.8818, 0.4409, 20.3004, 115.122], c: 1.2298, }),
        Atom { el: V, ionization: 3, }     => Some(Scatter { a_s: [9.43141, 7.7419, 2.15343, 0.016865], b_s: [6.39535, 0.383349, 15.1908, 63.969], c: 0.656565, }),
        Atom { el: V, ionization: 5, }     => Some(Scatter { a_s: [15.6887, 8.14208, 2.03081, -9.576], b_s: [0.679003, 5.40135, 9.97278, 0.940464], c: 1.7143, }),
        Atom { el: Cr, ionization: 0, }    => Some(Scatter { a_s: [10.6406, 7.3537, 3.324, 1.4922], b_s: [6.1038, 0.392, 20.2626, 98.7399], c: 1.1832, }),
        Atom { el: Cr, ionization: 2, }    => Some(Scatter { a_s: [9.54034, 7.7509, 3.58274, 0.509107], b_s: [5.66078, 0.344261, 13.3075, 32.4224], c: 0.616898, }),
        Atom { el: Cr, ionization: 3, }    => Some(Scatter { a_s: [9.6809, 7.81136, 2.87603, 0.113575], b_s: [5.59463, 0.334393, 12.8288, 32.8761], c: 0.518275, }),
        Atom { el: Mn, ionization: 0, }    => Some(Scatter { a_s: [11.2819, 7.3573, 3.0193, 2.2441], b_s: [5.3409, 0.3432, 17.8674, 83.7543], c: 1.0896, }),
        Atom { el: Mn, ionization: 2, }    => Some(Scatter { a_s: [10.8061, 7.362, 3.5268, 0.2184], b_s: [5.2796, 0.3435, 14.343, 41.3235], c: 1.0874, }),
        Atom { el: Mn, ionization: 3, }    => Some(Scatter { a_s: [9.84521, 7.87194, 3.56531, 0.323613], b_s: [4.91797, 0.294393, 10.8171, 24.1281], c: 0.393974, }),
        Atom { el: Mn, ionization: 4, }    => Some(Scatter { a_s: [9.96253, 7.97057, 2.76067, 0.054447], b_s: [4.8485, 0.283303, 10.4852, 27.573], c: 0.251877, }),
        Atom { el: Fe, ionization: 0, }    => Some(Scatter { a_s: [11.7695, 7.3573, 3.5222, 2.3045], b_s: [4.7611, 0.3072, 15.3535, 76.8805], c: 1.0369, }),
        Atom { el: Fe, ionization: 2, }    => Some(Scatter { a_s: [11.0424, 7.374, 4.1346, 0.4399], b_s: [4.6538, 0.3053, 12.0546, 31.2809], c: 1.0097, }),
        Atom { el: Fe, ionization: 3, }    => Some(Scatter { a_s: [11.1764, 7.3863, 3.3948, 0.0724], b_s: [4.6147, 0.3005, 11.6729, 38.5566], c: 0.9707, }),
        Atom { el: Co, ionization: 0, }    => Some(Scatter { a_s: [12.2841, 7.3409, 4.0034, 2.3488], b_s: [4.2791, 0.2784, 13.5359, 71.1692], c: 1.0118, }),
        Atom { el: Co, ionization: 2, }    => Some(Scatter { a_s: [11.2296, 7.3883, 4.7393, 0.7108], b_s: [4.1231, 0.2726, 10.2443, 25.6466], c: 0.9324, }),
        Atom { el: Co, ionization: 3, }    => Some(Scatter { a_s: [10.338, 7.88173, 4.76795, 0.725591], b_s: [3.90969, 0.238668, 8.35583, 18.3491], c: 0.286667, }),
        Atom { el: Ni, ionization: 0, }    => Some(Scatter { a_s: [12.8376, 7.292, 4.4438, 2.38], b_s: [3.8785, 0.2565, 12.1763, 66.3421], c: 1.0341, }),
        Atom { el: Ni, ionization: 2, }    => Some(Scatter { a_s: [11.4166, 7.4005, 5.3442, 0.9773], b_s: [3.6766, 0.2449, 8.873, 22.1626], c: 0.8614, }),
        Atom { el: Ni, ionization: 3, }    => Some(Scatter { a_s: [10.7806, 7.75868, 5.22746, 0.847114], b_s: [3.5477, 0.22314, 7.64468, 16.9673], c: 0.386044, }),
        Atom { el: Cu, ionization: 0, }    => Some(Scatter { a_s: [13.338, 7.1676, 5.6158, 1.6735], b_s: [3.5828, 0.247, 11.3966, 64.8126], c: 1.191, }),
        Atom { el: Cu, ionization: 1, }    => Some(Scatter { a_s: [11.9475, 7.3573, 6.2455, 1.5578], b_s: [3.3669, 0.2274, 8.6625, 25.8487], c: 0.89, }),
        Atom { el: Cu, ionization: 2, }    => Some(Scatter { a_s: [11.8168, 7.11181, 5.78135, 1.14523], b_s: [3.37484, 0.244078, 7.9876, 19.897], c: 1.14431, }),
        Atom { el: Zn, ionization: 0, }    => Some(Scatter { a_s: [14.0743, 7.0318, 5.1652, 2.41], b_s: [3.2655, 0.2333, 10.3163, 58.7097], c: 1.3041, }),
        Atom { el: Zn, ionization: 2, }    => Some(Scatter { a_s: [11.9719, 7.3862, 6.4668, 1.394], b_s: [2.9946, 0.2031, 7.0826, 18.0995], c: 0.7807, }),
        Atom { el: Ga, ionization: 0, }    => Some(Scatter { a_s: [15.2354, 6.7006, 4.3591, 2.9623], b_s: [3.0669, 0.2412, 10.7805, 61.4135], c: 1.7189, }),
        Atom { el: Ga, ionization: 3, }    => Some(Scatter { a_s: [12.692, 6.69883, 6.06692, 1.0066], b_s: [2.81262, 0.22789, 6.36441, 14.4122], c: 1.53545, }),
        Atom { el: Ge, ionization: 0, }    => Some(Scatter { a_s: [16.0816, 6.3747, 3.7068, 3.683], b_s: [2.8509, 0.2516, 11.4468, 54.7625], c: 2.1313, }),
        Atom { el: Ge, ionization: 4, }    => Some(Scatter { a_s: [12.9172, 6.70003, 6.06791, 0.859041], b_s: [2.53718, 0.205855, 5.47913, 11.603], c: 1.45572, }),
        Atom { el: As, ionization: 0, }    => Some(Scatter { a_s: [16.6723, 6.0701, 3.4313, 4.2779], b_s: [2.6345, 0.2647, 12.9479, 47.7972], c: 2.531, }),
        Atom { el: Se, ionization: 0, }    => Some(Scatter { a_s: [17.0006, 5.8196, 3.9731, 4.3543], b_s: [2.4098, 0.2726, 15.2372, 43.8163], c: 2.8409, }),
        Atom { el: Br, ionization: 0, }    => Some(Scatter { a_s: [17.1789, 5.2358, 5.6377, 3.9851], b_s: [2.1723, 16.5796, 0.2609, 41.4328], c: 2.9557, }),
        Atom { el: Br, ionization: -1, }   => Some(Scatter { a_s: [17.1718, 6.3338, 5.5754, 3.7272], b_s: [2.2059, 19.3345, 0.2871, 58.1535], c: 3.1776, }),
        Atom { el: Kr, ionization: 0, }    => Some(Scatter { a_s: [17.3555, 6.7286, 5.5493, 3.5375], b_s: [1.9384, 16.5623, 0.2261, 39.3972], c: 2.825, }),
        Atom { el: Rb, ionization: 0, }    => Some(Scatter { a_s: [17.1784, 9.6435, 5.1399, 1.5292], b_s: [1.7888, 17.3151, 0.2748, 164.934], c: 3.4873, }),
        Atom { el: Rb, ionization: 1, }    => Some(Scatter { a_s: [17.5816, 7.6598, 5.8981, 2.7817], b_s: [1.7139, 14.7957, 0.1603, 31.2087], c: 2.0782, }),
        Atom { el: Sr, ionization: 0, }    => Some(Scatter { a_s: [17.5663, 9.8184, 5.422, 2.6694], b_s: [1.5564, 14.0988, 0.1664, 132.376], c: 2.5064, }),
        Atom { el: Sr, ionization: 2, }    => Some(Scatter { a_s: [18.0874, 8.1373, 2.5654, -34.193], b_s: [1.4907, 12.6963, 24.5651, -0.0138], c: 41.4025, }),
        Atom { el: Y, ionization: 0, }     => Some(Scatter { a_s: [17.776, 10.2946, 5.72629, 3.26588], b_s: [1.4029, 12.8006, 0.125599, 104.354], c: 1.91213, }),
        Atom { el: Y, ionization: 3, }     => Some(Scatter { a_s: [17.9268, 9.1531, 1.76795, -33.108], b_s: [1.35417, 11.2145, 22.6599, -0.01319], c: 40.2602, }),
        Atom { el: Zr, ionization: 0, }    => Some(Scatter { a_s: [17.8765, 10.948, 5.41732, 3.65721], b_s: [1.27618, 11.916, 0.117622, 87.6627], c: 2.06929, }),
        Atom { el: Zr, ionization: 4, }    => Some(Scatter { a_s: [18.1668, 10.0562, 1.01118, -2.6479], b_s: [1.2148, 10.1483, 21.6054, -0.10276], c: 9.41454, }),
        Atom { el: Nb, ionization: 0, }    => Some(Scatter { a_s: [17.6142, 12.0144, 4.04183, 3.53346], b_s: [1.18865, 11.766, 0.204785, 69.7957], c: 3.75591, }),
        Atom { el: Nb, ionization: 3, }    => Some(Scatter { a_s: [19.8812, 18.0653, 11.0177, 1.94715], b_s: [0.019175, 1.13305, 10.1621, 28.3389], c: -12.912, }),
        Atom { el: Nb, ionization: 5, }    => Some(Scatter { a_s: [17.9163, 13.3417, 10.799, 0.337905], b_s: [1.12446, 0.028781, 9.28206, 25.7228], c: -6.3934, }),
        Atom { el: Mo, ionization: 0, }    => Some(Scatter { a_s: [3.7025, 17.2356, 12.8876, 3.7429], b_s: [0.2772, 1.0958, 11.004, 61.6584], c: 4.3875, }),
        Atom { el: Mo, ionization: 3, }    => Some(Scatter { a_s: [21.1664, 18.2017, 11.7423, 2.30951], b_s: [0.014734, 1.03031, 9.53659, 26.6307], c: -14.421, }),
        Atom { el: Mo, ionization: 5, }    => Some(Scatter { a_s: [21.0149, 18.0992, 11.4632, 0.740625], b_s: [0.014345, 1.02238, 8.78809, 23.3452], c: -14.316, }),
        Atom { el: Mo, ionization: 6, }    => Some(Scatter { a_s: [17.8871, 11.175, 6.57891, 0.0], b_s: [1.03649, 8.48061, 0.058881, 0.0], c: 0.344941, }),
        Atom { el: Tc, ionization: 0, }    => Some(Scatter { a_s: [19.1301, 11.0948, 4.64901, 2.71263], b_s: [0.864132, 8.14487, 21.5707, 86.8472], c: 5.40428, }),
        Atom { el: Ru, ionization: 0, }    => Some(Scatter { a_s: [19.2674, 12.9182, 4.86337, 1.56756], b_s: [0.80852, 8.43467, 24.7997, 94.2928], c: 5.37874, }),
        Atom { el: Ru, ionization: 3, }    => Some(Scatter { a_s: [18.5638, 13.2885, 9.32602, 3.00964], b_s: [0.847329, 8.37164, 0.017662, 22.887], c: -3.1892, }),
        Atom { el: Ru, ionization: 4, }    => Some(Scatter { a_s: [18.5003, 13.1787, 4.71304, 2.18535], b_s: [0.844582, 8.12534, 0.36495, 20.8504], c: 1.42357, }),
        Atom { el: Rh, ionization: 0, }    => Some(Scatter { a_s: [19.2957, 14.3501, 4.73425, 1.28918], b_s: [0.751536, 8.21758, 25.8749, 98.6062], c: 5.328, }),
        Atom { el: Rh, ionization: 3, }    => Some(Scatter { a_s: [18.8785, 14.1259, 3.32515, -6.1989], b_s: [0.764252, 7.84438, 21.2487, -0.01036], c: 11.8678, }),
        Atom { el: Rh, ionization: 4, }    => Some(Scatter { a_s: [18.8545, 13.9806, 2.53464, -5.6526], b_s: [0.760825, 7.62436, 19.3317, -0.0102], c: 11.2835, }),
        Atom { el: Pd, ionization: 0, }    => Some(Scatter { a_s: [19.3319, 15.5017, 5.29537, 0.605844], b_s: [0.698655, 7.98929, 25.2052, 76.8986], c: 5.26593, }),
        Atom { el: Pd, ionization: 2, }    => Some(Scatter { a_s: [19.1701, 15.2096, 4.32234, 0.0], b_s: [0.696219, 7.55573, 22.5057, 0.0], c: 5.2916, }),
        Atom { el: Pd, ionization: 4, }    => Some(Scatter { a_s: [19.2493, 14.79, 2.89289, -7.9492], b_s: [0.683839, 7.14833, 17.9144, 0.005127], c: 13.0174, }),
        Atom { el: Ag, ionization: 0, }    => Some(Scatter { a_s: [19.2808, 16.6885, 4.8045, 1.0463], b_s: [0.6446, 7.4726, 24.6605, 99.8156], c: 5.179, }),
        Atom { el: Ag, ionization: 1, }    => Some(Scatter { a_s: [19.1812, 15.9719, 5.27475, 0.357534], b_s: [0.646179, 7.19123, 21.7326, 66.1147], c: 5.21572, }),
        Atom { el: Ag, ionization: 2, }    => Some(Scatter { a_s: [19.1643, 16.2456, 4.3709, 0.0], b_s: [0.645643, 7.18544, 21.4072, 0.0], c: 5.21404, }),
        Atom { el: Cd, ionization: 0, }    => Some(Scatter { a_s: [19.2214, 17.6444, 4.461, 1.6029], b_s: [0.5946, 6.9089, 24.7008, 87.4825], c: 5.0694, }),
        Atom { el: Cd, ionization: 2, }    => Some(Scatter { a_s: [19.1514, 17.2535, 4.47128, 0.0], b_s: [0.597922, 6.80639, 20.2521, 0.0], c: 5.11937, }),
        Atom { el: In, ionization: 0, }    => Some(Scatter { a_s: [19.1624, 18.5596, 4.2948, 2.0396], b_s: [0.5476, 6.3776, 25.8499, 92.8029], c: 4.9391, }),
        Atom { el: In, ionization: 3, }    => Some(Scatter { a_s: [19.1045, 18.1108, 3.78897, 0.0], b_s: [0.551522, 6.3247, 17.3595, 0.0], c: 4.99635, }),
        Atom { el: Sn, ionization: 0, }    => Some(Scatter { a_s: [19.1889, 19.1005, 4.4585, 2.4663], b_s: [5.8303, 0.5031, 26.8909, 83.9571], c: 4.7821, }),
        Atom { el: Sn, ionization: 2, }    => Some(Scatter { a_s: [19.1094, 19.0548, 4.5648, 0.487], b_s: [0.5036, 5.8378, 23.3752, 62.2061], c: 4.7861, }),
        Atom { el: Sn, ionization: 4, }    => Some(Scatter { a_s: [18.9333, 19.7131, 3.4182, 0.0193], b_s: [5.764, 0.4655, 14.0049, -0.7583], c: 3.9182, }),
        Atom { el: Sb, ionization: 0, }    => Some(Scatter { a_s: [19.6418, 19.0455, 5.0371, 2.6827], b_s: [5.3034, 0.4607, 27.9074, 75.2825], c: 4.5909, }),
        Atom { el: Sb, ionization: 3, }    => Some(Scatter { a_s: [18.9755, 18.933, 5.10789, 0.288753], b_s: [0.467196, 5.22126, 19.5902, 55.5113], c: 4.69626, }),
        Atom { el: Sb, ionization: 5, }    => Some(Scatter { a_s: [19.8685, 19.0302, 2.41253, 0.0], b_s: [5.44853, 0.467973, 14.1259, 0.0], c: 4.69263, }),
        Atom { el: Te, ionization: 0, }    => Some(Scatter { a_s: [19.9644, 19.0138, 6.14487, 2.5239], b_s: [4.81742, 0.420885, 28.5284, 70.8403], c: 4.352, }),
        Atom { el: I, ionization: 0, }     => Some(Scatter { a_s: [20.1472, 18.9949, 7.5138, 2.2735], b_s: [4.347, 0.3814, 27.766, 66.8776], c: 4.0712, }),
        Atom { el: I, ionization: -1, }    => Some(Scatter { a_s: [20.2332, 18.997, 7.8069, 2.8868], b_s: [4.3579, 0.3815, 29.5259, 84.9304], c: 4.0714, }),
        Atom { el: Xe, ionization: 0, }    => Some(Scatter { a_s: [20.2933, 19.0298, 8.9767, 1.99], b_s: [3.9282, 0.344, 26.4659, 64.2658], c: 3.7118, }),
        Atom { el: Cs, ionization: 0, }    => Some(Scatter { a_s: [20.3892, 19.1062, 10.662, 1.4953], b_s: [3.569, 0.3107, 24.3879, 213.904], c: 3.3352, }),
        Atom { el: Cs, ionization: 1, }    => Some(Scatter { a_s: [20.3524, 19.1278, 10.2821, 0.9615], b_s: [3.552, 0.3086, 23.7128, 59.4565], c: 3.2791, }),
        Atom { el: Ba, ionization: 0, }    => Some(Scatter { a_s: [20.3361, 19.297, 10.888, 2.6959], b_s: [3.216, 0.2756, 20.2073, 167.202], c: 2.7731, }),
        Atom { el: Ba, ionization: 2, }    => Some(Scatter { a_s: [20.1807, 19.1136, 10.9054, 0.77634], b_s: [3.21367, 0.28331, 20.0558, 51.746], c: 3.02902, }),
        Atom { el: La, ionization: 0, }    => Some(Scatter { a_s: [20.578, 19.599, 11.3727, 3.28719], b_s: [2.94817, 0.244475, 18.7726, 133.124], c: 2.14678, }),
        Atom { el: La, ionization: 3, }    => Some(Scatter { a_s: [20.2489, 19.3763, 11.6323, 0.336048], b_s: [2.9207, 0.250698, 17.8211, 54.9453], c: 2.4086, }),
        Atom { el: Ce, ionization: 0, }    => Some(Scatter { a_s: [21.1671, 19.7695, 11.8513, 3.33049], b_s: [2.81219, 0.226836, 17.6083, 127.113], c: 1.86264, }),
        Atom { el: Ce, ionization: 3, }    => Some(Scatter { a_s: [20.8036, 19.559, 11.9369, 0.612376], b_s: [2.77691, 0.23154, 16.5408, 43.1692], c: 2.09013, }),
        Atom { el: Ce, ionization: 4, }    => Some(Scatter { a_s: [20.3235, 19.8186, 12.1233, 0.144583], b_s: [2.65941, 0.21885, 15.7992, 62.2355], c: 1.5918, }),
        Atom { el: Pr, ionization: 0, }    => Some(Scatter { a_s: [22.044, 19.6697, 12.3856, 2.82428], b_s: [2.77393, 0.222087, 16.7669, 143.644], c: 2.0583, }),
        Atom { el: Pr, ionization: 3, }    => Some(Scatter { a_s: [21.3727, 19.7491, 12.1329, 0.97518], b_s: [2.6452, 0.214299, 15.323, 36.4065], c: 1.77132, }),
        Atom { el: Pr, ionization: 4, }    => Some(Scatter { a_s: [20.9413, 20.0539, 12.4668, 0.296689], b_s: [2.54467, 0.202481, 14.8137, 45.4643], c: 1.24285, }),
        Atom { el: Nd, ionization: 0, }    => Some(Scatter { a_s: [22.6845, 19.6847, 12.774, 2.85137], b_s: [2.66248, 0.210628, 15.885, 137.903], c: 1.98486, }),
        Atom { el: Nd, ionization: 3, }    => Some(Scatter { a_s: [21.961, 19.9339, 12.12, 1.51031], b_s: [2.52722, 0.199237, 14.1783, 30.8717], c: 1.47588, }),
        Atom { el: Pm, ionization: 0, }    => Some(Scatter { a_s: [23.3405, 19.6095, 13.1235, 2.87516], b_s: [2.5627, 0.202088, 15.1009, 132.721], c: 2.02876, }),
        Atom { el: Pm, ionization: 3, }    => Some(Scatter { a_s: [22.5527, 20.1108, 12.0671, 2.07492], b_s: [2.4174, 0.185769, 13.1275, 27.4491], c: 1.19499, }),
        Atom { el: Sm, ionization: 0, }    => Some(Scatter { a_s: [24.0042, 19.4258, 13.4396, 2.89604], b_s: [2.47274, 0.196451, 14.3996, 128.007], c: 2.20963, }),
        Atom { el: Sm, ionization: 3, }    => Some(Scatter { a_s: [23.1504, 20.2599, 11.9202, 2.71488], b_s: [2.31641, 0.174081, 12.1571, 24.8242], c: 0.954586, }),
        Atom { el: Eu, ionization: 0, }    => Some(Scatter { a_s: [24.6274, 19.0886, 13.7603, 2.9227], b_s: [2.3879, 0.1942, 13.7546, 123.174], c: 2.5745, }),
        Atom { el: Eu, ionization: 2, }    => Some(Scatter { a_s: [24.0063, 19.9504, 11.8034, 3.87243], b_s: [2.27783, 0.17353, 11.6096, 26.5156], c: 1.36389, }),
        Atom { el: Eu, ionization: 3, }    => Some(Scatter { a_s: [23.7497, 20.3745, 11.8509, 3.26503], b_s: [2.22258, 0.16394, 11.311, 22.9966], c: 0.759344, }),
        Atom { el: Gd, ionization: 0, }    => Some(Scatter { a_s: [25.0709, 19.0798, 13.8518, 3.54545], b_s: [2.25341, 0.181951, 12.9331, 101.398], c: 2.4196, }),
        Atom { el: Gd, ionization: 3, }    => Some(Scatter { a_s: [24.3466, 20.4208, 11.8708, 3.7149], b_s: [2.13553, 0.155525, 10.5782, 21.7029], c: 0.645089, }),
        Atom { el: Tb, ionization: 0, }    => Some(Scatter { a_s: [25.8976, 18.2185, 14.3167, 2.95354], b_s: [2.24256, 0.196143, 12.6648, 115.362], c: 3.58324, }),
        Atom { el: Tb, ionization: 3, }    => Some(Scatter { a_s: [24.9559, 20.3271, 12.2471, 3.773], b_s: [2.05601, 0.149525, 10.0499, 21.2773], c: 0.691967, }),
        Atom { el: Dy, ionization: 0, }    => Some(Scatter { a_s: [26.507, 17.6383, 14.5596, 2.96577], b_s: [2.1802, 0.202172, 12.1899, 111.874], c: 4.29728, }),
        Atom { el: Dy, ionization: 3, }    => Some(Scatter { a_s: [25.5395, 20.2861, 11.9812, 4.50073], b_s: [1.9804, 0.143384, 9.34972, 19.581], c: 0.68969, }),
        Atom { el: Ho, ionization: 0, }    => Some(Scatter { a_s: [26.9049, 17.294, 14.5583, 3.63837], b_s: [2.07051, 0.19794, 11.4407, 92.6566], c: 4.56796, }),
        Atom { el: Ho, ionization: 3, }    => Some(Scatter { a_s: [26.1296, 20.0994, 11.9788, 4.93676], b_s: [1.91072, 0.139358, 8.80018, 18.5908], c: 0.852795, }),
        Atom { el: Er, ionization: 0, }    => Some(Scatter { a_s: [27.6563, 16.4285, 14.9779, 2.98233], b_s: [2.07356, 0.223545, 11.3604, 105.703], c: 5.92046, }),
        Atom { el: Er, ionization: 3, }    => Some(Scatter { a_s: [26.722, 19.7748, 12.1506, 5.17379], b_s: [1.84659, 0.13729, 8.36225, 17.8974], c: 1.17613, }),
        Atom { el: Tm, ionization: 0, }    => Some(Scatter { a_s: [28.1819, 15.8851, 15.1542, 2.98706], b_s: [2.02859, 0.238849, 10.9975, 102.961], c: 6.75621, }),
        Atom { el: Tm, ionization: 3, }    => Some(Scatter { a_s: [27.3083, 19.332, 12.3339, 5.38348], b_s: [1.78711, 0.136974, 7.96778, 17.2922], c: 1.63929, }),
        Atom { el: Yb, ionization: 0, }    => Some(Scatter { a_s: [28.6641, 15.4345, 15.3087, 2.98963], b_s: [1.9889, 0.257119, 10.6647, 100.417], c: 7.56672, }),
        Atom { el: Yb, ionization: 2, }    => Some(Scatter { a_s: [28.1209, 17.6817, 13.3335, 5.14657], b_s: [1.78503, 0.15997, 8.18304, 20.39], c: 3.70983, }),
        Atom { el: Yb, ionization: 3, }    => Some(Scatter { a_s: [27.8917, 18.7614, 12.6072, 5.47647], b_s: [1.73272, 0.13879, 7.64412, 16.8153], c: 2.26001, }),
        Atom { el: Lu, ionization: 0, }    => Some(Scatter { a_s: [28.9476, 15.2208, 15.1, 3.71601], b_s: [1.90182, 9.98519, 0.261033, 84.3298], c: 7.97628, }),
        Atom { el: Lu, ionization: 3, }    => Some(Scatter { a_s: [28.4628, 18.121, 12.8429, 5.59415], b_s: [1.68216, 0.142292, 7.33727, 16.3535], c: 2.97573, }),
        Atom { el: Hf, ionization: 0, }    => Some(Scatter { a_s: [29.144, 15.1726, 14.7586, 4.30013], b_s: [1.83262, 9.5999, 0.275116, 72.029], c: 8.58154, }),
        Atom { el: Hf, ionization: 4, }    => Some(Scatter { a_s: [28.8131, 18.4601, 12.7285, 5.59927], b_s: [1.59136, 0.128903, 6.76232, 14.0366], c: 2.39699, }),
        Atom { el: Ta, ionization: 0, }    => Some(Scatter { a_s: [29.2024, 15.2293, 14.5135, 4.76492], b_s: [1.77333, 9.37046, 0.295977, 63.3644], c: 9.24354, }),
        Atom { el: Ta, ionization: 5, }    => Some(Scatter { a_s: [29.1587, 18.8407, 12.8268, 5.38695], b_s: [1.50711, 0.116741, 6.31524, 12.4244], c: 1.78555, }),
        Atom { el: W, ionization: 0, }     => Some(Scatter { a_s: [29.0818, 15.43, 14.4327, 5.11982], b_s: [1.72029, 9.2259, 0.321703, 57.056], c: 9.8875, }),
        Atom { el: W, ionization: 6, }     => Some(Scatter { a_s: [29.4936, 19.3763, 13.0544, 5.06412], b_s: [1.42755, 0.104621, 5.93667, 11.1972], c: 1.01074, }),
        Atom { el: Re, ionization: 0, }    => Some(Scatter { a_s: [28.7621, 15.7189, 14.5564, 5.44174], b_s: [1.67191, 9.09227, 0.3505, 52.0861], c: 10.472, }),
        Atom { el: Os, ionization: 0, }    => Some(Scatter { a_s: [28.1894, 16.155, 14.9305, 5.67589], b_s: [1.62903, 8.97948, 0.382661, 48.1647], c: 11.0005, }),
        Atom { el: Os, ionization: 4, }    => Some(Scatter { a_s: [30.419, 15.2637, 14.7458, 5.06795], b_s: [1.37113, 6.84706, 0.165191, 18.003], c: 6.49804, }),
        Atom { el: Ir, ionization: 0, }    => Some(Scatter { a_s: [27.3049, 16.7296, 15.6115, 5.83377], b_s: [1.59279, 8.86553, 0.417916, 45.0011], c: 11.4722, }),
        Atom { el: Ir, ionization: 3, }    => Some(Scatter { a_s: [30.4156, 15.862, 13.6145, 5.82008], b_s: [1.34323, 7.10909, 0.204633, 20.3254], c: 8.27903, }),
        Atom { el: Ir, ionization: 4, }    => Some(Scatter { a_s: [30.7058, 15.5512, 14.2326, 5.53672], b_s: [1.30923, 6.71983, 0.167252, 17.4911], c: 6.96824, }),
        Atom { el: Pt, ionization: 0, }    => Some(Scatter { a_s: [27.0059, 17.7639, 15.7131, 5.7837], b_s: [1.51293, 8.81174, 0.424593, 38.6103], c: 11.6883, }),
        Atom { el: Pt, ionization: 2, }    => Some(Scatter { a_s: [29.8429, 16.7224, 13.2153, 6.35234], b_s: [1.32927, 7.38979, 0.263297, 22.9426], c: 9.85329, }),
        Atom { el: Pt, ionization: 4, }    => Some(Scatter { a_s: [30.9612, 15.9829, 13.7348, 5.92034], b_s: [1.24813, 6.60834, 0.16864, 16.9392], c: 7.39534, }),
        Atom { el: Au, ionization: 0, }    => Some(Scatter { a_s: [16.8819, 18.5913, 25.5582, 5.86], b_s: [0.4611, 8.6216, 1.4826, 36.3956], c: 12.0658, }),
        Atom { el: Au, ionization: 1, }    => Some(Scatter { a_s: [28.0109, 17.8204, 14.3359, 6.58077], b_s: [1.35321, 7.7395, 0.356752, 26.4043], c: 11.2299, }),
        Atom { el: Au, ionization: 3, }    => Some(Scatter { a_s: [30.6886, 16.9029, 12.7801, 6.52354], b_s: [1.2199, 6.82872, 0.212867, 18.659], c: 9.0968, }),
        Atom { el: Hg, ionization: 0, }    => Some(Scatter { a_s: [20.6809, 19.0417, 21.6575, 5.9676], b_s: [0.545, 8.4484, 1.5729, 38.3246], c: 12.6089, }),
        Atom { el: Hg, ionization: 1, }    => Some(Scatter { a_s: [25.0853, 18.4973, 16.8883, 6.48216], b_s: [1.39507, 7.65105, 0.443378, 28.2262], c: 12.0205, }),
        Atom { el: Hg, ionization: 2, }    => Some(Scatter { a_s: [29.5641, 18.06, 12.8374, 6.89912], b_s: [1.21152, 7.05639, 0.284738, 20.7482], c: 10.6268, }),
        Atom { el: Tl, ionization: 0, }    => Some(Scatter { a_s: [27.5446, 19.1584, 15.538, 5.52593], b_s: [0.65515, 8.70751, 1.96347, 45.8149], c: 13.1746, }),
        Atom { el: Tl, ionization: 1, }    => Some(Scatter { a_s: [21.3985, 20.4723, 18.7478, 6.82847], b_s: [1.4711, 0.517394, 7.43463, 28.8482], c: 12.5258, }),
        Atom { el: Tl, ionization: 3, }    => Some(Scatter { a_s: [30.8695, 18.3481, 11.9328, 7.00574], b_s: [1.1008, 6.53852, 0.219074, 17.2114], c: 9.8027, }),
        Atom { el: Pb, ionization: 0, }    => Some(Scatter { a_s: [31.0617, 13.0637, 18.442, 5.9696], b_s: [0.6902, 2.3576, 8.618, 47.2579], c: 13.4118, }),
        Atom { el: Pb, ionization: 2, }    => Some(Scatter { a_s: [21.7886, 19.5682, 19.1406, 7.01107], b_s: [1.3366, 0.488383, 6.7727, 23.8132], c: 12.4734, }),
        Atom { el: Pb, ionization: 4, }    => Some(Scatter { a_s: [32.1244, 18.8003, 12.0175, 6.96886], b_s: [1.00566, 6.10926, 0.147041, 14.714], c: 8.08428, }),
        Atom { el: Bi, ionization: 0, }    => Some(Scatter { a_s: [33.3689, 12.951, 16.5877, 6.4692], b_s: [0.704, 2.9238, 8.7937, 48.0093], c: 13.5782, }),
        Atom { el: Bi, ionization: 3, }    => Some(Scatter { a_s: [21.8053, 19.5026, 19.1053, 7.10295], b_s: [1.2356, 6.24149, 0.469999, 20.3185], c: 12.4711, }),
        Atom { el: Bi, ionization: 5, }    => Some(Scatter { a_s: [33.5364, 25.0946, 19.2497, 6.91555], b_s: [0.91654, 0.39042, 5.71414, 12.8285], c: -6.7994, }),
        Atom { el: Po, ionization: 0, }    => Some(Scatter { a_s: [34.6726, 15.4733, 13.1138, 7.02588], b_s: [0.700999, 3.55078, 9.55642, 47.0045], c: 13.677, }),
        Atom { el: At, ionization: 0, }    => Some(Scatter { a_s: [35.3163, 19.0211, 9.49887, 7.42518], b_s: [0.68587, 3.97458, 11.3824, 45.4715], c: 13.7108, }),
        Atom { el: Rn, ionization: 0, }    => Some(Scatter { a_s: [35.5631, 21.2816, 8.0037, 7.4433], b_s: [0.6631, 4.0691, 14.0422, 44.2473], c: 13.6905, }),
        Atom { el: Fr, ionization: 0, }    => Some(Scatter { a_s: [35.9299, 23.0547, 12.1439, 2.11253], b_s: [0.646453, 4.17619, 23.1052, 150.645], c: 13.7247, }),
        Atom { el: Ra, ionization: 0, }    => Some(Scatter { a_s: [35.763, 22.9064, 12.4739, 3.21097], b_s: [0.616341, 3.87135, 19.9887, 142.325], c: 13.6211, }),
        Atom { el: Ra, ionization: 2, }    => Some(Scatter { a_s: [35.215, 21.67, 7.91342, 7.65078], b_s: [0.604909, 3.5767, 12.601, 29.8436], c: 13.5431, }),
        Atom { el: Ac, ionization: 0, }    => Some(Scatter { a_s: [35.6597, 23.1032, 12.5977, 4.08655], b_s: [0.589092, 3.65155, 18.599, 117.02], c: 13.5266, }),
        Atom { el: Ac, ionization: 3, }    => Some(Scatter { a_s: [35.1736, 22.1112, 8.19216, 7.05545], b_s: [0.579689, 3.41437, 12.9187, 25.9443], c: 13.4637, }),
        Atom { el: Th, ionization: 0, }    => Some(Scatter { a_s: [35.5645, 23.4219, 12.7473, 4.80703], b_s: [0.563359, 3.46204, 17.8309, 99.1722], c: 13.4314, }),
        Atom { el: Th, ionization: 4, }    => Some(Scatter { a_s: [35.1007, 22.4418, 9.78554, 5.29444], b_s: [0.555054, 3.24498, 13.4661, 23.9533], c: 13.376, }),
        Atom { el: Pa, ionization: 0, }    => Some(Scatter { a_s: [35.8847, 23.2948, 14.1891, 4.17287], b_s: [0.547751, 3.41519, 16.9235, 105.251], c: 13.4287, }),
        Atom { el: U, ionization: 0, }     => Some(Scatter { a_s: [36.0228, 23.4128, 14.9491, 4.188], b_s: [0.5293, 3.3253, 16.0927, 100.613], c: 13.3966, }),
        Atom { el: U, ionization: 3, }     => Some(Scatter { a_s: [35.5747, 22.5259, 12.2165, 5.37073], b_s: [0.52048, 3.12293, 12.7148, 26.3394], c: 13.3092, }),
        Atom { el: U, ionization: 4, }     => Some(Scatter { a_s: [35.3715, 22.5326, 12.0291, 4.7984], b_s: [0.516598, 3.05053, 12.5723, 23.4582], c: 13.2671, }),
        Atom { el: U, ionization: 6, }     => Some(Scatter { a_s: [34.8509, 22.7584, 14.0099, 1.21457], b_s: [0.507079, 2.8903, 13.1767, 25.2017], c: 13.1665, }),
        Atom { el: Np, ionization: 0, }    => Some(Scatter { a_s: [36.1874, 23.5964, 15.6402, 4.1855], b_s: [0.511929, 3.25396, 15.3622, 97.4908], c: 13.3573, }),
        Atom { el: Np, ionization: 3, }    => Some(Scatter { a_s: [35.7074, 22.613, 12.9898, 5.43227], b_s: [0.502322, 3.03807, 12.1449, 25.4928], c: 13.2544, }),
        Atom { el: Np, ionization: 4, }    => Some(Scatter { a_s: [35.5103, 22.5787, 12.7766, 4.92159], b_s: [0.498626, 2.96627, 11.9484, 22.7502], c: 13.2116, }),
        Atom { el: Np, ionization: 6, }    => Some(Scatter { a_s: [35.0136, 22.7286, 14.3884, 1.75669], b_s: [0.48981, 2.81099, 12.33, 22.6581], c: 13.113, }),
        Atom { el: Pu, ionization: 0, }    => Some(Scatter { a_s: [36.5254, 23.8083, 16.7707, 3.47947], b_s: [0.499384, 3.26371, 14.9455, 105.98], c: 13.3812, }),
        Atom { el: Pu, ionization: 3, }    => Some(Scatter { a_s: [35.84, 22.7169, 13.5807, 5.66016], b_s: [0.484938, 2.96118, 11.5331, 24.3992], c: 13.1991, }),
        Atom { el: Pu, ionization: 4, }    => Some(Scatter { a_s: [35.6493, 22.646, 13.3595, 5.18831], b_s: [0.481422, 2.8902, 11.316, 21.8301], c: 13.1555, }),
        Atom { el: Pu, ionization: 6, }    => Some(Scatter { a_s: [35.1736, 22.7181, 14.7635, 2.28678], b_s: [0.473204, 2.73848, 11.553, 20.9303], c: 13.0582, }),
        Atom { el: Am, ionization: 0, }    => Some(Scatter { a_s: [36.6706, 24.0992, 17.3415, 3.49331], b_s: [0.483629, 3.20647, 14.3136, 102.273], c: 13.3592, }),
        Atom { el: Cm, ionization: 0, }    => Some(Scatter { a_s: [36.6488, 24.4096, 17.399, 4.21665], b_s: [0.465154, 3.08997, 13.4346, 88.4834], c: 13.2887, }),
        Atom { el: Bk, ionization: 0, }    => Some(Scatter { a_s: [36.7881, 24.7736, 17.8919, 4.23284], b_s: [0.451018, 3.04619, 12.8946, 86.003], c: 13.2754, }),
        Atom { el: Cf, ionization: 0, }    => Some(Scatter { a_s: [36.9185, 25.1995, 18.3317, 4.24391], b_s: [0.437533, 3.00775, 12.4044, 83.7881], c: 13.2674, }),
        _                                  => None,
    };

    ret
}
