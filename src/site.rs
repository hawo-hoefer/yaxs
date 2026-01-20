use crate::element::Element;
use crate::math::linalg::{Mat3, Vec3};
use crate::scatter::{get_scatter_or_base_elem, Scatter};
use std::str::FromStr;

#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct Atom {
    pub el: Element,
    pub ionization: i16,
}

impl Atom {
    /// acquire atomic scattering parameters for an atom with oxidation number
    /// at a specified sin(\theta) / \lambda value (in 1/amstrong)
    ///
    /// * `a`: atom
    /// * `sin_theta_over_lambda`: sin theta over lambda for scattering parameters
    pub fn scattering_params(&self) -> Option<Scatter> {
        get_scatter_or_base_elem(self)
    }
}

impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.el)?;
        match self.ionization {
            0 => (),
            ..0 => write!(f, "{}-", self.ionization.abs())?,
            1.. => write!(f, "{}+", self.ionization)?,
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub coords: Vec3<f64>,
    pub site_label: SiteLabel,
    pub occu: f64,
    pub displacement: Option<AtomicDisplacement>,
}

impl Site {
    pub fn normalized(&self) -> Site {
        let coords = self.coords.map(|x| {
            let mut x = x - x.round();
            if x < 0.0 {
                // map negative positions to positive end of unit cell
                x += 1.0
            }
            x
        });
        Self {
            coords,
            site_label: self.site_label.clone(),
            occu: self.occu,
            displacement: self.displacement.clone(),
        }
    }

    pub fn weight_contribution(&self) -> f64 {
        let wt_dalton = self
            .site_label
            .0
            .iter()
            .map(|x| x.el.atomic_weight())
            .sum::<f64>();
        let site_wt_with_duplicates = wt_dalton * self.occu;
        // let edge_positions = self
        //     .coords
        //     .iter_values()
        //     .map(|x| (*x == 0.0 || *x == 1.0) as u8)
        //     .sum::<u8>();
        // let shared_in_cells = match edge_positions {
        //     0 => 1.0,
        //     1 => 2.0,
        //     2 => 4.0,
        //     3 => 8.0,
        //     _ => unreachable!("vector has length of 3, not more than 3 elements can be 0 or 1"),
        // };
        // site_wt_with_duplicates / shared_in_cells
        site_wt_with_duplicates
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteLabel(pub Vec<Atom>);

impl SiteLabel {
    pub fn try_parse_single_element(val: &str) -> Result<(Element, &str), String> {
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
            Some('+') => {
                // chop the plus
                let mut rest =
                    std::str::from_utf8(&val.as_bytes()[1..]).expect("at least one character");

                if rest.is_empty() {
                    // output is finished, stop
                    return Some((1, rest));
                }

                let Some((ion, _)) = rest.split_once(|x: char| !x.is_ascii_digit()) else {
                    // only numbers left
                    let ion: i16 = rest.parse().expect("at least one number");
                    rest = "";
                    return Some((ion, rest));
                };

                if ion.is_empty() {
                    // no numbers left
                    return Some((1, rest));
                }

                let num: i16 = ion.parse().expect("");
                rest = std::str::from_utf8(&rest.as_bytes()[ion.len()..]).unwrap();
                return Some((num, rest));
            }
            Some('-') => {
                // chop the minus
                let mut rest =
                    std::str::from_utf8(&val.as_bytes()[1..]).expect("at least one character");

                if rest.is_empty() {
                    // output is finished, stop
                    return Some((1, rest));
                }

                let Some((ion, _)) = rest.split_once(|x: char| !x.is_ascii_digit()) else {
                    // only numbers left
                    let ion: i16 = rest.parse().expect("at least one number");
                    rest = "";
                    return Some((-ion, rest));
                };

                if ion.is_empty() {
                    // no numbers left
                    return Some((-1, rest));
                }

                let num: i16 = ion.parse().expect("");
                rest = std::str::from_utf8(&rest.as_bytes()[ion.len()..]).unwrap();
                return Some((-num, rest));
            }
            Some('*') => unimplemented!("'*' - ionization specifier"),
            Some('(') => {
                unimplemented!("ionization starting with '(' - ionization string: '{val}'")
            }
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

impl std::fmt::Display for SiteLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SiteLabel(")?;
        for atom in self.0.iter() {
            write!(f, "{atom}")?;
        }
        f.write_str(")")
    }
}

impl IntoIterator for SiteLabel {
    type Item = Atom;

    type IntoIter = <Vec<Atom> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a SiteLabel {
    type Item = &'a Atom;

    type IntoIter = <&'a Vec<Atom> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl FromStr for SiteLabel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut label = Vec::new();
        let mut val = value;
        loop {
            if val.is_empty() {
                break;
            }
            let (el, v) = SiteLabel::try_parse_single_element(val)?;
            let (ionization, v) = SiteLabel::try_parse_ionization(v).unwrap_or((0, v));
            val = v;
            label.push(Atom { el, ionization })
        }
        Ok(Self(label))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AtomicDisplacement {
    // https://www.iucr.org/resources/commissions/crystallographic-nomenclature/adp
    Uiso(f64),
    Biso(f64),
    Uani(Mat3<f64>),
    Bani(Mat3<f64>),
    // Uovl,
    // Umpe,
    // Bovl,
}

impl AtomicDisplacement {
    pub fn debye_waller_factor(&self, h: &Vec3<f64>, sin_theta_over_lambda: f64) -> f64 {
        use std::f64::consts::PI;
        match self {
            AtomicDisplacement::Uiso(u) => {
                // original formula from link above is
                // T(|h|) = exp(- 8 pi^2 * <u^2> (sin^2 theta / lambda^2) )
                let x = (-8.0 * PI * PI * u * sin_theta_over_lambda.powi(2)).exp();
                x
            }
            AtomicDisplacement::Biso(b) => {
                // the same as Uiso, only that b = u / (8 pi^2)
                let x = (-b * sin_theta_over_lambda.powi(2)).exp();
                x
            }
            AtomicDisplacement::Uani(u) => {
                let t = h.transpose().matmul(u).matmul(h)[0];
                (-2.0 * PI * PI * t).exp()
            }
            AtomicDisplacement::Bani(b) => {
                let mut t = 0.0;
                for j in 0..3 {
                    for l in 0..3 {
                        t += h[j] * b[(j, l)] / (8.0 * PI * PI) * h[l];
                    }
                }
                (-2.0 * PI * PI * t).exp()
            }
        }
    }

    pub fn fmt_kind(&self) -> &'static str {
        match self {
            AtomicDisplacement::Uiso(_) => "Uiso",
            AtomicDisplacement::Biso(_) => "Biso",
            AtomicDisplacement::Uani(_) => "Uani",
            AtomicDisplacement::Bani(_) => "Bani",
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_single_element_ok() {
        let (el, rest) = SiteLabel::try_parse_single_element("Fe").unwrap();
        assert_eq!(el, Element::Fe);
        assert_eq!(rest, "");
    }

    #[test]
    fn parse_single_element_ion_ok() {
        let (el, rest) = SiteLabel::try_parse_single_element("Fe3+").unwrap();
        assert_eq!(el, Element::Fe);
        assert_eq!(rest, "3+");
    }

    #[test]
    fn parse_single_element_ion_rest_ok() {
        let (el, rest) = SiteLabel::try_parse_single_element("Fe3+C-").unwrap();
        assert_eq!(el, Element::Fe);
        assert_eq!(rest, "3+C-");
    }

    #[test]
    fn parse_single_element_ion_rest_wrong_symbol() {
        SiteLabel::try_parse_single_element("Fear+C-").expect_err("");
    }

    #[test]
    fn parse_element_two_non_ions() {
        let (el, rest) = SiteLabel::try_parse_single_element("NiFe").unwrap();
        assert_eq!(el, Element::Ni);
        assert_eq!(rest, "Fe");
    }

    #[test]
    fn parse_ionization_positive() {
        let (el, rest) = SiteLabel::try_parse_ionization("3+").unwrap();
        assert_eq!(el, 3);
        assert_eq!(rest, "");
    }

    #[test]
    fn parse_ionization_negative() {
        let (el, rest) = SiteLabel::try_parse_ionization("3-").unwrap();
        assert_eq!(el, -3);
        assert_eq!(rest, "");
    }

    #[test]
    fn parse_ionization_positive_no_number() {
        let (el, rest) = SiteLabel::try_parse_ionization("+").unwrap();
        assert_eq!(el, 1);
        assert_eq!(rest, "");
    }

    #[test]
    fn parse_site_no_charge() {
        let site: SiteLabel = "FeNi".parse().unwrap();
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
        let label: SiteLabel = "Fe3+Ni2-".parse().unwrap();
        assert_eq!(
            label.0,
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

    #[test]
    fn parse_na_plus1() {
        let _: SiteLabel = "Na+1".parse().unwrap();
    }

    #[test]
    fn parse_na_minus1() {
        let _: SiteLabel = "Na-1".parse().unwrap();
    }

    #[test]
    fn normalization() {
        let s = Site {
            coords: Vec3::new(1.52, 0.2, -1.2),
            site_label: SiteLabel::from_str("Fe+").unwrap(),
            occu: 1.0,
            displacement: None,
        };
        let s2 = s.normalized();
        assert_eq!(s2.coords, Vec3::new(0.52, 0.2, 0.8))
    }
}
