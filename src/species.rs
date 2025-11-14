use std::str::FromStr;

use crate::scatter::{get_scatter_or_base_elem, Scatter};

use crate::element::Element;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Species(pub Vec<Atom>);

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

impl std::fmt::Display for Species {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Species(")?;
        for atom in self.0.iter() {
            write!(f, "{atom}")?;
        }
        f.write_str(")")
    }
}

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
            if val.is_empty() {
                break;
            }
            let (el, v) = Species::try_parse_single_element(val)?;
            let (ionization, v) = Species::try_parse_ionization(v).unwrap_or((0, v));
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

    #[test]
    fn parse_na_plus1() {
        let _: Species = "Na+1".parse().unwrap();
    }

    #[test]
    fn parse_na_minus1() {
        let _: Species = "Na-1".parse().unwrap();
    }
}
