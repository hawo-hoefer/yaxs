use std::str::FromStr;

use crate::element::Element;
#[derive(Eq, PartialEq, Debug)]
pub struct Atom {
    el: Element,
    ionization: i8,
}

pub struct Species(pub Vec<Atom>);

impl FromStr for Species {
    type Err= String;

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

    fn try_parse_ionization(val: &str) -> Option<(i8, &str)> {
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
            let mut ionization: i8 = ion.parse().ok()?;
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
