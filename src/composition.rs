use std::str::FromStr;

use crate::element::Element;
use crate::site::SiteLabel;

#[derive(Debug, Clone, PartialEq)]
/// A Composition by Mole
///
/// what you probably think of as a composition, like H2 O
/// or Fe2 O3
///
/// We only support parsing from sum formula right now.
pub struct Composition(Vec<(Element, f64)>);

/// a structure's composition by weight fraction
#[derive(Debug, Clone, PartialEq)]
pub struct FractionalComposition(pub Vec<(Element, f64)>);
impl FractionalComposition {
    pub fn new(Composition(mut elements): Composition) -> Self {
        let total_mass = elements
            .iter()
            .map(|(el, n)| el.atomic_weight() * n)
            .sum::<f64>();

        for (el, n) in elements.iter_mut() {
            let mass = *n * el.atomic_weight();
            *n = mass / total_mass;
        }

        Self(elements)
    }

    pub fn get_mac_at_energy(&self, energy_kev: f64) -> Result<f64, String> {
        let mut val = 0.0;
        for (el, n) in self.0.iter() {
            let mac = el.mac_at_energy(energy_kev)?;
            val += n * mac;
        }

        Ok(val)
    }

    pub fn get_lac_at_energy(&self, energy_kev: f64, density: f64) -> Result<f64, String> {
        let mac = self.get_mac_at_energy(energy_kev)?;

        return Ok(density * mac)
    }
}

impl FromStr for Composition {
    type Err = String;

    fn from_str(mut s: &str) -> Result<Self, Self::Err> {
        let mut composition = Vec::new();
        loop {
            let v = SiteLabel::try_parse_single_element(s)?;
            let element = v.0;
            s = v.1;
            let (number, rest) = match s.split_once(' ') {
                Some((number, rest)) => (number, rest),
                None => (s, ""),
            };

            s = rest;
            let n = if number.len() == 0 {
                1.0
            } else {
                number
                    .parse()
                    .map_err(|err| format!("Could not parse atom count: {err}"))?
            };

            if n <= 0.0 {
                return Err(format!("Element number must be larger than 0. Got {n}."));
            }

            composition.push((element, n));

            s = s.trim_start();
            if s.len() == 0 {
                break;
            }
        }

        return Ok(Composition(composition));
    }
}

impl FromStr for FractionalComposition {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let c = Composition::from_str(s)?;
        Ok(Self::new(c))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::element::Element;

    #[test]
    fn simple_ok() {
        assert_eq!(
            Composition(vec![(Element::N, 1.0)]),
            Composition::from_str("N").unwrap()
        );
    }

    #[test]
    fn two_single_elements() {
        assert_eq!(
            Composition(vec![(Element::Na, 1.0), (Element::Fe, 1.0)]),
            Composition::from_str("Na Fe").unwrap()
        );
    }

    #[test]
    fn fe2_o3() {
        assert_eq!(
            Composition(vec![(Element::Fe, 2.0), (Element::O, 3.0)]),
            Composition::from_str("Fe2 O3").unwrap()
        );
    }

    #[test]
    fn fe2_7_o3_1() {
        assert_eq!(
            Composition(vec![(Element::Fe, 2.7), (Element::O, 3.1)]),
            Composition::from_str("Fe2.7 O3.1").unwrap()
        );
    }

    #[test]
    fn empty() {
        _ = Composition::from_str("").expect_err("invalid element");
    }

    #[test]
    fn negative_number() {
        _ = Composition::from_str("Fe-1").expect_err("negative number should not work");
    }
}
