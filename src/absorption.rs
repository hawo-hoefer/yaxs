use std::collections::HashMap;

use itertools::Itertools;
use lazy_static::lazy_static;
use ordered_float::NotNan;

use crate::composition::FractionalComposition;
use crate::element::Element;
use crate::pattern::adxrd::PrecomputedLACs;
use crate::structure::Structure;

lazy_static! {
    static ref MAC_DATA: [MACData; NUM_ENTRIES] = {
        // TODO: Verify file format, add version etc
        // extract MACData for element from binary blob
        // binary blob file format
        // [   # Header
        //     number of elements:       u32 (num_el)
        //     offset for each element: [num_el * u32]
        //
        //     # Data section
        //     representation for each MACData element
        //     with format: [
        //        atomic_number:              u8,
        //        num_entries:               u32,
        //        energies:    num_entries * f64,
        //        macs:        num_entries * f64,
        //     ]
        // ]

        let mut offset = 0;
        let num_el = u8_at_byte_offset(&mut offset);

        for _ in 0..num_el {
            // ignore since we get all of the entries anyway
            _ = u32_at_byte_offset(&mut offset);
        }

        let mut mac_data = core::array::from_fn::<MACData, NUM_ENTRIES, _>(|_| MACData { energies: Vec::new(), macs: Vec::new() });
        for i in 0..num_el {
            let entry = &mut mac_data[i as usize];
            let element_z = u8_at_byte_offset(&mut offset);
            assert_eq!(element_z, i + 1 as u8, "entry is what we expect");

            let num_samples = u32_at_byte_offset(&mut offset);
            for _ in 0..num_samples {
                let energy = NotNan::new(f64_at_byte_offset(&mut offset)).expect("we did not write NaNs to the file");
                // assert ascending
                assert!(entry.energies.first().map(|x| *x < energy).unwrap_or(true));
                entry.energies.push(energy);
            }

            for _ in 0..num_samples {
                let mac = NotNan::new(f64_at_byte_offset(&mut offset)).expect("we did not write NaNs to the file");
                entry.macs.push(mac);
            }
        }

        mac_data
    };
}

const MAC_DATA_BYTES: &'static [u8] = include_bytes!("macdata.bin");
const NUM_ENTRIES: usize = MAC_DATA_BYTES[0] as usize;
#[derive(Clone, PartialEq)]
pub struct MACData {
    pub energies: Vec<NotNan<f64>>,
    pub macs: Vec<NotNan<f64>>,
}

impl MACData {
    /// interpolate the Mass attenuation coefficient for a given energy
    /// by interpolating the loaded data
    ///
    /// * `energy_kev`: energy in keV
    pub fn interpolate(&self, energy_kev: f64) -> Result<NotNan<f64>, String> {
        let (e0, e1) = self.energy_limits();
        if energy_kev < *e0 || energy_kev > *e1 {
            return Err(format!("No data to interpolate mass attenuation coefficient outside of energy range [{e0}, {e1}] keV. Got {energy_kev} keV"));
        }
        match self
            .energies
            .binary_search(&NotNan::new(energy_kev).expect("energy is not NaN"))
        {
            Ok(found_index) => {
                // if the exact value was found in the data,
                // just return it, no need to interpolate
                return Ok(self.macs[found_index].into());
            }
            Err(insert_idx) => {
                let xlo = self.energies[insert_idx - 1];
                let xhi = self.energies[insert_idx];

                let ylo = self.macs[insert_idx - 1];
                let yhi = self.macs[insert_idx];

                let t = (energy_kev - *xlo) / (*xhi - *xlo);
                return Ok(NotNan::new(ylo + (yhi - ylo) * t).expect("xlo and xhi are not equal"));
            }
        }
    }

    pub fn get_for(el: Element) -> Result<&'static Self, String> {
        MAC_DATA.get(el as usize).ok_or(format!(
            "Could not eg mass attenuation data for element {el}. Not present in database."
        ))
    }

    pub fn energy_limits(&self) -> (NotNan<f64>, NotNan<f64>) {
        let emin = *self
            .energies
            .first()
            .expect("at least one energy in database");
        let emax = *self
            .energies
            .last()
            .expect("at least one energy in database");

        (emin, emax)
    }

    /// interpolate a range of energies
    ///
    /// may fail in case the given energy range is outside of the limits of the database
    ///
    /// `energies_kev`: energies to interpolate
    pub fn interpolate_slice(&self, energies_kev: &[NotNan<f64>]) -> Result<Box<[f64]>, String> {
        let emin = energies_kev.first().expect("at least one energy");
        let emax = energies_kev.last().expect("at least one energy");
        let (e0, e1) = self.energy_limits();

        if *emin < e0 || *emax > e1 {
            return Err(format!("No data to interpolate mass attenuation coefficient outside of energy range [{e0}, {e1}] keV. Got [{emin}, {emax}] keV"));
        }

        let mut ret = Vec::with_capacity(energies_kev.len());
        for e in energies_kev {
            ret.push(*self.interpolate(**e).expect("values are in range"));
        }

        Ok(ret.into())
    }
}
pub struct MACGenerator {
    macs: HashMap<Element, Box<[f64]>>,
    energies: Box<[NotNan<f64>]>,
}

impl MACGenerator {
    pub fn from_structures_energy(
        structures: &[Structure],
        (emin, emax): (f64, f64),
    ) -> Result<Self, String> {
        let energies = {
            // TODO: figure out how to deal with no structures
            // if no structures, there should be no error
            let el = structures
                .first()
                .expect("at least one structure")
                .wt_composition
                .into_iter()
                .next()
                .expect("structure is not empty")
                .0;

            let emin = NotNan::new(emin).expect("emin is not NaN");
            let emax = NotNan::new(emax).expect("emax is not NaN");

            let mac_data = MACData::get_for(el)?;
            // check for errors beforehand, and use interpolate_slice so that
            // the error messages are the same
            let _ = mac_data.interpolate_slice(&[emin, emax])?;

            let (smallest, _) = mac_data
                .energies
                .iter()
                .rev()
                .find_position(|&&x| x <= emin)
                .expect("values are in range, we check above");
            let (largest, _) = mac_data
                .energies
                .iter()
                .find_position(|&&x| x >= emax)
                .expect("values are in range, we check above");

            mac_data.energies[smallest..=largest]
                .to_vec()
                .into_boxed_slice()
        };

        let mut macs = HashMap::new();
        for s in structures {
            for (el, _) in s.wt_composition.0.iter() {
                if macs.contains_key(el) {
                    break;
                }

                let mac_data = MACData::get_for(*el)?;
                macs.insert(*el, mac_data.interpolate_slice(&energies)?);
            }
        }

        Ok(MACGenerator { macs, energies })
    }

    pub fn get_mixture<'a>(
        &self,
        components_by_wt: impl Iterator<Item = (&'a FractionalComposition, f64)>,
    ) -> MACData {
        let mut all_elements = std::collections::HashMap::new();

        for (composition, wf) in components_by_wt {
            for (el, frac) in composition.into_iter() {
                *all_elements.entry(*el).or_insert(0.0) += frac * wf;
            }
        }

        let mut macs = vec![NotNan::new(0.0).expect("0.0"); self.energies.len()];

        for (el, wt_frac) in all_elements.iter() {
            let data = self
                .macs
                .get(el)
                .ok_or(format!("Missing element {el}"))
                .expect("Error in MACGenerator construction");

            for (dst, src) in macs.iter_mut().zip(data) {
                *dst += NotNan::new(src * wt_frac).expect("not nan");
            }
        }

        MACData {
            macs: macs,
            energies: self.energies.clone().to_vec(),
        }
    }
}

/// Compute the linear attenuation coefficient for mixtures of phases present in PrecomputedLACs
///
/// Returns one value per emission line present in lacs.
///
/// * `volume_fractions`: volume fractions of the phases
/// * `lacs`: precomputed linear absorption coefficients for pure components
///
/// computation is done according to Equation 6 in Chapter 2 of X-Ray Mass Attenuation Coefficients
/// <https://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html>
///
/// $$ \frac{\mu_\text{m}}{\rho_\text{m}} = \sum_{i} w_i \left(\frac{\mu}{\rho}\right)_i $$
///
/// using the relation of volume and mass fraction $$ w_i = \frac{\rho_i v_i}{\rho_m} \Longrightarrow v_i = \frac{w_i}{\rho_i} \rho_m, $$
/// we can shuffle the above relation a bit and arrive at
/// $$
/// \mu_\text{m} = \sum_i \frac{w_i \rho_\text{m}}{\rho_i} \mu_i = \sum_i v_i \mu_i.
/// $$
pub fn compute_mixture_attenuation_coef(
    volume_fractions: &[f64],
    lacs: &PrecomputedLACs,
) -> Box<[f64]> {
    for emission_line_data in lacs.0.iter() {
        assert_eq!(
            volume_fractions.len(),
            emission_line_data.len(),
            "lacs are present for all structures"
        );
    }

    let mut mixture_attenuation_coefs = Vec::with_capacity(lacs.0.len());
    for by_emission_line in lacs.0.iter() {
        let mix_lac = volume_fractions
            .iter()
            .zip(by_emission_line.iter())
            .map(|(vf, lac)| vf * lac)
            .sum::<f64>();
        mixture_attenuation_coefs.push(mix_lac);
    }
    mixture_attenuation_coefs.into()
}

macro_rules! le_bytes_to_ty {
    ($offset:ident, $T:ty) => {{
        const SIZE: usize = std::mem::size_of::<$T>();
        let bytes = &MAC_DATA_BYTES[*$offset..*$offset + SIZE];
        *$offset += SIZE;
        <$T>::from_le_bytes(*bytes.first_chunk().unwrap())
    }};
}

fn u8_at_byte_offset(offset: &mut usize) -> u8 {
    let b = MAC_DATA_BYTES[*offset];
    *offset += 1;
    b
}

fn u32_at_byte_offset(offset: &mut usize) -> u32 {
    le_bytes_to_ty!(offset, u32)
}

fn f64_at_byte_offset(offset: &mut usize) -> f64 {
    le_bytes_to_ty!(offset, f64)
}
