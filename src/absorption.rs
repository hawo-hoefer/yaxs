use lazy_static::lazy_static;
use ordered_float::NotNan;

use crate::element::Element;

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
    pub(crate) fn interpolate(&self, energy_kev: f64) -> Option<f64> {
        match self
            .energies
            .binary_search(&NotNan::new(energy_kev).expect("energy is not NaN"))
        {
            Ok(found_index) => {
                // if the exact value was found in the data,
                // just return it, no need to interpolate
                return Some(self.macs[found_index].into());
            }
            Err(insert_idx) => {
                let xlo = self.energies[insert_idx];
                let xhi = self.energies[insert_idx + 1];

                let ylo = self.macs[insert_idx + 1];
                let yhi = self.macs[insert_idx + 1];

                let t = (energy_kev - *xlo) / (*xhi - *xlo);
                return Some(ylo + (yhi - ylo) * t);
            }
        }
    }
}

pub fn get_mac(el: Element) -> Option<&'static MACData> {
    MAC_DATA.get(el as usize)
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
