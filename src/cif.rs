use std::collections::HashMap;

use crate::math::linalg::{Mat3, Vec3};
use itertools::Itertools;
use log::{debug, warn};

use crate::site::{AtomicDisplacement, Site};
use crate::species::Species;
use crate::structure::{Lattice, SGClass};
use crate::symop::SymOp;

// TODO: make this case-insensitive
const DATA_HEADER_START: &str = "data_";
const LOOP_HEADER_START: &str = "loop_";
const DENSITY_KEY: &str = "_exptl_crystal_density_diffrn";
const ANGLE_KEYS: [&str; 3] = ["_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"];
const LENGTH_KEYS: [&str; 3] = ["_cell_length_a", "_cell_length_b", "_cell_length_c"];
const SITE_DIST_TOL: f64 = 1e-6;
const FRAC_TOL_POS_ATOL: f64 = 1e-4;
const IMPORTANT_FRACTIONS: [f64; 4] = [1.0 / 3.0, 2.0 / 3.0, 1.0 / 6.0, 5.0 / 6.0];

// in the cif-definition, the elements ase entered by row, and they use the top
// right half of the symmetric matrix. In yaxs, we usually use the bottom left
// half, so we change the order here to reflect that.
const ANISO_ADP_U_LABELS: [&str; 6] = [
    "_atom_site_aniso_U_11",
    "_atom_site_aniso_U_12",
    "_atom_site_aniso_U_22",
    "_atom_site_aniso_U_13",
    "_atom_site_aniso_U_23",
    "_atom_site_aniso_U_33",
];

const ANISO_ADP_B_LABELS: [&str; 6] = [
    "_atom_site_aniso_B_11",
    "_atom_site_aniso_B_12",
    "_atom_site_aniso_B_22",
    "_atom_site_aniso_B_13",
    "_atom_site_aniso_B_23",
    "_atom_site_aniso_B_33",
];

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct CifParser<'a> {
    file_path: Option<String>,
    c: &'a str,
}

#[derive(PartialEq, Debug, Clone)]
pub enum Value {
    Inapplicable,
    Unknown,
    Float(f64),
    Int(i32),
    Text(String),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Inapplicable => f.write_str("Inapplicable"),
            Value::Unknown => f.write_str("Unknown"),
            Value::Float(v) => write!(f, "Float({v})"),
            Value::Int(v) => write!(f, "Integer({v})"),
            Value::Text(v) => write!(f, "Text({v})"),
        }
    }
}

impl Value {
    pub fn try_to_f64(&self) -> Result<f64, String> {
        match self {
            Value::Inapplicable => Err("Could not get float from Inapplicable".to_string()),
            Value::Unknown => Err("Could not get float from Unknown".to_string()),
            Value::Float(v) => Ok(*v),
            Value::Int(v) => Ok(f64::from(*v)),
            Value::Text(text) => Err(format!("Could not get value from Text '{text}'")),
        }
    }
}

pub type Table = HashMap<String, Vec<Value>>;

#[derive(Debug, Clone, PartialEq)]
pub enum DataItem {
    KV(String, Value),
    Table(Table),
}

#[derive(Debug)]
pub struct CIFContents<'a> {
    pub block_name: String,
    pub kvs: HashMap<String, Value>,
    pub tables: Vec<HashMap<String, Vec<Value>>>,
    pub file_path: Option<&'a str>,
}

fn parse_matrix_from_symmetric_order_labels(
    t: &Table,
    mat_labels: &[&str; 6],
    base_label: &str,
    label: &str,
    idx: usize,
) -> Result<Mat3<f64>, String> {
    let mut values = [0.0; 6];

    for ((key, v), tgt) in mat_labels
        .iter()
        .map(|key| (key, t.get(*key).map(|x| &x[idx])))
        .zip(values.iter_mut())
    {
        if let Some(v) = v {
            *tgt = v.try_to_f64().map_err(|_| {
                format!("Could not get {key} for {label}: Needs to be Integer or Float, got {v}.")
            })?;
        } else {
            return Err(format!("Could not acquire anisotropic atomic displacement parameters. Could not find column '{key}' in table containing '{base_label}'"));
        }
    }

    let [v11, v21, v22, v31, v32, v33] = values;
    return Ok(Mat3::from_rows([
        [v11, v21, v31],
        [v21, v22, v32],
        [v31, v32, v33],
    ]));
}
fn extract_aniso_adp(
    label: &str,
    atom_site_aniso_table: Option<&Table>,
) -> Result<Option<AtomicDisplacement>, String> {
    let Some(atom_site_aniso_table) = atom_site_aniso_table else {
        return Ok(None);
    };

    let mut idx = None;
    for (i, site_label) in atom_site_aniso_table
        .get("_atom_site_aniso_label")
        .expect("we check this above")
        .iter()
        .enumerate()
    {
        match site_label {
            Value::Text(v) => {
                if v == label {
                    idx = Some(i)
                }
            }
            _ => {
                return Err(format!(
                    "Invalid type for _atom_site_aniso_label: Must be Text, got {site_label}"
                ))
            }
        }
    }

    let Some(idx) = idx else {
        // TODO: should this be ok?
        return Ok(None);
    };

    let aniso_u_ident = ANISO_ADP_U_LABELS[0];
    if atom_site_aniso_table.contains_key(aniso_u_ident) {
        let mat = parse_matrix_from_symmetric_order_labels(
            atom_site_aniso_table,
            &ANISO_ADP_U_LABELS,
            aniso_u_ident,
            label,
            idx,
        )?;

        return Ok(Some(AtomicDisplacement::Uani(mat)));
    }

    let aniso_b_ident = ANISO_ADP_B_LABELS[0];
    if atom_site_aniso_table.contains_key(aniso_b_ident) {
        let mat = parse_matrix_from_symmetric_order_labels(
            atom_site_aniso_table,
            &ANISO_ADP_B_LABELS,
            aniso_b_ident,
            label,
            idx,
        )?;

        return Ok(Some(AtomicDisplacement::Bani(mat)));
    }

    let mut table_labels = String::from("[");
    let n_keys = atom_site_aniso_table.len();
    for (i, key) in atom_site_aniso_table.keys().enumerate() {
        table_labels.push_str(key);

        if i == n_keys - 1 {
            table_labels.push_str("]");
        } else {
            table_labels.push_str(", ");
        }
    }

    return Err(
                format!("Could not extract anisotropic ADP from table with _atom_site_aniso_label. Currently, we only support anisotropic U and B parameters. Table contains labels {table_labels}"),
            );
}

enum IsotropicDisplacementInFile {
    FoundUnlabeled,
    FoundLabeled,
}

fn extract_iso_adp(
    index: usize,
    label: &str,
    site_table: &Table,
    file_path: &str,
) -> Result<Option<(AtomicDisplacement, IsotropicDisplacementInFile)>, String> {
    use AtomicDisplacement::*;
    use IsotropicDisplacementInFile::*;
    let site_table_adp_type = site_table.get("_atom_site_adp_type").map(|x| &x[index]);
    if let Some(Value::Text(v)) = site_table_adp_type {
        return match v.as_str() {
            "Uiso" => {
                let v = site_table["_atom_site_U_iso_or_equiv"][index].try_to_f64()?;
                Ok(Some((Uiso(v), FoundLabeled)))
            },
            "Biso" => {
                let v = site_table.get("_atom_site_B_iso_or_equiv").ok_or(format!("Site {label} specified ADP 'Biso', but could not find '_atom_site_B_iso_or_equiv' in table."))?[index].try_to_f64()?;
                Ok(Some((Biso(v), FoundLabeled)))
            }
            "Uovl" => unimplemented!("Parse atomic displacement parameter Uovl"),
            "Umpe" => unimplemented!("Parse atomic displacement parameter Umpe"),
            "Bovl" => unimplemented!("Parse atomic displacement parameter Bovl"),
            "Uani" | "Bani" => return Ok(None), // if nothing is present here, try to parse anisotropic parameters
            v => Err(format!("Unknown ADP type: '{v}'. Must be one of ['Uani', 'Uiso', 'Uovl', 'Umpe', 'Bani', 'Biso', 'Bovl']"))
        };
    }

    if let Some(v) = site_table_adp_type {
        return Err(format!("ADP type must be string. Got '{v}'"));
    }

    if site_table.get("_atom_site_thermal_displace_type").is_some() {
        warn!(
            "found deprecated '_atom_site_thermal_displace_type'. Ignoring until it is implemented"
        );
        // try parsing anisotropic displacement anyway
        return Ok(None);
    }

    debug!("{file_path}: site '{label}': No ADP type explicitly declared. Trying to parse from table header.");
    if let Some(v) = site_table
        .get("_atom_site_B_iso_or_equiv")
        .map(|x| &x[index])
    {
        let v = match v {
            Value::Unknown | Value::Text(_) => return Err(format!("Could not acquire atomic displacement parameters for site {label}: '_atom_site_B_iso_or_equiv' needs to be Integer, Float or Inapplicable. Got {v}")),
            Value::Float(v) => Some((Biso(*v), FoundUnlabeled)),
            Value::Int(v) => Some((Biso(*v as f64), FoundUnlabeled)),
            Value::Inapplicable => None, 
        };
        return Ok(v);
    }

    if let Some(v) = site_table
        .get("_atom_site_U_iso_or_equiv")
        .map(|x| &x[index])
    {
        let v = match v {
            Value::Unknown | Value::Text(_) => return Err(format!("Could not acquire atomic displacement parameters for site {label}: '_atom_site_U_iso_or_equiv' needs to be Integer, Float or Inapplicable. Got {v}")),
            Value::Float(v) => Some((Uiso(*v), FoundUnlabeled)),
            Value::Int(v) => Some((Uiso(*v as f64), FoundUnlabeled)),
            Value::Inapplicable => None,
        };
        return Ok(v);
    }

    Ok(None)
}

impl<'a> CIFContents<'a> {
    pub fn get_symops(&self) -> Result<Vec<SymOp>, String> {
        let mut symop_label = "";
        let Some(symops_table) = self.tables.iter().find(|t: &&Table| {
            const SITE_KEYS: [&str; 2] = [
                "_space_group_symop_operation_xyz",
                "_symmetry_equiv_pos_as_xyz",
            ];
            let symop_operation_xyz = t.contains_key(SITE_KEYS[0]);
            if !symop_operation_xyz {
                let equiv_pos_as_xyz = t.contains_key(SITE_KEYS[1]);

                if equiv_pos_as_xyz {
                    symop_label = "_symmetry_equiv_pos_as_xyz";
                }
                equiv_pos_as_xyz
            } else {
                symop_label = "_space_group_symop_operation_xyz";
                true
            }
        }) else {
            return Err("No atom site label info in CIF".to_string());
        };

        let mut ret = Vec::with_capacity(symops_table[symop_label].len());
        for s in symops_table[symop_label].iter() {
            let Value::Text(s) = s else {
                return Err("Invalid symmetry operation. needs to be a string".to_string());
            };

            ret.push(s.parse::<SymOp>()?)
        }

        Ok(ret)
    }

    pub fn get_lattice(&self) -> Lattice {
        let Some((a, b, c)) = LENGTH_KEYS
            .iter()
            .map(|k: &&str| self.kvs.get(*k).unwrap().try_to_f64().unwrap())
            .collect_tuple::<(f64, f64, f64)>()
        else {
            panic!("This has to be three values")
        };
        let Some((alpha, beta, gamma)) = ANGLE_KEYS
            .iter()
            .map(|k: &&str| self.kvs.get(*k).unwrap().try_to_f64().unwrap().to_radians())
            .collect_tuple::<(f64, f64, f64)>()
        else {
            panic!("This has to be three values")
        };

        // from pymatgen.core.Lattice.from_parameters
        let val = alpha.cos() * beta.cos() - gamma.cos() / (alpha.sin() * beta.sin());
        let val = val.clamp(-1.0, 1.0);
        let gamma_star = val.acos();
        let va = [a * beta.sin(), 0.0, a * beta.cos()];
        let vb = [
            -b * alpha.sin() * gamma_star.cos(),
            b * alpha.sin() * gamma_star.sin(),
            b * alpha.cos(),
        ];
        let vc = [0.0, 0.0, c];
        Lattice {
            mat: Mat3::from_cols([va, vb, vc]),
        }
    }

    pub fn get_sites(&self, lattice: &Lattice) -> Result<Vec<Site>, String> {
        let Some(site_table) = self.tables.iter().find(|t: &&Table| {
            const SITE_KEYS: [&str; 6] = [
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "_atom_site_occupancy",
            ];
            SITE_KEYS.iter().map(|&k| t.contains_key(k)).all(|x| x)
        }) else {
            panic!("No atom site label info in CIF")
        };
        let n = site_table["_atom_site_type_symbol"].len();
        let symops = self.get_symops()?;

        let atom_site_aniso_table = self
            .tables
            .iter()
            .find(|t: &&Table| t.contains_key("_atom_site_aniso_label"));

        let site_at_index = |i: usize| -> Result<Site, String> {
            let sp: Species = match &site_table["_atom_site_type_symbol"][i] {
                Value::Text(label) => label.parse().unwrap(),
                v => return Err(format!("Invalid _atom_site_type_symbol: {v}")),
            };

            let label = match &site_table["_atom_site_label"][i] {
                Value::Text(label) => label,
                _ => return Err("Invalid site label".to_string()),
            };

            let occu = site_table["_atom_site_occupancy"][i].try_to_f64().unwrap();
            // TODO: remove unwraps, proper error handling
            let coords = Vec3::new(
                site_table["_atom_site_fract_x"][i].try_to_f64().unwrap(),
                site_table["_atom_site_fract_y"][i].try_to_f64().unwrap(),
                site_table["_atom_site_fract_z"][i].try_to_f64().unwrap(),
            );

            let iso_adp = extract_iso_adp(i, label, site_table, &self.file_path.unwrap_or("in-mem"))?;
            let aniso_adp = extract_aniso_adp(label, atom_site_aniso_table)?;
            let adp = match (iso_adp, aniso_adp) {
                (Some((_iso_adp, how_found)), Some(aniso_adp)) => {
                    use IsotropicDisplacementInFile::*;
                    match how_found {
                        FoundUnlabeled => {
                            warn!("{p}: site '{label}': Both isotropic and anisotropic atomic displacement parameters are defined. Using isotropic ADP.", p = self.file_path.unwrap_or("in-mem"));
                            Some(aniso_adp)
                        }
                        FoundLabeled => {
                            warn!("{p}: site '{label}': Both isotropic and anisotropic atomic displacement parameters are defined. Using isotropic ADP, because ADP is labeled as isotropic in site table.", p = self.file_path.unwrap_or("in-mem"));
                            Some(_iso_adp)
                        },
                    }
                }
                (Some((iso_adp, _)), None) => Some(iso_adp),
                (None, Some(aniso_adp)) => Some(aniso_adp),
                (None, None) => None,
            };

            debug!("{p}: site '{label}': got atomic displacement parameter of type {k}", p=self.file_path.unwrap_or("in-mem"), k=adp.as_ref().map(|x| x.fmt_kind()).unwrap_or("None"));

            let coords = coords.map(|x| {
                for frac in IMPORTANT_FRACTIONS {
                    if (x - frac).abs() < FRAC_TOL_POS_ATOL {
                        warn!("Rounded fractional coordinate {x} to {frac}");
                        return frac;
                    }
                }
                *x
            });

            Ok(Site {
                species: sp,
                coords,
                occu,
                displacement: adp,
            })
        };

        fn site_exists_periodic(site: &Site, sites: &[Site]) -> bool {
            // adapted from pymavtgen.util.coord.find_in_coord_list
            sites
                .iter()
                .map(|ps| {
                    let delta = &site.coords - &ps.coords;
                    let inside_dist_tol = delta
                        .iter_values()
                        .map(|x| (x - x.round()).powi(2))
                        .sum::<f64>()
                        < SITE_DIST_TOL.powi(2);

                    inside_dist_tol && (ps.species == site.species) && (ps.occu == site.occu)
                })
                .any(|x| x)
        }

        // we parsed the symops, but still need to remove duplicate sites
        let mut sites = Vec::new();
        for base_site in (0..n).map(site_at_index) {
            let base_site = base_site?;

            let base_site = base_site.normalized();
            if site_exists_periodic(&base_site, &sites) {
                continue;
            }

            for op in symops.iter() {
                let s = Site {
                    coords: op.apply(&base_site.coords),
                    species: base_site.species.clone(),
                    occu: base_site.occu,
                    displacement: match base_site.displacement {
                        Some(AtomicDisplacement::Uiso(_) | AtomicDisplacement::Biso(_)) | None => {
                            base_site.displacement.clone()
                        }
                        Some(AtomicDisplacement::Uani(ref v)) => {
                            Some(AtomicDisplacement::Uani(op.transform_orientation(&v)))
                        }
                        Some(AtomicDisplacement::Bani(ref v)) => {
                            Some(AtomicDisplacement::Bani(op.transform_orientation(&v)))
                        }
                    },
                };

                if site_exists_periodic(&s, &sites) {
                    continue;
                }
                sites.push(s.normalized());
            }
        }

        Ok(sites)
    }

    pub fn get_sg_no_and_class(&self) -> Result<(u8, SGClass), String> {
        let sg_no = self
            .kvs
            .get("_space_group_IT_number")
            .or_else(|| self.kvs.get("_symmetry_Int_Tables_number"))
            .ok_or_else(|| "No symmetry group in cif. checked '_symmetry_Int_Tables_number' and '_space_group_IT_number'.".to_string())?;
        let sg_no = match *sg_no {
            // TODO: add test for invalid sg_no
            Value::Int(sg_no) if !(1..=230).contains(&sg_no) => {
                return Err(format!(
                    "space group number is out of range. Needs to be in [1, 230], got {sg_no}"
                ))
            }
            Value::Inapplicable | Value::Unknown | Value::Float(_) | Value::Text(_) => {
                return Err(format!(
                    "space group number is of wrong type. Required Integer, got {sg_no:?}"
                ))
            }
            Value::Int(sg_no) => sg_no as u8,
        };

        let sg_class = SGClass::try_from(sg_no).expect("we test this above");
        Ok((sg_no, sg_class))
    }

    pub fn get_density(&self) -> Result<Option<f64>, String> {
        let value = match self.kvs.get(DENSITY_KEY) {
            Some(v) => v,
            None => return Ok(None),
        };
        match value {
            Value::Inapplicable | Value::Unknown => Ok(None),
            Value::Float(v) => Ok(Some(*v)),
            Value::Int(v) => Ok(Some(*v as f64)),
            Value::Text(t) => Err(format!(
                "Invalid value of type Text for {DENSITY_KEY} in CIF: '{t}'"
            )),
        }
    }
}

impl<'a> CifParser<'a> {
    pub fn new(data: &'a str) -> Self {
        Self {
            c: data,
            file_path: None,
        }
    }

    pub fn with_file(mut self, file_path: String) -> Self {
        self.file_path = Some(file_path);
        self
    }

    pub fn parse(&'a mut self) -> Result<CIFContents<'a>, String> {
        self.skip_ws_comments();
        let bn = self.parse_block_name()?.to_string();
        let mut kvs = HashMap::new();
        let mut tables = Vec::new();
        while !self.c.is_empty() {
            self.skip_ws_comments();
            if self.c.is_empty() {
                break;
            }
            if self.c.starts_with(DATA_HEADER_START) {
                let second_bn = self
                    .parse_block_name()
                    .expect("we are at the start of a data descriptor");

                return Err(format!(
                    "Multiple structures per CIF is ambiguous. First block name: '{bn}', second: '{second_bn}'"
                ));
            }
            match self.parse_data_item()? {
                DataItem::KV(k, v) => {
                    if kvs.contains_key(&k) {
                        return Err(format!("Duplicate Key '{k}' in CIF"));
                    };
                    let _ = kvs.insert(k, v);
                }
                DataItem::Table(table) => {
                    tables.push(table);
                }
            }
        }
        Ok(CIFContents {
            block_name: bn,
            kvs,
            tables,
            file_path: self.file_path.as_ref().map(|x| x.as_str()),
        })
    }

    fn skip_comments(&mut self) {
        while self.c.starts_with('#') {
            // skip comments
            let next_line = self.c.find('\n').unwrap_or(self.c.len() - 1) + 1;
            self.c = std::str::from_utf8(&self.c.as_bytes()[next_line..]).unwrap();
        }
    }

    fn skip_whitespace(&mut self) {
        self.c = self.c.trim_start();
    }

    fn skip_ws_comments(&mut self) {
        while self.c.starts_with(|x: char| x == '#' || x.is_whitespace()) {
            self.skip_comments();
            self.skip_whitespace();
        }
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        self.skip_whitespace();
        match self
            .c
            .chars()
            .next()
            .ok_or_else(|| "Cannot parse value from empty contents".to_string())?
        {
            '+' | '-' | '0'..='9' => {
                // try to parse number. if it fails, we parse as string
                match self.parse_number() {
                    Ok(num) => Ok(num),
                    Err(_) => Ok(self.parse_text()?),
                }
            }
            '.' => {
                let _ = self.consume_once('.');
                Ok(Value::Inapplicable)
            }
            '?' => {
                let _ = self.consume_once('?');
                Ok(Value::Unknown)
            }
            _ => Ok(self.parse_text()?), // this includes ';'
        }
    }

    fn parse_block_name(&mut self) -> Result<&str, String> {
        if !self.c.starts_with(DATA_HEADER_START) {
            let (line, _) = self.c.split_once('\n').unwrap_or((self.c, ""));
            return Err(format!(
                "No Data header found. First line after comments is '{line}'..."
            ));
        }

        self.c = self.c.trim_start_matches(DATA_HEADER_START);

        // TODO: this should probably be the correct CIF whitespace, but i won't bother right now
        let (block_header, rest) = self
            .c
            .split_once(|x: char| x.is_whitespace())
            .unwrap_or((self.c, ""));
        self.c = rest;

        Ok(block_header)
    }

    fn parse_tag(&mut self) -> Result<&str, String> {
        let Some((tag, c0)) = self.c.split_once(|x: char| x.is_whitespace()) else {
            return Err("Could not parse data item tag".to_string());
        };
        self.c = c0;
        Ok(tag)
    }

    fn parse_data_item(&mut self) -> Result<DataItem, String> {
        if self.c.starts_with('_') {
            // we are reading a tag
            let tag = self.parse_tag()?.to_string();
            let val = self.parse_value();
            return Ok(DataItem::KV(tag.to_string(), val?));
        } else if self.c.starts_with(LOOP_HEADER_START) {
            return Ok(DataItem::Table(self.parse_loop()?));
        }
        Err(format!(
            "Invalid parser state. Parser is at: {c}",
            c = self.c
        ))
    }

    fn consume_once(&mut self, c: char) -> bool {
        let Some(sc) = self.c.chars().next() else {
            // at end
            return false;
        };
        if sc == c {
            self.c = std::str::from_utf8(&self.c.as_bytes()[c.len_utf8()..])
                .expect("we use c's utf-8-len");
            return true;
        }

        false
    }

    fn parse_text(&mut self) -> Result<Value, String> {
        match self.c.chars().next() {
            Some(c) if c == ';' => {
                // multiline string ';'
                assert!(self.consume_once(c));
                let (text, rest) = self
                    .c
                    .split_once("\n;")
                    .ok_or_else(|| format!("unterminated '{c}'-string"))?;
                self.c = rest;
                // re-append the newline we stripped off before
                let mut text = text.to_string();
                text.push('\n');
                Ok(Value::Text(text))
            }
            Some(c) if matches!(c, '\'' | '\"') => {
                assert!(self.consume_once(c));
                let (text, rest) = self
                    .c
                    .split_once(c)
                    .ok_or_else(|| format!("unterminated '{c}'-string"))?;
                self.c = rest;
                Ok(Value::Text(text.to_string()))
            }
            Some(_) => {
                // UnquotedString
                let (text, rest) = self
                    .c
                    .split_once(|x: char| x.is_whitespace())
                    .unwrap_or((self.c, ""));
                self.c = rest;
                Ok(Value::Text(text.to_string()))
            }
            None => Err("Cannot Parse CIF value from empty contents".to_string()),
        }
    }

    fn parse_number(&mut self) -> Result<Value, String> {
        let (mut text, rest) = self
            .c
            .split_once(|x: char| x.is_whitespace())
            .unwrap_or((self.c, ""));

        if let Some((num, p_range)) = text.split_once('(') {
            if !p_range.ends_with(')') {
                return Err("Handle unterminated precision. missing ')'".to_string());
            }

            // NOTE: we probably are handling this wrong.
            // what if the input number is an integer and has precision - like 12312(12)
            // or if there is a float like 1.234e17(123)?
            // in these cases, we crash and burn, we should probably error gracefully
            if num.find('e').is_some() // float in scientific notation
                || num.find(['.', 'e']).is_none()
            // is an integer because no decimal point or scientific notation 'e'
            {
                return Err(format!(
                    "Precision parentheses are only valid after decimal notation. Tried to parse '{num}({p_range}'"
                ));
            }

            text = num;
        }

        if let Ok(v) = text.parse::<i32>() {
            self.c = rest;
            return Ok(Value::Int(v));
        }

        let v = text.parse::<f64>().map_err(|err| err.to_string())?;
        self.c = rest;
        Ok(Value::Float(v))
    }

    fn parse_loop(&mut self) -> Result<Table, String> {
        self.c = std::str::from_utf8(&self.c.as_bytes()[LOOP_HEADER_START.len()..]).unwrap();
        self.skip_whitespace();
        let mut kvs = Vec::new();
        while self.c.starts_with('_') {
            // while tokens start with '_', we are reading column names

            kvs.push((self.parse_tag()?.to_string(), Vec::new()));
            self.skip_ws_comments();
        }

        while !self.c.starts_with('_')
            && !self.c.starts_with(LOOP_HEADER_START)
            && !self.c.starts_with(DATA_HEADER_START)
            && !self.c.is_empty()
        {
            for (_, v) in kvs.iter_mut() {
                let val = self.parse_value()?;
                v.push(val);
            }
            self.skip_ws_comments();
        }

        Ok(kvs.drain(..).collect())
    }
}

#[cfg(test)]
mod test {
    use crate::structure::Structure;

    use super::*;

    #[test]
    fn whitespace_comments() {
        let mut p = CifParser::new("#arstoirest\n    \nabc");
        p.skip_ws_comments();
        assert_eq!(p.c, "abc");
    }

    #[test]
    fn block_name() {
        let mut p = CifParser::new("data_ABABAB\n");
        let bn = p.parse_block_name().unwrap();
        assert_eq!(bn, "ABABAB");
    }

    #[test]
    fn parse_float_test() {
        let mut p = CifParser::new("1.2123 aroistena");
        let v = p.parse_number().expect("valid float");
        assert_eq!(v, Value::Float(1.2123));
        assert_eq!(p.c, "aroistena");
    }

    #[test]
    fn parse_int_test() {
        let mut p = CifParser::new("12123 arstr");
        let v = p.parse_number().expect("valid int");
        assert_eq!(v, Value::Int(12123));
        assert_eq!(p.c, "arstr");
    }

    #[test]
    fn parse_loop() {
        let mut p = CifParser::new(
            "loop_
_col_a
_col_b
_col_c
_col_d
A 1 2.0 123.2(15)
B 2.0(32) 1.0 test",
        );
        use Value::*;
        let table = p.parse_loop().expect("valid loop definition");
        assert_eq!(
            table["_col_a"],
            [Text("A".to_string()), Text("B".to_string())]
        );
        assert_eq!(table["_col_b"], [Int(1), Float(2.0)]);
        assert_eq!(table["_col_c"], [Float(2.0), Float(1.0)]);
        assert_eq!(table["_col_d"], [Float(123.2), Text("test".to_string())]);
    }

    #[test]
    fn parse_text_field() {
        let mut p = CifParser::new(
            "_test 
;
Test Test Test
test test
test
;",
        );
        let di = p.parse_data_item().expect("valid data item definition");
        assert_eq!(
            di,
            DataItem::KV(
                "_test".to_string(),
                Value::Text("\nTest Test Test\ntest test\ntest\n".to_string())
            )
        )
    }

    #[test]
    fn parse_cif_small_no_loop() {
        let mut p = CifParser::new(
            "data_dummy_block_name
_integer_value 12
_float_value 123.12(12)
_date_val 2012-02-01
_unquoted_string test
_single_quote 'Test A B'
_double_quote \"Test A B\"
_text_field ;
hello hello
;",
        );
        let vals = p.parse().expect("valid cif contents");
        assert_eq!(vals.block_name, "dummy_block_name");
        let kvs = vals.kvs;
        assert_eq!(kvs["_integer_value"], Value::Int(12));
        assert_eq!(kvs["_float_value"], Value::Float(123.12));
        assert_eq!(kvs["_date_val"], Value::Text("2012-02-01".to_string()));
        assert_eq!(kvs["_unquoted_string"], Value::Text("test".to_string()));
        assert_eq!(kvs["_single_quote"], Value::Text("Test A B".to_string()));
        assert_eq!(kvs["_double_quote"], Value::Text("Test A B".to_string()));
        assert_eq!(
            kvs["_text_field"],
            Value::Text("\nhello hello\n".to_string())
        );
        assert!(vals.tables.is_empty())
    }

    #[test]
    fn parse_loops_string_end() {
        let data = "loop_
_sym
_ox
Hf2+ 2
He2- '-2'
loop_
_a
_b
hello 1.0 
hell  -2";
        use Value::*;

        let mut p = CifParser::new(data);
        let DataItem::Table(item) = p.parse_data_item().expect("valid data item") else {
            panic!()
        };
        assert_eq!(
            item["_sym"],
            [Text("Hf2+".to_string()), Text("He2-".to_string())]
        );
        assert_eq!(item["_ox"], [Int(2), Text("-2".to_string())]);
        let DataItem::Table(item) = p.parse_data_item().expect("valid table header") else {
            panic!()
        };
        assert_eq!(
            item["_a"],
            [Text("hello".to_string()), Text("hell".to_string())]
        );
        assert_eq!(item["_b"], [Float(1.0), Int(-2)]);
    }

    #[test]
    fn parse_float_ending_in_dot() {
        let mut p = CifParser::new("_test 1.");
        let item = p.parse_data_item().expect("valid kv pair");
        assert_eq!(item, DataItem::KV("_test".to_string(), Value::Float(1.0)))
    }

    #[test]
    fn parse_loops_float_dot_end() {
        let data = "loop_
_sym
_ox
Hf2+ 2
He2- 2.
loop_
_a
_b
hello 1.0 
hell  -2";
        use Value::*;

        let mut p = CifParser::new(data);
        let DataItem::Table(item) = p.parse_data_item().expect("valid table") else {
            panic!()
        };
        assert_eq!(
            item["_sym"],
            [Text("Hf2+".to_string()), Text("He2-".to_string())]
        );
        assert_eq!(item["_ox"], [Int(2), Float(2.0)]);
        let DataItem::Table(item) = p.parse_data_item().expect("valid table header") else {
            panic!()
        };
        assert_eq!(
            item["_a"],
            [Text("hello".to_string()), Text("hell".to_string())]
        );
        assert_eq!(item["_b"], [Float(1.0), Int(-2)]);
    }

    #[test]
    fn parse_unknown_inapplicable() {
        let mut p = CifParser::new("_inapplicable .\n _unknown ?\n");
        let DataItem::KV(_, Value::Inapplicable) =
            p.parse_data_item().expect("valid kv definition")
        else {
            panic!()
        };

        p.skip_whitespace();
        let DataItem::KV(_, Value::Unknown) = p.parse_data_item().expect("valid kv definition")
        else {
            panic!()
        };
        assert_eq!(p.c, "\n");
    }

    #[test]
    fn parse_loop_end_comment() {
        let data = "data_test
_data 1234
loop_
_sym
_ox
Hf2+ 2
He2- 2.
#arst
";
        let mut p = CifParser::new(data);
        let CIFContents { kvs, tables, .. } = p.parse().expect("valid cif contents");
        let kvs_exp = HashMap::from([("_data".to_string(), Value::Int(1234))]);
        assert_eq!(kvs, kvs_exp);
        assert_eq!(tables.len(), 1);
    }

    #[test]
    fn parse_loop_end_comment_without_trailing_newline() {
        let data = "data_test
_data 1234
loop_
_sym
_ox
Hf2+ 2
He2- 2.
#arst";
        let mut p = CifParser::new(data);
        let CIFContents { kvs, tables, .. } = p.parse().expect("valid cif contents");
        let kvs_exp = HashMap::from([("_data".to_string(), Value::Int(1234))]);
        assert_eq!(kvs, kvs_exp);
        assert_eq!(tables.len(), 1);
    }

    #[test]
    fn parse_leading_new_line() {
        let mut p = CifParser::new("_test 1.");
        let item = p.parse_data_item().expect("valid kv definition");
        assert_eq!(item, DataItem::KV("_test".to_string(), Value::Float(1.0)))
    }

    #[test]
    fn parse_kw_after_loop() {
        let mut p = CifParser::new(
            "data_dummy_block_name
loop_
_test
_test2
A B
A C
_cell_length_a 4
_cell_length_b 8
_cell_length_c 12
",
        );

        let CIFContents {
            block_name,
            kvs,
            tables,
            ..
        } = p.parse().expect("valid cif contents");
        assert_eq!(block_name, "dummy_block_name");
        let mut table = HashMap::new();
        table.insert(
            "_test".to_string(),
            vec![Value::Text("A".to_string()), Value::Text("A".to_string())],
        );

        table.insert(
            "_test2".to_string(),
            vec![Value::Text("B".to_string()), Value::Text("C".to_string())],
        );
        assert_eq!(tables, vec![table]);

        assert_eq!(kvs.get("_cell_length_a"), Some(&Value::Int(4)));
        assert_eq!(kvs.get("_cell_length_b"), Some(&Value::Int(8)));
        assert_eq!(kvs.get("_cell_length_c"), Some(&Value::Int(12)));
    }

    #[test]
    fn space_before_loop_labels() {
        let mut p = CifParser::new(
            "data_HfO2
loop_
  _a
  _b
  _c
  1 2 3",
        );
        p.parse().expect("valid cif contents");
    }

    #[test]
    fn string_at_end() {
        let mut p = CifParser::new(
            r"data_HfO2
_citation_title

;
Test test
test
;
",
        );
        p.parse().expect("valid cif contents");
    }
    #[test]
    fn semicolon_in_multiline_string() {
        let mut p = CifParser::new(
            r"data_HfO2
_citation_title

;
Test test; test
;
",
        );
        p.parse().expect("valid cif contents");
    }

    #[test]
    fn parse_number_int_precision() {
        let mut p = CifParser::new("123(12)");
        p.parse_number()
            .expect_err("integer with precision parens is illegal");
    }

    #[test]
    fn parse_number_scientific_precision() {
        let mut p = CifParser::new("1.23e12(12)");
        p.parse_number()
            .expect_err("scientific number notation with precision parens is illegal");
    }

    #[test]
    fn parse_struct_multiple_frac_occu_site() {
        const CIF_WITH_OCCUPANCY: &'static str = "# generated using pymatgen
data_sites_with_occupancy
_symmetry_space_group_name_H-M   'P 1'
_symmetry_Int_Tables_number 1
_cell_length_a   2.92800000
_cell_length_b   2.92800000
_cell_length_c   11.18000000
_cell_angle_alpha   90.00000000
_cell_angle_beta   90.00000000
_cell_angle_gamma   120.00000000
_cell_volume   83.00697361
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Na+  Na1  1  0.00000000  0.00000000  0.25000000  0.25
  Fe-  Fe1  1  0.00000000  0.00000000  0.25000000  0.75";

        let mut p = CifParser::new(CIF_WITH_OCCUPANCY);
        let d = p.parse().expect("valid cif contents");
        let s = Structure::try_from(&d).expect("valid cif contents");
        let mut sites = s.sites.iter();
        assert_eq!(
            &Site {
                coords: Vec3::new(0.0, 0.0, 0.25),
                species: "Na+".parse().unwrap(),
                occu: 0.25,
                displacement: None,
            },
            sites.next().unwrap()
        );

        assert_eq!(
            &Site {
                coords: Vec3::new(0.0, 0.0, 0.25),
                species: "Fe-".parse().unwrap(),
                occu: 0.75,
                displacement: None,
            },
            sites.next().unwrap()
        );
    }

    #[test]
    fn test_multiple_structures_should_err() {
        let input = "data_phase_1
_chemical_name_mineral 'phase_1'
_cell_length_a  2.898009
_cell_length_b  2.898009
_cell_length_c  11.1751
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 120
_cell_volume 81.27959
_symmetry_space_group_name_H-M P63/mmc
_space_group_IT_number 194
loop_
_symmetry_equiv_pos_as_xyz
	 'x, y, z '
	 '-x, -x+y, z+1/2 '
	 '-x, -y, z+1/2 '
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_symmetry_multiplicity
_atom_site_adp_type
_atom_site_U_iso_or_equiv
Na_e Na+1 0.3333333 0.6666667 0.75 0.45134   2 Biso 5
O1 O-2 0.3333333 0.6666667 0.091 1   4 Biso 0.7

data_phase_2
_chemical_name_mineral 'phase_2'
_cell_length_a  2.898009
_cell_length_b  2.898009
_cell_length_c  11.1751
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 120
_cell_volume 81.27959
_symmetry_space_group_name_H-M P63/mmc
_space_group_IT_number 194
loop_
_symmetry_equiv_pos_as_xyz
	 'x, y, z '
	 '-x, -x+y, z+1/2 '
	 '-x, -y, z+1/2 '
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_symmetry_multiplicity
_atom_site_adp_type
_atom_site_U_iso_or_equiv
Na_e Na+1 0.3333333 0.6666667 0.75 0.45134   2 Biso 5
O1 O-2 0.3333333 0.6666667 0.091 1   4 Biso 0.7";

        let s = CifParser::new(input)
            .parse()
            .expect_err("should error on multiple structures in one cif");

        assert_eq!(s, "Multiple structures per CIF is ambiguous. First block name: 'phase_1', second: 'phase_2'")
    }

    #[test]
    fn test_thirds_site_coords() {
        let input = "data_site_tolerance_thirds
_chemical_name_mineral 'bug_rock'
_cell_length_a  2.898009
_cell_length_b  2.898009
_cell_length_c  11.1751
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 120
_cell_volume 81.27959
_symmetry_space_group_name_H-M P63/mmc
_space_group_IT_number 194
loop_
_symmetry_equiv_pos_as_xyz
	 'x, y, z '
	 '-x, -x+y, z+1/2 '
	 '-x, -y, z+1/2 '
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_symmetry_multiplicity
_atom_site_adp_type
_atom_site_B_iso_or_equiv
Na_e Na+1 0.3333333 0.6666667 0.75 0.45134   2 Biso 5
O1 O-2 0.3333333 0.6666667 0.091 1   4 Biso 0.7";
        let mut p = CifParser::new(input);
        let contents = p.parse().unwrap();
        let s = Structure::try_from(&contents).unwrap();
        for site in s.sites.iter() {
            println!("{:?}", site);
        }
        assert_eq!(s.sites.len(), 4)
    }

    #[test]
    fn test_biso_specified_but_not_present() {
        let input = "data_biso_wrong
_chemical_name_mineral 'bug_rock'
_cell_length_a  2.898009
_cell_length_b  2.898009
_cell_length_c  11.1751
_cell_angle_alpha 90
_cell_angle_beta  90
_cell_angle_gamma 120
_cell_volume 81.27959
_symmetry_space_group_name_H-M P63/mmc
_space_group_IT_number 194
loop_
_symmetry_equiv_pos_as_xyz
	 'x, y, z '
	 '-x, -x+y, z+1/2 '
	 '-x, -y, z+1/2 '
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_symmetry_multiplicity
_atom_site_adp_type
_atom_site_U_iso_or_equiv
Na_e Na+1 0.3333333 0.6666667 0.75 0.45134   2 Biso 5
O1 O-2 0.3333333 0.6666667 0.091 1   4 Biso 0.7";
        let mut p = CifParser::new(input);
        let contents = p.parse().unwrap();
        let s = Structure::try_from(&contents).expect_err(
            "this should fail because _atom_site_B_iso_or_equiv is missing in the sites table",
        );
        assert_eq!(s, "Site Na_e specified ADP 'Biso', but could not find '_atom_site_B_iso_or_equiv' in table.")
    }

    #[test]
    fn test_t_direction_space_group_symop_operation() {
        let input = "#======================================================================
# CRYSTAL DATA
#----------------------------------------------------------------------
data_VESTA_phase_2

_chemical_name_common                  ''
_cell_length_a                         4.625(14)
_cell_length_b                         3.491(10)
_cell_length_c                         5.080(15)
_cell_angle_alpha                      90.000000
_cell_angle_beta                       99.10(18)
_cell_angle_gamma                      90.000000
_cell_volume                           80.988710
_space_group_name_H-M_alt              'C 2/c'
_space_group_IT_number                 15

loop_
_space_group_symop_operation_xyz
   'x, y, z'
   '-x, -y, -z'
   '-x, y, -z+1/2t'
   'x, -y, z+1/2t'
   'x+1/2t, y+1/2t, z'
   '-x+1/2t, -y+1/2t, -z'
   '-x+1/2t, y+1/2t, -z+1/2t'
   'x+1/2t, -y+1/2t, z+1/2t'

loop_
   _atom_site_label
   _atom_site_occupancy
   _atom_site_fract_x
   _atom_site_fract_y
   _atom_site_fract_z
   _atom_site_adp_type
   _atom_site_B_iso_or_equiv
   _atom_site_type_symbol
   Cu1        1.0     0.250000     0.250000     0.000000    Biso  1.000000 Cu
   O1         1.0     0.000000     0.418400     0.250000    Biso  1.000000 O

";
        let _ = CifParser::new(input)
            .parse()
            .expect("in symop t should be treated as constant and not throw error");
    }

    #[test]
    fn parse_cif_comment_in_loop_header() {
        let input = "data_comment_loop_header
_chemical_name_common                  ''
_cell_length_a                         4.625(14)
_cell_length_b                         3.491(10)
_cell_length_c                         5.080(15)
_cell_angle_alpha                      90.000000
_cell_angle_beta                       99.10(18)
_cell_angle_gamma                      90.000000
_cell_volume                           80.988710
_space_group_name_H-M_alt              'C 2/c'
_space_group_IT_number                 15

loop_
_space_group_symop_operation_xyz
   'x, y, z'
   '-x, -y, -z'
   '-x, y, -z+1/2'
   'x, -y, z+1/2'
   'x+1/2, y+1/2, z'
   '-x+1/2, -y+1/2, -z'
   '-x+1/2, y+1/2, -z+1/2'
   'x+1/2, -y+1/2, z+1/2'

loop_
   _atom_site_label
   _atom_site_occupancy
   _atom_site_fract_x
   _atom_site_fract_y
   _atom_site_fract_z
   _atom_site_adp_type
# this is a comment
   _atom_site_B_iso_or_equiv
   _atom_site_type_symbol
   Cu1        1.0     0.250000     0.250000     0.000000    Biso  1.000000 Cu
   O1         1.0     0.000000     0.418400     0.250000    Biso  1.000000 O

";
        let _ = CifParser::new(input)
            .parse()
            .expect("comment in loop header should be ok");
    }

    #[test]
    fn parse_cif_comment_in_loop() {
        let input = "data_comment_loop
_chemical_name_common                  ''
_cell_length_a                         4.625(14)
_cell_length_b                         3.491(10)
_cell_length_c                         5.080(15)
_cell_angle_alpha                      90.000000
_cell_angle_beta                       99.10(18)
_cell_angle_gamma                      90.000000
_cell_volume                           80.988710
_space_group_name_H-M_alt              'C 2/c'
_space_group_IT_number                 15

loop_
_space_group_symop_operation_xyz
   'x, y, z'
   '-x, -y, -z'
   '-x, y, -z+1/2'
   'x, -y, z+1/2'
   'x+1/2, y+1/2, z'
   '-x+1/2, -y+1/2, -z'
   '-x+1/2, y+1/2, -z+1/2'
   'x+1/2, -y+1/2, z+1/2'

loop_
   _atom_site_label
   _atom_site_occupancy
   _atom_site_fract_x
   _atom_site_fract_y
   _atom_site_fract_z
   _atom_site_adp_type
   _atom_site_B_iso_or_equiv
   _atom_site_type_symbol
# this is a comment
   Cu1        1.0     0.250000     0.250000     0.000000    Biso  1.000000 Cu # this is a comment
# another one
   O1         1.0     0.000000     0.418400     0.250000    Biso  1.000000 O

";
        let _ = CifParser::new(input)
            .parse()
            .expect("comment in loop header should be ok");
    }
}
