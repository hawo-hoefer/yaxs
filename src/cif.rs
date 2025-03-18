use std::collections::HashMap;
use std::num::ParseFloatError;

use itertools::Itertools;
use nalgebra::{Matrix3, Vector3};

use crate::species::Species;
use crate::structure::Lattice;
use crate::site::Site;
use crate::symop::SymOp;

// TODO: make this case-insensitive
const DATA_HEADER_START: &'static str = "data_";
const LOOP_HEADER_START: &'static str = "loop_";
const ANGLE_KEYS: [&'static str; 3] =
    ["_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"];
const LENGTH_KEYS: [&'static str; 3] = ["_cell_length_a", "_cell_length_b", "_cell_length_c"];
const SITE_DIST_TOL: f64 = 1e-8;

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct CifParser<'a> {
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

pub struct CIFContents {
    pub block_name: String,
    pub kvs: HashMap<String, Value>,
    pub tables: Vec<HashMap<String, Vec<Value>>>,
}

impl CIFContents {
    pub fn get_symops(&self) -> Vec<SymOp> {
        let Some(symops_table) = self.tables.iter().find(|t: &&Table| {
            const SITE_KEYS: [&'static str; 2] = [
                "_space_group_symop_id",
                "_space_group_symop_operation_xyz", // TODO: handle alternative table with _space_group_symop_equiv_pos_as_xyz
            ];
            SITE_KEYS.iter().map(|&k| t.contains_key(k)).all(|x| x)
        }) else {
            panic!("No atom site label info in CIF")
        };
        symops_table["_space_group_symop_operation_xyz"]
            .iter()
            .map(|s| {
                let Value::Text(s) = s else {
                    // TODO: error handling
                    panic!("Invalid symmetry operation. needs to be a string");
                };
                // TODO: error handling of symop invalid
                s.parse::<SymOp>().unwrap()
            })
            .collect_vec()
    }
    pub fn get_volume(&self) -> f64 {
        self.kvs.get("_cell_volume").unwrap().try_to_f64().unwrap()
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
        let va = Vector3::new(a * beta.sin(), 0.0, a * beta.cos());
        let vb = Vector3::new(
            -b * alpha.sin() * gamma_star.cos(),
            b * alpha.sin() * gamma_star.sin(),
            b * alpha.cos(),
        );
        let vc = Vector3::new(0.0, 0.0, c);
        Lattice {
            mat: Matrix3::from_columns(&[va, vb, vc]),
        }
    }

    pub fn get_sites(&self) -> Vec<Site> {
        let Some(site_table) = self.tables.iter().find(|t: &&Table| {
            const SITE_KEYS: [&'static str; 5] = [
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
        let symops = self.get_symops();

        let site_at_index = |i: usize| -> Site {
            let label = &site_table["_atom_site_type_symbol"][i];
            let sp: Species = match label {
                Value::Text(label) => label.parse().unwrap(),
                _ => todo!("Invalid site label"),
            };
            let occu = site_table["_atom_site_occupancy"][i].try_to_f64().unwrap();
            // TODO: remove unwraps, proper error handling
            let coords = Vector3::new(
                site_table["_atom_site_fract_x"][i].try_to_f64().unwrap(),
                site_table["_atom_site_fract_y"][i].try_to_f64().unwrap(),
                site_table["_atom_site_fract_z"][i].try_to_f64().unwrap(),
            );
            Site {
                species: sp,
                coords,
                occu,
            }
        };

        fn site_exists_periodic(site: &Site, sites: &[Site]) -> bool {
            // adapted from pymatgen.util.coord.find_in_coord_list
            sites
                .iter()
                .map(|ps| {
                    let dist = site.coords - ps.coords;
                    dist.map(|x| (x - x.round()).abs() < SITE_DIST_TOL)
                        .iter()
                        .all(|x| *x)
                })
                .any(|x| x)
        }

        // we parsed the symops, but still need to remove duplicate sites
        let mut sites = Vec::new();
        for base_site in (0..n).map(site_at_index) {
            // if site_exists_periodic(&base_site, &sites) {
            //     continue;
            // }
            sites.push(base_site.normalized());

            for op in symops.iter() {
                let s = Site {
                    coords: op.apply(base_site.coords.clone()),
                    species: base_site.species.clone(),
                    occu: base_site.occu.clone(),
                };

                if site_exists_periodic(&base_site, &sites) {
                    continue;
                }
                sites.push(s.normalized());
            }
        }

        for site in (0..n)
            .map(site_at_index)
            .map(|base_site| {
                symops.iter().map(move |s| Site {
                    coords: s.apply(base_site.coords.clone()),
                    species: base_site.species.clone(),
                    occu: base_site.occu.clone(),
                })
            })
            .flatten()
        {
            if site_exists_periodic(&site, &sites) {
                continue;
            }
            sites.push(site.normalized())
        }
        sites
    }
}

impl<'a> CifParser<'a> {
    pub fn new(data: &'a str) -> Self {
        Self { c: data }
    }

    pub fn parse(&mut self) -> CIFContents {
        self.skip_ws_comments();
        let bn = self
            .parse_block_name()
            .expect("need block name")
            .to_string();
        let mut kvs = HashMap::new();
        let mut tables = Vec::new();
        while !self.c.is_empty() {
            self.skip_ws_comments();
            if self.c.starts_with(DATA_HEADER_START) {
                todo!("handle multiple data blocks")
            }
            match self.parse_data_item() {
                DataItem::KV(k, v) => {
                    if kvs.contains_key(&k) {
                        todo!("Duplicate Key '{k}' in CIF");
                    };
                    let _ = kvs.insert(k, v);
                }
                DataItem::Table(table) => {
                    tables.push(table);
                }
            }
        }
        CIFContents {
            block_name: bn,
            kvs,
            tables,
        }
    }

    fn skip_comments(&mut self) {
        while self.c.starts_with('#') {
            // skip comments
            let next_line = self
                .c
                .find('\n')
                .expect("only a comment line is left - cannot parse a block name")
                + 1;
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

    fn parse_value(&mut self) -> Value {
        self.skip_whitespace();
        match self
            .c
            .chars()
            .next()
            .expect("cannot parse anything on empty string")
        {
            '+' | '-' | '0'..='9' => {
                // try to parse number. if it fails, we parse as string
                self.parse_number().unwrap_or_else(|_e| self.parse_text())
            }
            '.' => {
                let _ = self.consume_once('.');
                Value::Inapplicable
            }
            '?' => {
                let _ = self.consume_once('?');
                Value::Unknown
            }
            _ => self.parse_text(), // this includes ';'
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

    fn parse_tag(&mut self) -> &str {
        let Some((tag, c0)) = self.c.split_once(|x: char| x.is_whitespace()) else {
            panic!("Incomplete value")
        };
        self.c = c0;
        tag
    }

    fn parse_data_item(&mut self) -> DataItem {
        if self.c.starts_with('_') {
            // we are reading a tag
            let tag = self.parse_tag().to_string();
            let val = self.parse_value();
            return DataItem::KV(tag.to_string(), val);
        } else if self.c.starts_with(LOOP_HEADER_START) {
            return DataItem::Table(self.parse_loop());
        }

        panic!("WTF Where are we? '{d}'", d = self.c)
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

    fn parse_text(&mut self) -> Value {
        match self.c.chars().next() {
            Some(c) if matches!(c, '\'' | '\"' | ';') => {
                assert!(self.consume_once(c));
                let (text, rest) = self
                    .c
                    .split_once(c)
                    .unwrap_or_else(|| todo!("handle unterminated '{c}'-string"));
                self.c = rest;
                Value::Text(text.to_string())
            }
            Some(_) => {
                // UnquotedString
                let (text, rest) = self
                    .c
                    .split_once(|x: char| x.is_whitespace())
                    .unwrap_or((self.c, ""));
                self.c = rest;
                Value::Text(text.to_string())
            }
            None => todo!("Cannot Parse from empty contents"),
        }
    }

    fn parse_number(&mut self) -> Result<Value, ParseFloatError> {
        let (mut text, rest) = self
            .c
            .split_once(|x: char| x.is_whitespace())
            .unwrap_or((self.c, ""));

        if let Some((num, p_range)) = text.split_once('(') {
            if !p_range.ends_with(')') {
                todo!("Handle unterminated precision. missing ')'")
            }

            // NOTE: we probably are handling this wrong.
            // what if the input number is an integer and has precision - like 12312(12)
            // or if there is a float like 1.234e17(123)?
            // in these cases, we crash and burn, we should probably error gracefully
            if num.find(|x: char| x == 'e').is_some() // float in scientific notation
                || num.find(|x: char| x == '.' || x == 'e').is_none()
            // is an integer because no decimal point or scientific notation 'e'
            {
                todo!("Handle precision brackets after integer or scientific notation")
            }

            text = num;
        }

        if let Ok(v) = text.parse::<i32>() {
            self.c = rest;
            return Ok(Value::Int(v));
        }

        let v = text.parse::<f64>()?;
        self.c = rest;
        Ok(Value::Float(v))
    }

    fn parse_loop(&mut self) -> Table {
        self.c = std::str::from_utf8(&self.c.as_bytes()[LOOP_HEADER_START.len()..]).unwrap();
        self.skip_whitespace();
        let mut kvs = Vec::new();
        while self.c.starts_with('_') {
            // while tokens start with '_', we are reading column names
            kvs.push((self.parse_tag().to_string(), Vec::new()));
        }

        while !self.c.starts_with('_')
            && !self.c.starts_with(LOOP_HEADER_START)
            && !self.c.is_empty()
        {
            for (_, v) in kvs.iter_mut() {
                let val = self.parse_value();
                v.push(val);
            }
            self.skip_ws_comments();
        }

        kvs.drain(..).collect()
    }
}

#[cfg(test)]
mod test {
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
        let v = p.parse_number().unwrap();
        assert_eq!(v, Value::Float(1.2123));
        assert_eq!(p.c, "aroistena");
    }

    #[test]
    fn parse_int_test() {
        let mut p = CifParser::new("12123 arstr");
        let v = p.parse_number().unwrap();
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
        let table = p.parse_loop();
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
        let di = p.parse_data_item();
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
        let vals = p.parse();
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
        let DataItem::Table(item) = p.parse_data_item() else {
            panic!()
        };
        assert_eq!(
            item["_sym"],
            [Text("Hf2+".to_string()), Text("He2-".to_string())]
        );
        assert_eq!(item["_ox"], [Int(2), Text("-2".to_string())]);
        let DataItem::Table(item) = p.parse_data_item() else {
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
        let item = p.parse_data_item();
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
        let DataItem::Table(item) = p.parse_data_item() else {
            panic!()
        };
        assert_eq!(
            item["_sym"],
            [Text("Hf2+".to_string()), Text("He2-".to_string())]
        );
        assert_eq!(item["_ox"], [Int(2), Float(2.0)]);
        let DataItem::Table(item) = p.parse_data_item() else {
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
        let DataItem::KV(_, Value::Inapplicable) = p.parse_data_item() else {
            panic!()
        };

        p.skip_whitespace();
        let DataItem::KV(_, Value::Unknown) = p.parse_data_item() else {
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
        let CIFContents {
            block_name,
            kvs,
            tables,
        } = p.parse();
        let kvs_exp = HashMap::from([("_data".to_string(), Value::Int(1234))]);
        assert_eq!(kvs, kvs_exp);
        assert_eq!(tables.len(), 1);
    }
}
