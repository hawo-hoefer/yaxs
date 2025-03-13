use std::collections::HashMap;
use std::num::ParseFloatError;

// TODO: make this case-insensitive
const DATA_HEADER_START: &'static str = "data_";
const LOOP_HEADER_START: &'static str = "loop_";
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
        match self
            .c
            .chars()
            .next()
            .expect("cannot parse anything on empty string")
        {
            '+' | '-' | '0'..='9' => {
                // try to parse number. if it fails, we parse as string
                self.parse_number().unwrap_or_else(|e| self.parse_text())
            }
            '.' => Value::Inapplicable,
            '?' => Value::Unknown,
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
            self.skip_whitespace();
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
                self.skip_whitespace();
                v.push(self.parse_value());
            }
            self.skip_whitespace();
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
}
