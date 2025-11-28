use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
#[serde(deny_unknown_fields)]
pub struct Linspace {
    pub range: (f64, f64),
    pub steps: usize,
}

#[derive(Clone)]
pub struct LinspaceIter {
    inner: Linspace,
    pos: usize,
}

impl Iterator for LinspaceIter {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.inner.steps {
            return None;
        }

        if self.inner.steps == 1 {
            self.pos += 1;
            return Some(self.inner.range.0);
        }

        let ret = (self.inner.range.1 - self.inner.range.0) * self.pos as f64
            / (self.inner.steps - 1) as f64
            + self.inner.range.0;

        self.pos += 1;

        Some(ret)
    }
}

impl IntoIterator for Linspace {
    type Item = f64;

    type IntoIter = LinspaceIter;

    fn into_iter(self) -> Self::IntoIter {
        LinspaceIter {
            inner: self,
            pos: 0,
        }
    }
}



#[derive(Serialize, Deserialize, Clone, Debug, Copy)]
#[serde(deny_unknown_fields)]
pub struct TextureMeasurement {
    pub phi: Linspace,
    pub chi: Linspace,
}

impl TextureMeasurement {
    pub fn stride(&self) -> usize {
        self.chi.steps * self.phi.steps
    }
}


