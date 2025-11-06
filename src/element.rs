#[derive(PartialEq, Eq, Debug, Clone, Copy)]
#[rustfmt::skip]
pub enum Element {
    //   1   2                                                           3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
        H,                                                                                                                          He,
        Li, Be,                                                                                                 B,  C,  N,  O,  F,  Ne,
        Na, Mg,                                                                                                 Al, Si, P,  S,  Cl, Ar,
        K,  Ca,                                                         Sc, Ti, V,  Cr, Mn, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Br, Kr,
        Rb, Sr,                                                         Y,  Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I,  Xe,
        Cs, Ba, La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf, Ta, W,  Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn,
        Fr, Ra, Ac, Th, Pa, U,  Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr, Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Nh, Fl, Mc, Lv, Ts, Og,
}

impl std::fmt::Display for Element {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Element {
    /// return atomic number
    /// ```
    /// use yaxs::element::Element;
    ///
    /// assert_eq!(Element::H.z(), 1);
    /// assert_eq!(Element::He.z(), 2);
    /// assert_eq!(Element::Li.z(), 3);
    /// assert_eq!(Element::Db.z(), 105);
    /// ```
    pub fn z(&self) -> i32 {
        *self as i32 + 1
    }
}

impl TryFrom<&str> for Element {
    type Error = String;

    fn try_from(val: &str) -> Result<Self, Self::Error> {
        use Element::*;
        let el = match val {
            "H" => H,
            "He" => He,
            "Li" => Li,
            "Be" => Be,
            "B" => B,
            "C" => C,
            "N" => N,
            "O" => O,
            "F" => F,
            "Ne" => Ne,
            "Na" => Na,
            "Mg" => Mg,
            "Al" => Al,
            "Si" => Si,
            "P" => P,
            "S" => S,
            "Cl" => Cl,
            "Ar" => Ar,
            "K" => K,
            "Ca" => Ca,
            "Sc" => Sc,
            "Ti" => Ti,
            "V" => V,
            "Cr" => Cr,
            "Mn" => Mn,
            "Fe" => Fe,
            "Co" => Co,
            "Ni" => Ni,
            "Cu" => Cu,
            "Zn" => Zn,
            "Ga" => Ga,
            "Ge" => Ge,
            "As" => As,
            "Se" => Se,
            "Br" => Br,
            "Kr" => Kr,
            "Rb" => Rb,
            "Sr" => Sr,
            "Y" => Y,
            "Zr" => Zr,
            "Nb" => Nb,
            "Mo" => Mo,
            "Tc" => Tc,
            "Ru" => Ru,
            "Rh" => Rh,
            "Pd" => Pd,
            "Ag" => Ag,
            "Cd" => Cd,
            "In" => In,
            "Sn" => Sn,
            "Sb" => Sb,
            "Te" => Te,
            "I" => I,
            "Xe" => Xe,
            "Cs" => Cs,
            "Ba" => Ba,
            "La" => La,
            "Ce" => Ce,
            "Pr" => Pr,
            "Nd" => Nd,
            "Pm" => Pm,
            "Sm" => Sm,
            "Eu" => Eu,
            "Gd" => Gd,
            "Tb" => Tb,
            "Dy" => Dy,
            "Ho" => Ho,
            "Er" => Er,
            "Tm" => Tm,
            "Yb" => Yb,
            "Lu" => Lu,
            "Hf" => Hf,
            "Ta" => Ta,
            "W" => W,
            "Re" => Re,
            "Os" => Os,
            "Ir" => Ir,
            "Pt" => Pt,
            "Au" => Au,
            "Hg" => Hg,
            "Tl" => Tl,
            "Pb" => Pb,
            "Bi" => Bi,
            "Po" => Po,
            "At" => At,
            "Rn" => Rn,
            "Fr" => Fr,
            "Ra" => Ra,
            "Ac" => Ac,
            "Th" => Th,
            "Pa" => Pa,
            "U" => U,
            "Np" => Np,
            "Pu" => Pu,
            "Am" => Am,
            "Cm" => Cm,
            "Bk" => Bk,
            "Cf" => Cf,
            "Es" => Es,
            "Fm" => Fm,
            "Md" => Md,
            "No" => No,
            "Lr" => Lr,
            "Rf" => Rf,
            "Db" => Db,
            "Sg" => Sg,
            "Bh" => Bh,
            "Hs" => Hs,
            "Mt" => Mt,
            "Ds" => Ds,
            "Rg" => Rg,
            "Cn" => Cn,
            "Nh" => Nh,
            "Fl" => Fl,
            "Mc" => Mc,
            "Lv" => Lv,
            "Ts" => Ts,
            "Og" => Og,
            _ => return Err(format!("Invalid Element Symbol '{val}'")),
        };
        Ok(el)
    }
}
