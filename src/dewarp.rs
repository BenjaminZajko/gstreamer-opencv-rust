//! Rovný obraz (dewarp): odstráni skreslenie fisheye objektívu a vyrovná náklon kamery.
//!
//! Parametre pripraví `calib/make_dewarp.py` do `dewarp.json` (pre záber 3840x2160):
//! objektív (`lens_f`, `lens_k1`, `lens_k2`, `lens_cx`, `lens_cy`), poloha skutočnej kamery
//! (`level_pitch`, `level_roll`), sklon virtuálnej kamery (`virtual_pitch`) a výstup
//! (`out_f`, `out_cx`, `out_cy`). Výstup je zvolený tak, aby nemal čierne okraje.
//!
//! Prevod sa pri prvej snímke predpočíta do tabuliek (remap), potom stojí pár ms na snímku.

use opencv::{
    core::{Mat, Scalar, Size, CV_16SC2, CV_32FC1},
    imgproc,
    prelude::*,
};

use crate::{speed::json_number, Result};

/// Parametre prevodu (v pixeloch záberu `width` x `height`).
#[derive(Debug, Clone, PartialEq)]
pub struct DewarpParams {
    pub width: f64,
    pub height: f64,
    pub lens_f: f64,
    pub lens_k1: f64,
    pub lens_k2: f64,
    pub lens_cx: f64,
    pub lens_cy: f64,
    pub level_pitch: f64,
    pub level_roll: f64,
    pub virtual_pitch: f64,
    pub out_f: f64,
    pub out_cx: f64,
    pub out_cy: f64,
}

impl DewarpParams {
    pub fn from_json(text: &str) -> Result<Self> {
        let n = |k: &str| json_number(text, k);
        Ok(Self {
            width: n("width")?,
            height: n("height")?,
            lens_f: n("lens_f")?,
            lens_k1: n("lens_k1")?,
            lens_k2: n("lens_k2")?,
            lens_cx: n("lens_cx")?,
            lens_cy: n("lens_cy")?,
            level_pitch: n("level_pitch")?,
            level_roll: n("level_roll")?,
            virtual_pitch: n("virtual_pitch").unwrap_or(0.0),
            out_f: n("out_f")?,
            out_cx: n("out_cx")?,
            out_cy: n("out_cy")?,
        })
    }

    pub fn load(path: &str) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("dewarp '{path}': {e}"))?;
        Self::from_json(&text).map_err(|e| format!("dewarp '{path}': {e}").into())
    }

    /// Pixel rovného obrazu -> pixel pôvodného (oba v rozlíšení `width` x `height`).
    /// `None`, ak lúč smeruje za kameru.
    pub fn source_pixel(&self, u: f64, v: f64) -> Option<(f64, f64)> {
        let (a, b) = ((u - self.out_cx) / self.out_f, (v - self.out_cy) / self.out_f);
        // smer vo svete (X doprava, Y dopredu, Z hore) z virtuálnej kamery
        let (svp, cvp) = self.virtual_pitch.sin_cos();
        let (dx, dy, dz) = (a, -b * svp + cvp, -b * cvp - svp);
        // do skutočnej kamery (sklon, potom náklon)
        let (sp, cp) = self.level_pitch.sin_cos();
        let fwd = dy * cp - dz * sp;
        if fwd <= 1e-6 {
            return None;
        }
        let down = -dy * sp - dz * cp;
        let (xr, yr) = (dx / fwd, down / fwd);
        let (s, c) = self.level_roll.sin_cos();
        let (x, y) = (c * xr + s * yr, -s * xr + c * yr);
        // skreslenie objektívu: hľadá sa rd, pre ktoré rd (1 + k1 rd² + k2 rd⁴) = ru
        let ru = x.hypot(y);
        let mut rd = ru;
        for _ in 0..30 {
            let r2 = rd * rd;
            let g = rd * (1.0 + self.lens_k1 * r2 + self.lens_k2 * r2 * r2) - ru;
            let dg = 1.0 + 3.0 * self.lens_k1 * r2 + 5.0 * self.lens_k2 * r2 * r2;
            if dg <= 0.0 {
                return None;
            }
            rd -= g / dg;
        }
        let k = if ru > 1e-12 { rd / ru } else { 1.0 };
        Some((self.lens_cx + self.lens_f * x * k, self.lens_cy + self.lens_f * y * k))
    }
}

/// Predpočítaný prevod pre jedno rozlíšenie snímok.
pub struct Dewarp {
    map1: Mat,
    map2: Mat,
    size: Size,
}

impl Dewarp {
    /// Tabuľky pre snímky `size` (výstup má rovnaký rozmer).
    pub fn new(p: &DewarpParams, size: Size) -> Result<Self> {
        let k = size.width as f64 / p.width;
        let (mut mx, mut my) = (
            Mat::new_size_with_default(size, CV_32FC1, Scalar::all(-1.0))?,
            Mat::new_size_with_default(size, CV_32FC1, Scalar::all(-1.0))?,
        );
        for v in 0..size.height {
            for u in 0..size.width {
                if let Some((su, sv)) = p.source_pixel(u as f64 / k, v as f64 / k) {
                    *mx.at_2d_mut::<f32>(v, u)? = (su * k) as f32;
                    *my.at_2d_mut::<f32>(v, u)? = (sv * k) as f32;
                }
            }
        }
        // rýchlejší formát tabuliek (celé čísla + zlomky)
        let (mut map1, mut map2) = (Mat::default(), Mat::default());
        imgproc::convert_maps(&mx, &my, &mut map1, &mut map2, CV_16SC2, false)?;
        Ok(Self { map1, map2, size })
    }

    pub fn size(&self) -> Size {
        self.size
    }

    /// Rovný obraz zo snímky (musí mať rozmer `size`).
    pub fn apply(&self, frame: &Mat) -> Result<Mat> {
        let mut out = Mat::default();
        imgproc::remap_def(frame, &mut out, &self.map1, &self.map2, imgproc::INTER_LINEAR)?;
        Ok(out)
    }
}

/// Prevod, ktorý si tabuľky pripraví sám pri prvej snímke (a znova, ak sa zmení rozmer).
pub struct LazyDewarp {
    params: DewarpParams,
    inner: Option<Dewarp>,
}

impl LazyDewarp {
    pub fn new(params: DewarpParams) -> Self {
        Self { params, inner: None }
    }

    pub fn apply(&mut self, frame: &Mat) -> Result<Mat> {
        let size = frame.size()?;
        if self.inner.as_ref().is_none_or(|d| d.size() != size) {
            self.inner = Some(Dewarp::new(&self.params, size)?);
        }
        self.inner.as_ref().unwrap().apply(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> DewarpParams {
        DewarpParams {
            width: 3840.0,
            height: 2160.0,
            lens_f: 2715.0,
            lens_k1: 0.34,
            lens_k2: 1.15,
            lens_cx: 2007.0,
            lens_cy: 1070.0,
            level_pitch: 0.115,
            level_roll: -0.016,
            virtual_pitch: 0.115,
            out_f: 2549.0,
            out_cx: 2111.0,
            out_cy: 1080.0,
        }
    }

    #[test]
    fn nacita_json() {
        let text = r#"{"width": 3840, "height": 2160, "lens_f": 2715.0, "lens_k1": 0.34, "lens_k2": 1.15,
            "lens_cx": 2007, "lens_cy": 1070, "level_pitch": 0.115, "level_roll": -0.016, "virtual_pitch": 0.115,
            "out_f": 2549, "out_cx": 2111, "out_cy": 1080}"#;
        assert_eq!(DewarpParams::from_json(text).unwrap(), params());
        assert!(DewarpParams::from_json(r#"{"width": 1}"#).is_err());
    }

    #[test]
    fn bez_skreslenia_a_naklonu_je_to_len_zmena_mierky() {
        let p = DewarpParams { lens_k1: 0.0, lens_k2: 0.0, level_pitch: 0.0, level_roll: 0.0, virtual_pitch: 0.0,
            lens_cx: 1920.0, lens_cy: 1080.0, out_cx: 1920.0, out_cy: 1080.0, lens_f: 2000.0, out_f: 1000.0, ..params() };
        let (u, v) = p.source_pixel(2020.0, 1130.0).unwrap();
        assert!((u - 2120.0).abs() < 1e-6 && (v - 1180.0).abs() < 1e-6, "{u} {v}");
    }

    #[test]
    fn rovny_obraz_ide_dalej_do_rohov_ako_fisheye() {
        // okraj rovného obrazu zodpovedá bodu bližšie k stredu pôvodného (fisheye ho stláča)
        let p = params();
        let (u, _) = p.source_pixel(0.0, 1080.0).unwrap();
        assert!(u > 0.0 && u < 1000.0, "{u}");
        // stred objektívu sa zobrazí sám na seba pri rovnakom ohnisku a bez natočenia
        let q = DewarpParams { level_pitch: 0.0, level_roll: 0.0, virtual_pitch: 0.0, out_f: 2715.0, out_cx: 2007.0, out_cy: 1070.0, ..p };
        let (u, v) = q.source_pixel(2007.0, 1070.0).unwrap();
        assert!((u - 2007.0).abs() < 1e-6 && (v - 1070.0).abs() < 1e-6);
    }

    #[test]
    fn tabulky_maju_rozmer_snimky_a_posuvaju_obraz() {
        let p = params();
        let size = Size::new(384, 216);
        let d = Dewarp::new(&p, size).unwrap();
        assert_eq!(d.size(), size);
        // vodorovný gradient: po prevode musí ostať rastúci zľava doprava v strede
        let mut img = Mat::new_size_with_default(size, opencv::core::CV_8UC1, Scalar::all(0.0)).unwrap();
        for v in 0..216 {
            for u in 0..384 {
                *img.at_2d_mut::<u8>(v, u).unwrap() = (u * 255 / 383) as u8;
            }
        }
        let out = d.apply(&img).unwrap();
        assert_eq!(out.size().unwrap(), size);
        let (a, b) = (*out.at_2d::<u8>(108, 50).unwrap(), *out.at_2d::<u8>(108, 330).unwrap());
        assert!(a < b && a > 0, "{a} {b}");
    }
}
