//! Orezanie záberu (región záujmu): zo širokého záberu ulice vezmeme len časť s cestou.
//! Zadáva sa v zlomkoch pôvodného obrazu (0-1), takže nezávisí od rozlíšenia videa.

use crate::Result;

/// Obdĺžnik v zlomkoch obrazu: ľavý horný roh (x, y) a rozmery (w, h).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Crop {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Koľko pixelov treba odrezať z každej strany (formát prvku `videocrop` v GStreamer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropPixels {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl Crop {
    /// Celý obraz (bez orezania).
    pub const FULL: Crop = Crop { x: 0.0, y: 0.0, w: 1.0, h: 1.0 };

    /// Spracuje text "x,y,w,h", napr. "0.0,0.4,0.5,0.4".
    pub fn parse(s: &str) -> Result<Crop> {
        let v: Vec<f64> = s
            .split(',')
            .map(|p| p.trim().parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| format!("--crop '{s}': očakávam štyri čísla x,y,w,h (napr. 0,0.4,0.5,0.4)"))?;
        let [x, y, w, h] = v[..] else {
            return Err(format!("--crop '{s}': očakávam presne štyri čísla x,y,w,h").into());
        };
        let c = Crop { x, y, w, h };
        c.validate()?;
        Ok(c)
    }

    fn validate(&self) -> Result<()> {
        const EPS: f64 = 1e-9;
        if !(self.x >= 0.0 && self.y >= 0.0) {
            return Err("--crop: x a y nesmú byť záporné".into());
        }
        if !(self.w > 0.0 && self.h > 0.0) {
            return Err("--crop: w a h musia byť kladné".into());
        }
        if self.x + self.w > 1.0 + EPS || self.y + self.h > 1.0 + EPS {
            return Err("--crop: obdĺžnik presahuje obraz (x+w a y+h musia byť najviac 1)".into());
        }
        Ok(())
    }

    /// Prepočet na pixely pre video `src_w` x `src_h`. Hodnoty sa zaokrúhľujú na párne čísla
    /// (formáty videa so zníženým rozlíšením farieb – napr. NV12 – to vyžadujú).
    pub fn to_pixels(&self, src_w: i32, src_h: i32) -> Result<CropPixels> {
        self.validate()?;
        let even = |v: f64| ((v.round() as i32) / 2) * 2;
        let left = even(self.x * src_w as f64);
        let top = even(self.y * src_h as f64);
        let right = src_w - even((self.x + self.w) * src_w as f64).min(src_w);
        let bottom = src_h - even((self.y + self.h) * src_h as f64).min(src_h);
        let (w, h) = (src_w - left - right, src_h - top - bottom);
        if w < 32 || h < 32 {
            return Err(format!("--crop je príliš malý: vychádza {w}x{h} px").into());
        }
        Ok(CropPixels { left, top, right: right.max(0), bottom: bottom.max(0) })
    }
}

impl CropPixels {
    /// Rozmery orezaného obrazu.
    pub fn size(&self, src_w: i32, src_h: i32) -> (i32, i32) {
        (src_w - self.left - self.right, src_h - self.top - self.bottom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsuje_styri_cisla() {
        assert_eq!(Crop::parse("0,0.4,0.5,0.4").unwrap(), Crop { x: 0.0, y: 0.4, w: 0.5, h: 0.4 });
        assert_eq!(Crop::parse(" 0.1 , 0.2 , 0.3 , 0.4 ").unwrap().w, 0.3);
    }

    #[test]
    fn chybne_vstupy() {
        for bad in ["", "1,2,3", "0,0,1,1,1", "a,b,c,d", "-0.1,0,0.5,0.5", "0,0,0,0.5", "0,0,0.5,0", "0.6,0,0.5,0.5", "0,0.6,0.5,0.5"] {
            assert!(Crop::parse(bad).is_err(), "'{bad}' mal zlyhať");
        }
    }

    #[test]
    fn cely_obraz_neodrezava_nic() {
        let p = Crop::FULL.to_pixels(3840, 2160).unwrap();
        assert_eq!(p, CropPixels { left: 0, top: 0, right: 0, bottom: 0 });
        assert_eq!(p.size(3840, 2160), (3840, 2160));
    }

    #[test]
    fn prepocet_na_pixely_4k() {
        // ľavá polovica, vodorovný pás od 40 % do 80 % výšky
        let p = Crop { x: 0.0, y: 0.4, w: 0.5, h: 0.4 }.to_pixels(3840, 2160).unwrap();
        assert_eq!(p.left, 0);
        assert_eq!(p.right, 1920);
        assert_eq!(p.top, 864);
        assert_eq!(p.bottom, 2160 - 1728);
        assert_eq!(p.size(3840, 2160), (1920, 864));
    }

    #[test]
    fn hodnoty_su_parne() {
        let p = Crop { x: 0.1234, y: 0.3333, w: 0.4321, h: 0.2222 }.to_pixels(1281, 721).unwrap();
        for v in [p.left, p.top] {
            assert_eq!(v % 2, 0, "{p:?}");
        }
    }

    #[test]
    fn prilis_maly_vyrez_zlyha() {
        assert!(Crop { x: 0.0, y: 0.0, w: 0.001, h: 0.001 }.to_pixels(3840, 2160).is_err());
    }

    #[test]
    fn pravy_dolny_okraj_nepretecie() {
        let p = Crop { x: 0.5, y: 0.5, w: 0.5, h: 0.5 }.to_pixels(1000, 1000).unwrap();
        assert!(p.right >= 0 && p.bottom >= 0);
        assert_eq!(p.size(1000, 1000), (500, 500));
    }
}
