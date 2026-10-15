//! Kreslenie výsledkov detekcie do snímky.

use opencv::{
    core::{Mat, Point, Rect, Scalar},
    imgproc,
    prelude::*,
};

use crate::{detector::Detection, Result};

/// Nakreslí zelené rámčeky s popisom okolo každej detekcie a počítadlo vpravo hore.
pub fn draw(frame: &mut Mat, dets: &[Detection], idx: u64) -> Result<()> {
    let green = Scalar::new(0.0, 255.0, 0.0, 0.0); // farba je v poradí B, G, R
    for det in dets {
        let r = det.rect;
        // rámček hrúbky 2 px
        imgproc::rectangle(frame, r, green, 2, imgproc::LINE_8, 0)?;
        // popis + istota v percentách (ak ju detektor má)
        let label = match det.score {
            Some(s) => format!("{} {:.0}%", det.label, s * 100.0),
            None => det.label.clone(),
        };
        // popis tesne nad rámčekom (max(14) zabráni, aby text vyšiel mimo obrazu)
        imgproc::put_text(
            frame,
            &label,
            Point::new(r.x, (r.y - 6).max(14)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6, // veľkosť písma
            green,
            2,                // hrúbka
            imgproc::LINE_AA, // vyhladené hrany
            false,
        )?;
    }
    // Počítadlo v pravom hornom rohu (vľavo hore je časová pečiatka kamery).
    // Tmavé pozadie, aby bolo čitateľné aj na svetlom obraze.
    imgproc::rectangle(
        frame,
        Rect::new(frame.cols() - 440, 0, 440, 42),
        Scalar::all(0.0),
        imgproc::FILLED,
        imgproc::LINE_8,
        0,
    )?;
    let text = format!("frame {idx}  objektov: {}", dets.len());
    imgproc::put_text(
        frame,
        &text,
        Point::new(frame.cols() - 430, 28),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.8,
        Scalar::new(255.0, 255.0, 255.0, 0.0), // žltá  //benji, prve cislo som vymenil z 0.0 na 255.0
        2,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Scalar, Vec3b};

    fn det(r: Rect, label: &str, score: Option<f32>) -> Detection {
        Detection { rect: r, label: label.into(), score }
    }

    const W: i32 = 640;
    const H: i32 = 480;

    fn black() -> Mat {
        Mat::new_rows_cols_with_default(H, W, opencv::core::CV_8UC3, Scalar::all(0.0)).unwrap()
    }

    fn px(m: &Mat, x: i32, y: i32) -> [u8; 3] {
        let v = m.at_2d::<Vec3b>(y, x).unwrap();
        [v[0], v[1], v[2]]
    }

    /// Je v obdĺžniku aspoň jeden nenulový pixel?
    fn has_ink(m: &Mat, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
        (y0..y1).any(|y| (x0..x1).any(|x| px(m, x, y) != [0, 0, 0]))
    }

    #[test]
    fn nakresli_zeleny_ramcek() {
        let mut f = black();
        draw(&mut f, &[det(Rect::new(100, 200, 80, 60), "pohyb", None)], 0).unwrap();
        assert_eq!(px(&f, 100, 230), [0, 255, 0], "ľavá hrana rámčeka (BGR zelená)");
        assert_eq!(px(&f, 140, 200), [0, 255, 0], "horná hrana rámčeka");
        assert_eq!(px(&f, 140, 230), [0, 0, 0], "vnútro rámčeka ostáva nedotknuté");
    }

    #[test]
    fn nakresli_popis_nad_ramcek() {
        let mut f = black();
        draw(&mut f, &[det(Rect::new(100, 200, 80, 60), "pohyb", None)], 0).unwrap();
        assert!(has_ink(&f, 100, 180, 180, 199), "text má byť tesne nad rámčekom");
    }

    #[test]
    fn pocitadlo_je_vzdy_vpravo_hore() {
        let mut f = black();
        draw(&mut f, &[], 7).unwrap();
        assert!(has_ink(&f, W - 430, 5, W, 40));
        assert!(!has_ink(&f, 0, 0, 200, 40), "vľavo hore je pečiatka kamery, nesmie sa prekrývať");
    }

    #[test]
    fn popis_nevyjde_mimo_obrazu_pri_hornom_okraji() {
        let mut f = black();
        // rámček úplne hore: y - 6 by bolo záporné, text sa musí orezať na 14
        draw(&mut f, &[det(Rect::new(50, 0, 40, 40), "x", None)], 0).unwrap();
        assert!(has_ink(&f, 50, 0, 100, 20));
    }

    #[test]
    fn viac_ramcekov_naraz() {
        let mut f = black();
        let boxes = [
            det(Rect::new(10, 100, 50, 50), "clovek", Some(0.9)),
            det(Rect::new(300, 300, 50, 50), "clovek", Some(0.8)),
        ];
        draw(&mut f, &boxes, 3).unwrap();
        assert_eq!(px(&f, 10, 120), [0, 255, 0]);
        assert_eq!(px(&f, 300, 320), [0, 255, 0]);
    }

    #[test]
    fn popis_obsahuje_istotu_v_percentach() {
        // Detekcia s istotou má dlhší popis ("clovek 93%") než bez nej ("clovek").
        let ink_width = |score: Option<f32>| {
            let mut f = black();
            draw(&mut f, &[det(Rect::new(100, 200, 300, 60), "clovek", score)], 0).unwrap();
            (0..W).filter(|&x| (180..199).any(|y| px(&f, x, y) != [0, 0, 0])).count()
        };
        assert!(ink_width(Some(0.93)) > ink_width(None) + 15);
    }

    #[test]
    fn pocitadlo_ma_tmave_pozadie_na_svetlom_obraze() {
        let mut f = Mat::new_rows_cols_with_default(H, W, opencv::core::CV_8UC3, Scalar::all(255.0)).unwrap();
        draw(&mut f, &[], 0).unwrap();
        assert_eq!(px(&f, W - 435, 3), [0, 0, 0], "pod počítadlom má byť čierny pásik");
        assert_eq!(px(&f, 100, 300), [255, 255, 255], "zvyšok obrazu sa nemení");
    }
}
