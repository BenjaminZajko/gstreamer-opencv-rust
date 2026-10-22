//! Kreslenie výsledkov detekcie do snímky.

use opencv::{
    core::{Mat, Point, Rect, Scalar},
    imgproc,
    prelude::*,
};

use crate::{
    counter::{CountsView, Direction, Group},
    detector::Detection,
    Result,
};

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

/// Detekcia, ktorá je zároveň sledovaná (má stále ID) – vstup pre `draw_counting`.
#[derive(Debug, Clone)]
pub struct Counted {
    pub rect: Rect,
    pub label: String,
    pub score: Option<f32>,
    pub id: u32,
    pub group: Group,
}

/// Nakreslí zvislú čiaru, sledované objekty (farba podľa skupiny, s ID) a panel s počtami.
/// Text je bez diakritiky, lebo písmo OpenCV ju nevie vykresliť.
pub fn draw_counting(frame: &mut Mat, items: &[Counted], counts: &CountsView) -> Result<()> {
    // čiara (žltá): zadané krajné body (naklonená po otočení kamery), inak zvislá čiara
    // cez celú výšku alebo len v pásme, kde sa počíta
    let ((x0, y0), (x1, y1)) = match counts.line_pts {
        Some((a, b)) => (
            (a.0.round() as i32, a.1.round() as i32),
            (b.0.round() as i32, b.1.round() as i32),
        ),
        None => {
            let x = counts.line_x.round() as i32;
            let (y0, y1) = match counts.span {
                Some((a, b)) => (a.round() as i32, b.round() as i32),
                None => (0, frame.rows()),
            };
            ((x, y0), (x, y1))
        }
    };
    imgproc::line(
        frame,
        Point::new(x0, y0),
        Point::new(x1, y1),
        Scalar::new(0.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;

    for it in items {
        let color = match it.group {
            Group::Vehicle => Scalar::new(0.0, 140.0, 255.0, 0.0), // oranžová (BGR)
            Group::Person => Scalar::new(0.0, 255.0, 0.0, 0.0),    // zelená
        };
        imgproc::rectangle(frame, it.rect, color, 2, imgproc::LINE_8, 0)?;
        let text = match it.score {
            Some(s) => format!("{} #{} {:.0}%", it.label, it.id, s * 100.0),
            None => format!("{} #{}", it.label, it.id),
        };
        imgproc::put_text(
            frame,
            &text,
            Point::new(it.rect.x, (it.rect.y - 6).max(14)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            imgproc::LINE_AA,
            false,
        )?;
    }

    // panel s počtami vľavo hore na tmavom pozadí
    imgproc::rectangle(
        frame,
        Rect::new(0, 0, 430, 96),
        Scalar::all(0.0),
        imgproc::FILLED,
        imgproc::LINE_8,
        0,
    )?;
    let row = |g: Group, name: &str| {
        format!(
            "{name}: {}  (-> {}, <- {})",
            counts.total(g),
            counts.count(g, Direction::LeftToRight),
            counts.count(g, Direction::RightToLeft)
        )
    };
    let lines = [
        (row(Group::Vehicle, "AUTA"), Scalar::new(0.0, 140.0, 255.0, 0.0)),
        (row(Group::Person, "LUDIA"), Scalar::new(0.0, 255.0, 0.0, 0.0)),
        (
            counts.footer.clone(),
            if counts.warning { Scalar::new(0.0, 0.0, 255.0, 0.0) } else { Scalar::all(200.0) },
        ),
    ];
    for (i, (text, color)) in lines.iter().enumerate() {
        imgproc::put_text(
            frame,
            text,
            Point::new(10, 28 + i as i32 * 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.75,
            *color,
            2,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(())
}

/// Obrys oblasti, v ktorej beží detekcia (tenký sivý rámik).
pub fn draw_roi(frame: &mut Mat, roi: Rect) -> Result<()> {
    imgproc::rectangle(frame, roi, Scalar::all(170.0), 1, imgproc::LINE_8, 0)?;
    Ok(())
}

/// Obrys výrezu ako štvoruholník (po otočení kamery už nie je obdĺžnikom).
pub fn draw_quad(frame: &mut Mat, quad: &[(f64, f64); 4]) -> Result<()> {
    let pts: opencv::core::Vector<Point> =
        quad.iter().map(|p| Point::new(p.0.round() as i32, p.1.round() as i32)).collect();
    imgproc::polylines(frame, &pts, true, Scalar::all(170.0), 1, imgproc::LINE_8, 0)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::LineCounter;
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

    // ---- draw_counting ----

    fn counted(id: u32, g: Group) -> Counted {
        Counted { rect: Rect::new(100, 200, 80, 60), label: "car".into(), score: Some(0.9), id, group: g }
    }

    #[test]
    fn cara_je_zlta_a_cez_celu_vysku() {
        let mut f = black();
        let c = LineCounter::new(300.0, 10.0);
        draw_counting(&mut f, &[], &c.view()).unwrap();
        for y in [100, 240, H - 1] {
            assert_eq!(px(&f, 300, y), [0, 255, 255], "žltá (BGR) na riadku {y}");
        }
        assert_eq!(px(&f, 250, 300), [0, 0, 0]);
    }

    #[test]
    fn auto_je_oranzove_a_clovek_zeleny() {
        let mut f = black();
        let c = LineCounter::new(600.0, 10.0);
        let mut p = counted(2, Group::Person);
        p.rect = Rect::new(300, 200, 50, 50);
        draw_counting(&mut f, &[counted(1, Group::Vehicle), p], &c.view()).unwrap();
        assert_eq!(px(&f, 100, 230), [0, 140, 255], "auto");
        assert_eq!(px(&f, 300, 225), [0, 255, 0], "človek");
    }

    #[test]
    fn panel_je_vlavo_hore_na_tmavom_pozadi() {
        let mut f = Mat::new_rows_cols_with_default(H, W, opencv::core::CV_8UC3, Scalar::all(255.0)).unwrap();
        let c = LineCounter::new(600.0, 10.0);
        draw_counting(&mut f, &[], &c.view()).unwrap();
        assert_eq!(px(&f, 425, 90), [0, 0, 0], "pozadie panelu");
        assert!(has_ink(&f, 10, 5, 420, 90));
        assert_eq!(px(&f, 100, 300), [255, 255, 255], "zvyšok obrazu sa nemení");
    }

    #[test]
    fn panel_ukazuje_pocty() {
        let mut c = LineCounter::new(500.0, 10.0);
        let render = |c: &LineCounter| {
            let mut f = black();
            draw_counting(&mut f, &[], &c.view()).unwrap();
            (0..430).filter(|&x| (0..96).any(|y| px(&f, x, y) != [0, 0, 0])).count()
        };
        let before = render(&c);
        for t in 1..=11 {
            c.observe(0, 0.0, t, Group::Vehicle, 400.0);
            c.observe(1, 0.0, t, Group::Vehicle, 600.0);
        }
        assert_ne!(render(&c), before, "po 11 autách sa text panelu zmení (dvojciferné číslo)");
    }

    #[test]
    fn cara_s_pasmom_je_nakreslena_len_v_pasme() {
        let mut f = black();
        let c = LineCounter::new(300.0, 10.0).with_span(200.0, 400.0);
        draw_counting(&mut f, &[], &c.view()).unwrap();
        assert_eq!(px(&f, 300, 300), [0, 255, 255], "v pásme");
        assert_eq!(px(&f, 300, 150), [0, 0, 0], "nad pásmom");
        assert_eq!(px(&f, 300, 450), [0, 0, 0], "pod pásmom");
    }

    #[test]
    fn roi_obrys_sa_nakresli_a_vnutro_nechá() {
        let mut f = black();
        draw_roi(&mut f, Rect::new(100, 100, 200, 100)).unwrap();
        assert_ne!(px(&f, 100, 150), [0, 0, 0]);
        assert_eq!(px(&f, 200, 150), [0, 0, 0]);
    }

    #[test]
    fn pata_panela_ukazuje_zadany_text() {
        let ink_width = |footer: &str| {
            let mut f = black();
            let v = LineCounter::new(300.0, 10.0).view().with_footer(footer);
            draw_counting(&mut f, &[], &v).unwrap();
            (0..430).filter(|&x| (66..96).any(|y| px(&f, x, y) != [0, 0, 0])).count()
        };
        assert_eq!(ink_width(""), 0, "bez textu nič v päte");
        assert!(ink_width("od zapnutia 1:23") > ink_width("0:05"), "dlhší text zaberie viac miesta");
    }

    #[test]
    fn panel_ukazuje_zadane_pocty_nie_celkove() {
        // panel sa kreslí z CountsView: dva rôzne pohľady -> rôzny obraz
        let mut total = LineCounter::new(500.0, 10.0);
        for t in 1..=3 {
            total.observe(0, 0.0, t, Group::Vehicle, 400.0);
            total.observe(1, 0.0, t, Group::Vehicle, 600.0);
        }
        let render = |v: &CountsView| {
            let mut f = black();
            draw_counting(&mut f, &[], v).unwrap();
            let mut pixels = vec![];
            for y in 0..40 {
                for x in 0..430 {
                    pixels.push(px(&f, x, y));
                }
            }
            pixels
        };
        let empty = LineCounter::new(500.0, 10.0).view();
        assert_ne!(render(&total.view()), render(&empty), "3 autá vs 0 áut");
    }

    #[test]
    fn naklonena_cara_podla_krajnych_bodov() {
        let mut f = black();
        let mut v = LineCounter::new(300.0, 10.0).with_span(100.0, 300.0).view();
        v.line_pts = Some(((100.0, 100.0), (300.0, 300.0))); // uhlopriečka
        draw_counting(&mut f, &[], &v).unwrap();
        assert_eq!(px(&f, 200, 200), [0, 255, 255], "stred uhlopriečky");
        assert_eq!(px(&f, 150, 150), [0, 255, 255]);
        assert_eq!(px(&f, 300, 150), [0, 0, 0], "tam by bola zvislá čiara, tu už nie je");
    }

    #[test]
    fn bez_krajnych_bodov_je_cara_zvisla() {
        let mut f = black();
        let v = LineCounter::new(300.0, 10.0).view();
        draw_counting(&mut f, &[], &v).unwrap();
        assert_eq!(px(&f, 300, 250), [0, 255, 255]);
    }

    #[test]
    fn varovanie_zafarbi_patu_na_cerveno() {
        let red_pixels = |warning: bool| {
            let mut f = black();
            let mut v = LineCounter::new(500.0, 10.0).view().with_footer("REF STRATENA");
            v.warning = warning;
            draw_counting(&mut f, &[], &v).unwrap();
            (66..96)
                .flat_map(|y| (0..430).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    let p = px(&f, x, y);
                    p[2] > 150 && p[0] < 80 && p[1] < 80
                })
                .count()
        };
        assert_eq!(red_pixels(false), 0);
        assert!(red_pixels(true) > 50);
    }

    #[test]
    fn stvoruholnik_vyrezu_sa_nakresli_aj_naklonený() {
        let mut f = black();
        draw_quad(&mut f, &[(100.0, 120.0), (300.0, 100.0), (310.0, 250.0), (90.0, 260.0)]).unwrap();
        assert_ne!(px(&f, 100, 120), [0, 0, 0], "roh");
        assert_ne!(px(&f, 200, 110), [0, 0, 0], "horná hrana medzi rohmi");
        assert_eq!(px(&f, 200, 200), [0, 0, 0], "vnútro ostáva čisté");
    }
}
