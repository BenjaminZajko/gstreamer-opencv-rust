//! Detektory objektov (OpenCV): pohyb (MOG2) a neurónová sieť YOLOX (ľudia / všetky objekty).

use opencv::{
    core::{self, Mat, Point, Rect, Size, Vector},
    geometry, imgproc,
    prelude::*,
    video,
};

use crate::{
    args::Mode,
    labels::{COCO, PERSON},
    yolox::Yolox,
    Result,
};

/// Jedna detekcia: kde je (rámček), čo to je (popis) a ako si je detektor istý.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub rect: Rect,
    pub label: String,
    /// Istota 0-1; `None` ak ju detektor nemá (pohyb).
    pub score: Option<f32>,
}

/// Detektor: zo snímky (BGR) vráti nájdené objekty.
/// Každý režim si drží vlastný stav (napr. MOG2 si pamätá pozadie z predošlých snímok).
pub enum Detector {
    Motion {
        mog: opencv::core::Ptr<video::BackgroundSubtractorMOG2>, // model pozadia
        kernel: Mat, // eliptický "štetec" pre čistenie masky (morfológia)
        warmup: u32, // koľko snímok ešte ostáva na naučenie pozadia
    },
    Yolo {
        net: Yolox,
        classes: Option<Vec<usize>>, // ktoré triedy hľadať (`None` = všetkých 80)
        people_label: bool,          // true = popis "clovek" namiesto "person"
    },
}

impl Detector {
    /// Vytvorí detektor pre zvolený režim. `model` a `conf` sa použijú len pri YOLOX režimoch.
    pub fn new(mode: Mode, model: &str, conf: f32) -> Result<Self> {
        Ok(match mode {
            Mode::Motion => Detector::Motion {
                // MOG2: history=500 snímok, prah=25 (citlivosť), true = rozpoznávaj tiene
                mog: video::create_background_subtractor_mog2(500, 25.0, true)?,
                // elipsa 5x5 px; Point(-1,-1) = stred jadra
                kernel: imgproc::get_structuring_element(
                    imgproc::MORPH_ELLIPSE,
                    Size::new(5, 5),
                    Point::new(-1, -1),
                )?,
                warmup: 30, // prvých pár snímok sa pozadie učí, nič nehlásime
            },
            Mode::People => Detector::Yolo {
                net: Yolox::load(model, conf)?,
                classes: Some(vec![PERSON]),
                people_label: true,
            },
            Mode::Objects => Detector::Yolo {
                net: Yolox::load(model, conf)?,
                classes: None,
                people_label: false,
            },
        })
    }

    /// YOLOX detektor, ktorý hľadá len vybrané triedy (čísla tried COCO, pozri `labels`).
    pub fn yolo_with_classes(model: &str, conf: f32, classes: &[usize]) -> Result<Self> {
        Ok(Detector::Yolo {
            net: Yolox::load(model, conf)?,
            classes: Some(classes.to_vec()),
            people_label: false,
        })
    }

    /// Nájde objekty na snímke.
    pub fn detect(&mut self, frame: &Mat) -> Result<Vec<Detection>> {
        match self {
            Detector::Motion { mog, kernel, warmup } => Ok(detect_motion(mog, kernel, warmup, frame)?
                .into_iter()
                .map(|rect| Detection { rect, label: "pohyb".into(), score: None })
                .collect()),
            Detector::Yolo { net, classes, people_label } => {
                Ok(net
                    .detect(frame, classes.as_deref())?
                    .into_iter()
                    .map(|c| Detection {
                        rect: c.rect,
                        label: if *people_label { "clovek".into() } else { COCO[c.class].into() },
                        score: Some(c.score),
                    })
                    .collect())
            }
        }
    }
}

/// Detekcia pohybu: odčítanie pozadia (MOG2) + čistenie masky + kontúry.
fn detect_motion(
    mog: &mut opencv::core::Ptr<video::BackgroundSubtractorMOG2>,
    kernel: &Mat,
    warmup: &mut u32,
    frame: &Mat,
) -> Result<Vec<Rect>> {
    // 1) Maska popredia: biele pixely = niečo sa zmenilo oproti pozadiu.
    //    Parameter -1.0 = automatická rýchlosť učenia pozadia.
    let mut fg = Mat::default();
    BackgroundSubtractorTrait::apply(mog, frame, &mut fg, -1.0)?;
    if *warmup > 0 {
        *warmup -= 1;
        return Ok(vec![]); // ešte sa učí, nič nehlásime
    }
    // 2) Binarizácia: 255 = pohyb, 127 = tieň -> tiene vyhodíme prahom 200
    let mut bin = Mat::default();
    imgproc::threshold(&fg, &mut bin, 200.0, 255.0, imgproc::THRESH_BINARY)?;
    // 3) MORPH_OPEN odstráni drobný šum (osamotené bodky)
    let mut opened = Mat::default();
    imgproc::morphology_ex(
        &bin,
        &mut opened,
        imgproc::MORPH_OPEN,
        kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    // 4) Dilatácia (3 iterácie) zlepí blízke časti jedného objektu dokopy
    let mut dilated = Mat::default();
    imgproc::dilate(
        &opened,
        &mut dilated,
        kernel,
        Point::new(-1, -1),
        3,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    // 5) Nájdenie súvislých bielych oblastí (kontúr); RETR_EXTERNAL = len vonkajšie
    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        &dilated,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;
    // 6) Ignorujeme príliš malé oblasti (menej než 0,1 % plochy obrazu = šum)
    let min_area = (frame.rows() * frame.cols()) as f64 * 0.001;
    let mut out = vec![];
    for c in contours.iter() {
        if geometry::contour_area(&c, false)? >= min_area {
            out.push(geometry::bounding_rect(&c)?); // obdĺžnik okolo kontúry
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Scalar;

    const W: i32 = 320;
    const H: i32 = 240;

    /// Jednofarebná snímka (sivé pozadie).
    fn background() -> Mat {
        Mat::new_rows_cols_with_default(H, W, core::CV_8UC3, Scalar::all(100.0)).unwrap()
    }

    /// Pozadie s bielym štvorcom `side` x `side` na pozícii (x, y).
    fn with_square(x: i32, y: i32, side: i32) -> Mat {
        let mut f = background();
        imgproc::rectangle(
            &mut f,
            Rect::new(x, y, side, side),
            Scalar::all(255.0),
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )
        .unwrap();
        f
    }

    /// Detektor pohybu, ktorý už prešiel zahrievaním na statickom pozadí.
    fn warmed_up_motion() -> Detector {
        let mut d = Detector::new(Mode::Motion, "", 0.4).unwrap();
        for _ in 0..40 {
            d.detect(&background()).unwrap();
        }
        d
    }

    #[test]
    fn motion_pocas_zahrievania_nehlasi_nic() {
        let mut d = Detector::new(Mode::Motion, "", 0.4).unwrap();
        // aj keď je v obraze objekt, prvých 30 snímok sa pozadie len učí
        for _ in 0..30 {
            assert!(d.detect(&with_square(100, 80, 60)).unwrap().is_empty());
        }
    }

    #[test]
    fn motion_staticky_obraz_bez_detekcii() {
        let mut d = warmed_up_motion();
        assert!(d.detect(&background()).unwrap().is_empty());
    }

    #[test]
    fn motion_najde_novy_objekt() {
        let mut d = warmed_up_motion();
        let boxes: Vec<Rect> = d.detect(&with_square(100, 80, 60)).unwrap().into_iter().map(|d| d.rect).collect();
        assert!(!boxes.is_empty(), "objekt mal byť nájdený");
        // stred štvorca (130, 110) musí byť v niektorom z rámčekov
        let center = Point::new(130, 110);
        assert!(boxes.iter().any(|r| r.contains(center)), "{boxes:?}");
    }

    #[test]
    fn motion_ignoruje_drobny_sum() {
        let mut d = warmed_up_motion();
        // 3x3 px bodka je pod prahom (otvorenie ju odstráni)
        assert!(d.detect(&with_square(100, 80, 3)).unwrap().is_empty());
    }

    #[test]
    fn motion_rámcek_sedí_na_objekt() {
        let mut d = warmed_up_motion();
        let boxes: Vec<Rect> = d.detect(&with_square(200, 50, 60)).unwrap().into_iter().map(|d| d.rect).collect();
        let r = boxes.iter().max_by_key(|r| r.area()).expect("detekcia");
        // dilatácia rámček mierne zväčší, ale nesmie ísť ďaleko od objektu
        assert!(r.x >= 200 - 20 && r.x <= 200 + 5, "{r:?}");
        assert!(r.y >= 50 - 20 && r.y <= 50 + 5, "{r:?}");
        assert!(r.width >= 55 && r.width <= 100, "{r:?}");
    }


    /// Na veľkom obraze (1280x720) je minimálna plocha 0,1 % = ~921 px. Malý objekt, ktorý
    /// prežije morfologické čistenie, ale je pod touto plochou, sa musí zahodiť filtrom plochy.
    #[test]
    fn motion_filter_plochy_zahodi_maly_objekt_na_velkom_obraze() {
        let big = |square: Option<(i32, i32, i32)>| {
            let mut f = Mat::new_rows_cols_with_default(720, 1280, core::CV_8UC3, Scalar::all(100.0)).unwrap();
            if let Some((x, y, side)) = square {
                imgproc::rectangle(&mut f, Rect::new(x, y, side, side), Scalar::all(255.0), imgproc::FILLED, imgproc::LINE_8, 0).unwrap();
            }
            f
        };
        let mut d = Detector::new(Mode::Motion, "", 0.4).unwrap();
        for _ in 0..40 {
            d.detect(&big(None)).unwrap();
        }
        // 8x8 px: prežije otvorenie, po dilatácii ~20x20 = 400 px < 921 px -> zahodené
        assert!(d.detect(&big(Some((600, 300, 8)))).unwrap().is_empty());
        // 80x80 px je jasne nad prahom -> nájdené
        assert!(!d.detect(&big(Some((600, 300, 80)))).unwrap().is_empty());
    }

    #[test]
    fn motion_detekcia_ma_popis_pohyb_bez_istoty() {
        let mut d = warmed_up_motion();
        let dets = d.detect(&with_square(100, 80, 60)).unwrap();
        assert!(dets.iter().all(|x| x.label == "pohyb" && x.score.is_none()));
    }

    #[test]
    fn yolo_bez_modelu_zlyha_s_napovedou() {
        let e = Detector::new(Mode::People, "/neexistuje/model.onnx", 0.4).err().expect("má zlyhať").to_string();
        assert!(e.contains("neexistuje") && e.contains("yolox_s.onnx"), "{e}");
    }

    /// Cesta k modelu; ak nie je stiahnutý, testy s modelom sa preskočia.
    fn model_path() -> Option<String> {
        let p = format!("{}/models/yolox_s.onnx", env!("CARGO_MANIFEST_DIR"));
        std::path::Path::new(&p).exists().then_some(p)
    }

    #[test]
    fn yolo_prazdna_snimka_bez_detekcii() {
        let Some(model) = model_path() else { return eprintln!("model chýba – test preskočený") };
        for mode in [Mode::People, Mode::Objects] {
            let mut d = Detector::new(mode, &model, 0.4).unwrap();
            assert!(d.detect(&background()).unwrap().is_empty(), "{mode:?}");
        }
    }
}
