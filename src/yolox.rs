//! YOLOX (neurónová sieť na detekciu objektov) spustená cez OpenCV DNN.
//!
//! Model: yolox_s.onnx (https://github.com/Megvii-BaseDetection/YOLOX, licencia Apache-2.0).
//! Vstup: BGR obraz 640x640 (hodnoty 0-255, bez normalizácie), zmenšený so zachovaním pomeru
//! strán a doplnený sivým okrajom (letterbox).
//! Výstup: tensor [1, 8400, 85]. Každý z 8400 riadkov je jeden kandidát:
//!   [x, y, w, h, istota_objektu, 80 skóre tried]
//! Súradnice sú "surové" (relatívne k mriežke) a treba ich dekódovať – pozri `decode`.

use std::path::Path;

use opencv::{
    core::{self, Mat, Rect, Scalar, Size, Vector},
    dnn, imgproc,
    prelude::*,
};

use crate::Result;

/// Rozmer vstupu siete (štvorec).
pub const INPUT: i32 = 640;
/// Kroky troch výstupných mriežok: 80x80, 40x40 a 20x20 buniek (spolu 8400 kandidátov).
const STRIDES: [i32; 3] = [8, 16, 32];
/// Prah pre potlačenie prekrývajúcich sa rámčekov (NMS).
const NMS_THRESHOLD: f32 = 0.45;

/// Jeden kandidát po dekódovaní: rámček v súradniciach pôvodného obrazu, číslo triedy, skóre.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub rect: Rect,
    pub class: usize,
    pub score: f32,
}

/// Načítaná sieť.
pub struct Yolox {
    net: dnn::Net,
    conf: f32,
}

impl Yolox {
    /// Načíta ONNX model. `conf` je minimálna istota (0-1), pod ktorou sa detekcie zahodia.
    pub fn load(path: &str, conf: f32) -> Result<Self> {
        if !Path::new(path).exists() {
            return Err(format!(
                "model '{path}' neexistuje. Stiahni yolox_s.onnx z \
                 https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
            )
            .into());
        }
        let mut net = dnn::read_net_from_onnx(path, dnn::ENGINE_CLASSIC)?; // klasický engine: rešpektuje nastavenie backendu/cieľa
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        net.set_preferable_target(dnn::DNN_TARGET_CPU)?;
        Ok(Self { net, conf })
    }

    /// Nájde objekty na snímke (BGR). Ak je `only_class` zadané, vráti len túto triedu.
    pub fn detect(&mut self, frame: &Mat, only_class: Option<usize>) -> Result<Vec<Candidate>> {
        // 1) Letterbox: zmenšenie na 640 px (zachová pomer strán) + sivý okraj vpravo a dole.
        let ratio = (INPUT as f64 / frame.rows() as f64).min(INPUT as f64 / frame.cols() as f64);
        let (nw, nh) = ((frame.cols() as f64 * ratio) as i32, (frame.rows() as f64 * ratio) as i32);
        let mut resized = Mat::default();
        imgproc::resize(frame, &mut resized, Size::new(nw, nh), 0.0, 0.0, imgproc::INTER_LINEAR)?;
        let mut padded = Mat::default();
        core::copy_make_border(
            &resized,
            &mut padded,
            0,
            INPUT - nh,
            0,
            INPUT - nw,
            core::BORDER_CONSTANT,
            Scalar::all(114.0), // farba, akú YOLOX používa pri trénovaní
        )?;
        // 2) Mat -> blob [1, 3, 640, 640]; BGR poradie ostáva (swap_rb=false), bez normalizácie.
        let blob = dnn::blob_from_image(
            &padded,
            1.0,
            Size::new(INPUT, INPUT),
            Scalar::default(),
            false,
            false,
            core::CV_32F,
        )?;
        // 3) Inferencia
        self.net.set_input(&blob, "", 1.0, Scalar::default())?;
        let out = self.net.forward_single("")?;
        let data = out.data_typed::<f32>()?;

        // 4) Dekódovanie + odfiltrovanie slabých + NMS
        let mut cands = decode(data, ratio, self.conf, only_class);
        for c in &mut cands {
            c.rect = clamp(c.rect, frame.cols(), frame.rows());
        }
        Ok(nms(cands, NMS_THRESHOLD)?)
    }
}

/// Dekóduje surový výstup siete na kandidátov v súradniciach pôvodného obrazu.
///
/// Riadky idú po mriežkach: najprv 80x80 (krok 8), potom 40x40 (krok 16), potom 20x20 (krok 32),
/// v každej po riadkoch zľava doprava. Pre bunku (gx, gy) s krokom s platí:
///   stred = ((x + gx) * s, (y + gy) * s),   veľkosť = (exp(w) * s, exp(h) * s)
/// Skóre = istota_objektu * skóre_triedy. `ratio` je zmenšenie z letterboxu (súradnice sa
/// vydelia, aby sedeli na pôvodný obraz).
pub fn decode(data: &[f32], ratio: f64, conf: f32, only_class: Option<usize>) -> Vec<Candidate> {
    const COLS: usize = 85; // 4 súradnice + 1 istota + 80 tried
    let mut out = vec![];
    let mut row = 0;
    for stride in STRIDES {
        let cells = (INPUT / stride) as usize;
        for gy in 0..cells {
            for gx in 0..cells {
                let Some(r) = data.get(row * COLS..(row + 1) * COLS) else {
                    return out; // kratší tensor než očakávame – nehádžeme paniku
                };
                row += 1;
                let obj = r[4];
                if obj < conf {
                    continue; // rýchle zahodenie: objekt tu nie je
                }
                // trieda s najvyšším skóre
                let (class, cls_score) = r[5..]
                    .iter()
                    .enumerate()
                    .fold((0, 0.0f32), |best, (i, &s)| if s > best.1 { (i, s) } else { best });
                let score = obj * cls_score;
                if score < conf || only_class.is_some_and(|c| c != class) {
                    continue;
                }
                let s = stride as f32;
                let (cx, cy) = ((r[0] + gx as f32) * s, (r[1] + gy as f32) * s);
                let (w, h) = (r[2].exp() * s, r[3].exp() * s);
                let ratio = ratio as f32;
                out.push(Candidate {
                    rect: Rect::new(
                        ((cx - w / 2.0) / ratio) as i32,
                        ((cy - h / 2.0) / ratio) as i32,
                        (w / ratio) as i32,
                        (h / ratio) as i32,
                    ),
                    class,
                    score,
                });
            }
        }
    }
    out
}

/// Orežе rámček, aby nevyšiel mimo obrazu.
fn clamp(r: Rect, w: i32, h: i32) -> Rect {
    let x0 = r.x.clamp(0, w - 1);
    let y0 = r.y.clamp(0, h - 1);
    let x1 = (r.x + r.width).clamp(x0 + 1, w);
    let y1 = (r.y + r.height).clamp(y0 + 1, h);
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

/// Non-maximum suppression: z prekrývajúcich sa rámčikov nechá len ten s najvyšším skóre.
fn nms(cands: Vec<Candidate>, iou: f32) -> Result<Vec<Candidate>> {
    if cands.is_empty() {
        return Ok(cands);
    }
    let boxes: Vector<Rect> = cands.iter().map(|c| c.rect).collect();
    let scores: Vector<f32> = cands.iter().map(|c| c.score).collect();
    let mut keep = Vector::<i32>::new();
    dnn::nms_boxes(&boxes, &scores, 0.0, iou, &mut keep, 1.0, 0)?;
    Ok(keep.iter().map(|i| cands[i as usize].clone()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const COLS: usize = 85;
    const ROWS: usize = 8400;

    /// Prázdny výstup siete (všetko s nulovou istotou).
    fn empty() -> Vec<f32> {
        vec![0.0; ROWS * COLS]
    }

    /// Nastaví riadok `row`: súradnice, istotu objektu a skóre jednej triedy.
    fn set(data: &mut [f32], row: usize, xywh: [f32; 4], obj: f32, class: usize, cls: f32) {
        let r = &mut data[row * COLS..(row + 1) * COLS];
        r[..4].copy_from_slice(&xywh);
        r[4] = obj;
        r[5 + class] = cls;
    }

    #[test]
    fn prazdny_vystup_nema_detekcie() {
        assert!(decode(&empty(), 1.0, 0.4, None).is_empty());
    }

    #[test]
    fn dekoduje_bunku_s_krokom_8() {
        // riadok 81 = mriežka 80x80, gy=1, gx=1 (krok 8)
        let mut d = empty();
        // stred = ((2+1)*8, (3+1)*8) = (24, 32); veľkosť = exp(ln 4)*8 = 32
        set(&mut d, 81, [2.0, 3.0, 4f32.ln(), 4f32.ln()], 0.9, 0, 0.8);
        let c = decode(&d, 1.0, 0.4, None);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].rect, Rect::new(8, 16, 32, 32)); // stred ± 16
        assert_eq!(c[0].class, 0);
        assert!((c[0].score - 0.72).abs() < 1e-5, "skóre = 0.9 * 0.8");
    }

    #[test]
    fn druha_mriezka_ma_krok_16() {
        // prvý riadok druhej mriežky (po 6400) = gx=0, gy=0, krok 16
        let mut d = empty();
        set(&mut d, 6400, [1.0, 1.0, 2f32.ln(), 2f32.ln()], 0.9, 0, 0.9);
        let c = decode(&d, 1.0, 0.4, None);
        // stred (16,16), veľkosť 32 -> x=0,y=0
        assert_eq!(c[0].rect, Rect::new(0, 0, 32, 32));
    }

    #[test]
    fn tretia_mriezka_ma_krok_32() {
        // prvý riadok tretej mriežky (po 6400 + 1600)
        let mut d = empty();
        set(&mut d, 8000, [1.0, 1.0, 2f32.ln(), 2f32.ln()], 0.9, 0, 0.9);
        let c = decode(&d, 1.0, 0.4, None);
        assert_eq!(c[0].rect, Rect::new(0, 0, 64, 64)); // stred (32,32), veľkosť 64
    }

    #[test]
    fn suradnice_sa_vratia_do_povodneho_rozlisenia() {
        let mut d = empty();
        set(&mut d, 81, [2.0, 3.0, 4f32.ln(), 4f32.ln()], 0.9, 0, 0.8);
        // ratio 0.5 = obraz bol zmenšený na polovicu -> rámček je 2x väčší a posunutý
        let c = decode(&d, 0.5, 0.4, None);
        assert_eq!(c[0].rect, Rect::new(16, 32, 64, 64));
    }

    #[test]
    fn nizka_istota_sa_zahodi() {
        let mut d = empty();
        set(&mut d, 0, [0.0, 0.0, 0.0, 0.0], 0.3, 0, 0.9); // obj pod prahom
        set(&mut d, 1, [0.0, 0.0, 0.0, 0.0], 0.9, 0, 0.3); // trieda pod prahom
        set(&mut d, 2, [0.0, 0.0, 0.0, 0.0], 0.6, 0, 0.6); // 0.36 < 0.4
        assert!(decode(&d, 1.0, 0.4, None).is_empty());
        // pri nižšom prahu prejdú všetky tri
        assert_eq!(decode(&d, 1.0, 0.1, None).len(), 3);
    }

    #[test]
    fn vyberie_najlepsiu_triedu() {
        let mut d = empty();
        set(&mut d, 0, [0.0, 0.0, 0.0, 0.0], 0.9, 16, 0.6); // 16 = dog
        d[5 + 2] = 0.3; // aj slabšia trieda 2 (car)
        let c = decode(&d, 1.0, 0.4, None);
        assert_eq!(c[0].class, 16);
    }

    #[test]
    fn filter_triedy_necha_len_zvolenu() {
        let mut d = empty();
        set(&mut d, 0, [0.0, 0.0, 0.0, 0.0], 0.9, 0, 0.9); // person
        set(&mut d, 1, [0.0, 0.0, 0.0, 0.0], 0.9, 2, 0.9); // car
        assert_eq!(decode(&d, 1.0, 0.4, None).len(), 2);
        let only = decode(&d, 1.0, 0.4, Some(0));
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].class, 0);
    }

    #[test]
    fn kratky_tensor_nezpanikuje() {
        // menej riadkov, než očakávame: nesmie to spadnúť
        let d = vec![0.0; 10 * COLS];
        assert!(decode(&d, 1.0, 0.4, None).is_empty());
    }

    #[test]
    fn clamp_orezava_rámcek_do_obrazu() {
        assert_eq!(clamp(Rect::new(-10, -20, 50, 60), 100, 100), Rect::new(0, 0, 40, 40));
        assert_eq!(clamp(Rect::new(90, 90, 50, 50), 100, 100), Rect::new(90, 90, 10, 10));
        // rámček celý mimo obrazu sa zmenší na 1 px, nie na zápornú veľkosť
        let r = clamp(Rect::new(500, 500, 10, 10), 100, 100);
        assert!(r.width >= 1 && r.height >= 1 && r.x < 100 && r.y < 100, "{r:?}");
    }

    #[test]
    fn nms_odstrani_prekryvajuce_sa_rámčeky() {
        let cands = vec![
            Candidate { rect: Rect::new(10, 10, 100, 100), class: 0, score: 0.9 },
            Candidate { rect: Rect::new(12, 12, 100, 100), class: 0, score: 0.8 }, // takmer rovnaký
            Candidate { rect: Rect::new(300, 300, 50, 50), class: 0, score: 0.7 }, // inde
        ];
        let kept = nms(cands, 0.45).unwrap();
        assert_eq!(kept.len(), 2);
        assert!(kept.iter().any(|c| c.score == 0.9), "ostáva silnejší z duplicít");
        assert!(!kept.iter().any(|c| c.score == 0.8));
    }

    #[test]
    fn nms_prazdny_zoznam() {
        assert!(nms(vec![], 0.45).unwrap().is_empty());
    }
}
