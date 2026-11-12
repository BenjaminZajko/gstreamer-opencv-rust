//! Neurónová sieť na detekciu objektov (YOLOX alebo YOLO11) spustená cez OpenCV DNN.
//! Typ modelu sa rozpozná automaticky podľa tvaru výstupu (pozri `ModelKind`).
//!
//! YOLOX – yolox_s.onnx (https://github.com/Megvii-BaseDetection/YOLOX, licencia Apache-2.0):
//!   Vstup: BGR obraz 640x640 (hodnoty 0-255, bez normalizácie), zmenšený so zachovaním pomeru
//!   strán a doplnený sivým okrajom vpravo a dole (letterbox).
//!   Výstup: tensor [1, 8400, 85]. Každý z 8400 riadkov je jeden kandidát:
//!     [x, y, w, h, istota_objektu, 80 skóre tried]
//!   Súradnice sú "surové" (relatívne k mriežke) a treba ich dekódovať – pozri `decode`.
//!
//! YOLO11 – yolo11s.onnx (Ultralytics, licencia AGPL-3.0; export z yolo11s.pt):
//!   Vstup: RGB obraz 640x640, hodnoty 0-1, letterbox so sivým okrajom rovnomerne z oboch strán.
//!   Výstup: tensor [1, 84, 8400] – "po stĺpcoch": pre každého z 8400 kandidátov
//!     [cx, cy, w, h, 80 skóre tried] (bez istoty objektu), súradnice už v pixeloch vstupu 640x640.
//!   Pozri `decode_yolo11`.

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

/// Na čom beží výpočet siete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    /// OpenCV DNN na CPU (pôvodný spôsob)
    OpenCv,
    /// ONNX Runtime na CPU
    OrtCpu,
    /// ONNX Runtime + Core ML; Core ML si sám vyberie CPU, GPU alebo Neural Engine
    CoreMl,
    /// ONNX Runtime + Core ML, len CPU a GPU
    CoreMlGpu,
    /// ONNX Runtime + Core ML, len CPU a Neural Engine (NPU)
    CoreMlAne,
}

impl Engine {
    /// Z textu voľby `--engine`.
    pub fn parse(s: &str) -> Result<Engine> {
        Ok(match s {
            "opencv" => Engine::OpenCv,
            "ort-cpu" => Engine::OrtCpu,
            "coreml" => Engine::CoreMl,
            "coreml-gpu" => Engine::CoreMlGpu,
            "coreml-ane" => Engine::CoreMlAne,
            _ => return Err(format!("--engine '{s}': opencv, ort-cpu, coreml, coreml-gpu alebo coreml-ane").into()),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Engine::OpenCv => "opencv",
            Engine::OrtCpu => "ort-cpu",
            Engine::CoreMl => "coreml",
            Engine::CoreMlGpu => "coreml-gpu",
            Engine::CoreMlAne => "coreml-ane",
        }
    }
}

/// Prostredie ONNX Runtime (jedno na celý program).
fn ort_env() -> Result<&'static ort::environment::Environment> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    if let Some(e) = ENV.get() {
        return Ok(e);
    }
    let e = ort::init().with_name("object_detect").build().map_err(|e| format!("ONNX Runtime: {e}"))?;
    Ok(ENV.get_or_init(|| e))
}

/// Typ modelu: líši sa príprava vstupu aj formát výstupu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// YOLOX: výstup [1, 8400, 85]
    Yolox,
    /// YOLO11 (Ultralytics, aj YOLOv8): výstup [1, 84, 8400]
    Yolo11,
}

impl ModelKind {
    pub fn name(self) -> &'static str {
        match self {
            ModelKind::Yolox => "YOLOX",
            ModelKind::Yolo11 => "YOLO11",
        }
    }

    /// Rozpozná typ podľa rozmerov výstupu siete.
    pub fn from_output_dims(dims: &[i32]) -> Option<ModelKind> {
        match dims {
            [1, 8400, 85] => Some(ModelKind::Yolox),
            [1, 84, 8400] => Some(ModelKind::Yolo11),
            _ => None,
        }
    }
}

/// Parametre letterboxu: zmenšenie a umiestnenie obrazu v štvorci `INPUT` x `INPUT`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// zmenšenie (vstup siete / pôvodný obraz)
    pub ratio: f64,
    /// rozmery zmenšeného obrazu
    pub nw: i32,
    pub nh: i32,
    /// okraj vľavo a hore (YOLOX 0, YOLO11 polovica voľného miesta)
    pub left: i32,
    pub top: i32,
}

/// Vypočíta letterbox pre obraz `w` x `h` podľa typu modelu.
pub fn letterbox(w: i32, h: i32, kind: ModelKind) -> Letterbox {
    let ratio = (INPUT as f64 / h as f64).min(INPUT as f64 / w as f64);
    let (nw, nh) = ((w as f64 * ratio) as i32, (h as f64 * ratio) as i32);
    let (left, top) = match kind {
        ModelKind::Yolox => (0, 0),
        ModelKind::Yolo11 => ((INPUT - nw) / 2, (INPUT - nh) / 2),
    };
    Letterbox { ratio, nw, nh, left, top }
}

/// Jeden kandidát po dekódovaní: rámček v súradniciach pôvodného obrazu, číslo triedy, skóre.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub rect: Rect,
    pub class: usize,
    pub score: f32,
}

/// Spustiteľná sieť: OpenCV DNN alebo ONNX Runtime.
enum Backend {
    OpenCv(dnn::Net),
    Ort(ort::session::Session),
}

/// Načítaná sieť (YOLOX alebo YOLO11).
pub struct Yolox {
    backend: Backend,
    conf: f32,
    kind: ModelKind,
    engine: Engine,
}

impl Yolox {
    /// Načíta ONNX model. `conf` je minimálna istota (0-1), pod ktorou sa detekcie zahodia.
    pub fn load(path: &str, conf: f32) -> Result<Self> {
        Self::load_with_target(path, conf, dnn::DNN_TARGET_CPU)
    }

    /// OpenCV DNN na zvolenom cieli výpočtu (`dnn::DNN_TARGET_CPU`, `DNN_TARGET_OPENCL`...).
    pub fn load_with_target(path: &str, conf: f32, target: i32) -> Result<Self> {
        Self::check_path(path)?;
        let mut net = dnn::read_net_from_onnx(path, dnn::ENGINE_CLASSIC)?; // klasický engine: rešpektuje nastavenie backendu/cieľa
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        net.set_preferable_target(target)?;
        Self::finish(path, conf, Backend::OpenCv(net), Engine::OpenCv)
    }

    /// Načíta model na zvolenom spôsobe výpočtu (pozri `Engine`).
    pub fn load_with_engine(path: &str, conf: f32, engine: Engine) -> Result<Self> {
        if engine == Engine::OpenCv {
            return Self::load(path, conf);
        }
        Self::check_path(path)?;
        use ort::ep::coreml::{ComputeUnits, ModelFormat};
        let builder = ort::session::Session::builder(ort_env()?).map_err(|e| format!("ONNX Runtime: {e}"))?;
        let units = match engine {
            Engine::CoreMl => Some(ComputeUnits::All),
            Engine::CoreMlGpu => Some(ComputeUnits::CPUAndGPU),
            Engine::CoreMlAne => Some(ComputeUnits::CPUAndNeuralEngine),
            _ => None,
        };
        let mut builder = match units {
            Some(u) => {
                // skompilovaný model si Core ML odloží, ďalšie spustenie je rýchlejšie
                let cache = std::env::temp_dir().join("object_detect_coreml_cache");
                let ep = ort::ep::CoreML::default()
                    .with_compute_units(u)
                    .with_model_format(ModelFormat::MLProgram)
                    .with_model_cache_dir(cache.display())
                    .build()
                    .error_on_failure();
                builder.with_execution_providers([ep]).map_err(|e| format!("Core ML: {e}"))?
            }
            None => builder,
        };
        let session = builder.commit_from_file(path).map_err(|e| format!("model '{path}' sa nepodarilo načítať: {e}"))?;
        Self::finish(path, conf, Backend::Ort(session), engine)
    }

    fn check_path(path: &str) -> Result<()> {
        if !Path::new(path).exists() {
            return Err(format!(
                "model '{path}' neexistuje. Stiahni yolox_s.onnx z \
                 https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
            )
            .into());
        }
        Ok(())
    }

    /// Rozpozná typ modelu podľa tvaru výstupu (jedna skúšobná inferencia na prázdnom obraze).
    fn finish(path: &str, conf: f32, backend: Backend, engine: Engine) -> Result<Self> {
        let mut me = Self { backend, conf, kind: ModelKind::Yolox, engine };
        let blank = Mat::new_rows_cols_with_default(INPUT, INPUT, core::CV_8UC3, Scalar::all(114.0))?;
        let blob = dnn::blob_from_image(&blank, 1.0, Size::new(INPUT, INPUT), Scalar::default(), false, false, core::CV_32F)?;
        let (dims, _) = me.forward(&blob)?;
        me.kind = ModelKind::from_output_dims(&dims).ok_or_else(|| {
            format!("model '{path}' má neznámy formát výstupu {dims:?} (podporované: YOLOX [1,8400,85], YOLO11 [1,84,8400])")
        })?;
        Ok(me)
    }

    /// Spustí sieť na blobe [1, 3, 640, 640]. Vráti rozmery výstupu a jeho hodnoty.
    fn forward(&mut self, blob: &Mat) -> Result<(Vec<i32>, Vec<f32>)> {
        match &mut self.backend {
            Backend::OpenCv(net) => {
                net.set_input(blob, "", 1.0, Scalar::default())?;
                let out = net.forward_single("")?;
                let dims: Vec<i32> = (0..out.dims() as usize).map(|i| out.mat_size().get(i)).collect::<std::result::Result<_, _>>()?;
                Ok((dims, out.data_typed::<f32>()?.to_vec()))
            }
            Backend::Ort(session) => {
                let input = ort::value::Tensor::from_array(([1usize, 3, INPUT as usize, INPUT as usize], blob.data_typed::<f32>()?.to_vec()))
                    .map_err(|e| format!("ONNX Runtime: {e}"))?;
                let outputs = session.run(ort::inputs![input]).map_err(|e| format!("ONNX Runtime: {e}"))?;
                let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(|e| format!("ONNX Runtime: {e}"))?;
                Ok((shape.iter().map(|&d| d as i32).collect(), data.to_vec()))
            }
        }
    }

    /// Na čom beží výpočet.
    pub fn engine(&self) -> Engine {
        self.engine
    }

    /// Typ načítaného modelu.
    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    /// Nájde objekty na snímke (BGR). Ak je `classes` zadané, vráti len tieto triedy.
    pub fn detect(&mut self, frame: &Mat, classes: Option<&[usize]>) -> Result<Vec<Candidate>> {
        // 1) Letterbox: zmenšenie na 640 px (zachová pomer strán) + sivý okraj
        //    (YOLOX vpravo a dole, YOLO11 rovnomerne z oboch strán – tak ako pri trénovaní).
        let lb = letterbox(frame.cols(), frame.rows(), self.kind);
        let mut resized = Mat::default();
        imgproc::resize(frame, &mut resized, Size::new(lb.nw, lb.nh), 0.0, 0.0, imgproc::INTER_LINEAR)?;
        let mut padded = Mat::default();
        core::copy_make_border(
            &resized,
            &mut padded,
            lb.top,
            INPUT - lb.nh - lb.top,
            lb.left,
            INPUT - lb.nw - lb.left,
            core::BORDER_CONSTANT,
            Scalar::all(114.0), // farba okraja, akú oba modely používajú pri trénovaní
        )?;
        // 2) Mat -> blob [1, 3, 640, 640].
        //    YOLOX: BGR, hodnoty 0-255.  YOLO11: RGB (swap_rb), hodnoty 0-1 (x 1/255).
        let (scale, swap_rb) = match self.kind {
            ModelKind::Yolox => (1.0, false),
            ModelKind::Yolo11 => (1.0 / 255.0, true),
        };
        let blob = dnn::blob_from_image(
            &padded,
            scale,
            Size::new(INPUT, INPUT),
            Scalar::default(),
            swap_rb,
            false,
            core::CV_32F,
        )?;
        // 3) Inferencia (OpenCV DNN alebo ONNX Runtime)
        let (_, data) = self.forward(&blob)?;
        let data = &data[..];

        // 4) Dekódovanie + odfiltrovanie slabých + NMS
        let mut cands = match self.kind {
            ModelKind::Yolox => decode(data, lb.ratio, self.conf, classes),
            ModelKind::Yolo11 => decode_yolo11(data, &lb, self.conf, classes),
        };
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
pub fn decode(data: &[f32], ratio: f64, conf: f32, classes: Option<&[usize]>) -> Vec<Candidate> {
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
                if score < conf || classes.is_some_and(|cs| !cs.contains(&class)) {
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

/// Dekóduje výstup YOLO11 [1, 84, 8400] na kandidátov v súradniciach pôvodného obrazu.
///
/// Výstup je "po stĺpcoch": hodnota `k` kandidáta `i` je na indexe `k * 8400 + i`, kde
/// k = 0..4 sú cx, cy, w, h (pixely vstupu 640x640) a k = 4..84 sú skóre 80 tried (už 0-1).
/// Skóre kandidáta = najvyššie skóre triedy (YOLO11 nemá samostatnú istotu objektu).
/// Súradnice sa prepočítajú späť cez letterbox: odpočíta sa okraj a vydelí zmenšením.
pub fn decode_yolo11(data: &[f32], lb: &Letterbox, conf: f32, classes: Option<&[usize]>) -> Vec<Candidate> {
    const N: usize = 8400;
    const NC: usize = 80;
    if data.len() < (4 + NC) * N {
        return vec![]; // kratší tensor než očakávame – nehádžeme paniku
    }
    let mut out = vec![];
    for i in 0..N {
        // trieda s najvyšším skóre
        let (mut class, mut score) = (0usize, 0f32);
        for c in 0..NC {
            let s = data[(4 + c) * N + i];
            if s > score {
                (class, score) = (c, s);
            }
        }
        if score < conf || classes.is_some_and(|cs| !cs.contains(&class)) {
            continue;
        }
        let (cx, cy, w, h) = (data[i], data[N + i], data[2 * N + i], data[3 * N + i]);
        let r = lb.ratio as f32;
        out.push(Candidate {
            rect: Rect::new(
                ((cx - w / 2.0 - lb.left as f32) / r) as i32,
                ((cy - h / 2.0 - lb.top as f32) / r) as i32,
                (w / r) as i32,
                (h / r) as i32,
            ),
            class,
            score,
        });
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
        let only = decode(&d, 1.0, 0.4, Some(&[0]));
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].class, 0);
    }

    #[test]
    fn filter_viacerych_tried() {
        let mut d = empty();
        set(&mut d, 0, [0.0, 0.0, 0.0, 0.0], 0.9, 0, 0.9); // person
        set(&mut d, 1, [0.0, 0.0, 0.0, 0.0], 0.9, 2, 0.9); // car
        set(&mut d, 2, [0.0, 0.0, 0.0, 0.0], 0.9, 16, 0.9); // dog
        let c = decode(&d, 1.0, 0.4, Some(&[0, 2]));
        let mut classes: Vec<usize> = c.iter().map(|x| x.class).collect();
        classes.sort();
        assert_eq!(classes, vec![0, 2], "pes sa vyradí");
        assert!(decode(&d, 1.0, 0.4, Some(&[])).is_empty(), "prázdny zoznam = nič");
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

    // ---- YOLO11 ----

    const N11: usize = 8400;

    /// Prázdny výstup YOLO11 [1, 84, 8400].
    fn empty11() -> Vec<f32> {
        vec![0.0; 84 * N11]
    }

    /// Nastaví kandidáta `i` vo výstupe YOLO11 (formát po stĺpcoch).
    fn set11(d: &mut [f32], i: usize, cxcywh: [f32; 4], class: usize, score: f32) {
        for (k, v) in cxcywh.iter().enumerate() {
            d[k * N11 + i] = *v;
        }
        d[(4 + class) * N11 + i] = score;
    }

    #[test]
    fn typ_modelu_podla_vystupu() {
        assert_eq!(ModelKind::from_output_dims(&[1, 8400, 85]), Some(ModelKind::Yolox));
        assert_eq!(ModelKind::from_output_dims(&[1, 84, 8400]), Some(ModelKind::Yolo11));
        assert_eq!(ModelKind::from_output_dims(&[1, 25200, 85]), None);
        assert_eq!(ModelKind::from_output_dims(&[]), None);
    }

    #[test]
    fn letterbox_yolox_vlavo_hore_yolo11_v_strede() {
        // široký obraz 1280x505 -> 640x252
        let x = letterbox(1280, 505, ModelKind::Yolox);
        assert_eq!((x.nw, x.nh, x.left, x.top), (640, 252, 0, 0));
        assert!((x.ratio - 0.5).abs() < 1e-9);
        let y = letterbox(1280, 505, ModelKind::Yolo11);
        assert_eq!((y.nw, y.nh, y.left, y.top), (640, 252, 0, 194), "okraj rovnomerne hore aj dole");
        // vysoký obraz -> okraj vľavo a vpravo
        let t = letterbox(300, 600, ModelKind::Yolo11);
        assert_eq!((t.nw, t.nh, t.left, t.top), (320, 640, 160, 0));
    }

    #[test]
    fn yolo11_dekoduje_kandidata_a_prepocita_cez_letterbox() {
        let mut d = empty11();
        // v 640x640: stred (100, 300), rozmer 50x80, trieda 2 (auto) so skóre 0.9
        set11(&mut d, 1234, [100.0, 300.0, 50.0, 80.0], 2, 0.9);
        let lb = Letterbox { ratio: 0.5, nw: 640, nh: 252, left: 0, top: 194 };
        let c = decode_yolo11(&d, &lb, 0.4, None);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].class, 2);
        assert!((c[0].score - 0.9).abs() < 1e-6);
        // x = (100 - 25 - 0) / 0.5 = 150, y = (300 - 40 - 194) / 0.5 = 132, w = 100, h = 160
        assert_eq!(c[0].rect, Rect::new(150, 132, 100, 160));
    }

    #[test]
    fn yolo11_filtruje_slabe_a_triedy() {
        let mut d = empty11();
        set11(&mut d, 0, [50.0, 50.0, 10.0, 10.0], 0, 0.35); // pod prahom
        set11(&mut d, 1, [50.0, 50.0, 10.0, 10.0], 0, 0.8); // človek
        set11(&mut d, 2, [50.0, 50.0, 10.0, 10.0], 16, 0.8); // pes
        let lb = letterbox(640, 640, ModelKind::Yolo11);
        assert_eq!(decode_yolo11(&d, &lb, 0.4, None).len(), 2);
        let only = decode_yolo11(&d, &lb, 0.4, Some(&[0]));
        assert_eq!(only.len(), 1);
        assert_eq!(only[0].class, 0);
        assert_eq!(decode_yolo11(&d, &lb, 0.3, None).len(), 3, "nižší prah pustí aj slabý");
    }

    #[test]
    fn yolo11_vyberie_najlepsiu_triedu() {
        let mut d = empty11();
        set11(&mut d, 7, [50.0, 50.0, 10.0, 10.0], 7, 0.6); // nákladiak
        d[(4 + 2) * N11 + 7] = 0.5; // aj auto, ale slabšie
        let c = decode_yolo11(&d, &letterbox(640, 640, ModelKind::Yolo11), 0.4, None);
        assert_eq!(c[0].class, 7);
    }

    #[test]
    fn yolo11_kratky_tensor_nezpanikuje() {
        assert!(decode_yolo11(&[0.0; 100], &letterbox(640, 640, ModelKind::Yolo11), 0.4, None).is_empty());
    }

    /// Načíta model, ak je stiahnutý (inak test preskočí).
    fn try_load(file: &str) -> Option<Yolox> {
        let p = format!("{}/models/{file}", env!("CARGO_MANIFEST_DIR"));
        Path::new(&p).exists().then(|| Yolox::load(&p, 0.4).unwrap())
    }

    #[test]
    fn skutocne_modely_sa_rozpoznaju_spravne() {
        if let Some(m) = try_load("yolox_s.onnx") {
            assert_eq!(m.kind(), ModelKind::Yolox);
        }
        if let Some(m) = try_load("yolo11s.onnx") {
            assert_eq!(m.kind(), ModelKind::Yolo11);
        }
    }

    #[test]
    fn skutocne_modely_na_prazdnom_obraze_nic_nenajdu() {
        let blank = Mat::new_rows_cols_with_default(505, 1280, core::CV_8UC3, Scalar::all(120.0)).unwrap();
        for file in ["yolox_s.onnx", "yolo11s.onnx"] {
            if let Some(mut m) = try_load(file) {
                assert!(m.detect(&blank, None).unwrap().is_empty(), "{file}");
            }
        }
    }
}
