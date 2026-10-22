//! Spracovanie argumentov z príkazového riadku.
//cargo run --release -- video.mp4 --mode motion --output out.mp4 --width 1280

use std::env;

use crate::{roi::Crop, video_io::Codec, Result};

/// Režim detekcie, vyberá sa cez `--mode`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    Motion,  // pohybujúce sa objekty (MOG2)
    People,  // len ľudia (YOLOX)
    Objects, // všetkých 80 tried COCO (YOLOX)
}

/// Argumenty z príkazového riadku.
#[derive(Debug)]
pub struct Args {
    pub input: String,  // cesta k vstupnému videu
    pub output: String, // cesta k výstupnému videu s rámčekmi
    pub mode: Mode,
    pub width: i32, // na akú šírku sa video zmenší (výška sa dopočíta)
    pub model: String, // cesta k ONNX modelu YOLOX
    pub conf: f32,     // minimálna istota detekcie (0-1)
}

/// Prevedie text z `--mode` na režim.
fn parse_mode(v: Option<&str>) -> Result<Mode> {
    match v {
        Some("motion") => Ok(Mode::Motion),
        Some("people") => Ok(Mode::People),
        Some("objects") => Ok(Mode::Objects),
        _ => Err("--mode: motion, people alebo objects".into()),
    }
}

/// Spracuje argumenty programu. Prvý argument bez `--` je vstupný súbor.
pub fn parse_args() -> Result<Args> {
    parse_args_from(env::args().skip(1)) // skip(1) preskočí názov programu
}

/// To isté, ale z ľubovoľného zoznamu argumentov (bez názvu programu) – kvôli testom.
pub fn parse_args_from<I: IntoIterator<Item = String>>(args: I) -> Result<Args> {
    let mut it = args.into_iter();
    let mut input = None;
    let mut output = "annotated.mp4".to_string(); // predvolený výstup
    let mut mode = Mode::Motion; // predvolený režim
    let mut width = 1280; // predvolená šírka (720p)
    let mut model = "models/yolox_s.onnx".to_string();
    let mut conf = 0.4;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mode" => mode = parse_mode(it.next().as_deref())?,
            "--output" => output = it.next().ok_or("--output chýba hodnota")?,
            "--width" => width = it.next().ok_or("--width chýba hodnota")?.parse()?,
            "--model" => model = it.next().ok_or("--model chýba hodnota")?,
            "--conf" => {
                conf = it.next().ok_or("--conf chýba hodnota")?.parse()?;
                if !(0.0..=1.0).contains(&conf) {
                    return Err("--conf musí byť medzi 0 a 1".into());
                }
            }
            _ if input.is_none() => input = Some(a), // prvý "voľný" argument = vstupný súbor
            _ => return Err(format!("neznámy argument: {a}").into()),
        }
    }
    let input = input.ok_or("použitie: object_detect <vstup.mp4> [--mode motion|people|objects] [--output out.mp4] [--width 1280] [--model models/yolox_s.onnx] [--conf 0.4]")?;
    Ok(Args { input, output, mode, width, model, conf })
}

/// Argumenty živého režimu (kamera -> detekcia -> vlastný RTSP server).
#[derive(Debug)]
pub struct LiveArgs {
    pub input: String, // rtsp://... (kamera) alebo test:ball
    pub mode: Mode,
    pub width: i32,
    pub model: String,
    pub conf: f32,
    pub port: u16,      // port RTSP servera
    pub mount: String,  // cesta v URL, napr. /detect
    pub codec: Codec,   // kodek kamery
    pub fps: i32,       // tempo výstupného streamu
    pub bitrate: u32,   // bitrate výstupu v kbps
    // --- počítanie prechodov cez čiaru (zapne sa voľbou --line) ---
    pub line: Option<f64>,       // poloha zvislej čiary ako zlomok šírky záberu
    pub line_span: (f64, f64),   // pásmo výšky čiary (zlomky výšky)
    pub roi: Crop,               // výrez, na ktorom beží detekcia (zobrazuje sa celý záber)
    pub detect_width: i32,       // šírka snímok na detekciu (väčšia = viac detailov)
    pub margin: f64,             // pásmo okolo čiary v px (proti chveniu)
    pub max_missed: u32,         // po koľkých snímkach bez nálezu sa objekt zabudne
    pub out_dir: String,         // kam sa ukladá log prechodov
    // --- ukotvenie čiary k scéne (pri pohybe kamery) ---
    pub anchor: bool,                // zapnuté ukotvenie
    pub anchor_every: u32,           // ukotvenie sa prepočíta každú N-tú snímku
    pub anchor_features: i32,        // koľko referenčných bodov najviac hľadať
    pub anchor_min_inliers: usize,   // minimum zhodných bodov pre spoľahlivú polohu (ORB)
    pub anchor_points: usize,        // koľko stabilných bodov sa sleduje (0 = len ORB)
    pub anchor_calib: usize,         // koľko aktualizácií trvá kalibrácia (výber bodov)
    pub anchor_deadband: f64,        // mŕtve pásmo v px: čiara sa nehýbe, kým sa kamera neposunie viac
    pub anchor_log: bool,            // zapisovať polohu čiary každú aktualizáciu do out/anchor_log.csv
    pub output: Option<String>,      // video_count: kam uložiť výsledné video
}

/// Argumenty živého režimu z príkazového riadku. Zdroj sa dá zadať aj cez premennú prostredia
/// `LIVE_INPUT` (aby heslo ku kamere nebolo vidieť v zozname procesov).
pub fn parse_live_args() -> Result<LiveArgs> {
    parse_live_args_from(env::args().skip(1), env::var("LIVE_INPUT").ok())
}

/// To isté z ľubovoľného zoznamu argumentov a hodnoty `LIVE_INPUT` – kvôli testom.
pub fn parse_live_args_from<I: IntoIterator<Item = String>>(
    args: I,
    env_input: Option<String>,
) -> Result<LiveArgs> {
    let mut it = args.into_iter();
    let mut a = LiveArgs {
        input: String::new(),
        mode: Mode::People,
        width: 1280,
        model: "models/yolox_s.onnx".into(),
        conf: 0.4,
        port: 8554,
        mount: "/detect".into(),
        codec: Codec::H265,
        fps: 20,
        bitrate: 4000,
        line: None,
        line_span: (0.0, 1.0),
        roi: Crop::FULL,
        detect_width: 2560,
        margin: 12.0,
        max_missed: 12,
        out_dir: "out".into(),
        anchor: true,
        anchor_every: 2,
        anchor_features: 1500,
        anchor_min_inliers: 20,
        anchor_points: 10,
        anchor_calib: 12,
        anchor_deadband: 1.0,
        anchor_log: false,
        output: None,
    };
    let mut input = None;
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} chýba hodnota"));
        match arg.as_str() {
            "--input" => input = Some(val("--input")?),
            "--mode" => a.mode = parse_mode(val("--mode")?.as_str().into())?,
            "--width" => a.width = val("--width")?.parse()?,
            "--model" => a.model = val("--model")?,
            "--conf" => {
                a.conf = val("--conf")?.parse()?;
                if !(0.0..=1.0).contains(&a.conf) {
                    return Err("--conf musí byť medzi 0 a 1".into());
                }
            }
            "--port" => a.port = val("--port")?.parse()?,
            "--mount" => {
                a.mount = val("--mount")?;
                if !a.mount.starts_with('/') {
                    return Err("--mount musí začínať lomkou, napr. /detect".into());
                }
            }
            "--codec" => {
                a.codec = match val("--codec")?.as_str() {
                    "h264" => Codec::H264,
                    "h265" => Codec::H265,
                    _ => return Err("--codec: h264 alebo h265".into()),
                }
            }
            "--fps" => {
                a.fps = val("--fps")?.parse()?;
                if a.fps < 1 {
                    return Err("--fps musí byť aspoň 1".into());
                }
            }
            "--bitrate" => a.bitrate = val("--bitrate")?.parse()?,
            "--line" => {
                let l: f64 = val("--line")?.parse()?;
                if !(l > 0.0 && l < 1.0) {
                    return Err("--line musí byť medzi 0 a 1 (zlomok šírky záberu)".into());
                }
                a.line = Some(l);
            }
            "--line-span" => {
                let v = val("--line-span")?;
                let (y0, y1) = v
                    .split_once(',')
                    .and_then(|(p, q)| Some((p.trim().parse::<f64>().ok()?, q.trim().parse::<f64>().ok()?)))
                    .ok_or("--line-span: očakávam dve čísla od,do (napr. 0.52,0.85)")?;
                if !(0.0 <= y0 && y0 < y1 && y1 <= 1.0) {
                    return Err("--line-span: musí platiť 0 <= od < do <= 1".into());
                }
                a.line_span = (y0, y1);
            }
            "--roi" => a.roi = Crop::parse(&val("--roi")?)?,
            "--detect-width" => a.detect_width = val("--detect-width")?.parse()?,
            "--margin" => {
                a.margin = val("--margin")?.parse()?;
                if a.margin < 0.0 {
                    return Err("--margin nesmie byť záporný".into());
                }
            }
            "--max-missed" => a.max_missed = val("--max-missed")?.parse()?,
            "--out-dir" => a.out_dir = val("--out-dir")?,
            "--anchor" => {
                a.anchor = match val("--anchor")?.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => return Err("--anchor: on alebo off".into()),
                }
            }
            "--anchor-every" => {
                a.anchor_every = val("--anchor-every")?.parse()?;
                if a.anchor_every < 1 {
                    return Err("--anchor-every musí byť aspoň 1".into());
                }
            }
            "--anchor-features" => {
                a.anchor_features = val("--anchor-features")?.parse()?;
                if a.anchor_features < 100 {
                    return Err("--anchor-features musí byť aspoň 100".into());
                }
            }
            "--anchor-min-inliers" => {
                a.anchor_min_inliers = val("--anchor-min-inliers")?.parse()?;
                if a.anchor_min_inliers < 8 {
                    return Err("--anchor-min-inliers musí byť aspoň 8".into());
                }
            }
            "--anchor-log" => a.anchor_log = true,
            "--output" => a.output = Some(val("--output")?),
            "--anchor-points" => {
                a.anchor_points = val("--anchor-points")?.parse()?;
                if a.anchor_points != 0 && a.anchor_points < 6 {
                    return Err("--anchor-points: 0 (vypnuté) alebo aspoň 6".into());
                }
            }
            "--anchor-calib" => {
                a.anchor_calib = val("--anchor-calib")?.parse()?;
                if a.anchor_calib < 3 {
                    return Err("--anchor-calib musí byť aspoň 3".into());
                }
            }
            "--anchor-deadband" => {
                a.anchor_deadband = val("--anchor-deadband")?.parse()?;
                if a.anchor_deadband < 0.0 {
                    return Err("--anchor-deadband nesmie byť záporný".into());
                }
            }
            _ => return Err(format!("neznámy argument: {arg}").into()),
        }
    }
    a.input = input
        .or(env_input)
        .filter(|s| !s.is_empty())
        .ok_or("použitie: object_detect_live --input rtsp://... (alebo premenná LIVE_INPUT) [--mode people] [--port 8554] [--mount /detect] [--codec h265|h264] [--width 1280] [--conf 0.4] [--fps 20] [--bitrate 4000] [--line 0.235 --line-span 0.52,0.85 --roi 0,0.52,0.47,0.33] [--anchor on|off] [--anchor-points 10]")?;
    Ok(a)
}

/// Argumenty programu na počítanie prechodov cez čiaru (`line_count`).
#[derive(Debug)]
pub struct CountArgs {
    pub input: String,
    pub crop: Crop,       // ktorá časť záberu sa spracuje (zlomky pôvodného obrazu)
    pub line: f64,        // poloha zvislej čiary ako zlomok šírky orezaného obrazu (0-1)
    pub width: i32,       // šírka spracovaného obrazu v px
    pub model: String,
    pub conf: f32,
    pub margin: f64,      // pásmo okolo čiary v px, ktoré sa nepočíta (proti chveniu)
    pub max_missed: u32,  // po koľkých snímkach bez nálezu sa objekt zabudne
    pub every: u32,       // spracuje sa každá N-tá snímka (1 = všetky; vyššie = rýchlejšie)
    pub out_dir: String,  // kam sa uložia výsledky
    pub name: Option<String>, // základ názvov výstupných súborov (predvolene názov videa)
}

/// Argumenty z príkazového riadku.
pub fn parse_count_args() -> Result<CountArgs> {
    parse_count_args_from(env::args().skip(1))
}

/// To isté z ľubovoľného zoznamu argumentov – kvôli testom.
pub fn parse_count_args_from<I: IntoIterator<Item = String>>(args: I) -> Result<CountArgs> {
    let mut it = args.into_iter();
    let mut a = CountArgs {
        input: String::new(),
        crop: Crop::FULL,
        line: 0.5,
        width: 1280,
        model: "models/yolox_s.onnx".into(),
        conf: 0.4,
        margin: 12.0,
        max_missed: 12,
        every: 1,
        out_dir: "out".into(),
        name: None,
    };
    let mut input = None;
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} chýba hodnota"));
        match arg.as_str() {
            "--crop" => a.crop = Crop::parse(&val("--crop")?)?,
            "--line" => {
                a.line = val("--line")?.parse()?;
                if !(a.line > 0.0 && a.line < 1.0) {
                    return Err("--line musí byť medzi 0 a 1 (zlomok šírky orezaného obrazu)".into());
                }
            }
            "--width" => a.width = val("--width")?.parse()?,
            "--model" => a.model = val("--model")?,
            "--conf" => {
                a.conf = val("--conf")?.parse()?;
                if !(0.0..=1.0).contains(&a.conf) {
                    return Err("--conf musí byť medzi 0 a 1".into());
                }
            }
            "--margin" => {
                a.margin = val("--margin")?.parse()?;
                if a.margin < 0.0 {
                    return Err("--margin nesmie byť záporný".into());
                }
            }
            "--max-missed" => a.max_missed = val("--max-missed")?.parse()?,
            "--every" => {
                a.every = val("--every")?.parse()?;
                if a.every < 1 {
                    return Err("--every musí byť aspoň 1".into());
                }
            }
            "--out-dir" => a.out_dir = val("--out-dir")?,
            "--name" => a.name = Some(val("--name")?),
            _ if arg.starts_with("--") => return Err(format!("neznámy argument: {arg}").into()),
            _ if input.is_none() => input = Some(arg),
            _ => return Err(format!("neznámy argument: {arg}").into()),
        }
    }
    a.input = input.ok_or("použitie: line_count <video.mp4> [--crop x,y,w,h] [--line 0.5] [--width 1280] [--conf 0.4] [--margin 12] [--every 1] [--out-dir out] [--name nazov]")?;
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: &[&str]) -> Result<Args> {
        parse_args_from(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn predvolene_hodnoty() {
        let a = parse(&["video.mp4"]).unwrap();
        assert_eq!(a.input, "video.mp4");
        assert_eq!(a.output, "annotated.mp4");
        assert_eq!(a.mode, Mode::Motion);
        assert_eq!(a.width, 1280);
        assert_eq!(a.model, "models/yolox_s.onnx");
        assert_eq!(a.conf, 0.4);
    }

    #[test]
    fn vsetky_prepinace() {
        let a = parse(&["--mode", "people", "v.mp4", "--output", "o.mp4", "--width", "640", "--model", "m.onnx", "--conf", "0.6"]).unwrap();
        assert_eq!(a.input, "v.mp4"); // vstup môže byť aj za prepínačmi
        assert_eq!(a.output, "o.mp4");
        assert_eq!(a.mode, Mode::People);
        assert_eq!(a.width, 640);
        assert_eq!(a.model, "m.onnx");
        assert_eq!(a.conf, 0.6);
    }

    #[test]
    fn chyba_bez_vstupneho_suboru() {
        let e = parse(&[]).unwrap_err().to_string();
        assert!(e.contains("použitie"), "{e}");
    }

    #[test]
    fn chyba_neznamy_rezim() {
        assert!(parse(&["v.mp4", "--mode", "yolo"]).is_err());
        assert!(parse(&["v.mp4", "--mode"]).is_err()); // chýba hodnota
    }

    #[test]
    fn chyba_nevalidna_sirka() {
        assert!(parse(&["v.mp4", "--width", "abc"]).is_err());
        assert!(parse(&["v.mp4", "--width"]).is_err());
    }

    #[test]
    fn chyba_druhy_volny_argument() {
        let e = parse(&["a.mp4", "b.mp4"]).unwrap_err().to_string();
        assert!(e.contains("neznámy argument"), "{e}");
    }

    #[test]
    fn rezim_objects() {
        assert_eq!(parse(&["v.mp4", "--mode", "objects"]).unwrap().mode, Mode::Objects);
    }

    #[test]
    fn chyba_conf_mimo_rozsah() {
        assert!(parse(&["v.mp4", "--conf", "1.5"]).is_err());
        assert!(parse(&["v.mp4", "--conf", "-0.1"]).is_err());
        assert!(parse(&["v.mp4", "--conf", "abc"]).is_err());
        assert!(parse(&["v.mp4", "--conf", "0"]).is_ok() && parse(&["v.mp4", "--conf", "1"]).is_ok());
    }

    #[test]
    fn chyba_model_bez_hodnoty() {
        assert!(parse(&["v.mp4", "--model"]).is_err());
    }

    // ---- živý režim ----

    fn live(v: &[&str]) -> Result<LiveArgs> {
        parse_live_args_from(v.iter().map(|s| s.to_string()), None)
    }

    #[test]
    fn live_predvolene_hodnoty() {
        let a = live(&["--input", "rtsp://cam/s"]).unwrap();
        assert_eq!(a.input, "rtsp://cam/s");
        assert_eq!(a.mode, Mode::People);
        assert_eq!((a.port, a.mount.as_str()), (8554, "/detect"));
        assert_eq!(a.codec, Codec::H265);
        assert_eq!((a.width, a.fps, a.bitrate), (1280, 20, 4000));
        assert_eq!(a.conf, 0.4);
    }

    #[test]
    fn live_vsetky_prepinace() {
        let a = live(&[
            "--input", "test:ball", "--mode", "objects", "--width", "640", "--model", "m.onnx",
            "--conf", "0.7", "--port", "9000", "--mount", "/x", "--codec", "h264", "--fps", "15",
            "--bitrate", "2000",
        ])
        .unwrap();
        assert_eq!(a.mode, Mode::Objects);
        assert_eq!((a.width, a.port, a.fps, a.bitrate), (640, 9000, 15, 2000));
        assert_eq!((a.model.as_str(), a.mount.as_str(), a.codec, a.conf), ("m.onnx", "/x", Codec::H264, 0.7));
    }

    #[test]
    fn live_vstup_z_premennej_prostredia() {
        let a = parse_live_args_from(Vec::<String>::new(), Some("rtsp://tajne".into())).unwrap();
        assert_eq!(a.input, "rtsp://tajne");
    }

    #[test]
    fn live_argument_ma_prednost_pred_premennou() {
        let a = parse_live_args_from(["--input".to_string(), "test:ball".to_string()], Some("rtsp://env".into())).unwrap();
        assert_eq!(a.input, "test:ball");
    }

    #[test]
    fn live_chyba_bez_vstupu() {
        assert!(live(&[]).unwrap_err().to_string().contains("použitie"));
        // prázdna premenná prostredia sa počíta ako chýbajúca
        assert!(parse_live_args_from(Vec::<String>::new(), Some(String::new())).is_err());
    }

    #[test]
    fn live_chybne_hodnoty() {
        let base = ["--input", "test:ball"];
        for bad in [
            &["--port", "70000"][..],  // mimo rozsahu u16
            &["--port", "abc"],
            &["--mount", "detect"],    // bez lomky
            &["--codec", "av1"],
            &["--fps", "0"],
            &["--conf", "2"],
            &["--mode", "yolo"],
            &["--width"],              // chýba hodnota
            &["--neznamy", "1"],
        ] {
            let v: Vec<&str> = base.iter().chain(bad.iter()).copied().collect();
            assert!(live(&v).is_err(), "{bad:?} mal zlyhať");
        }
    }

    // ---- počítanie prechodov ----

    fn count(v: &[&str]) -> Result<CountArgs> {
        parse_count_args_from(v.iter().map(|s| s.to_string()))
    }

    #[test]
    fn count_predvolene_hodnoty() {
        let a = count(&["street.mp4"]).unwrap();
        assert_eq!(a.input, "street.mp4");
        assert_eq!(a.crop, Crop::FULL);
        assert_eq!((a.line, a.width, a.conf, a.margin, a.max_missed), (0.5, 1280, 0.4, 12.0, 12));
        assert_eq!(a.every, 1);
        assert_eq!(a.out_dir, "out");
        assert_eq!(a.name, None);
    }

    #[test]
    fn count_vsetky_prepinace() {
        let a = count(&[
            "--crop", "0,0.4,0.5,0.4", "s.mp4", "--line", "0.3", "--width", "960", "--model", "m.onnx",
            "--conf", "0.55", "--margin", "20", "--max-missed", "5", "--out-dir", "vysledky", "--name", "ulica",
        ])
        .unwrap();
        assert_eq!(a.crop, Crop { x: 0.0, y: 0.4, w: 0.5, h: 0.4 });
        assert_eq!((a.line, a.width, a.conf, a.margin, a.max_missed), (0.3, 960, 0.55, 20.0, 5));
        assert_eq!((a.model.as_str(), a.out_dir.as_str()), ("m.onnx", "vysledky"));
        assert_eq!(a.name.as_deref(), Some("ulica"));
    }

    #[test]
    fn count_chybne_hodnoty() {
        for bad in [
            &["--line", "0"][..], &["--line", "1"], &["--line", "-0.2"], &["--line", "abc"],
            &["--crop", "1,2,3"], &["--crop", "0,0,2,2"], &["--conf", "7"], &["--margin", "-1"],
            &["--width"], &["--neznamy", "1"], &["--every", "0"], &["--every", "x"],
        ] {
            let v: Vec<&str> = ["s.mp4"].iter().chain(bad.iter()).copied().collect();
            assert!(count(&v).is_err(), "{bad:?} mal zlyhať");
        }
    }

    #[test]
    fn count_chyba_bez_videa_a_druhe_video() {
        assert!(count(&[]).unwrap_err().to_string().contains("použitie"));
        assert!(count(&["a.mp4", "b.mp4"]).is_err());
    }

    // ---- živé počítanie ----

    #[test]
    fn live_pocitanie_je_predvolene_vypnute() {
        let a = live(&["--input", "test:ball"]).unwrap();
        assert_eq!(a.line, None);
        assert_eq!(a.line_span, (0.0, 1.0));
        assert_eq!(a.roi, Crop::FULL);
        assert_eq!((a.detect_width, a.margin, a.max_missed, a.out_dir.as_str()), (2560, 12.0, 12, "out"));
    }

    #[test]
    fn live_pocitanie_prepinace() {
        let a = live(&[
            "--input", "file:/v.mp4", "--line", "0.235", "--line-span", "0.52,0.85", "--roi", "0,0.52,0.47,0.33",
            "--detect-width", "3000", "--margin", "20", "--max-missed", "8", "--out-dir", "logy",
        ])
        .unwrap();
        assert_eq!(a.line, Some(0.235));
        assert_eq!(a.line_span, (0.52, 0.85));
        assert_eq!(a.roi, Crop { x: 0.0, y: 0.52, w: 0.47, h: 0.33 });
        assert_eq!((a.detect_width, a.margin, a.max_missed, a.out_dir.as_str()), (3000, 20.0, 8, "logy"));
    }

    #[test]
    fn live_pocitanie_chybne_hodnoty() {
        for bad in [
            &["--line", "0"][..], &["--line", "1"], &["--line", "x"],
            &["--line-span", "0.5"], &["--line-span", "0.8,0.2"], &["--line-span", "0.2,1.5"], &["--line-span", "a,b"],
            &["--roi", "0,0,2,2"], &["--roi", "1,2,3"], &["--margin", "-3"],
        ] {
            let v: Vec<&str> = ["--input", "test:ball"].iter().chain(bad.iter()).copied().collect();
            assert!(live(&v).is_err(), "{bad:?} mal zlyhať");
        }
    }

    // ---- ukotvenie ----

    #[test]
    fn ukotvenie_je_predvolene_zapnute() {
        let a = live(&["--input", "test:ball"]).unwrap();
        assert!(a.anchor);
        assert_eq!((a.anchor_every, a.anchor_features, a.anchor_min_inliers), (2, 1500, 20));
        assert_eq!((a.anchor_points, a.anchor_calib, a.anchor_deadband), (10, 12, 1.0));
        assert!(!a.anchor_log);
        assert_eq!(a.output, None);
    }

    #[test]
    fn live_output() {
        assert_eq!(live(&["--input", "x", "--output", "v.mp4"]).unwrap().output.as_deref(), Some("v.mp4"));
        assert!(live(&["--input", "x", "--output"]).is_err());
    }

    #[test]
    fn ukotvenie_log_je_prepinac_bez_hodnoty() {
        let a = live(&["--input", "x", "--anchor-log", "--port", "9000"]).unwrap();
        assert!(a.anchor_log);
        assert_eq!(a.port, 9000, "argument za prepínačom sa nepovažuje za jeho hodnotu");
    }

    #[test]
    fn ukotvenie_prepinace() {
        let a = live(&["--input", "x", "--anchor", "off", "--anchor-every", "5", "--anchor-features", "800", "--anchor-min-inliers", "12"]).unwrap();
        assert!(!a.anchor);
        assert_eq!((a.anchor_every, a.anchor_features, a.anchor_min_inliers), (5, 800, 12));
        assert!(live(&["--input", "x", "--anchor", "on"]).unwrap().anchor);
    }

    #[test]
    fn ukotvenie_chybne_hodnoty() {
        for bad in [
            &["--anchor", "maybe"][..], &["--anchor-every", "0"], &["--anchor-every", "x"],
            &["--anchor-features", "50"], &["--anchor-min-inliers", "3"], &["--anchor"],
        ] {
            let v: Vec<&str> = ["--input", "x"].iter().chain(bad.iter()).copied().collect();
            assert!(live(&v).is_err(), "{bad:?} mal zlyhať");
        }
    }

    #[test]
    fn count_every() {
        assert_eq!(count(&["s.mp4", "--every", "3"]).unwrap().every, 3);
    }

    #[test]
    fn ukotvenie_stabilne_body_prepinace() {
        let a = live(&["--input", "x", "--anchor-points", "8", "--anchor-calib", "20", "--anchor-deadband", "1.2"]).unwrap();
        assert_eq!((a.anchor_points, a.anchor_calib, a.anchor_deadband), (8, 20, 1.2));
        assert_eq!(live(&["--input", "x", "--anchor-points", "0"]).unwrap().anchor_points, 0, "0 = len ORB");
    }

    #[test]
    fn ukotvenie_stabilne_body_chybne_hodnoty() {
        for bad in [
            &["--anchor-points", "3"][..], &["--anchor-points", "x"], &["--anchor-calib", "2"],
            &["--anchor-deadband", "-1"], &["--anchor-deadband", "x"], &["--anchor-points"],
        ] {
            let v: Vec<&str> = ["--input", "x"].iter().chain(bad.iter()).copied().collect();
            assert!(live(&v).is_err(), "{bad:?} mal zlyhať");
        }
    }
}
