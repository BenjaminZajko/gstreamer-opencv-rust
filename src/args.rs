//! Spracovanie argumentov z príkazového riadku.
//cargo run --release -- video.mp4 --mode motion --output out.mp4 --width 1280

use std::env;

use crate::{video_io::Codec, Result};

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
            _ => return Err(format!("neznámy argument: {arg}").into()),
        }
    }
    a.input = input
        .or(env_input)
        .filter(|s| !s.is_empty())
        .ok_or("použitie: object_detect_live --input rtsp://... (alebo premenná LIVE_INPUT) [--mode people] [--port 8554] [--mount /detect] [--codec h265|h264] [--width 1280] [--conf 0.4] [--fps 20] [--bitrate 4000]")?;
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
}
