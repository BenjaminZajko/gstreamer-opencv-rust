//! End-to-end testy: spustia skutočnú binárku `object_detect` na vygenerovanom videu
//! a skontrolujú výstupný súbor. Vyžadujú GStreamer s pluginmi x264enc, mp4mux, vtenc_h264_hw (macOS).

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;

const WIDTH: i32 = 320;
const HEIGHT: i32 = 240;
const FRAMES: usize = 90; // 3 s pri 30 fps; prvých 30 snímok sa MOG2 zahrieva

/// Dočasný priečinok, ktorý sa po teste zmaže.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("object_detect_e2e_{}_{name}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    fn path(&self, file: &str) -> PathBuf {
        self.0.join(file)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Vygeneruje testovacie video: biela lopta sa pohybuje po čiernom pozadí.
fn make_test_video(path: &Path) {
    gst::init().unwrap();
    let pipeline = gst::parse::launch(&format!(
        "videotestsrc pattern=ball num-buffers={FRAMES} \
         ! video/x-raw,width={WIDTH},height={HEIGHT},framerate=30/1 \
         ! videoconvert ! x264enc ! h264parse ! mp4mux ! filesink location=\"{}\"",
        path.display()
    ))
    .unwrap();
    pipeline.set_state(gst::State::Playing).unwrap();
    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        match msg.view() {
            gst::MessageView::Eos(_) => break,
            gst::MessageView::Error(e) => panic!("generovanie videa zlyhalo: {}", e.error()),
            _ => {}
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();
}

/// Cesta k modelu YOLOX; ak nie je stiahnutý, testy, ktoré ho potrebujú, sa preskočia.
fn model() -> Option<String> {
    let p = format!("{}/models/yolox_s.onnx", env!("CARGO_MANIFEST_DIR"));
    Path::new(&p).exists().then_some(p)
}

/// Spustí binárku s danými argumentmi.
fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_object_detect"))
        .args(args)
        .output()
        .expect("binárka sa nespustila")
}

/// Prečíta video: vráti (počet snímok, šírka, výška).
fn probe(path: &Path) -> (usize, i32, i32) {
    gst::init().unwrap();
    let p = gst::parse::launch(&format!(
        "filesrc location=\"{}\" ! decodebin ! videoconvert \
         ! video/x-raw,format=BGR ! appsink name=sink sync=false",
        path.display()
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();
    let sink = p.by_name("sink").unwrap().downcast::<gst_app::AppSink>().unwrap();
    p.set_state(gst::State::Playing).unwrap();
    let (mut n, mut w, mut h) = (0, 0, 0);
    while let Ok(sample) = sink.pull_sample() {
        let info = gst_video::VideoInfo::from_caps(sample.caps().unwrap()).unwrap();
        (w, h) = (info.width() as i32, info.height() as i32);
        n += 1;
    }
    p.set_state(gst::State::Null).unwrap();
    (n, w, h)
}

/// Z výpisu "Hotovo: ... , N detekcií celkom -> ..." vytiahne N.
fn detections(stdout: &str) -> u64 {
    let line = stdout.lines().find(|l| l.starts_with("Hotovo")).unwrap_or_else(|| panic!("chýba riadok Hotovo:\n{stdout}"));
    let before = line.split(" detekcií celkom").next().unwrap();
    before.rsplit([',', ' ']).next().unwrap().parse().unwrap()
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn motion_spracuje_video_a_najde_pohyb() {
    let dir = TempDir::new("motion");
    let (input, output) = (dir.path("in.mp4"), dir.path("out.mp4"));
    make_test_video(&input);

    let o = run(&[input.to_str().unwrap(), "--mode", "motion", "--output", output.to_str().unwrap(), "--width", "320"]);
    assert!(o.status.success(), "stderr: {}", stderr(&o));

    // výstup existuje, nie je prázdny a je prehrateľný
    assert!(output.exists());
    assert!(fs::metadata(&output).unwrap().len() > 1000);
    let (frames, w, h) = probe(&output);
    assert_eq!(frames, FRAMES, "výstup má mať rovnaký počet snímok ako vstup");
    assert_eq!((w, h), (WIDTH, HEIGHT), "rozmery sa nemajú zmeniť");

    // lopta sa po zahriatí hýbe, takže detekcie musia byť
    assert!(detections(&stdout(&o)) > 0, "{}", stdout(&o));
}

#[test]
fn people_nenajde_ludi_v_lopte_ale_dobehne() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let dir = TempDir::new("people");
    let (input, output) = (dir.path("in.mp4"), dir.path("out.mp4"));
    make_test_video(&input);

    let o = run(&[input.to_str().unwrap(), "--mode", "people", "--model", &model, "--output", output.to_str().unwrap(), "--width", "320"]);
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    assert_eq!(probe(&output).0, FRAMES);
    assert_eq!(detections(&stdout(&o)), 0, "lopta nie je človek");
}

#[test]
fn objects_dobehne_a_zachova_video() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let dir = TempDir::new("objects");
    let (input, output) = (dir.path("in.mp4"), dir.path("out.mp4"));
    make_test_video(&input);

    let o = run(&[input.to_str().unwrap(), "--mode", "objects", "--model", &model, "--output", output.to_str().unwrap(), "--width", "320"]);
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let (frames, w, h) = probe(&output);
    assert_eq!((frames, w, h), (FRAMES, WIDTH, HEIGHT));
}

#[test]
fn people_bez_modelu_zlyha_s_napovedou() {
    let dir = TempDir::new("nomodel");
    let (input, output) = (dir.path("in.mp4"), dir.path("out.mp4"));
    make_test_video(&input);

    let o = run(&[input.to_str().unwrap(), "--mode", "people", "--model", "/neexistuje/m.onnx", "--output", output.to_str().unwrap()]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("yolox_s.onnx"), "{}", stderr(&o));
    assert!(!output.exists());
}

#[test]
fn width_zmeni_rozmer_vystupu_a_zachova_pomer_stran() {
    let dir = TempDir::new("width");
    let (input, output) = (dir.path("in.mp4"), dir.path("out.mp4"));
    make_test_video(&input);

    let o = run(&[input.to_str().unwrap(), "--output", output.to_str().unwrap(), "--width", "160"]);
    assert!(o.status.success(), "stderr: {}", stderr(&o));
    let (_, w, h) = probe(&output);
    assert_eq!((w, h), (160, 120), "320x240 zmenšené na šírku 160 má mať výšku 120");
}

#[test]
fn bez_argumentov_zlyha_s_napovedou() {
    let o = run(&[]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("použitie"), "{}", stderr(&o));
}

#[test]
fn neznamy_rezim_zlyha() {
    let o = run(&["video.mp4", "--mode", "yolo"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("--mode"), "{}", stderr(&o));
}

#[test]
fn neexistujuci_vstup_zlyha_a_nevytvori_vystup() {
    let dir = TempDir::new("missing");
    let output = dir.path("out.mp4");
    let o = run(&["/neexistuje/video.mp4", "--output", output.to_str().unwrap()]);
    assert!(!o.status.success());
    assert!(!output.exists(), "pri chybe nemá vzniknúť výstupný súbor");
}
