//! E2E testy programu `video_count` (súbor -> počítanie s ukotvenou čiarou -> video).

use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, CV_8UC1},
    imgproc,
    prelude::*,
    videoio::VideoWriter,
};

const W: i32 = 640;
const H: i32 = 360;

struct Dir(PathBuf);
impl Dir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("video_count_e2e_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
    fn f(&self, n: &str) -> String {
        self.0.join(n).to_string_lossy().into_owned()
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn model() -> Option<String> {
    let p = format!("{}/models/yolox_s.onnx", env!("CARGO_MANIFEST_DIR"));
    Path::new(&p).exists().then_some(p)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: u64) -> i32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n) as i32
    }
}

/// Plátno s množstvom tvarov (aby malo ukotvenie z čoho vyberať body).
fn canvas(seed: u64, w: i32, h: i32) -> Mat {
    let mut img = Mat::new_rows_cols_with_default(h, w, CV_8UC1, Scalar::all(120.0)).unwrap();
    let mut r = Lcg(seed);
    for _ in 0..420 {
        let c = Scalar::all(30.0 + r.next(200) as f64);
        let (x, y) = (r.next(w as u64 - 60), r.next(h as u64 - 60));
        match r.next(3) {
            0 => imgproc::rectangle(&mut img, Rect::new(x, y, 8 + r.next(50), 8 + r.next(50)), c, -1, imgproc::LINE_8, 0).unwrap(),
            1 => imgproc::circle(&mut img, Point::new(x + 30, y + 30), 5 + r.next(25), c, -1, imgproc::LINE_8, 0).unwrap(),
            _ => imgproc::line(&mut img, Point::new(x, y), Point::new(x + r.next(80), y + r.next(80)), c, 2, imgproc::LINE_8, 0).unwrap(),
        }
    }
    img
}

fn to_bgr(gray: &Mat) -> Mat {
    let mut out = Mat::default();
    imgproc::cvt_color_def(gray, &mut out, imgproc::COLOR_GRAY2BGR).unwrap();
    out
}

fn write_video(path: &str, frames: &[Mat]) -> bool {
    let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v').unwrap();
    let Ok(mut w) = VideoWriter::new_def(path, fourcc, 20.0, Size::new(W, H)) else { return false };
    if !w.is_opened().unwrap_or(false) {
        return false;
    }
    for f in frames {
        w.write(f).unwrap();
    }
    w.release().unwrap();
    true
}

/// Kamera 2 s stojí, potom sa posunie o 90 px a ostane stáť (spolu 160 snímok pri 20 fps).
fn moved_video(path: &str) -> bool {
    let big = canvas(44, W + 130, H + 40);
    let view = |off: i32| to_bgr(&Mat::roi(&big, Rect::new(off, 20, W, H)).unwrap().try_clone().unwrap());
    let (a, b) = (view(0), view(90));
    let frames: Vec<Mat> = (0..160).map(|i| if i < 40 { a.try_clone().unwrap() } else { b.try_clone().unwrap() }).collect();
    write_video(path, &frames)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_video_count")).args(args).output().unwrap()
}

/// (šírka, výška, počet snímok) videa.
fn probe(path: &str) -> (i32, i32, usize) {
    gst::init().unwrap();
    let p = gst::parse::launch(&format!(
        "filesrc location=\"{path}\" ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink sync=false"
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();
    let sink = p.by_name("sink").unwrap().downcast::<gst_app::AppSink>().unwrap();
    p.set_state(gst::State::Playing).unwrap();
    let (mut n, mut w, mut h) = (0, 0, 0);
    while let Ok(s) = sink.pull_sample() {
        let i = gst_video::VideoInfo::from_caps(s.caps().unwrap()).unwrap();
        (w, h) = (i.width() as i32, i.height() as i32);
        n += 1;
    }
    p.set_state(gst::State::Null).unwrap();
    (w, h, n)
}

#[test]
fn spracuje_video_s_posunom_kamery_a_ulozi_vysledky() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let d = Dir::new("moved");
    if !moved_video(&d.f("cam.mp4")) {
        return eprintln!("OpenCV nevie zapisovať video – test preskočený");
    }
    let out = d.f("o");
    let o = run(&[&d.f("cam.mp4"), "--line", "0.5", "--width", "640", "--detect-width", "640", "--model", &model, "--out-dir", &out]);
    let stdout = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{}\n{stdout}", String::from_utf8_lossy(&o.stderr));

    // výstupné video: všetky snímky, rozmer záberu
    let (w, h, n) = probe(&format!("{out}/cam_counted.mp4"));
    assert_eq!((w, h, n), (640, 360, 160));
    // ukotvenie: kalibrácia, referenčná fotka a nový záber po posune kamery
    assert!(stdout.contains("vybraných 10 stabilných bodov"), "{stdout}");
    assert!(stdout.contains("nový referenčný záber"), "po posune kamery sa mal vytvoriť nový záber:\n{stdout}");
    assert!(Path::new(&format!("{out}/cam_reference.jpg")).exists());
    // CSV a súhrn
    let csv = std::fs::read_to_string(format!("{out}/cam_events.csv")).unwrap();
    assert_eq!(csv.lines().next().unwrap(), "time_s,frame,track_id,group,direction");
    let summary = std::fs::read_to_string(format!("{out}/cam_summary.txt")).unwrap();
    assert!(summary.contains("Snímok: 160") && summary.contains("ukotvenie zapnuté"), "{summary}");
}

#[test]
fn vlastna_cesta_vystupu_a_vypnute_ukotvenie() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let d = Dir::new("output");
    if !moved_video(&d.f("cam.mp4")) {
        return eprintln!("OpenCV nevie zapisovať video – test preskočený");
    }
    let target = d.f("vysledok.mp4");
    let o = run(&[&d.f("cam.mp4"), "--anchor", "off", "--width", "320", "--detect-width", "640", "--model", &model, "--out-dir", &d.f("o"), "--output", &target]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(probe(&target), (320, 180, 160));
    assert!(String::from_utf8_lossy(&o.stdout).contains("ukotvenie vypnuté"));
}

#[test]
fn neexistujuce_video_zlyha() {
    let o = run(&["/neexistuje/video.mp4"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("neexistuje"));
}

#[test]
fn bez_videa_ukaze_pouzitie() {
    let o = run(&[]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("video_count"));
}

#[test]
fn zla_hodnota_ciary_zlyha() {
    let o = run(&["x.mp4", "--line", "3"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--line"));
}
