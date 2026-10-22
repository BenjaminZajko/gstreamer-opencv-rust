//! E2E testy programu `line_count` (spúšťa skutočnú binárku).

use std::{fs, path::PathBuf, process::{Command, Output}};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;

struct TempDir(PathBuf);
impl TempDir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("line_count_e2e_{}_{name}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    fn path(&self, f: &str) -> String {
        self.0.join(f).to_string_lossy().into_owned()
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn model() -> Option<String> {
    let p = format!("{}/models/yolox_s.onnx", env!("CARGO_MANIFEST_DIR"));
    std::path::Path::new(&p).exists().then_some(p)
}

/// Video 640x360 s pohybujúcou sa loptou (lopta nie je auto ani človek).
fn make_video(path: &str) {
    gst::init().unwrap();
    let p = gst::parse::launch(&format!(
        "videotestsrc pattern=ball num-buffers=60 ! video/x-raw,width=640,height=360,framerate=12/1 \
         ! videoconvert ! x264enc ! h264parse ! mp4mux ! filesink location=\"{path}\""
    ))
    .unwrap();
    p.set_state(gst::State::Playing).unwrap();
    for m in p.bus().unwrap().iter_timed(gst::ClockTime::NONE) {
        match m.view() {
            gst::MessageView::Eos(_) => break,
            gst::MessageView::Error(e) => panic!("{}", e.error()),
            _ => {}
        }
    }
    p.set_state(gst::State::Null).unwrap();
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_line_count")).args(args).output().unwrap()
}

fn size_and_frames(path: &str) -> (i32, i32, usize) {
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
fn vytvori_video_csv_a_suhrn_s_orezanim() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let d = TempDir::new("ok");
    let input = d.path("in.mp4");
    make_video(&input);
    let out = d.path("out");

    // ľavá polovica, horné dve tretiny: 320x240 zo 640x360, zmenšené na šírku 160 -> 160x120
    let o = run(&[&input, "--crop", "0,0,0.5,0.6667", "--width", "160", "--model", &model, "--out-dir", &out, "--name", "t"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));

    let (w, h, n) = size_and_frames(&format!("{out}/t_counted.mp4"));
    assert_eq!((w, h), (160, 120), "výstup má mať rozmery orezaného záberu");
    assert_eq!(n, 60);

    let csv = fs::read_to_string(format!("{out}/t_events.csv")).unwrap();
    assert_eq!(csv.lines().next().unwrap(), "time_s,frame,track_id,group,direction");
    assert_eq!(csv.lines().count(), 1, "lopta nie je auto ani človek: žiadne udalosti");

    let summary = fs::read_to_string(format!("{out}/t_summary.txt")).unwrap();
    assert!(summary.contains("Autá: spolu 0") && summary.contains("Ľudia: spolu 0"), "{summary}");
    assert!(summary.contains("Snímok: 60"), "{summary}");
    assert!(!summary.contains("\\n"), "súhrn nesmie obsahovať doslovné \\n: {summary}");
}

#[test]
fn nazov_vystupu_je_predvolene_nazov_videa() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let d = TempDir::new("name");
    let input = d.path("moje_video.mp4");
    make_video(&input);
    let out = d.path("o");
    assert!(run(&[&input, "--width", "160", "--model", &model, "--out-dir", &out]).status.success());
    assert!(fs::metadata(format!("{out}/moje_video_counted.mp4")).is_ok());
    assert!(fs::metadata(format!("{out}/moje_video_events.csv")).is_ok());
}

#[test]
fn chybajuce_video_zlyha_zrozumitelne() {
    let o = run(&["/neexistuje/video.mp4", "--model", "/neexistuje/m.onnx"]);
    assert!(!o.status.success());
}

#[test]
fn zly_vyrez_zlyha_a_nevytvori_vystup() {
    let d = TempDir::new("badcrop");
    let input = d.path("in.mp4");
    make_video(&input);
    let out = d.path("o");
    let o = run(&[&input, "--crop", "0.8,0.8,0.5,0.5", "--out-dir", &out]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--crop"), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(!std::path::Path::new(&format!("{out}/in_counted.mp4")).exists());
}

#[test]
fn zla_ciara_zlyha() {
    let o = run(&["x.mp4", "--line", "1.5"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--line"));
}

#[test]
fn bez_argumentov_ukaze_pouzitie() {
    let o = run(&[]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("použitie"));
}

#[test]
fn every_spracuje_len_kazdu_n_tu_snimku() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let d = TempDir::new("every");
    let input = d.path("in.mp4");
    make_video(&input); // 60 snímok
    let out = d.path("o");
    for (every, expected) in [("1", 60), ("2", 30), ("3", 20), ("4", 15)] {
        let name = format!("e{every}");
        let o = run(&[&input, "--width", "160", "--every", every, "--model", &model, "--out-dir", &out, "--name", &name]);
        assert!(o.status.success(), "every={every}: {}", String::from_utf8_lossy(&o.stderr));
        let (_, _, n) = size_and_frames(&format!("{out}/{name}_counted.mp4"));
        assert_eq!(n, expected, "--every {every}: výstup má mať {expected} snímok");
        let summary = fs::read_to_string(format!("{out}/{name}_summary.txt")).unwrap();
        assert!(summary.contains(&format!("Snímok: {expected}")), "{summary}");
    }
}

#[test]
fn zla_hodnota_every_zlyha() {
    let o = run(&["x.mp4", "--every", "0"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--every"));
}
