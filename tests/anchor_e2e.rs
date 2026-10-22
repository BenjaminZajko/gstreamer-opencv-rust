//! E2E testy ukotvenia čiary k scéne: spúšťajú `object_detect_live --line ...` na videu, v ktorom sa
//! "kamera" hýbe, a z výstupného RTSP streamu zisťujú, kde je žltá čiara.

use std::{
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread::sleep,
    time::{Duration, Instant},
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
const FRAMES: i32 = 60;
const PAN: i32 = 120; // o koľko pixelov sa kamera posunie

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: u64) -> i32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n) as i32
    }
}

/// Veľké plátno s množstvom tvarov (aby mala scéna dosť výrazných bodov).
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

/// Zapíše video z postupnosti snímok. Vráti `false`, ak OpenCV na tomto stroji video zapisovať nevie.
fn write_video(path: &Path, frames: &[Mat]) -> bool {
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

/// Video, kde sa kamera plynulo posúva doprava o `PAN` px (scéna sa v obraze posúva doľava).
fn panning_video(path: &Path) -> bool {
    let big = canvas(11, W + PAN + 20, H + 40);
    let frames: Vec<Mat> = (0..FRAMES)
        .map(|i| {
            let off = i * PAN / (FRAMES - 1);
            to_bgr(&Mat::roi(&big, Rect::new(off, 20, W, H)).unwrap().try_clone().unwrap())
        })
        .collect();
    write_video(path, &frames)
}

/// Video, kde sa po polovici scéna úplne zmení (kamera sa otočila inam).
fn scene_change_video(path: &Path) -> bool {
    let (a, b) = (canvas(21, W, H), canvas(22, W, H));
    let frames: Vec<Mat> = (0..FRAMES).map(|i| to_bgr(if i < FRAMES / 2 { &a } else { &b })).collect();
    write_video(path, &frames)
}

/// Servery s neurónovou sieťou sú náročné na CPU, preto tieto testy bežia po jednom.
static HEAVY: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn heavy() -> std::sync::MutexGuard<'static, ()> {
    HEAVY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Širokouhlé sudovité skreslenie objektívu (ako má skutočná kamera).
fn barrel(img: &Mat, k: f64) -> Mat {
    let (w, h) = (img.cols(), img.rows());
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let mut mx = Mat::new_rows_cols_with_default(h, w, opencv::core::CV_32FC1, Scalar::all(0.0)).unwrap();
    let mut my = Mat::new_rows_cols_with_default(h, w, opencv::core::CV_32FC1, Scalar::all(0.0)).unwrap();
    for y in 0..h {
        for x in 0..w {
            let (nx, ny) = ((x as f64 - cx) / cx, (y as f64 - cy) / cx);
            let f = 1.0 + k * (nx * nx + ny * ny);
            *mx.at_2d_mut::<f32>(y, x).unwrap() = (cx + nx * f * cx) as f32;
            *my.at_2d_mut::<f32>(y, x).unwrap() = (cy + ny * f * cx) as f32;
        }
    }
    let mut out = Mat::default();
    imgproc::remap(
        img, &mut out, &mx, &my, imgproc::INTER_LINEAR, opencv::core::BORDER_REPLICATE, Scalar::default(),
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .unwrap();
    out
}

/// Kamera so skreslením objektívu: 2 s stojí, potom sa posunie o 90 px a 12 s stojí na novom mieste.
fn moved_and_held_video(path: &Path) -> bool {
    let big = canvas(33, W + 130, H + 40);
    let frame = |off: i32| to_bgr(&barrel(&Mat::roi(&big, Rect::new(off, 20, W, H)).unwrap().try_clone().unwrap(), 0.12));
    let (a, b) = (frame(0), frame(90));
    let frames: Vec<Mat> = (0..280).map(|i| if i < 40 { a.try_clone().unwrap() } else { b.try_clone().unwrap() }).collect();
    write_video(path, &frames)
}

fn model() -> Option<String> {
    let p = format!("{}/models/yolox_s.onnx", env!("CARGO_MANIFEST_DIR"));
    Path::new(&p).exists().then_some(p)
}

struct Dir(PathBuf);
impl Dir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("anchor_e2e_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        Dir(p)
    }
    fn file(&self, f: &str) -> PathBuf {
        self.0.join(f)
    }
    fn s(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Server {
    child: Child,
    port: u16,
}
impl Server {
    fn start(video: &Path, model: &str, out_dir: &str, extra: &[&str]) -> Self {
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let child = Command::new(env!("CARGO_BIN_EXE_object_detect_live"))
            .args(["--input", &format!("file:{}", video.display())])
            .args(["--line", "0.5", "--width", "640", "--detect-width", "640", "--model", model])
            .args(["--out-dir", out_dir, "--port", &port.to_string()])
            .args(extra)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut s = Server { child, port };
        let t = Instant::now();
        while TcpStream::connect(("127.0.0.1", port)).is_err() {
            assert!(s.child.try_wait().unwrap().is_none(), "server skončil predčasne");
            assert!(t.elapsed() < Duration::from_secs(30), "server sa nerozbehol");
            sleep(Duration::from_millis(200));
        }
        s
    }
    fn url(&self) -> String {
        format!("rtsp://127.0.0.1:{}/detect", self.port)
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Jedna prijatá snímka: stred žltej čiary (priemerné x) a počet červených bodov v päte panela.
struct Seen {
    line_x: Option<f64>,
    red_footer: usize,
}

fn watch(url: &str, want: usize) -> Vec<Seen> {
    gst::init().unwrap();
    let p = gst::parse::launch(&format!(
        "rtspsrc location={url} protocols=tcp latency=200 ! decodebin ! videoconvert \
         ! video/x-raw,format=BGR ! appsink name=sink sync=false"
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();
    let sink = p.by_name("sink").unwrap().downcast::<gst_app::AppSink>().unwrap();
    p.set_state(gst::State::Playing).unwrap();
    let (mut out, start) = (vec![], Instant::now());
    while out.len() < want && start.elapsed() < Duration::from_secs(60) {
        let Some(s) = sink.try_pull_sample(gst::ClockTime::from_seconds(2)) else { continue };
        let info = gst_video::VideoInfo::from_caps(s.caps().unwrap()).unwrap();
        let map = s.buffer().unwrap().map_readable().unwrap();
        let stride = info.stride()[0] as usize;
        let (mut sum, mut n, mut red) = (0.0, 0usize, 0usize);
        for y in 0..info.height() as usize {
            for x in 0..info.width() as usize {
                let px = &map.as_slice()[y * stride + x * 3..y * stride + x * 3 + 3];
                if px[0] < 100 && px[1] > 190 && px[2] > 190 {
                    sum += x as f64;
                    n += 1;
                }
                if y >= 66 && y < 96 && x < 430 && px[2] > 150 && px[0] < 80 && px[1] < 80 {
                    red += 1;
                }
            }
        }
        out.push(Seen { line_x: (n > 50).then(|| sum / n as f64), red_footer: red });
    }
    p.set_state(gst::State::Null).unwrap();
    out
}

fn xs(v: &[Seen]) -> Vec<f64> {
    v.iter().filter_map(|s| s.line_x).collect()
}

#[test]
fn cara_sa_posuva_spolu_so_scenou_ked_sa_kamera_hybe() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let _heavy = heavy();
    let d = Dir::new("pan");
    if !panning_video(&d.file("pan.mp4")) {
        return eprintln!("OpenCV nevie zapisovať video – test preskočený");
    }
    let s = Server::start(&d.file("pan.mp4"), &model, &d.s(), &[]);
    let seen = xs(&watch(&s.url(), 120));
    assert!(seen.len() > 60, "čiara bola viditeľná len v {} snímkach", seen.len());

    let (min, max) = (seen.iter().cloned().fold(f64::INFINITY, f64::min), seen.iter().cloned().fold(0.0, f64::max));
    // kamera sa posunula doprava o 120 px, čiara (ukotvená k scéne) sa v obraze posunie doľava
    assert!(max - min > 70.0, "čiara sa mala posunúť o ~{PAN} px, posun: {:.0} px (x od {min:.0} do {max:.0})", max - min);
    assert!(max < 330.0 && min > 320.0 - PAN as f64 - 15.0, "čiara mala ísť len doľava od stredu: {min:.0}..{max:.0}");
}

#[test]
fn bez_ukotvenia_ostava_cara_na_mieste() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let _heavy = heavy();
    let d = Dir::new("noanchor");
    if !panning_video(&d.file("pan.mp4")) {
        return eprintln!("OpenCV nevie zapisovať video – test preskočený");
    }
    let s = Server::start(&d.file("pan.mp4"), &model, &d.s(), &["--anchor", "off"]);
    let seen = xs(&watch(&s.url(), 100));
    assert!(seen.len() > 50);
    let (min, max) = (seen.iter().cloned().fold(f64::INFINITY, f64::min), seen.iter().cloned().fold(0.0, f64::max));
    assert!(max - min < 3.0, "bez ukotvenia sa čiara nemá hýbať: {min:.1}..{max:.1}");
}

#[test]
fn zmena_sceny_ukaze_cervene_varovanie_a_potom_sa_obnovi() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let _heavy = heavy();
    let d = Dir::new("lost");
    if !scene_change_video(&d.file("change.mp4")) {
        return eprintln!("OpenCV nevie zapisovať video – test preskočený");
    }
    let s = Server::start(&d.file("change.mp4"), &model, &d.s(), &[]);
    let seen = watch(&s.url(), 150);
    let warn = seen.iter().filter(|f| f.red_footer > 50).count();
    let ok = seen.iter().filter(|f| f.red_footer == 0).count();
    assert!(warn > 5, "po zmene scény sa malo objaviť červené varovanie (snímok: {warn})");
    assert!(ok > 5, "pri pôvodnej scéne varovanie byť nemá (snímok: {ok})");
}

#[test]
fn ukotvenie_na_scene_bez_detailov_sa_vypne_ale_server_bezi() {
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let _heavy = heavy();
    let d = Dir::new("blank");
    // jednofarebné video: nie sú v ňom body, ktoré by sa dali sledovať
    let blank = to_bgr(&Mat::new_rows_cols_with_default(H, W, CV_8UC1, Scalar::all(90.0)).unwrap());
    if !write_video(&d.file("blank.mp4"), &vec![blank; 30]) {
        return eprintln!("OpenCV nevie zapisovať video – test preskočený");
    }
    let mut s = Server::start(&d.file("blank.mp4"), &model, &d.s(), &[]);
    // vysiela sa ďalej, čiara je zvislá a nemení sa
    let seen = xs(&watch(&s.url(), 40));
    assert!(seen.len() > 20);
    assert!(seen.iter().all(|x| (x - 320.0).abs() < 3.0), "{seen:?}");
    assert!(s.child.try_wait().unwrap().is_none());
}

/// Načíta záznam polohy čiary (`--anchor-log`): riadky (čas, x čiary hore, počet záberov).
fn read_anchor_log(dir: &Path) -> Vec<(f64, f64, u32)> {
    let text = std::fs::read_to_string(dir.join("anchor_log.csv")).unwrap_or_default();
    text.lines()
        .skip(1)
        .filter_map(|l| {
            let c: Vec<&str> = l.split(',').collect();
            Some((c.get(1)?.parse().ok()?, c.get(5)?.parse().ok()?, c.get(11)?.parse().ok()?))
        })
        .collect()
}

#[test]
fn po_posune_kamery_a_ustaleni_cara_nepreblikava() {
    // Presne scenár z praxe: kamera sa pootočí, ostane stáť na novom mieste; čiara tam musí byť pokojná.
    let Some(model) = model() else { return eprintln!("model chýba – test preskočený") };
    let _heavy = heavy();
    let d = Dir::new("heldmove");
    if !moved_and_held_video(&d.file("held.mp4")) {
        return eprintln!("OpenCV nevie zapisovať video – test preskočený");
    }
    let mut s = Server::start(&d.file("held.mp4"), &model, &d.s(), &["--anchor-log"]);
    // video trvá 14 s; počkáme, kým sa nastaví nový záber a poloha ustáli (a kým sa video nezačne opakovať)
    sleep(Duration::from_secs(13));
    assert!(s.child.try_wait().unwrap().is_none());
    let rows = read_anchor_log(&d.0);
    drop(s);

    let settled: Vec<_> = rows.iter().filter(|r| r.0 >= 9.0 && r.0 <= 13.0).collect();
    assert!(settled.len() >= 8, "v ustálenom úseku je len {} záznamov", settled.len());
    let (min, max) = settled.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), r| (a.min(r.1), b.max(r.1)));
    assert!(max - min < 0.01, "čiara po ustálení kolíše: x od {min:.2} do {max:.2} (ustálených záznamov {})", settled.len());
    assert!(settled.iter().all(|r| r.2 >= 2), "nový referenčný záber sa mal vytvoriť");
    // čiara sa naozaj posunula so scénou doľava (kamera sa posunula doprava o 90 px)
    let first = rows.iter().find(|r| r.0 < 2.5).map(|r| r.1).unwrap();
    assert!(first - settled[0].1 > 60.0, "čiara sa mala posunúť o ~90 px: {first:.1} -> {:.1}", settled[0].1);
}
