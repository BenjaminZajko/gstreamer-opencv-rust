//! Živá detekcia: kamera (RTSP) -> detekcia (OpenCV/YOLOX) -> vlastný RTSP server.
//! Na výsledok sa pripojíš z VLC: rtsp://<ip-tohto-macu>:8554/detect
//!
//!   LIVE_INPUT=rtsp://user:heslo@192.168.0.100:554/stream1 \
//!     cargo run --release --bin object_detect_live -- [--mode people] [--port 8554] [--mount /detect]
//!
//! Dva režimy:
//!   * bez `--line`: detekcia s rámčekmi (people/objects/motion). Beží len kým sa niekto pozerá.
//!   * s `--line`: počítanie áut a ľudí, ktorí prejdú cez zvislú čiaru. Beží stále (aj keď sa nik
//!     nepozerá), vo videu je celý záber; detekcia sa robí na výreze `--roi`. Prechody sa ukladajú
//!     do `--out-dir` (live_<čas>_events.csv, live_counts.txt).
//!
//! Zdroj môže byť rtsp://... (kamera), file:/cesta/video.mp4 (súbor v reálnom tempe) alebo test:ball.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    thread::sleep,
    time::{Duration, Instant},
};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;
use object_detect::{
    anchor::AnchorConfig,
    args::{parse_live_args, LiveArgs},
    counter::{CrossEvent, Group, LineCounter, COUNTED_CLASSES},
    detector::Detector,
    draw,
    live_count::Session,
    rtsp_out::{ensure_port_free, RtspOutput},
    scene::{SceneConfig, SceneCounter},
    video_io, Result,
};

fn main() -> Result<()> {
    let args = parse_live_args()?;
    ensure_port_free(args.port)?; // chyba hneď, nie až po pripojení ku kamere
    gst::init()?;
    match args.line {
        Some(line) => run_counting(&args, line),
        None => run_detection(&args),
    }
}

/// Pri výpadku zdroja (kamera sa reštartovala, Wi-Fi...) sa po chvíli pripojíme znova.
fn retry_pause() {
    eprintln!("skúsim znova o 3 s");
    sleep(Duration::from_secs(3));
}

/// Režim bez počítania: rámčeky okolo detekcií, len kým sa niekto pozerá.
fn run_detection(args: &LiveArgs) -> Result<()> {
    let mut detector = Detector::new(args.mode, &args.model, args.conf)?;
    let mut output: Option<RtspOutput> = None; // vznikne pri prvej snímke (poznáme rozmery)
    let mut frames = 0u64;

    loop {
        match video_io::open_live_input(&args.input, args.codec, args.width) {
            Ok((pipeline, sink)) => {
                eprintln!("zdroj pripojený");
                let (mut window_start, mut window_frames) = (Instant::now(), 0u32);

                // pull_sample vráti Err, keď stream skončí alebo zlyhá
                while let Ok(sample) = sink.pull_sample() {
                    let buffer = sample.buffer().ok_or("snímka bez buffera")?;
                    let info = gst_video::VideoInfo::from_caps(sample.caps().ok_or("snímka bez caps")?)?;

                    // Server sa spustí až teraz, lebo potrebuje poznať rozmery snímok.
                    if output.is_none() {
                        output = Some(RtspOutput::start(args.port, &args.mount, &info, args.fps, args.bitrate)?);
                        println!(
                            "RTSP server beží: rtsp://<ip-tohto-macu>:{}{}  ({}x{}, režim {:?})",
                            args.port, args.mount, info.width(), info.height(), args.mode
                        );
                    }
                    let out = output.as_ref().unwrap();

                    // Nik sa nepozerá -> detekciu preskočíme (šetrí CPU), snímky len zahadzujeme.
                    if !out.has_client() {
                        continue;
                    }

                    let mut frame = video_io::buffer_to_mat(buffer, &info)?;
                    let dets = detector.detect(&frame)?;
                    draw::draw(&mut frame, &dets, frames)?;
                    out.push(video_io::mat_to_untimed_buffer(&frame, &info)?)?;

                    frames += 1;
                    window_frames += 1;
                    if window_start.elapsed() >= Duration::from_secs(10) {
                        let fps = window_frames as f64 / window_start.elapsed().as_secs_f64();
                        println!("{fps:.1} fps, {} detekcií na poslednej snímke", dets.len());
                        (window_start, window_frames) = (Instant::now(), 0);
                    }
                }
                let _ = pipeline.set_state(gst::State::Null);
                eprintln!("zdroj sa prerušil");
            }
            Err(e) => eprintln!("zdroj sa nepodarilo otvoriť ({e})"),
        }
        retry_pause();
    }
}

/// Zápis prechodov: CSV (každý prechod) a textový súhrn (aktuálne počty).
struct CountLog {
    csv_path: String,
    counts_path: String,
    started: String,
}

impl CountLog {
    fn new(out_dir: &str) -> Result<Self> {
        fs::create_dir_all(out_dir)?;
        let now = chrono::Local::now();
        let log = CountLog {
            csv_path: format!("{out_dir}/live_{}_events.csv", now.format("%Y%m%d_%H%M%S")),
            counts_path: format!("{out_dir}/live_counts.txt"),
            started: now.format("%Y-%m-%d %H:%M:%S").to_string(),
        };
        fs::write(&log.csv_path, "cas,track_id,skupina,smer,spolu_auta,spolu_ludia\n")?;
        Ok(log)
    }

    /// Pripíše prechod do CSV a prepíše súhrn.
    fn record(&self, ev: &CrossEvent, c: &LineCounter) -> Result<()> {
        let mut f = OpenOptions::new().append(true).open(&self.csv_path)?;
        writeln!(
            f,
            "{},{},{},{},{},{}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            ev.track,
            ev.group.name(),
            ev.direction.name(),
            c.total(Group::Vehicle),
            c.total(Group::Person)
        )?;
        self.write_counts(c)
    }

    fn write_counts(&self, c: &LineCounter) -> Result<()> {
        let text = format!(
            "Počítanie od: {}\nPosledná zmena: {}\n\n{}",
            self.started,
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            c.summary()
        );
        fs::write(&self.counts_path, text)?;
        Ok(())
    }
}

/// Režim počítania: celý záber vo výstupe, detekcia na výreze, stále zapnuté.
///
/// Čiara a výrez sú ukotvené k scéne (modul `anchor`, pozri `scene::SceneCounter`).
fn run_counting(args: &LiveArgs, line: f64) -> Result<()> {
    let log = CountLog::new(&args.out_dir)?;
    let cfg = scene_config(args, line);

    // všetko, čo závisí od rozmerov snímky, sa pripraví pri prvej snímke
    let mut scene: Option<(SceneCounter, RtspOutput)> = None;
    // voliteľný záznam polohy čiary (na meranie stability ukotvenia)
    let mut anchor_log: Option<std::fs::File> = if args.anchor_log {
        let mut f = std::fs::File::create(format!("{}/anchor_log.csv", args.out_dir))?;
        writeln!(f, "frame,cas_s,rezim,sledovanie,body,ciara_hore_x,ciara_hore_y,ciara_dole_x,ciara_dole_y,podpis_bodov,pocet_bodov,zabery,aktivny,zaloha,pohyb_px,kamera_stoji,zmrazene")?;
        Some(f)
    } else {
        None
    };
    // relácia diváka: panel a čísla objektov začínajú od nuly, keď sa pripojí divák
    let mut session: Option<Session> = None;
    let start = Instant::now();

    loop {
        match video_io::open_live_input(&args.input, args.codec, args.detect_width) {
            Ok((pipeline, sink)) => {
                eprintln!("zdroj pripojený");
                let (mut window_start, mut window_frames) = (Instant::now(), 0u32);

                while let Ok(sample) = sink.pull_sample() {
                    let buffer = sample.buffer().ok_or("snímka bez buffera")?;
                    let info = gst_video::VideoInfo::from_caps(sample.caps().ok_or("snímka bez caps")?)?;
                    let frame_hi = video_io::buffer_to_mat(buffer, &info)?;

                    if scene.is_none() {
                        let detector = Detector::yolo_with_classes(&args.model, args.conf, &COUNTED_CLASSES)?;
                        let (sc, notes) = SceneCounter::new(detector, &cfg, &frame_hi)?;
                        notes.iter().for_each(|n| eprintln!("{n}"));
                        let output = RtspOutput::start(args.port, &args.mount, sc.info_out(), args.fps, args.bitrate)?;
                        let (r, o) = (sc.roi_static(), sc.info_out());
                        println!(
                            "RTSP server beží: rtsp://<ip-tohto-macu>:{}{}  (záber {}x{}, detekcia na {}x{} px, čiara pri {:.0} % šírky, ukotvenie {})",
                            args.port, args.mount, o.width(), o.height(), r.width, r.height, line * 100.0,
                            if sc.anchor().is_some() { "zapnuté" } else { "vypnuté" }
                        );
                        println!("Počítanie beží stále; záznam prechodov: {}", log.csv_path);
                        log.write_counts(sc.counter())?;
                        scene = Some((sc, output));
                    }
                    let (sc, output) = scene.as_mut().unwrap();

                    // 0) + 1) ukotvenie, detekcia na výreze, počítanie (vždy, aj keď sa nik nepozerá)
                    let frame_idx = sc.frames();
                    let mut step = sc.step(&frame_hi, start.elapsed().as_secs_f64())?;
                    step.notes.iter().for_each(|n| println!("{n}"));
                    if let Some(photo) = &step.reference_photo {
                        let path = format!("{}/anchor_reference.jpg", args.out_dir);
                        opencv::imgcodecs::imwrite(&path, photo, &opencv::core::Vector::new())?;
                        println!("referenčná fotka: {path}");
                    }
                    if let (Some(f), true, Some(a)) = (anchor_log.as_mut(), step.anchor_updated, sc.anchor()) {
                        let line_ref = sc.line_ref();
                        let (t, bt) = (a.to_current(line_ref.0), a.to_current(line_ref.1));
                        // podpis zobrazených bodov: ak sa nemení, body sa nehýbu
                        let shown = a.points(40);
                        let sig: f64 = shown.iter().map(|p| (p.x + p.y) as f64).sum();
                        writeln!(
                            f,
                            "{frame_idx},{:.2},{:?},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{:.3},{},{}",
                            start.elapsed().as_secs_f64(), a.mode(), a.tracking(), a.inliers(), t.0, t.1, bt.0, bt.1, sig, shown.len(),
                            a.keyframes(), a.active_key(), a.in_fallback(), a.motion(), a.camera_static(), a.frozen()
                        )?;
                    }
                    for ev in &step.events {
                        log.record(ev, sc.counter())?;
                        println!("{} {}  (autá {}, ľudia {})", ev.group.name(), ev.direction.name(),
                            sc.counter().total(Group::Vehicle), sc.counter().total(Group::Person));
                    }

                    // 2) kreslenie a vysielanie len ak sa niekto pozerá
                    let watching = output.has_client();
                    if !watching {
                        session = None; // všetci odišli: ďalší divák začne od nuly
                    } else if session.is_none() {
                        session = Some(Session::start(sc.counter()));
                    }
                    if let (true, Some(sess)) = (watching, session.as_mut()) {
                        let items = sess.relabel(std::mem::take(&mut step.items));
                        let view = sess.view(sc.counter());
                        let out = sc.render(&frame_hi, &mut step, &items, view)?;
                        output.push(video_io::mat_to_untimed_buffer(&out, sc.info_out())?)?;
                    }

                    window_frames += 1;
                    if window_start.elapsed() >= Duration::from_secs(30) {
                        let fps = window_frames as f64 / window_start.elapsed().as_secs_f64();
                        let c = sc.counter();
                        println!("{fps:.1} fps | autá {} | ľudia {}", c.total(Group::Vehicle), c.total(Group::Person));
                        (window_start, window_frames) = (Instant::now(), 0);
                    }
                }
                let _ = pipeline.set_state(gst::State::Null);
                eprintln!("zdroj sa prerušil");
            }
            Err(e) => eprintln!("zdroj sa nepodarilo otvoriť ({e})"),
        }
        retry_pause();
    }
}

/// Nastavenie počítania z argumentov.
fn scene_config(args: &LiveArgs, line: f64) -> SceneConfig {
    SceneConfig {
        line,
        line_span: args.line_span,
        roi: args.roi,
        out_width: args.width,
        margin: args.margin,
        max_missed: args.max_missed,
        anchor: args.anchor.then_some(AnchorConfig {
            features: args.anchor_features,
            min_inliers: args.anchor_min_inliers,
            points: args.anchor_points,
            calib_frames: args.anchor_calib,
            deadband: args.anchor_deadband,
        }),
        anchor_every: args.anchor_every,
    }
}
