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
use opencv::{
    core::{Mat, Size},
    imgproc,
};

use object_detect::{
    anchor::{draw_points, rect_corners, render_reference, Anchor, AnchorConfig, AnchorMode},
    args::{parse_live_args, LiveArgs},
    counter::{CrossEvent, COUNTED_CLASSES},
    detector::Detector,
    draw,
    live_count::{detect_in_roi, roi_from_quad, roi_rect, CountingCore, Session},
    rtsp_out::{ensure_port_free, RtspOutput},
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
    fn record(&self, ev: &CrossEvent, core: &CountingCore) -> Result<()> {
        use object_detect::counter::Group;
        let c = core.counter();
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
        self.write_counts(core)
    }

    fn write_counts(&self, core: &CountingCore) -> Result<()> {
        let text = format!(
            "Počítanie od: {}\nPosledná zmena: {}\n\n{}",
            self.started,
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            core.counter().summary()
        );
        fs::write(&self.counts_path, text)?;
        Ok(())
    }
}

/// Režim počítania: celý záber vo výstupe, detekcia na výreze, stále zapnuté.
///
/// Čiara a výrez sú ukotvené k scéne: pri štarte sa uloží referenčný záber a v každej ďalšej snímke
/// sa hľadajú tie isté body (modul `anchor`). Keď sa kamera pootočí, čiara aj výrez sa posunú s ňou.
fn run_counting(args: &LiveArgs, line: f64) -> Result<()> {
    let mut detector = Detector::yolo_with_classes(&args.model, args.conf, &COUNTED_CLASSES)?;
    let log = CountLog::new(&args.out_dir)?;

    // všetko, čo závisí od rozmerov snímky, sa pripraví pri prvej snímke
    struct Setup {
        core: CountingCore,
        output: RtspOutput,
        info_out: gst_video::VideoInfo,
        scale: f64,
        out_size: Size,
        /// výrez na detekciu vo veľkej snímke (keď nie je ukotvenie)
        roi_static: opencv::core::Rect,
        /// výrez a čiara v súradniciach referencie (= výstupný záber pri štarte)
        roi_ref_quad: [(f64, f64); 4],
        line_ref: ((f64, f64), (f64, f64)),
        anchor: Option<Anchor>,
        was_tracking: bool,
        /// referenčný záber (farebný) na uloženie referenčnej fotky po kalibrácii
        ref_color: Option<Mat>,
        prev_mode: AnchorMode,
        prev_keys: usize,
    }
    let mut setup: Option<Setup> = None;
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
    let (mut frames, start) = (0u64, Instant::now());

    loop {
        match video_io::open_live_input(&args.input, args.codec, args.detect_width) {
            Ok((pipeline, sink)) => {
                eprintln!("zdroj pripojený");
                let (mut window_start, mut window_frames) = (Instant::now(), 0u32);

                while let Ok(sample) = sink.pull_sample() {
                    let buffer = sample.buffer().ok_or("snímka bez buffera")?;
                    let info = gst_video::VideoInfo::from_caps(sample.caps().ok_or("snímka bez caps")?)?;
                    let frame_hi = video_io::buffer_to_mat(buffer, &info)?;
                    let (hi_w, hi_h) = (info.width() as i32, info.height() as i32);

                    if setup.is_none() {
                        let out_w = args.width.min(hi_w);
                        let out_h = ((hi_h as f64 * out_w as f64 / hi_w as f64) as i32 / 2) * 2;
                        let size = Size::new(out_w, out_h);
                        let scale = out_w as f64 / hi_w as f64;
                        let info_out = gst_video::VideoInfo::builder(gst_video::VideoFormat::Bgr, out_w as u32, out_h as u32).build()?;
                        let (y0, y1) = args.line_span;
                        let roi_static = roi_rect(&args.roi, hi_w, hi_h);

                        // referenčný záber pre ukotvenie (výstupné rozlíšenie, šedotónový)
                        let mut ref_color = None;
                        let anchor = if args.anchor {
                            let mut small = Mat::default();
                            imgproc::resize(&frame_hi, &mut small, size, 0.0, 0.0, imgproc::INTER_AREA)?;
                            let mut gray = Mat::default();
                            imgproc::cvt_color_def(&small, &mut gray, imgproc::COLOR_BGR2GRAY)?;
                            let cfg = AnchorConfig {
                                features: args.anchor_features,
                                min_inliers: args.anchor_min_inliers,
                                points: args.anchor_points,
                                calib_frames: args.anchor_calib,
                                deadband: args.anchor_deadband,
                            };
                            match Anchor::new(&gray, cfg) {
                                Ok(a) => {
                                    ref_color = Some(small);
                                    Some(a)
                                }
                                Err(e) => {
                                    eprintln!("UPOZORNENIE: ukotvenie je vypnuté: {e}");
                                    None
                                }
                            }
                        } else {
                            None
                        };

                        let r = roi_static;
                        let roi_ref_quad = rect_corners(
                            r.x as f64 * scale,
                            r.y as f64 * scale,
                            r.width as f64 * scale,
                            r.height as f64 * scale,
                        );
                        let line_x = line * out_w as f64;
                        let s = Setup {
                            core: CountingCore::new(line_x, (y0 * out_h as f64, y1 * out_h as f64), args.margin, args.max_missed),
                            output: RtspOutput::start(args.port, &args.mount, &info_out, args.fps, args.bitrate)?,
                            info_out,
                            scale,
                            out_size: size,
                            roi_static,
                            roi_ref_quad,
                            line_ref: ((line_x, y0 * out_h as f64), (line_x, y1 * out_h as f64)),
                            prev_mode: anchor.as_ref().map(|a| a.mode()).unwrap_or(AnchorMode::OrbOnly),
                            prev_keys: 0,
                            anchor,
                            was_tracking: true,
                            ref_color,
                        };
                        println!(
                            "RTSP server beží: rtsp://<ip-tohto-macu>:{}{}  (záber {}x{}, detekcia na {}x{} px, čiara pri {:.0} % šírky, ukotvenie {})",
                            args.port, args.mount, out_w, out_h, roi_static.width, roi_static.height, line * 100.0,
                            if s.anchor.is_some() { "zapnuté" } else { "vypnuté" }
                        );
                        println!("Počítanie beží stále; záznam prechodov: {}", log.csv_path);
                        log.write_counts(&s.core)?;
                        setup = Some(s);
                    }
                    let s = setup.as_mut().unwrap();

                    // 0) ukotvenie: kde je kamera voči referencii (každú N-tú snímku)
                    let mut out_cache: Option<Mat> = None;
                    if let Some(a) = s.anchor.as_mut() {
                        if frames % args.anchor_every as u64 == 0 {
                            let mut out = Mat::default();
                            imgproc::resize(&frame_hi, &mut out, s.out_size, 0.0, 0.0, imgproc::INTER_AREA)?;
                            let mut gray = Mat::default();
                            imgproc::cvt_color_def(&out, &mut gray, imgproc::COLOR_BGR2GRAY)?;
                            let ok = a.update(&gray)?;
                            if ok != s.was_tracking {
                                if ok {
                                    println!("ukotvenie obnovené ({} bodov), počítanie pokračuje", a.inliers());
                                } else {
                                    println!("UKOTVENIE STRATENÉ (kamera sa otočila inam?), počítanie je pozastavené");
                                }
                                s.was_tracking = ok;
                            }
                            out_cache = Some(out);

                            if let Some(f) = anchor_log.as_mut() {
                                let (t, bt) = (a.to_current(s.line_ref.0), a.to_current(s.line_ref.1));
                                // podpis zobrazených bodov: ak sa nemení, body sa nehýbu
                                let shown = a.points(40);
                                let sig: f64 = shown.iter().map(|p| (p.x + p.y) as f64).sum();
                                writeln!(
                                    f,
                                    "{frames},{:.2},{:?},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{:.3},{},{}",
                                    start.elapsed().as_secs_f64(), a.mode(), a.tracking(), a.inliers(), t.0, t.1, bt.0, bt.1, sig, shown.len(),
                                    a.keyframes(), a.active_key(), a.in_fallback(), a.motion(), a.camera_static(), a.frozen()
                                )?;
                            }

                            // po skončení kalibrácie: uložíme referenčnú fotku s vybranými bodmi
                            if a.mode() != s.prev_mode {
                                if a.mode() == AnchorMode::Fine {
                                    if let Some(rc) = &s.ref_color {
                                        let img = render_reference(rc, a.chosen_points(), s.line_ref, &s.roi_ref_quad)?;
                                        let path = format!("{}/anchor_reference.jpg", args.out_dir);
                                        opencv::imgcodecs::imwrite(&path, &img, &opencv::core::Vector::new())?;
                                        println!("ukotvenie: vybraných {} stabilných bodov, referenčná fotka: {path}", a.chosen_points().len());
                                    }
                                } else if a.mode() == AnchorMode::OrbOnly {
                                    println!("UPOZORNENIE: nenašlo sa dosť stabilných bodov na jemné sledovanie, používa sa len ORB (čiara môže mierne kmitať)");
                                }
                                s.prev_mode = a.mode();
                            }
                            // kamera sa ustálila na novom mieste: nový referenčný záber s novými stabilnými bodmi
                            if a.keyframes() > s.prev_keys {
                                if s.prev_keys > 0 {
                                    println!("ukotvenie: kamera sa ustálila na novom mieste, nastavený nový referenčný záber (spolu {})", a.keyframes());
                                }
                                s.prev_keys = a.keyframes();
                            }
                        }
                    }
                    let tracking = s.anchor.as_ref().map(|a| a.tracking()).unwrap_or(true);

                    // výrez v aktuálnej snímke (štvoruholník, ak sa kamera pootočila)
                    let quad_cur = match &s.anchor {
                        Some(a) => s.roi_ref_quad.map(|p| a.to_current(p)),
                        None => s.roi_ref_quad,
                    };
                    let roi_hi = if s.anchor.is_some() {
                        roi_from_quad(&quad_cur, s.scale, hi_w, hi_h)
                    } else {
                        s.roi_static
                    };

                    // 1) detekcia na výreze + počítanie (vždy, aj keď sa nik nepozerá; ak je poloha
                    //    kamery stratená, počítanie stojí, aby sa nezrátalo nič nesprávne)
                    let time_s = start.elapsed().as_secs_f64();
                    let (items, events) = if tracking {
                        let dets = detect_in_roi(&mut detector, &frame_hi, roi_hi, s.scale)?;
                        match &s.anchor {
                            Some(a) => s.core.track_and_count_with(&dets, frames, time_s, |r| a.rect_to_reference(r)),
                            None => s.core.track_and_count(&dets, frames, time_s),
                        }
                    } else {
                        (vec![], vec![])
                    };
                    for ev in &events {
                        log.record(ev, &s.core)?;
                        println!("{} {}  (autá {}, ľudia {})", ev.group.name(), ev.direction.name(),
                            s.core.counter().total(object_detect::counter::Group::Vehicle),
                            s.core.counter().total(object_detect::counter::Group::Person));
                    }

                    // 2) kreslenie a vysielanie len ak sa niekto pozerá
                    let watching = s.output.has_client();
                    if !watching {
                        session = None; // všetci odišli: ďalší divák začne od nuly
                    } else if session.is_none() {
                        session = Some(Session::start(s.core.counter()));
                    }
                    if let (true, Some(sess)) = (watching, session.as_mut()) {
                        let items = sess.relabel(items);
                        let mut out = match out_cache.take() {
                            Some(o) => o,
                            None => {
                                let mut o = Mat::default();
                                imgproc::resize(&frame_hi, &mut o, s.out_size, 0.0, 0.0, imgproc::INTER_AREA)?;
                                o
                            }
                        };
                        let mut view = sess.view(s.core.counter());
                        match &s.anchor {
                            Some(a) => {
                                // čiara sa presunie spolu so scénou (môže byť naklonená)
                                view.line_pts = Some((a.to_current(s.line_ref.0), a.to_current(s.line_ref.1)));
                                if a.tracking() {
                                    view.footer = format!("{} | REF {}", view.footer, a.status_text());
                                } else {
                                    view.footer = format!("{} | REF STRATENE", view.footer);
                                    view.warning = true;
                                }
                                draw::draw_quad(&mut out, &quad_cur)?;
                                draw_points(&mut out, &a.points(40))?;
                            }
                            None => {
                                let q = s.roi_ref_quad;
                                draw::draw_quad(&mut out, &q)?;
                            }
                        }
                        draw::draw_counting(&mut out, &items, &view)?;
                        s.output.push(video_io::mat_to_untimed_buffer(&out, &s.info_out)?)?;
                    }

                    frames += 1;
                    window_frames += 1;
                    if window_start.elapsed() >= Duration::from_secs(30) {
                        let fps = window_frames as f64 / window_start.elapsed().as_secs_f64();
                        let c = s.core.counter();
                        println!(
                            "{fps:.1} fps | autá {} | ľudia {}",
                            c.total(object_detect::counter::Group::Vehicle),
                            c.total(object_detect::counter::Group::Person)
                        );
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
