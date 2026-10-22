//! Spracovanie videa zo súboru rovnako ako živý server: celý záber, detekcia na výreze, čiara
//! ukotvená k scéne (keď sa v zázname zmení uhol kamery, čiara ostane na tom istom mieste ulice),
//! počítanie áut a ľudí cez čiaru. Výsledok sa uloží ako video.
//!
//!   video_count video.mp4 --line 0.235 --line-span 0.52,0.85 --roi 0,0.52,0.47,0.33
//!
//! Výsledky v --out-dir (predvolene out/): <názov>_counted.mp4, <názov>_events.csv,
//! <názov>_summary.txt, <názov>_reference.jpg (referenčné body).
//! Voľby sú rovnaké ako pri object_detect_live (--line, --line-span, --roi, --conf, --anchor ...),
//! navyše --output cesta/k/videu.mp4.

use std::{fs, path::Path, time::Instant};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;

use object_detect::{
    anchor::AnchorConfig,
    args::parse_live_args_from,
    counter::{Group, COUNTED_CLASSES},
    detector::Detector,
    scene::{SceneConfig, SceneCounter},
    video_io, Result,
};

fn main() -> Result<()> {
    // prvý argument bez "--" je vstupné video
    let mut raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().is_some_and(|a| !a.starts_with("--")) {
        raw.insert(0, "--input".into());
    }
    let args = parse_live_args_from(raw, None).map_err(|e| {
        format!("{e}\npoužitie: video_count <video.mp4> [--line 0.5] [--line-span od,do] [--roi x,y,w,h] [--anchor on|off] [--conf 0.4] [--out-dir out] [--output video.mp4]")
    })?;
    gst::init()?;

    let input = args.input.strip_prefix("file:").unwrap_or(&args.input).to_string();
    if !Path::new(&input).exists() {
        return Err(format!("video '{input}' neexistuje").into());
    }
    let name = Path::new(&input).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "video".into());
    fs::create_dir_all(&args.out_dir)?;
    let video_path = args.output.clone().unwrap_or_else(|| format!("{}/{name}_counted.mp4", args.out_dir));
    let line = args.line.unwrap_or(0.5);
    let cfg = SceneConfig {
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
    };

    // model sa načíta hneď (chyba skôr než sa začne dekódovať video)
    let mut detector = Some(Detector::yolo_with_classes(&args.model, args.conf, &COUNTED_CLASSES)?);
    let (input_pipe, sink) = video_io::open_input(&input, args.detect_width)?;
    let mut scene: Option<SceneCounter> = None;
    let mut output = None;
    // výstupný formát: rozmery zobrazeného záberu, snímková frekvencia z pôvodného videa
    let mut info_file: Option<gst_video::VideoInfo> = None;
    let start = Instant::now();

    while let Ok(sample) = sink.pull_sample() {
        let buffer = sample.buffer().ok_or("snímka bez buffera")?;
        let info = gst_video::VideoInfo::from_caps(sample.caps().ok_or("snímka bez caps")?)?;
        let frame_hi = video_io::buffer_to_mat(buffer, &info)?;

        if scene.is_none() {
            let (sc, notes) = SceneCounter::new(detector.take().unwrap(), &cfg, &frame_hi)?;
            notes.iter().for_each(|n| eprintln!("{n}"));
            println!(
                "video {}x{} -> výstup {}x{}, detekcia na {}x{} px, čiara pri {:.0} % šírky, ukotvenie {}",
                info.width(), info.height(), sc.info_out().width(), sc.info_out().height(),
                sc.roi_static().width, sc.roi_static().height, line * 100.0,
                if sc.anchor().is_some() { "zapnuté" } else { "vypnuté" }
            );
            let o = sc.info_out();
            let fi = gst_video::VideoInfo::builder(gst_video::VideoFormat::Bgr, o.width(), o.height()).fps(info.fps()).build()?;
            output = Some(video_io::make_output(&video_path, &fi)?);
            info_file = Some(fi);
            scene = Some(sc);
        }
        let sc = scene.as_mut().unwrap();

        let time_s = buffer.pts().map(|t| t.nseconds() as f64 / 1e9).unwrap_or(0.0);
        let mut step = sc.step(&frame_hi, time_s)?;
        for n in &step.notes {
            println!("[{time_s:6.1} s] {n}");
        }
        if let Some(photo) = &step.reference_photo {
            let path = format!("{}/{name}_reference.jpg", args.out_dir);
            opencv::imgcodecs::imwrite(&path, photo, &opencv::core::Vector::new())?;
        }
        for ev in &step.events {
            println!(
                "[{time_s:6.1} s] {} {}  (autá {}, ľudia {})",
                ev.group.name(), ev.direction.name(),
                sc.counter().total(Group::Vehicle), sc.counter().total(Group::Person)
            );
        }

        let items = std::mem::take(&mut step.items);
        let view = sc.counter().view().with_footer(format!("{time_s:.1} s"));
        let out = sc.render(&frame_hi, &mut step, &items, view)?;
        let (_, src) = output.as_ref().unwrap();
        src.push_buffer(video_io::mat_to_buffer(&out, info_file.as_ref().unwrap(), buffer)?)?;

        if sc.frames() % 100 == 0 {
            println!("{} snímok, {:.1} fps", sc.frames(), sc.frames() as f64 / start.elapsed().as_secs_f64());
        }
    }

    if let Some((pipeline, src)) = output {
        video_io::finish_output(&pipeline, &src)?;
    }
    input_pipe.set_state(gst::State::Null)?;

    let sc = scene.ok_or("video neobsahuje žiadne snímky")?;
    fs::write(format!("{}/{name}_events.csv", args.out_dir), sc.counter().events_csv())?;
    let summary = format!(
        "Video: {input}\nSnímok: {}\nČiara: {:.0} % šírky záberu, ukotvenie {}\n\n{}",
        sc.frames(),
        line * 100.0,
        if sc.anchor().is_some() { "zapnuté" } else { "vypnuté" },
        sc.counter().summary()
    );
    fs::write(format!("{}/{name}_summary.txt", args.out_dir), &summary)?;
    println!("\n{summary}");
    println!("Uložené: {video_path}, {}/{name}_events.csv, {}/{name}_summary.txt", args.out_dir, args.out_dir);
    Ok(())
}
