//! Detekcia objektov: GStreamer dekóduje video, OpenCV detekuje, GStreamer zakóduje výsledok.
//!
//!   cargo run --release -- <vstup.mp4> [--mode motion|people|objects] [--output vystup.mp4] [--width 1280]
//!
//! Režimy:
//!   motion  pohybujúce sa objekty (MOG2 odčítanie pozadia + kontúry) – dobré pre statickú kameru
//!   people  len ľudia (neurónová sieť YOLOX cez OpenCV DNN, model models/yolox_s.onnx)
//!   objects všetkých 80 tried COCO (YOLOX): ľudia, zvieratá, vozidlá, predmety...
//!
//! Tok dát (každá snímka):
//!   súbor -> [GStreamer: dekódovanie + zmenšenie] -> appsink -> Mat (OpenCV)
//!         -> detekcia + kreslenie rámčekov -> appsrc -> [GStreamer: HW enkodér] -> MP4
//!
//! Moduly sú v knižnici (lib.rs).

use std::time::Instant;

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;

use object_detect::{
    args::parse_args,
    detector::Detector,
    draw, video_io, Result,
};

fn main() -> Result<()> {
    let args = parse_args()?;
    gst::init()?; // inicializácia GStreameru (nutné pred akýmkoľvek použitím)

    let (input, sink) = video_io::open_input(&args.input, args.width)?;
    let mut detector = Detector::new(args.mode, &args.model, args.conf)?;

    // Výstup sa vytvorí lenivo pri prvej snímke (pozri video_io::make_output)
    let mut output: Option<(gst::Pipeline, gst_app::AppSrc)> = None;
    let (mut frames, mut total_boxes) = (0u64, 0u64); // štatistika
    let start = Instant::now();

    // Hlavná slučka. pull_sample vráti Err na konci streamu (EOS), vtedy cyklus skončí.
    while let Ok(sample) = sink.pull_sample() {
        let buffer = sample.buffer().ok_or("snímka bez buffera")?;
        let info = gst_video::VideoInfo::from_caps(sample.caps().ok_or("snímka bez caps")?)?;

        // GStreamer buffer -> OpenCV Mat
        let mut frame = video_io::buffer_to_mat(buffer, &info)?;

        // Detekcia a kreslenie
        let boxes = detector.detect(&frame)?;
        total_boxes += boxes.len() as u64;
        draw::draw(&mut frame, &boxes, frames)?;

        // OpenCV Mat -> GStreamer buffer -> enkodér
        // pri prvej snímke vytvor výstup, potom ho už len používaj
        let (_, src) = match &mut output {
            Some(o) => o,
            None => output.insert(video_io::make_output(&args.output, &info)?),
        };
        src.push_buffer(video_io::mat_to_buffer(&frame, &info, buffer)?)?;

        frames += 1;
        if frames % 50 == 0 {
            // priebežný výpis každých 50 snímok
            println!("{frames} snímok, {:.1} fps", frames as f64 / start.elapsed().as_secs_f64());
        }
    }

    if let Some((pipeline, src)) = output {
        video_io::finish_output(&pipeline, &src)?;
    }
    input.set_state(gst::State::Null)?;

    let secs = start.elapsed().as_secs_f64();
    println!(
        "Hotovo: {frames} snímok za {secs:.1} s ({:.1} fps), {total_boxes} detekcií celkom -> {}",
        frames as f64 / secs,
        args.output
    );
    Ok(())
}
