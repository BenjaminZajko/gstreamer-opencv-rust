//! Živá detekcia: kamera (RTSP) -> detekcia (OpenCV/YOLOX) -> vlastný RTSP server.
//! Na výsledok sa pripojíš z VLC: rtsp://<ip-tohto-macu>:8554/detect
//!
//!   LIVE_INPUT=rtsp://user:heslo@192.168.0.100:554/stream1 \
//!     cargo run --release --bin object_detect_live -- [--mode people] [--port 8554] [--mount /detect]
//!
//! Nič sa neukladá. Detekcia beží len kým sa niekto pozerá (šetrí CPU).

use std::{thread::sleep, time::{Duration, Instant}};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_video as gst_video;

use object_detect::{
    args::parse_live_args,
    detector::Detector,
    draw,
    rtsp_out::{ensure_port_free, RtspOutput},
    video_io, Result,
};

fn main() -> Result<()> {
    let args = parse_live_args()?;
    ensure_port_free(args.port)?; // chyba hneď, nie až po pripojení ku kamere
    gst::init()?;

    let mut detector = Detector::new(args.mode, &args.model, args.conf)?;
    let mut output: Option<RtspOutput> = None; // vznikne pri prvej snímke (poznáme rozmery)
    let mut frames = 0u64;

    // Pri výpadku zdroja (kamera sa reštartovala, Wi-Fi...) sa po chvíli pripojíme znova.
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
                eprintln!("zdroj sa prerušil, skúsim znova o 3 s");
            }
            Err(e) => eprintln!("zdroj sa nepodarilo otvoriť ({e}), skúsim znova o 3 s"),
        }
        sleep(Duration::from_secs(3));
    }
}
