//! Počítanie prechodov cez zvislú čiaru: koľko áut a ľudí prešlo, zvlášť podľa smeru.
//!
//!   line_count street.mp4 --crop 0.0,0.45,0.5,0.35 --line 0.5
//!
//! 1. z 4K záberu sa orezá zadaná časť (cesta) – GStreamer `videocrop`, pred zmenšením,
//! 2. YOLOX nájde autá a ľudí,
//! 3. tracker každému objektu pridelí ID, aby sa jedno auto nepočítalo viackrát,
//! 4. počítadlo zaznamená prechod stredu objektu cez zvislú čiaru.
//!
//! Výsledky (v priečinku --out-dir): <názov>_counted.mp4 (video s čiarou a počtami),
//! <názov>_events.csv (každý prechod: čas, ID, skupina, smer), <názov>_summary.txt (súhrn).

use std::{fs, path::Path, time::Instant};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;

use object_detect::{
    args::parse_count_args,
    counter::{group_of, LineCounter, COUNTED_CLASSES},
    detector::Detector,
    draw::{self, Counted},
    tracker::Tracker,
    video_io, Result,
};

fn main() -> Result<()> {
    let args = parse_count_args()?;
    gst::init()?;

    // Rozmery videa -> pixely orezania
    let (src_w, src_h) = video_io::probe_size(&args.input)?;
    let crop = args.crop.to_pixels(src_w, src_h)?;
    let (crop_w, crop_h) = crop.size(src_w, src_h);
    println!("video {src_w}x{src_h}, orezanie na {crop_w}x{crop_h} px (vľavo {}, hore {})", crop.left, crop.top);

    let name = args.name.clone().unwrap_or_else(|| {
        Path::new(&args.input).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "video".into())
    });
    fs::create_dir_all(&args.out_dir)?;
    let video_path = format!("{}/{name}_counted.mp4", args.out_dir);

    let (input, sink) = video_io::open_input_cropped(&args.input, args.width, Some(crop))?;
    let mut detector = Detector::yolo_with_classes(&args.model, args.conf, &COUNTED_CLASSES)?;
    let mut tracker = Tracker::new(args.max_missed);
    let mut counter: Option<LineCounter> = None; // vznikne pri prvej snímke (poznáme šírku)
    let mut output: Option<(gst::Pipeline, gst_app::AppSrc)> = None;

    let (mut frames, mut seen, start) = (0u64, 0u64, Instant::now());
    while let Ok(sample) = sink.pull_sample() {
        // --every N: spracuje sa len každá N-tá snímka (ostatné sa zahodia, čas ostáva zachovaný)
        seen += 1;
        if (seen - 1) % args.every as u64 != 0 {
            continue;
        }
        let buffer = sample.buffer().ok_or("snímka bez buffera")?;
        let info = gst_video::VideoInfo::from_caps(sample.caps().ok_or("snímka bez caps")?)?;
        let mut frame = video_io::buffer_to_mat(buffer, &info)?;
        let time_s = buffer.pts().map(|t| t.nseconds() as f64 / 1e9).unwrap_or(frames as f64 / 12.0);

        let counter = counter.get_or_insert_with(|| {
            LineCounter::new(args.line * info.width() as f64, args.margin)
        });

        // detekcia -> len autá a ľudia -> sledovanie -> počítanie
        let dets: Vec<_> = detector
            .detect(&frame)?
            .into_iter()
            .filter_map(|d| group_of(&d.label).map(|g| (d, g)))
            .collect();
        let update = tracker.update(&dets.iter().map(|(d, g)| (d.rect, *g)).collect::<Vec<_>>());
        let mut items = Vec::with_capacity(dets.len());
        for ((d, group), id) in dets.into_iter().zip(update.ids) {
            let cx = d.rect.x as f64 + d.rect.width as f64 / 2.0;
            counter.observe(frames, time_s, id, group, cx);
            items.push(Counted { rect: d.rect, label: d.label, score: d.score, id, group });
        }
        for id in update.ended {
            counter.forget(id);
        }

        draw::draw_counting(&mut frame, &items, &counter.view().with_footer(format!("frame {frames}")))?;

        let (_, src) = match &mut output {
            Some(o) => o,
            None => output.insert(video_io::make_output(&video_path, &info)?),
        };
        src.push_buffer(video_io::mat_to_buffer(&frame, &info, buffer)?)?;

        frames += 1;
        if frames % 100 == 0 {
            println!("{frames} snímok, {:.1} fps", frames as f64 / start.elapsed().as_secs_f64());
        }
    }

    if let Some((pipeline, src)) = output {
        video_io::finish_output(&pipeline, &src)?;
    }
    input.set_state(gst::State::Null)?;

    let counter = counter.ok_or("video neobsahuje žiadne snímky")?;
    fs::write(format!("{}/{name}_events.csv", args.out_dir), counter.events_csv())?;
    let summary = format!(
        "Video: {}\nSnímok: {frames}\nČiara: {:.0} % šírky orezaného záberu\n\n{}",
        args.input,
        args.line * 100.0,
        counter.summary()
    );
    fs::write(format!("{}/{name}_summary.txt", args.out_dir), &summary)?;
    println!("\n{summary}");
    println!("Uložené do {}/ ({name}_counted.mp4, {name}_events.csv, {name}_summary.txt)", args.out_dir);
    Ok(())
}
