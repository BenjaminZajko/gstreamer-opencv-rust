//! Porovnanie dvoch modelov na tých istých snímkach videa (výrez ako pri počítaní).
//! Na každej snímke sa spustia oba modely hneď po sebe (poradie sa strieda), takže kolísanie
//! výkonu stroja ovplyvní oba rovnako. Zaznamená sa čas inferencie a nájdení ľudia.
//!
//!   cargo run --release --example compare_models -- video.mp4 models/yolox_s.onnx models/yolo11s.onnx vystup.csv
//!   (model sa dá zadať aj ako cesta:engine, napr. models/yolox_s.onnx:coreml)

use std::{io::Write, time::Instant};

use gstreamer as gst;
use gstreamer_video as gst_video;
use opencv::prelude::*;

use object_detect::{labels::PERSON, live_count::roi_rect, roi::Crop, video_io, yolox::Yolox};

fn main() -> object_detect::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [video, ma, mb, out] = &a[..] else {
        return Err("použitie: compare_models video.mp4 modelA.onnx modelB.onnx vystup.csv".into());
    };
    gst::init()?;
    let load = |spec: &str| -> object_detect::Result<Yolox> {
        let (path, eng) = spec.split_once(':').unwrap_or((spec, "opencv"));
        Yolox::load_with_engine(path, 0.4, object_detect::yolox::Engine::parse(eng)?)
    };
    let mut nets = [load(ma)?, load(mb)?];
    let names = [format!("{}_{}", nets[0].kind().name(), nets[0].engine().name()), format!("{}_{}", nets[1].kind().name(), nets[1].engine().name())];
    let (_p, sink) = video_io::open_input(video, 2560)?;
    let crop = Crop { x: 0.0, y: 0.52, w: 0.47, h: 0.33 };
    let mut f = std::fs::File::create(out)?;
    writeln!(f, "snimka,ms_{0},ms_{1},ludia_{0},ludia_{1},istota_{0},istota_{1},ine_{0},ine_{1}", names[0], names[1])?;
    let mut i = 0u32;
    while let Ok(s) = sink.pull_sample() {
        let info = gst_video::VideoInfo::from_caps(s.caps().ok_or("caps")?)?;
        let m = video_io::buffer_to_mat(s.buffer().ok_or("buffer")?, &info)?;
        let roi = Mat::roi(&m, roi_rect(&crop, m.cols(), m.rows()))?.try_clone()?;
        let mut res = [(0.0, 0usize, 0f32, 0usize); 2];
        // poradie sa strieda: párne snímky A->B, nepárne B->A
        let order = if i % 2 == 0 { [0, 1] } else { [1, 0] };
        for k in order {
            let t = Instant::now();
            let d = nets[k].detect(&roi, None)?;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let people: Vec<_> = d.iter().filter(|x| x.class == PERSON).collect();
            let best = people.iter().map(|x| x.score).fold(0f32, f32::max);
            res[k] = (ms, people.len(), best, d.len() - people.len());
        }
        writeln!(f, "{i},{:.2},{:.2},{},{},{:.3},{:.3},{},{}", res[0].0, res[1].0, res[0].1, res[1].1, res[0].2, res[1].2, res[0].3, res[1].3)?;
        i += 1;
    }
    println!("hotovo: {i} snímok -> {out}");
    Ok(())
}
