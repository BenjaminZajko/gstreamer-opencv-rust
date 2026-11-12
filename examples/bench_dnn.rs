//! Meranie rýchlosti samotnej siete YOLOX (bez videa): kolko ms trvá jedna detekcia na rôznych
//! cieľoch výpočtu (CPU, OpenCL/GPU) a pri rôznom počte vlákien.
//!
//!   cargo run --release --example bench_dnn -- obrazok.jpg [models/yolox_s.onnx] [počet opakovaní]

use std::time::Instant;

use opencv::{core, dnn, imgcodecs, prelude::*};

use object_detect::yolox::Yolox;

fn main() -> object_detect::Result<()> {
    let mut a = std::env::args().skip(1);
    let image = a.next().ok_or("použitie: bench_dnn obrazok.jpg [model.onnx] [n]")?;
    let model = a.next().unwrap_or_else(|| "models/yolox_s.onnx".into());
    let n: usize = a.next().map(|s| s.parse()).transpose()?.unwrap_or(30);
    let frame = imgcodecs::imread(&image, imgcodecs::IMREAD_COLOR)?;
    if frame.empty() {
        return Err(format!("obrázok '{image}' sa nepodarilo načítať").into());
    }
    println!("obrázok {}x{}, model {model}, {n} opakovaní", frame.cols(), frame.rows());
    println!("typ modelu: {}", Yolox::load(&model, 0.4)?.kind().name());
    println!("OpenCL dostupné: {}, vlákna OpenCV: {}", core::have_opencl()?, core::get_num_threads()?);

    use object_detect::yolox::Engine;
    let targets: [(&str, Box<dyn Fn() -> object_detect::Result<Yolox>>); 8] = [
        ("OpenCV CPU", Box::new(|| Yolox::load_with_target(&model, 0.4, dnn::DNN_TARGET_CPU))),
        ("OpenCV OpenCL", Box::new(|| Yolox::load_with_target(&model, 0.4, dnn::DNN_TARGET_OPENCL))),
        ("OpenCV OpenCL FP16", Box::new(|| Yolox::load_with_target(&model, 0.4, dnn::DNN_TARGET_OPENCL_FP16))),
        ("ORT CPU", Box::new(|| Yolox::load_with_engine(&model, 0.4, Engine::OrtCpu))),
        ("Core ML (všetko)", Box::new(|| Yolox::load_with_engine(&model, 0.4, Engine::CoreMl))),
        ("Core ML GPU", Box::new(|| Yolox::load_with_engine(&model, 0.4, Engine::CoreMlGpu))),
        ("Core ML Neural Eng.", Box::new(|| Yolox::load_with_engine(&model, 0.4, Engine::CoreMlAne))),
        ("OpenCV CPU (znova)", Box::new(|| Yolox::load_with_target(&model, 0.4, dnn::DNN_TARGET_CPU))),
    ];
    for (name, load) in targets {
        let t_load = Instant::now();
        let mut net = match load() {
            Ok(n) => n,
            Err(e) => {
                println!("{name:<20} nedostupné: {e}");
                continue;
            }
        };
        let load_s = t_load.elapsed().as_secs_f64();
        // zahriatie (prvé volanie je pomalšie: kompilácia kernelov, alokácie)
        let mut found = 0;
        for _ in 0..3 {
            found = match net.detect(&frame, None) {
                Ok(d) => d.len(),
                Err(e) => {
                    println!("{name:<20} zlyhalo: {e}");
                    found = usize::MAX;
                    break;
                }
            };
        }
        if found == usize::MAX {
            continue;
        }
        let t = Instant::now();
        for _ in 0..n {
            net.detect(&frame, None)?;
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let dets = net.detect(&frame, None)?;
        let best = dets.iter().map(|d| d.score).fold(0f32, f32::max);
        let what: Vec<String> = dets
            .iter()
            .map(|d| format!("{} {:.0} % @({},{} {}x{})", object_detect::labels::COCO[d.class], d.score * 100.0, d.rect.x, d.rect.y, d.rect.width, d.rect.height))
            .collect();
        println!(
            "{name:<20} {ms:6.1} ms na detekciu = {:5.1} fps   (načítanie {:.1} s, nájdených objektov: {found}, najvyššia istota {:.0} %)",
            1000.0 / ms,
            load_s,
            best * 100.0
        );
        println!("{:<20} {}", "", what.join(", "));
    }
    Ok(())
}
