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
    println!("OpenCL dostupné: {}, vlákna OpenCV: {}", core::have_opencl()?, core::get_num_threads()?);

    let targets = [
        ("CPU", dnn::DNN_TARGET_CPU),
        ("OpenCL (GPU)", dnn::DNN_TARGET_OPENCL),
        ("OpenCL FP16 (GPU)", dnn::DNN_TARGET_OPENCL_FP16),
    ];
    for (name, target) in targets {
        let mut net = match Yolox::load_with_target(&model, 0.4, target) {
            Ok(n) => n,
            Err(e) => {
                println!("{name:<20} nedostupné: {e}");
                continue;
            }
        };
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
        println!("{name:<20} {ms:6.1} ms na detekciu = {:5.1} fps   (nájdených objektov: {found})", 1000.0 / ms);
    }
    Ok(())
}
