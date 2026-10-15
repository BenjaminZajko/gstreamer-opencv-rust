//! Detekcia objektov: GStreamer dekóduje video, OpenCV detekuje, GStreamer zakóduje výsledok.
//!
//!   cargo run --release -- <vstup.mp4> [--mode motion|people] [--output vystup.mp4] [--width 1280]
//!
//! Režimy:
//!   motion  pohybujúce sa objekty (MOG2 odčítanie pozadia + kontúry) – dobré pre statickú kameru
//!   people  chodci/ľudia (HOG + predtrénovaný SVM detektor, ktorý je priamo v OpenCV)
//!
//! Tok dát (každá snímka):
//!   súbor -> [GStreamer: dekódovanie + zmenšenie] -> appsink -> Mat (OpenCV)
//!         -> detekcia + kreslenie rámčekov -> appsrc -> [GStreamer: HW enkodér] -> MP4

use std::{env, error::Error, time::Instant};

// GStreamer: základ, "app" elementy (appsink/appsrc = vstup/výstup dát do našej aplikácie)
// a "video" (informácie o snímke: rozmery, stride...).
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
// OpenCV: v OpenCV 5 sú niektoré funkcie v iných moduloch než v OpenCV 4
// (geometry = kontúry/obdĺžniky, xobjdetect = HOGDescriptor).
use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    geometry, imgproc,
    prelude::*,
    video,
    xobjdetect::HOGDescriptor,
};

/// Skrátený zápis: akákoľvek chyba (GStreamer, OpenCV, parsovanie) sa dá vrátiť cez `?`.
type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Režim detekcie, vyberá sa cez `--mode`.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Motion,
    People,
}

/// Argumenty z príkazového riadku.
struct Args {
    input: String,  // cesta k vstupnému videu
    output: String, // cesta k výstupnému videu s rámčekmi
    mode: Mode,
    width: i32, // na akú šírku sa video zmenší (výška sa dopočíta)
}

/// Spracuje argumenty programu. Prvý argument bez `--` je vstupný súbor.
fn parse_args() -> Result<Args> {
    let mut it = env::args().skip(1); // skip(1) preskočí názov programu
    let mut input = None;
    let mut output = "annotated.mp4".to_string(); // predvolený výstup
    let mut mode = Mode::Motion; // predvolený režim
    let mut width = 1280; // predvolená šírka (720p)
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mode" => {
                mode = match it.next().as_deref() {
                    Some("motion") => Mode::Motion,
                    Some("people") => Mode::People,
                    _ => return Err("--mode: motion alebo people".into()),
                }
            }
            "--output" => output = it.next().ok_or("--output chýba hodnota")?,
            "--width" => width = it.next().ok_or("--width chýba hodnota")?.parse()?,
            _ if input.is_none() => input = Some(a), // prvý "voľný" argument = vstupný súbor
            _ => return Err(format!("neznámy argument: {a}").into()),
        }
    }
    let input = input.ok_or("použitie: object_detect <vstup.mp4> [--mode motion|people] [--output out.mp4] [--width 1280]")?;
    Ok(Args { input, output, mode, width })
}

/// Detektor: zo snímky (BGR) vráti obdĺžniky nájdených objektov.
/// Každý režim si drží vlastný stav (napr. MOG2 si pamätá pozadie z predošlých snímok).
enum Detector {
    Motion {
        mog: opencv::core::Ptr<video::BackgroundSubtractorMOG2>, // model pozadia
        kernel: Mat, // eliptický "štetec" pre čistenie masky (morfológia)
        warmup: u32, // koľko snímok ešte ostáva na naučenie pozadia
    },
    People {
        hog: HOGDescriptor, // HOG detektor s natrénovaným modelom ľudí
    },
}

impl Detector {
    /// Vytvorí detektor pre zvolený režim.
    fn new(mode: Mode) -> Result<Self> {
        Ok(match mode {
            Mode::Motion => Detector::Motion {
                // MOG2: history=500 snímok, prah=25 (citlivosť), true = rozpoznávaj tiene
                mog: video::create_background_subtractor_mog2(500, 25.0, true)?,
                // elipsa 5x5 px; Point(-1,-1) = stred jadra
                kernel: imgproc::get_structuring_element(
                    imgproc::MORPH_ELLIPSE,
                    Size::new(5, 5),
                    Point::new(-1, -1),
                )?,
                warmup: 30, // prvých pár snímok sa pozadie učí, nič nehlásime
            },
            Mode::People => {
                let mut hog = HOGDescriptor::new_def()?; // predvolené parametre (okno 64x128)
                // nahrá hotové koeficienty SVM modelu na detekciu ľudí, ktoré sú súčasťou OpenCV
                hog.set_svm_detector(HOGDescriptor::get_default_people_detector()?);
                Detector::People { hog }
            }
        })
    }

    /// Nájde objekty na snímke. Vracia zoznam obdĺžnikov (x, y, šírka, výška).
    fn detect(&mut self, frame: &Mat) -> Result<Vec<Rect>> {
        match self {
            Detector::Motion { mog, kernel, warmup } => {
                // 1) Maska popredia: biele pixely = niečo sa zmenilo oproti pozadiu.
                //    Parameter -1.0 = automatická rýchlosť učenia pozadia.
                let mut fg = Mat::default();
                BackgroundSubtractorTrait::apply(mog, frame, &mut fg, -1.0)?;
                if *warmup > 0 {
                    *warmup -= 1;
                    return Ok(vec![]); // ešte sa učí, nič nehlásime
                }
                // 2) Binarizácia: 255 = pohyb, 127 = tieň -> tiene vyhodíme prahom 200
                let mut bin = Mat::default();
                imgproc::threshold(&fg, &mut bin, 200.0, 255.0, imgproc::THRESH_BINARY)?;
                // 3) MORPH_OPEN odstráni drobný šum (osamotené bodky)
                let mut opened = Mat::default();
                imgproc::morphology_ex(
                    &bin,
                    &mut opened,
                    imgproc::MORPH_OPEN,
                    kernel,
                    Point::new(-1, -1),
                    1,
                    core::BORDER_CONSTANT,
                    imgproc::morphology_default_border_value()?,
                )?;
                // 4) Dilatácia (3 iterácie) zlepí blízke časti jedného objektu dokopy
                let mut dilated = Mat::default();
                imgproc::dilate(
                    &opened,
                    &mut dilated,
                    kernel,
                    Point::new(-1, -1),
                    3,
                    core::BORDER_CONSTANT,
                    imgproc::morphology_default_border_value()?,
                )?;
                // 5) Nájdenie súvislých bielych oblastí (kontúr); RETR_EXTERNAL = len vonkajšie
                let mut contours = Vector::<Vector<Point>>::new();
                imgproc::find_contours(
                    &dilated,
                    &mut contours,
                    imgproc::RETR_EXTERNAL,
                    imgproc::CHAIN_APPROX_SIMPLE,
                    Point::new(0, 0),
                )?;
                // 6) Ignorujeme príliš malé oblasti (menej než 0,1 % plochy obrazu = šum)
                let min_area = (frame.rows() * frame.cols()) as f64 * 0.001;
                let mut out = vec![];
                for c in contours.iter() {
                    if geometry::contour_area(&c, false)? >= min_area {
                        out.push(geometry::bounding_rect(&c)?); // obdĺžnik okolo kontúry
                    }
                }
                Ok(out)
            }
            Detector::People { hog } => {
                // HOG je pomalý, preto ho púšťame na zmenšenom obraze (šírka 640 px)
                let scale = 640.0 / frame.cols() as f64;
                let mut small = Mat::default();
                imgproc::resize(frame, &mut small, Size::new(0, 0), scale, scale, imgproc::INTER_AREA)?;
                let mut found = Vector::<Rect>::new(); // nájdené obdĺžniky
                let mut weights = Vector::<f64>::new(); // istota každej detekcie
                hog.detect_multi_scale_weights(
                    &small,
                    &mut found,
                    &mut weights,
                    0.0,              // hit_threshold: minimálna istota SVM
                    Size::new(8, 8),  // win_stride: o koľko sa posúva okno
                    Size::new(16, 16), // padding okolo okna
                    1.05,             // scale: krok zmenšovania obrazu (viac mierok = nájde aj iné veľkosti)
                    2.0,              // final_threshold: koľko prekrývajúcich sa detekcií treba na potvrdenie
                    false,            // meanshift zoskupovanie vypnuté
                )?;
                // súradnice vrátime späť do pôvodného rozlíšenia snímky
                Ok(found
                    .iter()
                    .map(|r| {
                        Rect::new(
                            (r.x as f64 / scale) as i32,
                            (r.y as f64 / scale) as i32,
                            (r.width as f64 / scale) as i32,
                            (r.height as f64 / scale) as i32,
                        )
                    })
                    .collect())
            }
        }
    }
}

/// Nakreslí zelené rámčeky s popisom okolo každej detekcie a počítadlo vpravo hore.
fn draw(frame: &mut Mat, boxes: &[Rect], label: &str, idx: u64) -> Result<()> {
    let green = Scalar::new(0.0, 255.0, 0.0, 0.0); // farba je v poradí B, G, R
    for r in boxes {
        // rámček hrúbky 2 px
        imgproc::rectangle(frame, *r, green, 2, imgproc::LINE_8, 0)?;
        // popis tesne nad rámčekom (max(14) zabráni, aby text vyšiel mimo obrazu)
        imgproc::put_text(
            frame,
            label,
            Point::new(r.x, (r.y - 6).max(14)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6, // veľkosť písma
            green,
            2, // hrúbka
            imgproc::LINE_AA, // vyhladené hrany
            false,
        )?;
    }
    // Počítadlo v pravom hornom rohu (vľavo hore je časová pečiatka kamery)
    let text = format!("frame {idx}  objektov: {}", boxes.len());
    imgproc::put_text(
        frame,
        &text,
        Point::new(frame.cols() - 430, 28),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.8,
        Scalar::new(0.0, 255.0, 255.0, 0.0), // žltá
        2,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

/// Výstupná pipeline: appsrc (BGR) -> HW H.264 enkodér (VideoToolbox) -> MP4.
/// Volá sa až pri prvej snímke, lebo až vtedy poznáme rozmery videa.
fn make_output(path: &str, info: &gst_video::VideoInfo) -> Result<(gst::Pipeline, gst_app::AppSrc)> {
    let pipeline = gst::parse::launch(&format!(
        "appsrc name=src format=time ! videoconvert ! vtenc_h264_hw bitrate=8000 \
         ! h264parse ! mp4mux ! filesink location=\"{path}\""
    ))?
    .downcast::<gst::Pipeline>()
    .map_err(|_| "pipeline")?;
    // appsrc = element, do ktorého my posielame snímky
    let src = pipeline
        .by_name("src")
        .ok_or("appsrc")?
        .downcast::<gst_app::AppSrc>()
        .map_err(|_| "appsrc")?;
    // povieme mu formát snímok (BGR, rozmery, fps) podľa vstupu
    src.set_caps(Some(&info.to_caps()?));
    pipeline.set_state(gst::State::Playing)?;
    Ok((pipeline, src))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    gst::init()?; // inicializácia GStreameru (nutné pred akýmkoľvek použitím)

    // Vstupná pipeline: ľubovoľný formát -> BGR (formát, ktorý OpenCV používa), zmenšené na --width.
    // pixel-aspect-ratio=1/1 núti videoscale dopočítať výšku (inak by video zdeformoval).
    // sync=false = spracúvaj čo najrýchlejšie, nečakaj na reálny čas.
    let input = gst::parse::launch(&format!(
        "filesrc location=\"{}\" ! decodebin ! videoconvert ! videoscale \
         ! video/x-raw,format=BGR,width={},pixel-aspect-ratio=1/1 ! appsink name=sink sync=false",
        args.input, args.width
    ))?
    .downcast::<gst::Pipeline>()
    .map_err(|_| "pipeline")?;
    // appsink = element, z ktorého my vyberáme snímky
    let sink = input
        .by_name("sink")
        .ok_or("appsink")?
        .downcast::<gst_app::AppSink>()
        .map_err(|_| "appsink")?;
    input.set_state(gst::State::Playing)?;

    let mut detector = Detector::new(args.mode)?;
    let label = if args.mode == Mode::People { "clovek" } else { "pohyb" };

    // Výstup sa vytvorí lenivo pri prvej snímke (pozri make_output)
    let mut output: Option<(gst::Pipeline, gst_app::AppSrc)> = None;
    let (mut frames, mut total_boxes) = (0u64, 0u64); // štatistika
    let start = Instant::now();

    // Hlavná slučka. pull_sample vráti Err na konci streamu (EOS), vtedy cyklus skončí.
    while let Ok(sample) = sink.pull_sample() {
        let buffer = sample.buffer().ok_or("snímka bez buffera")?;
        let info = gst_video::VideoInfo::from_caps(sample.caps().ok_or("snímka bez caps")?)?;
        let (w, h) = (info.width() as i32, info.height() as i32);
        // stride = dĺžka jedného riadku v pamäti GStreameru v bajtoch (môže mať padding)
        let stride = info.stride()[0] as usize;

        // --- GStreamer buffer -> OpenCV Mat (kópia, aby sme mohli do obrazu kresliť) ---
        let map = buffer.map_readable()?; // prístup k bajtom snímky
        let mut frame = Mat::zeros(h, w, core::CV_8UC3)?.to_mat()?; // prázdna 3-kanálová (BGR) matica
        let row_bytes = (w * 3) as usize; // skutočné bajty jedného riadku (3 bajty na pixel)
        let dst = frame.data_bytes_mut()?; // Mat z Mat::zeros je súvislý (bez paddingu)
        for y in 0..h as usize {
            // kopírujeme po riadkoch, lebo stride sa môže líšiť od row_bytes
            dst[y * row_bytes..(y + 1) * row_bytes]
                .copy_from_slice(&map.as_slice()[y * stride..y * stride + row_bytes]);
        }
        drop(map); // uvoľníme mapovanie buffera

        // --- Detekcia a kreslenie ---
        let boxes = detector.detect(&frame)?;
        total_boxes += boxes.len() as u64;
        draw(&mut frame, &boxes, label, frames)?;

        // --- OpenCV Mat -> GStreamer buffer -> enkodér ---
        // pri prvej snímke vytvor výstup, potom ho už len používaj
        let (_, src) = match &mut output {
            Some(o) => o,
            None => output.insert(make_output(&args.output, &info)?),
        };
        let mut out_buf = gst::Buffer::with_size(stride * h as usize)?;
        {
            let b = out_buf.get_mut().ok_or("buffer")?;
            // prenesieme časové značky, aby mal výstup rovnaké tempo ako vstup
            b.set_pts(buffer.pts());
            b.set_duration(buffer.duration());
            let mut m = b.map_writable()?;
            let src_bytes = frame.data_bytes()?;
            for y in 0..h as usize {
                m.as_mut_slice()[y * stride..y * stride + row_bytes]
                    .copy_from_slice(&src_bytes[y * row_bytes..(y + 1) * row_bytes]);
            }
        }
        src.push_buffer(out_buf)?; // pošli snímku do enkodéra

        frames += 1;
        if frames % 50 == 0 {
            // priebežný výpis každých 50 snímok
            println!("{frames} snímok, {:.1} fps", frames as f64 / start.elapsed().as_secs_f64());
        }
    }

    // Koniec videa: povedz enkodéru, že už nepríde nič (EOS), a počkaj, kým dopíše súbor.
    // Bez EOS by MP4 nemal index a nešiel by prehrať.
    if let Some((pipeline, src)) = output {
        src.end_of_stream()?;
        let bus = pipeline.bus().ok_or("bus")?;
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(_) => break, // enkodér je hotový
                gst::MessageView::Error(e) => return Err(format!("výstup: {}", e.error()).into()),
                _ => {}
            }
        }
        pipeline.set_state(gst::State::Null)?;
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
