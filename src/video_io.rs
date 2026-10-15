//! Vstup a výstup videa cez GStreamer a prevod snímok medzi GStreamerom a OpenCV.

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use opencv::{core, core::Mat, prelude::*};

use crate::Result;

/// Vstupná pipeline: ľubovoľný formát -> BGR (formát, ktorý OpenCV používa), zmenšené na `width`.
/// pixel-aspect-ratio=1/1 núti videoscale dopočítať výšku (inak by video zdeformoval).
/// sync=false = spracúvaj čo najrýchlejšie, nečakaj na reálny čas.
/// Vráti pipeline a appsink, z ktorého sa vyberajú snímky.
pub fn open_input(path: &str, width: i32) -> Result<(gst::Pipeline, gst_app::AppSink)> {
    let input = gst::parse::launch(&format!(
        "filesrc location=\"{path}\" ! decodebin ! videoconvert ! videoscale \
         ! video/x-raw,format=BGR,width={width},pixel-aspect-ratio=1/1 ! appsink name=sink sync=false"
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
    Ok((input, sink))
}

/// Kodek videa z kamery.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Codec {
    H264,
    H265,
}

/// Spoločný koniec vstupných pipeline: BGR, zmenšené na `width`, appsink ktorý zahadzuje staré
/// snímky (max-buffers=1 drop=true), aby pri pomalšej detekcii nenarastalo oneskorenie.
fn bgr_tail(width: i32) -> String {
    format!(
        "videoconvert ! videoscale \
         ! video/x-raw,format=BGR,width={width},pixel-aspect-ratio=1/1 \
         ! appsink name=sink sync=false max-buffers=1 drop=true"
    )
}

/// Zloží popis pipeline pre živý zdroj:
///   * `rtsp://...`  kamera (TCP, HW dekódovanie VideoToolbox),
///   * `test:ball`   umelý zdroj s pohybujúcou sa loptou (na skúšanie bez kamery).
///
/// Pri H.265 z Tapo kamery sa stream musí pretočiť cez matroskamux/matroskademux, inak
/// VideoToolbox dekóduje len kľúčové snímky (0,5 fps) – pozri tapo.sh v GStream_configuration.
pub fn live_input_pipeline(source: &str, codec: Codec, width: i32) -> Result<String> {
    let tail = bgr_tail(width);
    if source == "test:ball" {
        return Ok(format!(
            "videotestsrc is-live=true pattern=ball ! video/x-raw,width=640,height=360,framerate=20/1 ! {tail}"
        ));
    }
    if source.starts_with("rtsp://") || source.starts_with("rtsps://") {
        let depay = match codec {
            Codec::H264 => "rtph264depay ! h264parse ! vtdec_hw",
            Codec::H265 => {
                "rtph265depay ! h265parse ! matroskamux streamable=true ! matroskademux \
                 ! h265parse ! vtdec_hw"
            }
        };
        return Ok(format!(
            "rtspsrc location=\"{source}\" protocols=tcp latency=200 ! {depay} ! {tail}"
        ));
    }
    Err(format!("neznámy zdroj '{source}': použi rtsp://... alebo test:ball").into())
}

/// Otvorí živý zdroj (pozri `live_input_pipeline`). Vráti pipeline a appsink so snímkami.
pub fn open_live_input(
    source: &str,
    codec: Codec,
    width: i32,
) -> Result<(gst::Pipeline, gst_app::AppSink)> {
    let desc = live_input_pipeline(source, codec, width)?;
    let pipeline = gst::parse::launch(&desc)?
        .downcast::<gst::Pipeline>()
        .map_err(|_| "pipeline")?;
    let sink = pipeline
        .by_name("sink")
        .ok_or("appsink")?
        .downcast::<gst_app::AppSink>()
        .map_err(|_| "appsink")?;
    pipeline.set_state(gst::State::Playing)?;
    Ok((pipeline, sink))
}

/// Výstupná pipeline: appsrc (BGR) -> HW H.264 enkodér (VideoToolbox) -> MP4.
/// Volá sa až pri prvej snímke, lebo až vtedy poznáme rozmery videa.
pub fn make_output(path: &str, info: &gst_video::VideoInfo) -> Result<(gst::Pipeline, gst_app::AppSrc)> {
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

/// GStreamer buffer -> OpenCV Mat (kópia, aby sme mohli do obrazu kresliť).
pub fn buffer_to_mat(buffer: &gst::BufferRef, info: &gst_video::VideoInfo) -> Result<Mat> {
    let (w, h) = (info.width() as i32, info.height() as i32);
    // stride = dĺžka jedného riadku v pamäti GStreameru v bajtoch (môže mať padding)
    let stride = info.stride()[0] as usize;
    let map = buffer.map_readable()?; // prístup k bajtom snímky
    let mut frame = Mat::zeros(h, w, core::CV_8UC3)?.to_mat()?; // prázdna 3-kanálová (BGR) matica
    let row_bytes = (w * 3) as usize; // skutočné bajty jedného riadku (3 bajty na pixel)
    let dst = frame.data_bytes_mut()?; // Mat z Mat::zeros je súvislý (bez paddingu)
    for y in 0..h as usize {
        // kopírujeme po riadkoch, lebo stride sa môže líšiť od row_bytes
        dst[y * row_bytes..(y + 1) * row_bytes]
            .copy_from_slice(&map.as_slice()[y * stride..y * stride + row_bytes]);
    }
    Ok(frame)
}

/// OpenCV Mat -> GStreamer buffer bez časových značiek (pre živý výstup, kde ich dopĺňa appsrc).
pub fn mat_to_untimed_buffer(frame: &Mat, info: &gst_video::VideoInfo) -> Result<gst::Buffer> {
    let h = info.height() as usize;
    let stride = info.stride()[0] as usize;
    let row_bytes = info.width() as usize * 3;
    let mut out_buf = gst::Buffer::with_size(stride * h)?;
    {
        let b = out_buf.get_mut().ok_or("buffer")?;
        let mut m = b.map_writable()?;
        let src_bytes = frame.data_bytes()?;
        for y in 0..h {
            m.as_mut_slice()[y * stride..y * stride + row_bytes]
                .copy_from_slice(&src_bytes[y * row_bytes..(y + 1) * row_bytes]);
        }
    }
    Ok(out_buf)
}

/// OpenCV Mat -> GStreamer buffer. Časové značky sa preberú zo vstupnej snímky,
/// aby mal výstup rovnaké tempo ako vstup.
pub fn mat_to_buffer(
    frame: &Mat,
    info: &gst_video::VideoInfo,
    original: &gst::BufferRef,
) -> Result<gst::Buffer> {
    let mut out_buf = mat_to_untimed_buffer(frame, info)?;
    let b = out_buf.get_mut().ok_or("buffer")?;
    b.set_pts(original.pts());
    b.set_duration(original.duration());
    Ok(out_buf)
}

/// Koniec videa: povedz enkodéru, že už nepríde nič (EOS), a počkaj, kým dopíše súbor.
/// Bez EOS by MP4 nemal index a nešiel by prehrať.
pub fn finish_output(pipeline: &gst::Pipeline, src: &gst_app::AppSrc) -> Result<()> {
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rozmery zámerne nie sú deliteľné 4: riadok má 5*3 = 15 B, ale stride je 16 B (padding).
    const W: u32 = 5;
    const H: u32 = 3;

    fn info() -> gst_video::VideoInfo {
        gst::init().unwrap();
        gst_video::VideoInfo::builder(gst_video::VideoFormat::Bgr, W, H)
            .build()
            .unwrap()
    }

    /// Očakávaná hodnota bajtu pre pixel (x, y), kanál c.
    fn val(x: usize, y: usize, c: usize) -> u8 {
        (y * 50 + x * 10 + c) as u8
    }

    /// GStreamer buffer so vzorom a paddingom (0xEE) na konci každého riadku.
    fn padded_buffer(info: &gst_video::VideoInfo) -> gst::Buffer {
        let stride = info.stride()[0] as usize;
        let mut buf = gst::Buffer::with_size(info.size()).unwrap();
        {
            let b = buf.get_mut().unwrap();
            b.set_pts(gst::ClockTime::from_mseconds(1234));
            b.set_duration(gst::ClockTime::from_mseconds(40));
            let mut m = b.map_writable().unwrap();
            m.as_mut_slice().fill(0xEE);
            for y in 0..H as usize {
                for x in 0..W as usize {
                    for c in 0..3 {
                        m.as_mut_slice()[y * stride + x * 3 + c] = val(x, y, c);
                    }
                }
            }
        }
        buf
    }

    #[test]
    fn stride_ma_padding() {
        // Test má zmysel len ak je padding naozaj prítomný.
        assert_eq!(info().stride()[0], 16);
    }

    #[test]
    fn buffer_na_mat_odstrani_padding() {
        let info = info();
        let buf = padded_buffer(&info);
        let mat = buffer_to_mat(buf.as_ref(), &info).unwrap();
        assert_eq!((mat.cols(), mat.rows()), (W as i32, H as i32));
        assert_eq!(mat.typ(), core::CV_8UC3);
        let bytes = mat.data_bytes().unwrap();
        assert_eq!(bytes.len(), (W * H * 3) as usize, "Mat je súvislý, bez paddingu");
        for y in 0..H as usize {
            for x in 0..W as usize {
                for c in 0..3 {
                    assert_eq!(bytes[(y * W as usize + x) * 3 + c], val(x, y, c), "x={x} y={y} c={c}");
                }
            }
        }
    }

    #[test]
    fn mat_na_buffer_zachova_pixely_a_casove_znacky() {
        let info = info();
        let original = padded_buffer(&info);
        let mat = buffer_to_mat(original.as_ref(), &info).unwrap();
        let out = mat_to_buffer(&mat, &info, original.as_ref()).unwrap();

        assert_eq!(out.pts(), original.pts());
        assert_eq!(out.duration(), original.duration());
        assert_eq!(out.size(), info.size());

        let stride = info.stride()[0] as usize;
        let m = out.map_readable().unwrap();
        for y in 0..H as usize {
            for x in 0..W as usize {
                for c in 0..3 {
                    assert_eq!(m.as_slice()[y * stride + x * 3 + c], val(x, y, c));
                }
            }
        }
    }

    #[test]
    fn upravy_v_mat_sa_prenesu_do_buffera() {
        let info = info();
        let original = padded_buffer(&info);
        let mut mat = buffer_to_mat(original.as_ref(), &info).unwrap();
        mat.data_bytes_mut().unwrap()[0] = 99; // prvý pixel, kanál B
        let out = mat_to_buffer(&mat, &info, original.as_ref()).unwrap();
        assert_eq!(out.map_readable().unwrap().as_slice()[0], 99);
    }

    #[test]
    fn vstup_z_neexistujuceho_suboru_zlyha_hned() {
        gst::init().unwrap();
        assert!(open_input("/neexistuje/video.mp4", 320).is_err());
    }

    #[test]
    fn mat_na_buffer_bez_casovych_znaciek() {
        let info = info();
        let original = padded_buffer(&info);
        let mat = buffer_to_mat(original.as_ref(), &info).unwrap();
        let out = mat_to_untimed_buffer(&mat, &info).unwrap();
        assert_eq!(out.pts(), None);
        assert_eq!(out.duration(), None);
        let stride = info.stride()[0] as usize;
        assert_eq!(out.map_readable().unwrap().as_slice()[stride + 3], val(1, 1, 0));
    }

    #[test]
    fn live_pipeline_rtsp_h265_pouziva_workaround_a_hw_dekoder() {
        let d = live_input_pipeline("rtsp://u:p@1.2.3.4:554/stream1", Codec::H265, 1280).unwrap();
        assert!(d.contains("location=\"rtsp://u:p@1.2.3.4:554/stream1\""), "{d}");
        assert!(d.contains("protocols=tcp"));
        assert!(d.contains("rtph265depay"));
        assert!(d.contains("matroskamux") && d.contains("matroskademux"), "workaround pre VideoToolbox");
        assert!(d.contains("vtdec_hw"));
        assert!(d.contains("width=1280") && d.contains("format=BGR"));
        assert!(d.contains("drop=true") && d.contains("max-buffers=1"), "živý vstup nesmie hromadiť snímky");
    }

    #[test]
    fn live_pipeline_rtsp_h264_bez_workaroundu() {
        let d = live_input_pipeline("rtsp://cam/stream2", Codec::H264, 640).unwrap();
        assert!(d.contains("rtph264depay") && d.contains("vtdec_hw"));
        assert!(!d.contains("matroska"), "H.264 workaround nepotrebuje");
    }

    #[test]
    fn live_pipeline_testovaci_zdroj() {
        let d = live_input_pipeline("test:ball", Codec::H265, 320).unwrap();
        assert!(d.contains("videotestsrc") && d.contains("is-live=true") && d.contains("width=320"));
    }

    #[test]
    fn live_pipeline_neznamy_zdroj() {
        let e = live_input_pipeline("http://x", Codec::H265, 320).unwrap_err().to_string();
        assert!(e.contains("neznámy zdroj"), "{e}");
        assert!(live_input_pipeline("", Codec::H265, 320).is_err());
    }

    #[test]
    fn live_pipeline_sa_da_naparsovat() {
        gst::init().unwrap();
        // aj rtsp pipeline (bez pripojenia) musí byť syntakticky platná
        for (src, codec) in [("rtsp://x/y", Codec::H265), ("rtsp://x/y", Codec::H264), ("test:ball", Codec::H265)] {
            let d = live_input_pipeline(src, codec, 320).unwrap();
            gst::parse::launch(&d).unwrap_or_else(|e| panic!("{d}\n{e}"));
        }
    }
}
