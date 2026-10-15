//! Vlastný RTSP server: anotované snímky z OpenCV sa kódujú HW H.264 enkodérom a vysielajú
//! klientom (VLC, ffplay...) na `rtsp://<ip>:<port><mount>`.
//!
//! Server beží vo vlastnom vlákne (GLib main loop). Hlavné vlákno posiela snímky cez `push`.
//! Keď sa pripojí klient, server vytvorí pipeline s `appsrc`, ktorý si tu zapamätáme;
//! keď sa odpojí, zabudneme ho (a detekcia sa môže pozastaviť – pozri `has_client`).

use std::sync::{Arc, Mutex};

use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_rtsp_server as rtsp;
use gstreamer_rtsp_server::prelude::*;
use gstreamer_video as gst_video;

use crate::Result;

/// Popis pipeline, ktorú server spustí pre každé médium (zdieľané medzi klientmi):
/// appsrc (BGR) -> HW H.264 enkodér (VideoToolbox) -> RTP.
///
/// * `leaky-type=downstream` + `max-buffers`: pri pomalom klientovi sa zahadzujú staré snímky,
///   oneskorenie nenarastá.
/// * `max-keyframe-interval`: kľúčová snímka každých `keyint` snímok, aby sa nový klient
///   (VLC) rýchlo "chytil" a nemusel čakať na ďalší keyframe.
/// * `config-interval=-1`: SPS/PPS sa posielajú pred každým keyframe.
pub fn launch_description(bitrate_kbps: u32, keyint: u32) -> String {
    format!(
        "( appsrc name=src is-live=true format=time do-timestamp=true \
           max-buffers=3 leaky-type=downstream \
         ! videoconvert \
         ! vtenc_h264_hw realtime=true allow-frame-reordering=false \
           bitrate={bitrate_kbps} max-keyframe-interval={keyint} \
         ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=-1 )"
    )
}

/// Overí, že sa dá otvoriť port na všetkých adresách (to robí aj RTSP server). Dá zrozumiteľnú
/// chybu hneď pri štarte, nie až po pripojení ku kamere. Server sa na obsadenom porte inak
/// nemusí sťažovať (macOS dovolí obsadiť rovnaký port na inej adrese).
pub fn ensure_port_free(port: u16) -> Result<()> {
    std::net::TcpListener::bind(("0.0.0.0", port)).map(drop).map_err(|e| {
        format!("port {port} je obsadený ({e}). Beží už iná inštancia? Skús ./live.sh stop alebo iný --port.").into()
    })
}

/// Bežiaci RTSP server s výstupom detekcie.
pub struct RtspOutput {
    /// appsrc aktuálne pripojeného média (`None` = nik sa nedíva).
    appsrc: Arc<Mutex<Option<gst_app::AppSrc>>>,
    // server musí žiť, kým beží program
    _server: rtsp::RTSPServer,
}

impl RtspOutput {
    /// Spustí server na `port` s bodom pripojenia `mount` (napr. "/detect").
    /// `info` určuje rozmery snímok, `fps` tempo výstupu.
    pub fn start(
        port: u16,
        mount: &str,
        info: &gst_video::VideoInfo,
        fps: i32,
        bitrate_kbps: u32,
    ) -> Result<Self> {
        ensure_port_free(port)?;
        let caps = gst_video::VideoInfo::builder(gst_video::VideoFormat::Bgr, info.width(), info.height())
            .fps(gst::Fraction::new(fps, 1))
            .build()?
            .to_caps()?;

        let appsrc: Arc<Mutex<Option<gst_app::AppSrc>>> = Arc::new(Mutex::new(None));

        let factory = rtsp::RTSPMediaFactory::new();
        factory.set_launch(&launch_description(bitrate_kbps, (fps as u32).max(1)));
        factory.set_shared(true); // viac klientov naraz vidí ten istý stream (jeden enkodér)

        let slot = Arc::clone(&appsrc);
        factory.connect_media_configure(move |_, media| {
            let Some(src) = media
                .element()
                .downcast::<gst::Bin>()
                .ok()
                .and_then(|bin| bin.by_name("src"))
                .and_then(|e| e.downcast::<gst_app::AppSrc>().ok())
            else {
                return;
            };
            src.set_caps(Some(&caps));
            *slot.lock().unwrap() = Some(src);

            // keď médium zanikne (odišiel posledný klient), appsrc zabudneme
            let slot_off = Arc::clone(&slot);
            media.connect_unprepared(move |_| {
                *slot_off.lock().unwrap() = None;
            });
        });

        let server = rtsp::RTSPServer::new();
        server.set_service(&port.to_string());
        server
            .mount_points()
            .ok_or("RTSP server nemá mount points")?
            .add_factory(mount, factory);
        server
            .attach(None)
            .map_err(|_| format!("RTSP server sa nepodarilo spustiť na porte {port} (je obsadený?)"))?;

        // GLib main loop obsluhuje pripojenia klientov
        std::thread::spawn(|| gst::glib::MainLoop::new(None, false).run());

        Ok(Self { appsrc, _server: server })
    }

    /// Pozerá sa práve niekto?
    pub fn has_client(&self) -> bool {
        self.appsrc.lock().unwrap().is_some()
    }

    /// Pošle snímku klientom. Ak sa nikto nedíva, snímka sa zahodí.
    pub fn push(&self, buffer: gst::Buffer) -> Result<()> {
        let src = self.appsrc.lock().unwrap().clone();
        if let Some(src) = src {
            // chyba (napr. flushing pri odchode klienta) nie je fatálna
            let _ = src.push_buffer(buffer);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_obsahuje_hw_enkoder_a_nastavenia() {
        let d = launch_description(4000, 20);
        assert!(d.starts_with("( ") && d.ends_with(" )"), "RTSP factory vyžaduje zátvorky: {d}");
        assert!(d.contains("appsrc name=src") && d.contains("is-live=true"));
        assert!(d.contains("vtenc_h264_hw") && d.contains("realtime=true"));
        assert!(d.contains("bitrate=4000") && d.contains("max-keyframe-interval=20"));
        assert!(d.contains("allow-frame-reordering=false"), "bez B-snímok = nízke oneskorenie");
        assert!(d.contains("rtph264pay name=pay0"), "RTSP server hľadá payloader pay0");
        assert!(d.contains("leaky-type=downstream"));
    }

    #[test]
    fn launch_sa_da_naparsovat() {
        gst::init().unwrap();
        // factory pipeline je v zátvorkách = bin; parse_bin to vie
        gst::parse::bin_from_description(
            launch_description(4000, 20).trim_start_matches("( ").trim_end_matches(" )"),
            false,
        )
        .unwrap();
    }

    #[test]
    fn obsadeny_port_je_odhaleny_aj_ked_ho_drzi_iny_na_inej_adrese() {
        // drží ho len 127.0.0.1, server by sa ale pokúsil o 0.0.0.0
        let l = std::net::TcpListener::bind("0.0.0.0:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let e = ensure_port_free(port).unwrap_err().to_string();
        assert!(e.contains(&port.to_string()) && e.contains("obsadený"), "{e}");
        drop(l);
        assert!(ensure_port_free(port).is_ok(), "po uvoľnení je port voľný");
    }
}
