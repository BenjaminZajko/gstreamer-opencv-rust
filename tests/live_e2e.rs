//! E2E testy živého režimu: spustia `object_detect_live` s umelým zdrojom `test:ball`
//! a pripoja sa naň ako skutoční RTSP klienti (GStreamer rtspsrc).

use std::{
    net::{TcpListener, TcpStream},
    process::{Child, Command, Stdio},
    thread::sleep,
    time::{Duration, Instant},
};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;

const WIDTH: i32 = 320; // test:ball je 640x360 -> po zmenšení 320x180
const HEIGHT: i32 = 180;

/// Voľný port (OS ho pridelí, hneď ho uvoľníme).
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// Bežiaci server; po teste (aj pri páde testu) sa proces ukončí.
struct Live {
    child: Child,
    port: u16,
}

impl Live {
    fn start(extra: &[&str]) -> Self {
        let port = free_port();
        let child = Command::new(env!("CARGO_BIN_EXE_object_detect_live"))
            .args(["--input", "test:ball", "--mode", "motion", "--width", &WIDTH.to_string(), "--port", &port.to_string()])
            .args(extra)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("server sa nespustil");
        Live { child, port }
    }

    /// Počká, kým server počúva na porte.
    fn wait_ready(&mut self) {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(20) {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!("server skončil predčasne: {status}");
            }
            if TcpStream::connect(("127.0.0.1", self.port)).is_ok() {
                return;
            }
            sleep(Duration::from_millis(200));
        }
        panic!("server sa nerozbehol do 20 s");
    }

    fn url(&self) -> String {
        format!("rtsp://127.0.0.1:{}/detect", self.port)
    }

    fn alive(&mut self) -> bool {
        self.child.try_wait().unwrap().is_none()
    }
}

impl Drop for Live {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Jedna prijatá snímka: rozmery a počet "zelených" pixelov (farba rámčeka detekcie).
struct Received {
    w: i32,
    h: i32,
    green: usize,
}

/// Pripojí sa ako klient a prijme až `want` snímok (najviac `timeout`).
fn receive(url: &str, want: usize, timeout: Duration) -> Vec<Received> {
    gst::init().unwrap();
    let p = gst::parse::launch(&format!(
        "rtspsrc location={url} protocols=tcp latency=200 ! decodebin ! videoconvert \
         ! video/x-raw,format=BGR ! appsink name=sink sync=false"
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();
    let sink = p.by_name("sink").unwrap().downcast::<gst_app::AppSink>().unwrap();
    p.set_state(gst::State::Playing).unwrap();

    let mut got = vec![];
    let start = Instant::now();
    while got.len() < want && start.elapsed() < timeout {
        let Some(sample) = sink.try_pull_sample(gst::ClockTime::from_seconds(2)) else { continue };
        let info = gst_video::VideoInfo::from_caps(sample.caps().unwrap()).unwrap();
        let map = sample.buffer().unwrap().map_readable().unwrap();
        let stride = info.stride()[0] as usize;
        let mut green = 0;
        for y in 0..info.height() as usize {
            let row = &map.as_slice()[y * stride..y * stride + info.width() as usize * 3];
            green += row.chunks_exact(3).filter(|px| px[1] > 180 && px[0] < 110 && px[2] < 110).count();
        }
        got.push(Received { w: info.width() as i32, h: info.height() as i32, green });
    }
    p.set_state(gst::State::Null).unwrap();
    got
}

#[test]
fn klient_dostane_anotovane_video() {
    let mut s = Live::start(&[]);
    s.wait_ready();
    let frames = receive(&s.url(), 80, Duration::from_secs(40));

    assert!(frames.len() >= 40, "prišlo len {} snímok", frames.len());
    assert!(frames.iter().all(|f| (f.w, f.h) == (WIDTH, HEIGHT)), "rozmery sa nemajú meniť");
    // po zahriatí MOG2 sa lopta označuje zeleným rámčekom -> v streame musia byť zelené pixely
    assert!(
        frames.iter().any(|f| f.green > 30),
        "žiadna snímka nemá zelený rámček – anotácie sa nedostali do streamu"
    );
}

#[test]
fn dvaja_klienti_naraz() {
    let mut s = Live::start(&[]);
    s.wait_ready();
    let (u1, u2) = (s.url(), s.url());
    let a = std::thread::spawn(move || receive(&u1, 20, Duration::from_secs(40)).len());
    let b = std::thread::spawn(move || receive(&u2, 20, Duration::from_secs(40)).len());
    let (na, nb) = (a.join().unwrap(), b.join().unwrap());
    assert!(na >= 20 && nb >= 20, "klient A: {na}, klient B: {nb}");
    assert!(s.alive());
}

#[test]
fn server_prezije_bez_klienta_a_klient_sa_moze_vratit() {
    let mut s = Live::start(&[]);
    s.wait_ready();
    sleep(Duration::from_secs(3)); // nik sa nepozerá, server nesmie spadnúť
    assert!(s.alive());

    let first = receive(&s.url(), 15, Duration::from_secs(30)).len();
    assert!(first >= 15, "prvý klient: {first}");

    sleep(Duration::from_secs(2)); // klient odišiel
    assert!(s.alive());
    let second = receive(&s.url(), 15, Duration::from_secs(30)).len();
    assert!(second >= 15, "klient po opätovnom pripojení: {second}");
}

#[test]
fn neexistujuca_cesta_vrati_chybu_klientovi() {
    let mut s = Live::start(&["--mount", "/detect"]);
    s.wait_ready();
    let bad = format!("rtsp://127.0.0.1:{}/inde", s.port);
    assert!(receive(&bad, 1, Duration::from_secs(8)).is_empty(), "na neexistujúcej ceste nemá byť video");
    assert!(s.alive());
}

#[test]
fn vlastna_cesta_mount() {
    let mut s = Live::start(&["--mount", "/moja/cesta"]);
    s.wait_ready();
    let url = format!("rtsp://127.0.0.1:{}/moja/cesta", s.port);
    assert!(receive(&url, 10, Duration::from_secs(30)).len() >= 10);
}

#[test]
fn nedostupna_kamera_nezhodi_program() {
    // na porte 1 nik nepočúva; program sa má vracať k opakovaným pokusom, nie skončiť
    let mut child = Command::new(env!("CARGO_BIN_EXE_object_detect_live"))
        .args(["--input", "rtsp://127.0.0.1:1/x", "--mode", "motion", "--port", &free_port().to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    sleep(Duration::from_secs(8));
    let alive = child.try_wait().unwrap().is_none();
    let _ = child.kill();
    let _ = child.wait();
    assert!(alive, "program skončil, hoci mal skúšať pripojenie znova");
}

#[test]
fn obsadeny_port_skonci_s_chybou() {
    // port drží iný proces na všetkých adresách (ako druhá inštancia servera)
    let blocker = TcpListener::bind("0.0.0.0:0").unwrap();
    let port = blocker.local_addr().unwrap().port().to_string();
    let mut child = Command::new(env!("CARGO_BIN_EXE_object_detect_live"))
        .args(["--input", "test:ball", "--mode", "motion", "--port", &port])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // musí skončiť rýchlo; ak by bežal ďalej, test zlyhá namiesto visenia
    let t = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        if t.elapsed() > Duration::from_secs(15) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("program mal skončiť s chybou obsadeného portu, ale beží");
        }
        sleep(Duration::from_millis(100));
    };
    assert!(!status.success());
    let mut err = String::new();
    std::io::Read::read_to_string(&mut child.stderr.take().unwrap(), &mut err).unwrap();
    assert!(err.contains("obsadený") && err.contains(&port), "{err}");
}

#[test]
fn bez_vstupu_zlyha_s_napovedou() {
    let out = Command::new(env!("CARGO_BIN_EXE_object_detect_live")).env_remove("LIVE_INPUT").output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("použitie"));
}

#[test]
fn vstup_z_premennej_prostredia() {
    let port = free_port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_object_detect_live"))
        .args(["--mode", "motion", "--width", "320", "--port", &port.to_string()])
        .env("LIVE_INPUT", "test:ball")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let t = Instant::now();
    let mut up = false;
    while t.elapsed() < Duration::from_secs(20) {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            up = true;
            break;
        }
        sleep(Duration::from_millis(200));
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(up, "server sa s LIVE_INPUT nerozbehol");
}

// ---------------------------------------------------------------------------
// Režim počítania (--line): celý záber vo výstupe, čiara, súbory s počtami
// ---------------------------------------------------------------------------

/// Servery s neurónovou sieťou sú náročné na CPU; testy, ktoré ich spúšťajú, bežia po jednom,
/// inak sa pri súbežnom behu navzájom pripravia o výkon a presiahnu časový limit.
static HEAVY: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn heavy() -> std::sync::MutexGuard<'static, ()> {
    HEAVY.lock().unwrap_or_else(|e| e.into_inner())
}

fn model_path() -> Option<String> {
    let p = format!("{}/models/yolox_s.onnx", env!("CARGO_MANIFEST_DIR"));
    std::path::Path::new(&p).exists().then_some(p)
}

/// Dočasný priečinok pre logy počítania.
struct Dir(std::path::PathBuf);
impl Dir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("live_count_e2e_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        Dir(p)
    }
    fn s(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Prijme snímky a vráti pre každú počet žltých pixelov v danom pásme stĺpcov a riadkov.
fn yellow_in(url: &str, want: usize, cols: std::ops::Range<usize>, rows: std::ops::Range<usize>) -> Vec<usize> {
    gst::init().unwrap();
    let p = gst::parse::launch(&format!(
        "rtspsrc location={url} protocols=tcp latency=200 ! decodebin ! videoconvert \
         ! video/x-raw,format=BGR ! appsink name=sink sync=false"
    ))
    .unwrap()
    .downcast::<gst::Pipeline>()
    .unwrap();
    let sink = p.by_name("sink").unwrap().downcast::<gst_app::AppSink>().unwrap();
    p.set_state(gst::State::Playing).unwrap();
    let (mut out, start) = (vec![], Instant::now());
    while out.len() < want && start.elapsed() < Duration::from_secs(40) {
        let Some(s) = sink.try_pull_sample(gst::ClockTime::from_seconds(2)) else { continue };
        let info = gst_video::VideoInfo::from_caps(s.caps().unwrap()).unwrap();
        let map = s.buffer().unwrap().map_readable().unwrap();
        let stride = info.stride()[0] as usize;
        let mut n = 0;
        for y in rows.clone().filter(|&y| y < info.height() as usize) {
            for x in cols.clone().filter(|&x| x < info.width() as usize) {
                let px = &map.as_slice()[y * stride + x * 3..y * stride + x * 3 + 3];
                // BGR: žltá = málo modrej, veľa zelenej a červenej
                if px[0] < 100 && px[1] > 190 && px[2] > 190 {
                    n += 1;
                }
            }
        }
        out.push(n);
    }
    p.set_state(gst::State::Null).unwrap();
    out
}

#[test]
fn pocitanie_vysiela_celu_scenu_so_zltou_ciarou_len_v_pasme() {
    let Some(model) = model_path() else { return eprintln!("model chýba – test preskočený") };
    let _heavy = heavy();
    let dir = Dir::new("line");
    // test:ball je 640x360; výstup 320x180; čiara v strede (x=160), pásmo 20–80 % výšky (y=36..144)
    let mut s = Live::start(&[
        "--line", "0.5", "--line-span", "0.2,0.8", "--detect-width", "640",
        "--model", &model, "--out-dir", &dir.s(),
    ]);
    s.wait_ready();
    let url = s.url();

    let on_line = yellow_in(&url, 25, 157..164, 36..145);
    assert!(on_line.len() >= 25, "prišlo len {} snímok", on_line.len());
    assert!(on_line.iter().skip(5).all(|&n| n > 50), "čiara má byť v pásme viditeľná: {on_line:?}");

    let above = yellow_in(&url, 15, 157..164, 0..30);
    let below = yellow_in(&url, 15, 157..164, 150..180);
    assert!(above.iter().all(|&n| n == 0), "nad pásmom nemá byť čiara: {above:?}");
    assert!(below.iter().all(|&n| n == 0), "pod pásmom nemá byť čiara: {below:?}");
}

#[test]
fn pocitanie_vytvori_subory_s_poctami_aj_bez_divaka() {
    let Some(model) = model_path() else { return eprintln!("model chýba – test preskočený") };
    let _heavy = heavy();
    let dir = Dir::new("files");
    let mut s = Live::start(&["--line", "0.5", "--detect-width", "640", "--model", &model, "--out-dir", &dir.s()]);
    s.wait_ready();
    sleep(Duration::from_secs(2)); // nik sa nepozerá

    let counts = std::fs::read_to_string(format!("{}/live_counts.txt", dir.s())).expect("súhrn počtov");
    assert!(counts.contains("Autá: spolu 0") && counts.contains("Ľudia: spolu 0"), "{counts}");
    assert!(counts.contains("Počítanie od:"), "{counts}");

    let csv = std::fs::read_dir(dir.s())
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().ends_with("_events.csv"))
        .expect("CSV s prechodmi");
    let text = std::fs::read_to_string(csv.path()).unwrap();
    assert_eq!(text.lines().next().unwrap(), "cas,track_id,skupina,smer,spolu_auta,spolu_ludia");
    assert!(s.alive());
}

#[test]
fn pocitanie_bez_modelu_skonci_s_napovedou() {
    let dir = Dir::new("nomodel");
    let out = Command::new(env!("CARGO_BIN_EXE_object_detect_live"))
        .args(["--input", "test:ball", "--line", "0.5", "--model", "/neexistuje/m.onnx", "--out-dir", &dir.s()])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("yolox_s.onnx"));
}

#[test]
fn zla_ciara_alebo_pasmo_skonci_pred_spustenim() {
    for bad in [&["--line", "2"][..], &["--line", "0.5", "--line-span", "0.9,0.1"]] {
        let out = Command::new(env!("CARGO_BIN_EXE_object_detect_live"))
            .args(["--input", "test:ball"])
            .args(bad)
            .output()
            .unwrap();
        assert!(!out.status.success(), "{bad:?}");
    }
}
