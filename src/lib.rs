//! Knižnica pre detekciu objektov (GStreamer + OpenCV). Binárka `object_detect` (main.rs)
//! je len tenký obal; takto sa dajú moduly testovať aj z priečinka `tests/`.
//!
//! Moduly:
//!   anchor    ukotvenie čiary a výrezu k scéne (sledovanie referenčných bodov)
//!   args      argumenty z príkazového riadku
//!   detector  detektory (OpenCV)
//!   draw      kreslenie rámčekov
//!   labels    názvy tried COCO
//!   live_count  živé počítanie (výrez pre detekciu, sledovanie, čiara)
//!   roi       orezanie záberu (región záujmu)
//!   tracker   sledovanie objektov medzi snímkami (ID)
//!   counter   počítanie prechodov cez zvislú čiaru
//!   rtsp_out  vlastný RTSP server s anotovaným videom (live)
//!   scene     počítanie na celom zábere s ukotvenou čiarou (live aj súbor)
//!   yolox     neurónová sieť YOLOX cez OpenCV DNN
//!   video_io  GStreamer vstup/výstup a prevod snímok GStreamer <-> OpenCV

pub mod anchor;
pub mod args;
pub mod counter;
pub mod detector;
pub mod draw;
pub mod labels;
pub mod live_count;
pub mod roi;
pub mod rtsp_out;
pub mod scene;
pub mod tracker;
pub mod video_io;
pub mod yolox;

use std::error::Error;

/// Skrátený zápis: akákoľvek chyba (GStreamer, OpenCV, parsovanie) sa dá vrátiť cez `?`.
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
