//! Knižnica pre detekciu objektov (GStreamer + OpenCV). Binárka `object_detect` (main.rs)
//! je len tenký obal; takto sa dajú moduly testovať aj z priečinka `tests/`.
//!
//! Moduly:
//!   args      argumenty z príkazového riadku
//!   detector  detektory (OpenCV)
//!   draw      kreslenie rámčekov
//!   labels    názvy tried COCO
//!   rtsp_out  vlastný RTSP server s anotovaným videom (live)
//!   yolox     neurónová sieť YOLOX cez OpenCV DNN
//!   video_io  GStreamer vstup/výstup a prevod snímok GStreamer <-> OpenCV

pub mod args;
pub mod detector;
pub mod draw;
pub mod labels;
pub mod rtsp_out;
pub mod video_io;
pub mod yolox;

use std::error::Error;

/// Skrátený zápis: akákoľvek chyba (GStreamer, OpenCV, parsovanie) sa dá vrátiť cez `?`.
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
