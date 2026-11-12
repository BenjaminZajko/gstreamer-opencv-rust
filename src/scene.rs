//! Počítanie na celom zábere s ukotvenou čiarou – spoločné pre živý server (`object_detect_live`)
//! aj spracovanie súboru (`video_count`).
//!
//! Pre každú snímku:
//!   0. ukotvenie: kde je kamera voči referencii (čiara a výrez sa posunú so scénou),
//!   1. detekcia na výreze (YOLOX vidí len cestu, nie celý 4K záber), sledovanie a počítanie,
//!   2. kreslenie celého záberu s čiarou, výrezom, referenčnými bodmi a panelom.

use opencv::{
    core::{Mat, Rect, Size},
    imgproc,
    prelude::*,
};

use gstreamer_video as gst_video;

use crate::{
    anchor::{align_images, draw_points, rect_corners, render_reference, Anchor, AnchorConfig, AnchorMode, Homography},
    counter::{CountsView, CrossEvent, LineCounter},
    detector::Detector,
    draw::{self, Counted},
    live_count::{detect_in_roi, roi_from_quad, roi_rect, CountingCore},
    roi::Crop,
    speed::{GroundMapper, GroundModel},
    Result,
};

/// Nastavenie počítania.
#[derive(Debug, Clone)]
pub struct SceneConfig {
    /// poloha zvislej čiary (zlomok šírky záberu)
    pub line: f64,
    /// pásmo výšky, v ktorom čiara platí (zlomky výšky)
    pub line_span: (f64, f64),
    /// výrez, na ktorom beží detekcia (zlomky záberu)
    pub roi: Crop,
    /// šírka výstupného (zobrazeného) záberu v px
    pub out_width: i32,
    pub margin: f64,
    pub max_missed: u32,
    /// ukotvenie čiary k scéne (`None` = vypnuté)
    pub anchor: Option<AnchorConfig>,
    /// ukotvenie sa prepočíta každú N-tú snímku
    pub anchor_every: u32,
    /// model kamery na meranie rýchlosti (`None` = nemerať)
    pub ground: Option<GroundModel>,
    /// čistý obraz: kreslia sa len autá a ľudia s rýchlosťou (bez čiary, panelu a bodov);
    /// počítanie a záznam prechodov bežia ďalej
    pub clean: bool,
}

/// Výsledok spracovania jednej snímky.
pub struct Step {
    /// sledované objekty na vykreslenie (súradnice výstupného záberu)
    pub items: Vec<Counted>,
    /// nové prechody cez čiaru
    pub events: Vec<CrossEvent>,
    /// správy pre používateľa (stratené/obnovené ukotvenie, nový referenčný záber...)
    pub notes: Vec<String>,
    /// referenčná fotka s vybranými bodmi (vznikne raz, po kalibrácii)
    pub reference_photo: Option<Mat>,
    /// prepočítalo sa v tejto snímke ukotvenie?
    pub anchor_updated: bool,
    /// výrez v aktuálnej snímke (štvoruholník, ak sa kamera pootočila)
    pub quad: [(f64, f64); 4],
    /// zmenšená snímka, ak už vznikla (aby sa nerobila dvakrát)
    out_cache: Option<Mat>,
}

/// Počítadlo na celom zábere.
pub struct SceneCounter {
    detector: Detector,
    core: CountingCore,
    anchor: Option<Anchor>,
    anchor_every: u32,
    scale: f64,
    out_size: Size,
    info_out: gst_video::VideoInfo,
    roi_static: Rect,
    roi_ref_quad: [(f64, f64); 4],
    line_ref: ((f64, f64), (f64, f64)),
    ref_color: Option<Mat>,
    /// referenčné body z merania (model rýchlosti) v súradniciach referencie
    calib_pts: Vec<(u32, (f64, f64))>,
    prev_mode: AnchorMode,
    /// čistý obraz (pozri `SceneConfig::clean`)
    clean: bool,
    prev_keys: usize,
    was_tracking: bool,
    frames: u64,
}

impl SceneCounter {
    /// Pripraví počítanie podľa prvej snímky (veľká snímka, BGR). Vráti aj správy (napr. že ukotvenie
    /// sa nedalo zapnúť).
    pub fn new(detector: Detector, cfg: &SceneConfig, first: &Mat) -> Result<(Self, Vec<String>)> {
        let (hi_w, hi_h) = (first.cols(), first.rows());
        let out_w = cfg.out_width.min(hi_w);
        let out_h = ((hi_h as f64 * out_w as f64 / hi_w as f64) as i32 / 2) * 2;
        let out_size = Size::new(out_w, out_h);
        let scale = out_w as f64 / hi_w as f64;
        let info_out = gst_video::VideoInfo::builder(gst_video::VideoFormat::Bgr, out_w as u32, out_h as u32).build()?;
        let (y0, y1) = cfg.line_span;
        let roi_static = roi_rect(&cfg.roi, hi_w, hi_h);
        let mut notes = vec![];

        // referenčný záber pre ukotvenie (výstupné rozlíšenie, šedotónový)
        let (mut anchor, mut ref_color) = (None, None);
        if let Some(acfg) = cfg.anchor {
            let small = resize(first, out_size)?;
            let mut gray = Mat::default();
            imgproc::cvt_color_def(&small, &mut gray, imgproc::COLOR_BGR2GRAY)?;
            match Anchor::new(&gray, acfg) {
                Ok(a) => {
                    anchor = Some(a);
                    ref_color = Some(small);
                }
                Err(e) => notes.push(format!("UPOZORNENIE: ukotvenie je vypnuté: {e}")),
            }
        }

        let r = roi_static;
        let roi_ref_quad = rect_corners(r.x as f64 * scale, r.y as f64 * scale, r.width as f64 * scale, r.height as f64 * scale);
        let line_x = cfg.line * out_w as f64;
        let mut core = CountingCore::new(line_x, (y0 * out_h as f64, y1 * out_h as f64), cfg.margin, cfg.max_missed).with_strict_display();
        // rýchlosť: sledovanie beží v súradniciach referencie (= výstupný záber pri štarte),
        // model kamery je pre plné rozlíšenie snímky, z ktorej sa počítal
        let mut calib_pts = vec![];
        if let Some(m) = &cfg.ground {
            let mut mapper = GroundMapper::new(m.clone(), out_w as f64, out_h as f64)?;
            // kamera sa mohla od merania pootočiť: zarovnáme záber pri štarte so snímkou z merania
            if let Some(path) = &m.snapshot {
                let (h, note) = align_to_snapshot(first, path, out_size)?;
                mapper = mapper.with_alignment(h);
                notes.push(note);
            }
            calib_pts = mapper.points_in_ref();
            core = core.with_speed(mapper);
        }
        let sc = SceneCounter {
            detector,
            core,
            prev_mode: anchor.as_ref().map(|a| a.mode()).unwrap_or(AnchorMode::OrbOnly),
            anchor,
            anchor_every: cfg.anchor_every.max(1),
            scale,
            out_size,
            info_out,
            roi_static,
            roi_ref_quad,
            line_ref: ((line_x, y0 * out_h as f64), (line_x, y1 * out_h as f64)),
            ref_color,
            calib_pts,
            clean: cfg.clean,
            prev_keys: 0,
            was_tracking: true,
            frames: 0,
        };
        Ok((sc, notes))
    }

    pub fn info_out(&self) -> &gst_video::VideoInfo {
        &self.info_out
    }

    pub fn counter(&self) -> &LineCounter {
        self.core.counter()
    }

    /// Zapne záznam polôh objektov (pozri `CountingCore::enable_track_log`).
    pub fn enable_track_log(&mut self) {
        self.core.enable_track_log();
    }

    pub fn track_log_csv(&self) -> Option<&str> {
        self.core.track_log_csv()
    }

    pub fn anchor(&self) -> Option<&Anchor> {
        self.anchor.as_ref()
    }

    /// Čiara v súradniciach referencie (= výstupný záber pri štarte).
    pub fn line_ref(&self) -> ((f64, f64), (f64, f64)) {
        self.line_ref
    }

    /// Výrez na detekciu vo veľkej snímke (bez ukotvenia).
    pub fn roi_static(&self) -> Rect {
        self.roi_static
    }

    /// Koľko snímok sa už spracovalo.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Spracuje jednu snímku: ukotvenie, detekcia na výreze, sledovanie a počítanie.
    pub fn step(&mut self, frame_hi: &Mat, time_s: f64) -> Result<Step> {
        let (hi_w, hi_h) = (frame_hi.cols(), frame_hi.rows());
        let (mut notes, mut reference_photo, mut out_cache, mut anchor_updated) = (vec![], None, None, false);

        // 0) ukotvenie: kde je kamera voči referencii (každú N-tú snímku)
        if let Some(a) = self.anchor.as_mut() {
            if self.frames % self.anchor_every as u64 == 0 {
                let out = resize(frame_hi, self.out_size)?;
                let mut gray = Mat::default();
                imgproc::cvt_color_def(&out, &mut gray, imgproc::COLOR_BGR2GRAY)?;
                let ok = a.update(&gray)?;
                anchor_updated = true;
                if ok != self.was_tracking {
                    notes.push(if ok {
                        format!("ukotvenie obnovené ({} bodov), počítanie pokračuje", a.inliers())
                    } else {
                        "UKOTVENIE STRATENÉ (kamera sa otočila inam?), počítanie je pozastavené".into()
                    });
                    self.was_tracking = ok;
                }
                out_cache = Some(out);
                if a.mode() != self.prev_mode {
                    match a.mode() {
                        AnchorMode::Fine => {
                            if let Some(rc) = &self.ref_color {
                                reference_photo = Some(render_reference(rc, a.chosen_points(), self.line_ref, &self.roi_ref_quad)?);
                            }
                            notes.push(format!("ukotvenie: vybraných {} stabilných bodov", a.chosen_points().len()));
                        }
                        AnchorMode::OrbOnly => notes.push(
                            "UPOZORNENIE: nenašlo sa dosť stabilných bodov na jemné sledovanie, používa sa len ORB".into(),
                        ),
                        AnchorMode::Calibrating => {}
                    }
                    self.prev_mode = a.mode();
                }
                if a.keyframes() > self.prev_keys {
                    if self.prev_keys > 0 {
                        notes.push(format!(
                            "ukotvenie: kamera sa ustálila na novom mieste, nový referenčný záber (spolu {})",
                            a.keyframes()
                        ));
                    }
                    self.prev_keys = a.keyframes();
                }
            }
        }
        let tracking = self.anchor.as_ref().map(|a| a.tracking()).unwrap_or(true);

        // výrez v aktuálnej snímke (štvoruholník, ak sa kamera pootočila)
        let quad = match &self.anchor {
            Some(a) => self.roi_ref_quad.map(|p| a.to_current(p)),
            None => self.roi_ref_quad,
        };
        let roi_hi = if self.anchor.is_some() { roi_from_quad(&quad, self.scale, hi_w, hi_h) } else { self.roi_static };

        // 1) detekcia na výreze + počítanie (ak je poloha kamery stratená, počítanie stojí,
        //    aby sa nezrátalo nič nesprávne)
        let (items, events) = if tracking {
            let dets = detect_in_roi(&mut self.detector, frame_hi, roi_hi, self.scale)?;
            match &self.anchor {
                Some(a) => self.core.track_and_count_with(&dets, self.frames, time_s, |r| a.rect_to_reference(r)),
                None => self.core.track_and_count(&dets, self.frames, time_s),
            }
        } else {
            (vec![], vec![])
        };
        self.frames += 1;
        Ok(Step { items, events, notes, reference_photo, anchor_updated, quad, out_cache })
    }

    /// Nakreslí celý záber: výrez, referenčné body, čiaru (ukotvenú) a panel s počtami z `view`.
    /// `items` sú objekty na vykreslenie (môžu mať prečíslované ID).
    pub fn render(&self, frame_hi: &Mat, step: &mut Step, items: &[Counted], mut view: CountsView) -> Result<Mat> {
        let mut out = match step.out_cache.take() {
            Some(o) => o,
            None => resize(frame_hi, self.out_size)?,
        };
        if self.clean {
            draw::draw_objects(&mut out, items)?;
            return Ok(out);
        }
        match &self.anchor {
            Some(a) => {
                // čiara sa presunie spolu so scénou (môže byť naklonená)
                view.line_pts = Some((a.to_current(self.line_ref.0), a.to_current(self.line_ref.1)));
                if a.tracking() {
                    view.footer = format!("{} | REF {}", view.footer, a.status_text());
                } else {
                    view.footer = format!("{} | REF STRATENE", view.footer);
                    view.warning = true;
                }
                draw::draw_quad(&mut out, &step.quad)?;
                draw_points(&mut out, &a.points(40))?;
            }
            None => draw::draw_quad(&mut out, &step.quad)?,
        }
        // referenčné body z merania (1–8): idú so scénou rovnako ako čiara
        if !self.calib_pts.is_empty() {
            let pts: Vec<(u32, (f64, f64))> = match &self.anchor {
                Some(a) => self.calib_pts.iter().map(|&(id, p)| (id, a.to_current(p))).collect(),
                None => self.calib_pts.clone(),
            };
            draw::draw_numbered_points(&mut out, &pts)?;
        }
        draw::draw_counting(&mut out, items, &view)?;
        Ok(out)
    }
}

/// Šírka (px), na ktorej sa hľadá zarovnanie so snímkou z merania.
const ALIGN_MAX_W: f64 = 1280.0;

/// Zarovná prvú snímku so snímkou z merania. Vráti homografiu referencia -> snímka z merania
/// (v rozlíšení `size`) a správu pre používateľa. Ak sa zarovnať nedá, použije sa identita.
fn align_to_snapshot(first: &Mat, snapshot: &str, size: Size) -> Result<(Homography, String)> {
    let snap = opencv::imgcodecs::imread(snapshot, opencv::imgcodecs::IMREAD_COLOR)?;
    if snap.empty() {
        return Ok((Homography::IDENTITY, format!("UPOZORNENIE: snímku z merania '{snapshot}' sa nepodarilo načítať, model rýchlosti sa nezarovnal")));
    }
    // ORB na zmenšenom zábere (najviac 1280 px) nájde viac stabilných zhôd než v plnom rozlíšení
    // (pri inom osvetlení ich v plnom rozlíšení býva tesne pod hranicou)
    let k = (ALIGN_MAX_W / size.width as f64).min(1.0);
    let small = Size::new((size.width as f64 * k).round() as i32, (size.height as f64 * k).round() as i32);
    let gray = |img: &Mat| -> Result<Mat> {
        let mut g = Mat::default();
        imgproc::cvt_color_def(&resize(img, small)?, &mut g, imgproc::COLOR_BGR2GRAY)?;
        Ok(g)
    };
    let (a, b) = (gray(first)?, gray(&snap)?);
    // robí sa raz pri štarte, takže sa oplatí skúsiť viac bodov a vziať výsledok s najviac
    // zhodami (pri 3000 bodoch býva zhôd tesne nad minimom a zarovnanie je nepresné)
    let mut found: Option<(Homography, usize)> = None;
    for features in [3000, 12000] {
        if let Some(r) = align_images(&a, &b, features, 25)? {
            if found.as_ref().is_none_or(|f| r.1 > f.1) {
                found = Some(r);
            }
        }
    }
    // homografia zo zmenšeného rozlíšenia späť do `size`
    let scale = |hm: Homography| {
        Homography([1.0 / k, 0.0, 0.0, 0.0, 1.0 / k, 0.0, 0.0, 0.0, 1.0])
            .compose(&hm)
            .compose(&Homography([k, 0.0, 0.0, 0.0, k, 0.0, 0.0, 0.0, 1.0]))
    };
    let (w, h) = (size.width as f64, size.height as f64);
    Ok(match found.map(|(hm, n)| (scale(hm), n)) {
        Some((hom, n)) => {
            // priemerný posun vo vnútri záberu (rohy obrazu by ho zveličili)
            let grid: Vec<(f64, f64)> =
                [0.25, 0.5, 0.75].iter().flat_map(|&fx| [0.25, 0.5, 0.75].map(|fy| (fx * w, fy * h))).collect();
            let shift = grid.iter().map(|&p| { let q = hom.apply(p); ((q.0 - p.0).powi(2) + (q.1 - p.1).powi(2)).sqrt() }).sum::<f64>() / grid.len() as f64;
            (hom, format!("model rýchlosti zarovnaný so snímkou z merania ({n} zhodných bodov, kamera posunutá o {shift:.1} px)"))
        }
        None => (
            Homography::IDENTITY,
            "UPOZORNENIE: záber sa nepodarilo zarovnať so snímkou z merania (kamera sa veľmi otočila?), rýchlosť môže byť nepresná".into(),
        ),
    })
}

fn resize(img: &Mat, size: Size) -> Result<Mat> {
    let mut out = Mat::default();
    imgproc::resize(img, &mut out, size, 0.0, 0.0, imgproc::INTER_AREA)?;
    Ok(out)
}
