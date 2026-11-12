//! Živé počítanie: spoločná logika pre `object_detect_live --line ...`.
//!
//! Detekcia beží na výreze (ROI) z veľkej snímky (napr. 2560 px široká), aby sa zachovali detaily,
//! ale zobrazuje sa celý záber zmenšený na výstupnú šírku. Súradnice detekcií sa preto prepočítavajú:
//! z výrezu do veľkej snímky a odtiaľ do výstupnej.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use opencv::{
    core::{Mat, Rect},
    prelude::*,
};

use crate::{
    counter::{group_of, CountsView, CrossEvent, Direction, Group, LineCounter},
    detector::{Detection, Detector},
    draw::Counted,
    roi::Crop,
    speed::{GroundMapper, SpeedTracker, WINDOW_S},
    tracker::Tracker,
    Result,
};

/// Koľko snímok sa objekt ešte kreslí na poslednom mieste, keď ho detektor na chvíľu nenájde
/// (aby rámček neblikal). Pri 20 fps je to 0,4 s.
pub const COAST_FRAMES: u32 = 8;

/// Autá sa berú až od tejto istoty, aj keď je `--conf` nižšie (ľudia sú malí a potrebujú nižší
/// prah, autá sú veľké; pri nízkom prahu sieť občas označí oblúk alebo plot s autom za ním ako
/// jedno veľké auto).
pub const VEHICLE_MIN_CONF: f32 = 0.4;

/// "Vozidlo", ktoré sa od objavenia ani raz nepohlo, sa po tomto čase (s) prestane kresliť:
/// sú to najmä omyly siete (elektrická skrinka ako "truck") a autá zaparkované od začiatku.
/// Auto, ktoré prišlo a zastavilo, sa kreslí ďalej (pohlo sa).
pub const STATIC_HIDE_S: f64 = 2.0;

/// Objekt sa kreslí až po toľkých detekciách (odfiltruje jednorazové omyly siete, napr. "auto"
/// na kúsku autobusu v jednej snímke).
pub const MIN_HITS: u32 = 3;

/// Sledovanie + počítanie (čistá logika bez videa a modelu – dobre sa testuje).
pub struct CountingCore {
    tracker: Tracker,
    counter: LineCounter,
    /// meranie rýchlosti (len ak je zadaný model kamery)
    speed: Option<(GroundMapper, SpeedTracker)>,
    /// záznam polôh objektov (CSV riadky), ak je zapnutý – na neskoršiu analýzu rýchlosti
    track_log: Option<String>,
    /// prechody, ktorým sa ešte dopočítava rýchlosť: (poradie v udalostiach, ID objektu, do kedy)
    pending: Vec<(usize, u32, f64)>,
    /// posledný vykreslený stav každého objektu (na kreslenie, keď sa na chvíľu stratí)
    last_items: HashMap<u32, Counted>,
    /// pohyb objektov: (čas objavenia, prvý stred v referencii, pohol sa už?, počet detekcií)
    motion: HashMap<u32, (f64, (f64, f64), bool, u32)>,
    /// prísnejšie kreslenie (pozri `with_strict_display`)
    strict: bool,
}

impl CountingCore {
    /// `line_x` a `span` sú v pixeloch výstupného záberu.
    pub fn new(line_x: f64, span: (f64, f64), margin: f64, max_missed: u32) -> Self {
        Self {
            tracker: Tracker::new(max_missed),
            counter: LineCounter::new(line_x, margin).with_span(span.0, span.1),
            speed: None,
            track_log: None,
            pending: vec![],
            last_items: HashMap::new(),
            motion: HashMap::new(),
            strict: false,
        }
    }

    /// Prísnejšie kreslenie pre živý obraz: objekt až po MIN_HITS detekciách a "človek" až keď
    /// sa pohne. Počítanie a meranie to nemení.
    pub fn with_strict_display(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Zapne záznam polôh sledovaných objektov (každá snímka, každý objekt).
    pub fn enable_track_log(&mut self) {
        self.track_log = Some(String::from("frame,time_s,track_id,group,x,y,w,h,foot_u,foot_v,ground_x,ground_y,speed_kmh\n"));
    }

    /// Záznam polôh ako CSV (x,y,w,h v súradniciach referencie; foot_u/v v px snímky z merania).
    pub fn track_log_csv(&self) -> Option<&str> {
        self.track_log.as_deref()
    }

    /// Zapne meranie rýchlosti: `mapper` prevádza súradnice sledovania na metre na zemi.
    pub fn with_speed(mut self, mapper: GroundMapper) -> Self {
        self.speed = Some((mapper, SpeedTracker::new()));
        self
    }

    pub fn counter(&self) -> &LineCounter {
        &self.counter
    }

    /// Spracuje detekcie jednej snímky: nepočítané triedy zahodí, ostatným pridelí ID a zaznamená
    /// prechody cez čiaru. Vráti objekty na vykreslenie a nové prechody.
    pub fn track_and_count(
        &mut self,
        dets: &[Detection],
        frame_idx: u64,
        time_s: f64,
    ) -> (Vec<Counted>, Vec<CrossEvent>) {
        self.track_and_count_with(dets, frame_idx, time_s, |r| r)
    }

    /// Jazdec na motorke ostane jedným objektom, aj keď ho sieť chvíľu vidí len ako človeka
    /// (motorku zakryje múr / oblúk) alebo len ako motorku:
    ///  - "človek" na mieste, kde sa práve pohybuje sledovaná motorka, je jej jazdec – dostane
    ///    názov "motorcycle" a pokračuje tá istá stopa,
    ///  - keď sa motorka (s jazdcom) objaví na mieste sledovaného "človeka", stopa sa preradí
    ///    medzi vozidlá (ID ostane).
    fn stick_riders(&mut self, dets: &[Detection], to_ref: &impl Fn(Rect) -> Rect) -> Vec<Detection> {
        let mut out = dets.to_vec();
        let near = |r: Rect, t: Rect| {
            let overlap = (r.x + r.width).min(t.x + t.width) - r.x.max(t.x);
            let cy = r.y + r.height / 2;
            overlap as f64 >= 0.4 * r.width.min(t.width) as f64 && cy >= t.y - t.height && cy <= t.y + 2 * t.height
        };
        // pohybujúce sa motorky, ktoré sa nedávno videli
        let bikes: Vec<Rect> = self
            .tracker
            .tracks()
            .iter()
            .filter(|t| t.group == Group::Vehicle && t.missed <= COAST_FRAMES)
            .filter(|t| self.last_items.get(&t.id).is_some_and(|it| it.label == "motorcycle"))
            .filter(|t| self.motion.get(&t.id).is_some_and(|m| m.2))
            .map(|t| {
                let k = (t.missed + 1) as f64;
                let mut r = t.rect;
                r.x += (t.vel.0 * k).round() as i32;
                r.y += (t.vel.1 * k).round() as i32;
                r
            })
            .collect();
        for d in out.iter_mut() {
            if d.label == "person" && bikes.iter().any(|&b| near(to_ref(d.rect), b)) {
                d.label = "motorcycle".into();
            }
        }
        // motorka na mieste sledovaného človeka: ten človek je jej jazdec
        let people: Vec<(u32, Rect)> = self
            .tracker
            .tracks()
            .iter()
            .filter(|t| t.group == Group::Person && t.missed <= 2)
            .map(|t| (t.id, t.rect))
            .collect();
        for d in out.iter().filter(|d| d.label == "motorcycle" && d.score.is_none_or(|s| s >= VEHICLE_MIN_CONF)) {
            let r = to_ref(d.rect);
            for &(id, p) in &people {
                if near(p, r) {
                    self.tracker.set_group(id, Group::Vehicle);
                }
            }
        }
        out
    }

    /// To isté, ale sledovanie a počítanie prebieha v iných súradniciach než zobrazenie:
    /// `to_ref` prevedie rámček z aktuálnej snímky do súradníc referencie (kam je ukotvená
    /// čiara). Vrátené objekty majú pôvodný rámček (kvôli kresleniu).
    pub fn track_and_count_with(
        &mut self,
        dets: &[Detection],
        frame_idx: u64,
        time_s: f64,
        to_ref: impl Fn(Rect) -> Rect,
    ) -> (Vec<Counted>, Vec<CrossEvent>) {
        let dets = self.stick_riders(dets, &to_ref);
        let counted: Vec<_> = dets
            .iter()
            .filter_map(|d| group_of(&d.label).map(|g| (d, g)))
            .filter(|(d, g)| *g != Group::Vehicle || d.score.is_none_or(|s| s >= VEHICLE_MIN_CONF))
            .map(|(d, g)| (d, g, to_ref(d.rect)))
            .collect();
        let update = self.tracker.update(&counted.iter().map(|(_, g, r)| (*r, *g)).collect::<Vec<_>>());

        let (mut items, mut events) = (Vec::with_capacity(counted.len()), vec![]);
        for ((d, group, rref), id) in counted.into_iter().zip(update.ids) {
            let cx = rref.x as f64 + rref.width as f64 / 2.0;
            let cy = rref.y as f64 + rref.height as f64 / 2.0;
            // pohol sa objekt od objavenia aspoň o polovicu svojej šírky?
            let m = self.motion.entry(id).or_insert((time_s, (cx, cy), false, 0));
            m.3 += 1;
            if !m.2 && ((cx - m.1.0).powi(2) + (cy - m.1.1).powi(2)).sqrt() > 0.5 * rref.width.max(10) as f64 {
                m.2 = true;
            }
            // rýchlosť z polohy na zemi pod objektom (spodok rámčeka v referenčných súradniciach)
            // (ľavý aj pravý okraj, pozri SpeedTracker)
            let speed_kmh = self.speed.as_mut().and_then(|(mapper, st)| {
                let (l, r) = mapper.foot_edges(rref.x as f64, rref.y as f64, rref.width as f64, rref.height as f64)?;
                st.observe_box(id, time_s, l, r, rref.height as f64)
            });
            // prechod tohto objektu ešte čaká na rýchlosť: spresníme ju
            if speed_kmh.is_some() {
                for &(idx, pid, _) in &self.pending {
                    if pid == id {
                        self.counter.set_event_speed(idx, speed_kmh);
                    }
                }
            }
            if let Some(log) = self.track_log.as_mut() {
                let (fu, fv, g) = match &self.speed {
                    Some((m, _)) => {
                        let (u, v) = m.foot_px(rref.x as f64, rref.y as f64, rref.width as f64, rref.height as f64);
                        (format!("{u:.1}"), format!("{v:.1}"), m.foot(rref.x as f64, rref.y as f64, rref.width as f64, rref.height as f64))
                    }
                    None => (String::new(), String::new(), None),
                };
                let (gx, gy) = g.map(|g| (format!("{:.3}", g.0), format!("{:.3}", g.1))).unwrap_or_default();
                let sp = speed_kmh.map(|v| format!("{v:.2}")).unwrap_or_default();
                *log += &format!(
                    "{frame_idx},{time_s:.3},{id},{},{},{},{},{},{fu},{fv},{gx},{gy},{sp}\n",
                    group.name(), rref.x, rref.y, rref.width, rref.height
                );
            }
            if let Some(mut ev) = self.counter.observe_xy(frame_idx, time_s, id, group, cx, cy) {
                self.counter.set_last_speed(speed_kmh);
                ev.speed_kmh = speed_kmh;
                if self.speed.is_some() {
                    // rýchlosť sa ešte pol okna spresňuje (aby bola z obdobia okolo čiary, nie len
                    // pred ňou); udalosť sa vráti, keď je hotová
                    self.pending.push((self.counter.events().len() - 1, id, time_s + WINDOW_S / 2.0));
                } else {
                    events.push(ev);
                }
            }
            items.push(Counted { rect: d.rect, label: d.label.clone(), score: d.score, id, group, speed_kmh });
        }
        // objekty, ktoré detektor v tejto snímke nenašiel, ale tracker ich ešte drží:
        // nakreslia sa na poslednom mieste (rámček neseká), najviac COAST_FRAMES snímok
        for it in &items {
            self.last_items.insert(it.id, it.clone());
        }
        for t in self.tracker.tracks() {
            if t.missed > 0 && t.missed <= COAST_FRAMES {
                if let Some(it) = self.last_items.get(&t.id) {
                    // posunie sa tam, kam sa objekt mal medzitým dostať (napr. za oblúkom)
                    let mut it = it.clone();
                    it.rect.x += (t.vel.0 * t.missed as f64).round() as i32;
                    it.rect.y += (t.vel.1 * t.missed as f64).round() as i32;
                    items.push(it);
                }
            }
        }
        for id in &update.ended {
            self.last_items.remove(id);
            self.motion.remove(id);
        }
        // čo sa nekreslí (počítanie tým nie je dotknuté):
        //  - objekt nájdený menej než MIN_HITS-krát (jednorazové omyly siete),
        //  - "vozidlo", ktoré sa od objavenia nepohlo (pozri STATIC_HIDE_S),
        //  - "človek", ktorý sa ešte nepohol (ozdoby brány, socha, skrinka...).
        //  (prvé a tretie len pri `with_strict_display`)
        let (motion, strict) = (&self.motion, self.strict);
        items.retain(|it| {
            motion.get(&it.id).is_none_or(|&(t0, _, moved, hits)| {
                (!strict || hits >= MIN_HITS)
                    && match it.group {
                        Group::Vehicle => moved || time_s - t0 < STATIC_HIDE_S,
                        Group::Person => moved || !strict,
                    }
            })
        });

        // hotové prechody: uplynul čas alebo objekt zmizol
        let counter = &self.counter;
        self.pending.retain(|&(idx, pid, until)| {
            let done = time_s >= until || update.ended.contains(&pid);
            if done {
                events.push(counter.events()[idx].clone());
            }
            !done
        });
        for id in update.ended {
            self.counter.forget(id);
            if let Some((_, st)) = self.speed.as_mut() {
                st.forget(id);
            }
        }
        (items, events)
    }
}

/// Relácia diváka: čo vidí ten, kto práve zapol stream. Začína pripojením prvého diváka.
///
/// Celkové počty (od spustenia servera) bežia ďalej v `LineCounter` a ukladajú sa do súborov;
/// relácia z nich len odpočíta počty z čias pred pripojením, a objektom prideľuje čísla od 1
/// podľa toho, ako ich divák videl (zvlášť pre autá a ľudí: "car #1", "car #2", "person #1"...).
pub struct Session {
    baseline: HashMap<(Group, Direction), u32>,
    ids: HashMap<u32, u32>,
    next: HashMap<Group, u32>,
    started: Instant,
}

impl Session {
    /// Nová relácia: počty v paneli začnú od nuly (teraz).
    pub fn start(counter: &LineCounter) -> Self {
        Self {
            baseline: counter.counts_snapshot(),
            ids: HashMap::new(),
            next: HashMap::new(),
            started: Instant::now(),
        }
    }

    /// Číslo objektu v rámci relácie: prvé zobrazené auto je 1, ďalšie 2... (ten istý objekt
    /// má stále to isté číslo).
    pub fn display_id(&mut self, global_id: u32, group: Group) -> u32 {
        if let Some(&id) = self.ids.get(&global_id) {
            return id;
        }
        let n = self.next.entry(group).or_insert(0);
        *n += 1;
        self.ids.insert(global_id, *n);
        *n
    }

    /// Prepíše ID objektov na čísla v rámci relácie.
    pub fn relabel(&mut self, items: Vec<Counted>) -> Vec<Counted> {
        items
            .into_iter()
            .map(|mut it| {
                it.id = self.display_id(it.id, it.group);
                it
            })
            .collect()
    }

    /// Počty pre panel: celkové mínus to, čo bolo pred pripojením, a čas od zapnutia.
    pub fn view(&self, counter: &LineCounter) -> CountsView {
        self.view_at(counter, self.started.elapsed())
    }

    /// To isté s explicitným časom od zapnutia (kvôli testom).
    pub fn view_at(&self, counter: &LineCounter, elapsed: Duration) -> CountsView {
        let counts = counter
            .counts_snapshot()
            .into_iter()
            .map(|(k, n)| (k, n.saturating_sub(self.baseline.get(&k).copied().unwrap_or(0))))
            .collect();
        CountsView::new(counter.line_x(), counter.span(), counts).with_footer(format_elapsed(elapsed))
    }
}

/// "od zapnutia 1:05" (bez diakritiky, písmo OpenCV ju nevykreslí).
pub fn format_elapsed(d: Duration) -> String {
    let s = d.as_secs();
    format!("od zapnutia {}:{:02}", s / 60, s % 60)
}

/// Výrez (ROI) v pixeloch veľkej snímky `w` x `h`.
pub fn roi_rect(roi: &Crop, w: i32, h: i32) -> Rect {
    let x = (roi.x * w as f64).round() as i32;
    let y = (roi.y * h as f64).round() as i32;
    let rw = ((roi.w * w as f64).round() as i32).min(w - x).max(1);
    let rh = ((roi.h * h as f64).round() as i32).min(h - y).max(1);
    Rect::new(x, y, rw, rh)
}

/// Výrez na detekciu zo štvoruholníka v súradniciach výstupného záberu (po posune/otočení kamery):
/// najmenší obdĺžnik, ktorý ho obsahuje, orezaný do obrazu a prepočítaný do veľkej snímky
/// (`scale` = výstup / veľká snímka). Aspoň 32 x 32 px.
pub fn roi_from_quad(quad: &[(f64, f64); 4], scale: f64, hi_w: i32, hi_h: i32) -> Rect {
    let (x0, y0, x1, y1) = crate::anchor::bounding_box(quad);
    let (x0, y0, x1, y1) = (x0 / scale, y0 / scale, x1 / scale, y1 / scale);
    let x = (x0.floor().max(0.0) as i32).min(hi_w - 32).max(0);
    let y = (y0.floor().max(0.0) as i32).min(hi_h - 32).max(0);
    let w = ((x1.ceil() as i32).min(hi_w) - x).max(32).min(hi_w - x);
    let h = ((y1.ceil() as i32).min(hi_h) - y).max(32).min(hi_h - y);
    Rect::new(x, y, w, h)
}

/// Rámček z výrezu -> súradnice výstupného záberu (posun o začiatok výrezu a zmenšenie `scale`).
pub fn to_output_coords(r: Rect, roi: Rect, scale: f64) -> Rect {
    Rect::new(
        ((r.x + roi.x) as f64 * scale).round() as i32,
        ((r.y + roi.y) as f64 * scale).round() as i32,
        ((r.width as f64 * scale).round() as i32).max(1),
        ((r.height as f64 * scale).round() as i32).max(1),
    )
}

/// Spustí detektor len na výreze veľkej snímky a vráti detekcie v súradniciach výstupu.
pub fn detect_in_roi(detector: &mut Detector, frame_hi: &Mat, roi: Rect, scale: f64) -> Result<Vec<Detection>> {
    let parts = tiles(roi);
    let mut all = vec![];
    for (i, tile) in parts.iter().enumerate() {
        // časť sa skopíruje, aby bola súvislá v pamäti
        let view = Mat::roi(frame_hi, *tile)?.try_clone()?;
        for d in detector.detect(&view)? {
            // rámček odrezaný okrajom časti (nie okrajom celého výrezu) – môže byť len kúsok objektu
            let cut_left = i > 0 && d.rect.x <= EDGE_PX;
            let cut_right = i + 1 < parts.len() && d.rect.x + d.rect.width >= tile.width - EDGE_PX;
            all.push(TileDet { det: Detection { rect: to_output_coords(d.rect, *tile, scale), ..d }, cut_left, cut_right });
        }
    }
    Ok(merge_riders(merge_tiles(all)))
}

/// Jazdec na motorke / mopede: sieť nájde "person" aj "motorcycle" – to je jeden objekt.
/// Človek, ktorý sedí na motorke (prekrývajú sa vodorovne a spodok človeka je v motorke),
/// sa spojí s ňou do jedného rámčeka "motorcycle" (vozidlo, rýchlosť od kolies).
/// Motorka sa spojí aj s nízkou istotou – jazdec nad ňou ju potvrdzuje; spojený objekt dostane
/// vyššiu z oboch istôt. Motorka bez jazdca s nízkou istotou sa neskôr zahodí (VEHICLE_MIN_CONF).
pub fn merge_riders(dets: Vec<Detection>) -> Vec<Detection> {
    let is_bike = |d: &Detection| d.label == "motorcycle";
    let mut out: Vec<Detection> = vec![];
    let mut riders = vec![false; dets.len()];
    let bikes: Vec<usize> = (0..dets.len()).filter(|&i| is_bike(&dets[i])).collect();
    for &b in &bikes {
        let m = dets[b].rect;
        for (i, p) in dets.iter().enumerate() {
            if riders[i] || p.label != "person" {
                continue;
            }
            let r = p.rect;
            let overlap = (r.x + r.width).min(m.x + m.width) - r.x.max(m.x);
            let bottom = r.y + r.height;
            let sits = bottom >= m.y + m.height / 4 && bottom <= m.y + m.height + m.height / 2;
            if overlap as f64 >= 0.4 * r.width as f64 && sits {
                riders[i] = true;
            }
        }
    }
    for (i, d) in dets.iter().enumerate() {
        if riders[i] {
            continue;
        }
        if bikes.contains(&i) {
            // rámček cez motorku aj jazdcov
            let mut rect = d.rect;
            let mut score = d.score;
            for (j, p) in dets.iter().enumerate() {
                if riders[j] {
                    let m = d.rect;
                    let r = p.rect;
                    let overlap = (r.x + r.width).min(m.x + m.width) - r.x.max(m.x);
                    if overlap as f64 >= 0.4 * r.width as f64 {
                        rect = union(rect, r);
                        score = match (score, p.score) {
                            (Some(a), Some(b)) => Some(a.max(b)),
                            (a, b) => a.or(b),
                        };
                    }
                }
            }
            out.push(Detection { rect, label: d.label.clone(), score });
        } else {
            out.push(d.clone());
        }
    }
    out
}

/// Ako blízko okraja časti (px) sa rámček považuje za odrezaný.
const EDGE_PX: i32 = 6;

/// Detekcia z jednej časti výrezu: či ju odrezal ľavý / pravý okraj časti.
#[derive(Debug, Clone)]
pub struct TileDet {
    pub det: Detection,
    pub cut_left: bool,
    pub cut_right: bool,
}

/// Široký výrez sa rozdelí na prekrývajúce sa časti: sieť každý vstup zmenší na 640 px, takže
/// pás cez celú šírku 4K záberu by zmenšila 4x a ľudia by mali len pár desiatok px. Časti majú
/// šírku 2,5x výšky a prekryv aspoň 1,2x výšky.
pub fn tiles(roi: Rect) -> Vec<Rect> {
    let tile_w = ((roi.height as f64 * 2.5) as i32).max(1);
    if roi.width <= tile_w + tile_w / 4 {
        return vec![roi];
    }
    let min_overlap = (roi.height as f64 * 1.2) as i32;
    let stride_max = (tile_w - min_overlap).max(1);
    let n = ((roi.width - tile_w) + stride_max - 1) / stride_max + 1;
    let stride = (roi.width - tile_w) as f64 / (n - 1) as f64;
    (0..n).map(|i| Rect::new(roi.x + (i as f64 * stride).round() as i32, roi.y, tile_w, roi.height)).collect()
}

fn union(a: Rect, b: Rect) -> Rect {
    let (x0, y0) = (a.x.min(b.x), a.y.min(b.y));
    let (x1, y1) = ((a.x + a.width).max(b.x + b.width), (a.y + a.height).max(b.y + b.height));
    Rect::new(x0, y0, x1 - x0, y1 - y0)
}

/// Spojí detekcie z prekrývajúcich sa častí:
///  1. kúsky toho istého objektu odrezané hranicou častí (vľavo a vpravo od nej, v rovnakej
///     výške, prekrývajú sa) sa zlepia do jedného rámčeka,
///  2. ten istý objekt nájdený vo viacerých častiach sa nechá raz (väčší rámček).
pub fn merge_tiles(dets: Vec<TileDet>) -> Vec<Detection> {
    // 1) zlepovanie odrezaných kúskov
    let mut items: Vec<(Rect, Detection, bool)> = dets.into_iter().map(|t| (t.det.rect, t.det, t.cut_left || t.cut_right)).collect();
    let mut changed = true;
    while changed {
        changed = false;
        'outer: for i in 0..items.len() {
            for j in 0..items.len() {
                if i == j || group_of(&items[i].1.label) != group_of(&items[j].1.label) || !(items[i].2 || items[j].2) {
                    continue;
                }
                let (a, b) = (items[i].0, items[j].0);
                let inter = a & b;
                let v_overlap = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);
                // kúsky toho istého objektu majú podobnú výšku a sú v rovnakej výške
                let (hmin, hmax) = (a.height.min(b.height) as f64, a.height.max(b.height) as f64);
                if inter.width > 0 && v_overlap as f64 > 0.7 * hmax && hmin > 0.75 * hmax {
                    let u = union(a, b);
                    let score = match (items[i].1.score, items[j].1.score) {
                        (Some(x), Some(y)) => Some(x.max(y)),
                        (x, y) => x.or(y),
                    };
                    // názov podľa väčšieho kúska (bus rozdelený na "bus" a "truck" ostane "bus")
                    let label = if a.area() >= b.area() { items[i].1.label.clone() } else { items[j].1.label.clone() };
                    let merged = Detection { rect: u, label, score };
                    // zlepený je stále "odrezaný", kým sa nevyrieši (môže pribudnúť ďalší kúsok)
                    let cut = items[i].2 && items[j].2;
                    let (lo, hi) = (i.min(j), i.max(j));
                    items.remove(hi);
                    items[lo] = (u, merged, cut);
                    changed = true;
                    break 'outer;
                }
            }
        }
    }
    // 2) duplicity: nechá sa väčší rámček. Porovnáva sa skupina, nie trieda: ten istý autobus
    //    býva raz "bus", raz "truck", a kúsky autobusu (okná, kolesá) sieť označí ako "car".
    let mut dets: Vec<Detection> = items.into_iter().map(|(_, d, _)| d).collect();
    dets.sort_by_key(|d| std::cmp::Reverse(d.rect.area()));
    let mut kept: Vec<Detection> = vec![];
    for d in dets {
        let dup = kept.iter().any(|k| {
            if group_of(&k.label) != group_of(&d.label) {
                return false;
            }
            let inter = (k.rect & d.rect).area() as f64;
            let iou = inter / ((k.rect.area() + d.rect.area()) as f64 - inter);
            iou > 0.5 || inter / d.rect.area().max(1) as f64 > 0.7
        });
        if !dup {
            kept.push(d);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::{Direction, Group};

    fn det(label: &str, x: i32, y: i32) -> Detection {
        Detection { rect: Rect::new(x, y, 100, 60), label: label.into(), score: Some(0.9) }
    }

    #[test]
    fn prisne_kreslenie_az_po_troch_detekciach() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 10.0, 5).with_strict_display();
        let mut shown = vec![];
        for f in 0..4u64 {
            let (items, _) = core.track_and_count(&[det("car", 100 + 30 * f as i32, 200)], f, f as f64 / 20.0);
            shown.push(items.len());
        }
        assert_eq!(shown, vec![0, 0, 1, 1]);
    }

    #[test]
    fn prisne_kreslenie_nehybny_clovek_sa_nekresli() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 10.0, 5).with_strict_display();
        // "človek" na bráne stojí na mieste: nikdy sa nenakreslí
        for f in 0..20u64 {
            let (items, _) = core.track_and_count(&[det("person", 300, 200)], f, f as f64 / 20.0);
            assert!(items.is_empty(), "snímka {f}");
        }
        // chodec sa pohne: nakreslí sa
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 10.0, 5).with_strict_display();
        let last = (0..10u64).map(|f| core.track_and_count(&[det("person", 100 + 20 * f as i32, 200)], f, f as f64 / 20.0).0.len()).last();
        assert_eq!(last, Some(1));
    }

    #[test]
    fn autobus_z_dvoch_casti_s_inou_triedou_sa_zlepi_a_auta_v_nom_zmiznu() {
        let t = |label: &str, x, w, cut_left, cut_right| TileDet {
            det: Detection { rect: Rect::new(x, 100, w, 200), label: label.into(), score: Some(0.8) },
            cut_left,
            cut_right,
        };
        let out = merge_tiles(vec![
            t("bus", 0, 600, false, true),     // ľavá časť autobusu, odrezaná pravým okrajom časti
            t("truck", 500, 400, true, false), // pravá časť, sieť ju nazvala inak
            t("car", 650, 150, false, false),  // okno autobusu ako "auto"
        ]);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!((out[0].label.as_str(), out[0].rect.x, out[0].rect.width), ("bus", 0, 900));
    }

    #[test]
    fn jazdec_na_mopede_je_jeden_objekt() {
        let d = |label: &str, x, y, w, h, s| Detection { rect: Rect::new(x, y, w, h), label: label.into(), score: Some(s) };
        // moped (spodok) a jazdec nad ním, prekrývajú sa
        let out = merge_riders(vec![d("motorcycle", 100, 160, 90, 60, 0.7), d("person", 115, 100, 50, 100, 0.9), d("person", 400, 100, 40, 110, 0.8)]);
        assert_eq!(out.len(), 2, "{out:?}");
        let bike = out.iter().find(|o| o.label == "motorcycle").unwrap();
        assert_eq!((bike.rect.y, bike.rect.y + bike.rect.height), (100, 220));
        assert!(out.iter().any(|o| o.label == "person" && o.rect.x == 400), "chodec vedľa ostane");
        // motorka s nízkou istotou sa s jazdcom spojí a dostane jeho istotu
        let out = merge_riders(vec![d("motorcycle", 100, 160, 90, 60, 0.2), d("person", 115, 100, 50, 100, 0.9)]);
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].label.as_str(), out[0].score), ("motorcycle", Some(0.9)));
        // človek stojaci za motorkou (spodok vysoko nad ňou) sa nespája
        let out = merge_riders(vec![d("motorcycle", 100, 300, 90, 60, 0.7), d("person", 115, 100, 50, 100, 0.9)]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn jazdec_za_mursom_ostane_motorkou() {
        // moped ide doprava; po pár snímkach z neho sieť vidí len jazdca (motorku zakryl múr)
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 10.0, 8);
        let d = |label: &str, x: i32, y: i32, w, h| Detection { rect: Rect::new(x, y, w, h), label: label.into(), score: Some(0.9) };
        let mut ids = std::collections::HashSet::new();
        for f in 0..30u64 {
            let x = 100 + 15 * f as i32;
            let det = if f < 10 || f > 20 { d("motorcycle", x, 100, 90, 120) } else { d("person", x + 20, 100, 50, 70) };
            let (items, _) = core.track_and_count(&[det], f, f as f64 / 20.0);
            for it in items {
                assert_eq!(it.group, Group::Vehicle, "snímka {f}");
                ids.insert(it.id);
            }
        }
        assert_eq!(ids.len(), 1, "jedna stopa: {ids:?}");
    }

    #[test]
    fn jazdec_najprv_ako_clovek_potom_s_motorkou_ma_jedno_id() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 10.0, 8);
        let d = |label: &str, x: i32, y: i32, w, h| Detection { rect: Rect::new(x, y, w, h), label: label.into(), score: Some(0.9) };
        let mut ids = std::collections::HashSet::new();
        for f in 0..20u64 {
            let x = 100 + 15 * f as i32;
            let det = if f < 8 { d("person", x + 20, 100, 50, 70) } else { d("motorcycle", x, 100, 90, 120) };
            let (items, _) = core.track_and_count(&[det], f, f as f64 / 20.0);
            ids.extend(items.iter().map(|it| it.id));
        }
        assert_eq!(ids.len(), 1, "{ids:?}");
    }

    #[test]
    fn chodec_vedla_zaparkovanej_motorky_ostane_clovekom() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 10.0, 8);
        let d = |label: &str, x: i32, y: i32, w, h| Detection { rect: Rect::new(x, y, w, h), label: label.into(), score: Some(0.9) };
        for f in 0..30u64 {
            let (items, _) = core.track_and_count(&[d("motorcycle", 300, 100, 90, 60), d("person", 200 + 5 * f as i32, 60, 40, 100)], f, f as f64 / 20.0);
            if f > 5 {
                assert!(items.iter().any(|it| it.group == Group::Person), "snímka {f}");
            }
        }
    }

    #[test]
    fn roi_v_pixeloch() {
        let r = roi_rect(&Crop { x: 0.0, y: 0.52, w: 0.47, h: 0.33 }, 2560, 1440);
        assert_eq!((r.x, r.y, r.width, r.height), (0, 749, 1203, 475));
    }

    #[test]
    fn roi_nepresahuje_obraz() {
        let r = roi_rect(&Crop { x: 0.9, y: 0.9, w: 0.5, h: 0.5 }, 1000, 1000);
        assert!(r.x + r.width <= 1000 && r.y + r.height <= 1000, "{r:?}");
    }

    #[test]
    fn prepocet_suradnic_z_vyrezu_na_vystup() {
        // výrez začína v (0, 800) veľkej snímky; výstup je 2x menší
        let out = to_output_coords(Rect::new(400, 100, 200, 120), Rect::new(0, 800, 1200, 475), 0.5);
        assert_eq!((out.x, out.y, out.width, out.height), (200, 450, 100, 60));
        // posun výrezu aj v osi x
        let out = to_output_coords(Rect::new(10, 0, 20, 20), Rect::new(100, 0, 500, 500), 1.0);
        assert_eq!(out.x, 110);
    }

    #[test]
    fn zhodny_prepocet_nezmeni_rozmer_na_nulu() {
        assert!(to_output_coords(Rect::new(0, 0, 1, 1), Rect::new(0, 0, 10, 10), 0.1).width >= 1);
    }

    /// Auto prejde zľava doprava cez čiaru v x=500 v pásme y=100..400.
    fn drive(core: &mut CountingCore, label: &str, y: i32, from: i32, to: i32, step: i32, first_frame: u64) -> usize {
        let (mut n, mut x, mut f) = (0, from, first_frame);
        while (step > 0 && x <= to) || (step < 0 && x >= to) {
            n += core.track_and_count(&[det(label, x, y)], f, f as f64 / 12.0).1.len();
            x += step;
            f += 1;
        }
        n
    }

    #[test]
    fn jadro_zapocita_auto_v_pasme() {
        let mut core = CountingCore::new(500.0, (100.0, 400.0), 12.0, 12);
        assert_eq!(drive(&mut core, "car", 200, 0, 900, 30, 0), 1);
        assert_eq!(core.counter().count(Group::Vehicle, Direction::LeftToRight), 1);
    }

    #[test]
    fn jadro_ignoruje_auto_mimo_pasma() {
        let mut core = CountingCore::new(500.0, (100.0, 400.0), 12.0, 12);
        assert_eq!(drive(&mut core, "car", 0, 0, 900, 30, 0), 0, "auto ide nad pásmom");
        assert_eq!(drive(&mut core, "car", 700, 0, 900, 30, 100), 0, "auto ide pod pásmom");
        assert_eq!(core.counter().total(Group::Vehicle), 0);
    }

    #[test]
    fn jadro_ignoruje_nepocitane_triedy_a_nevracia_ich() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 12);
        let (items, ev) = core.track_and_count(&[det("dog", 400, 100), det("car", 100, 100)], 0, 0.0);
        assert!(ev.is_empty());
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].label, "car");
    }

    #[test]
    fn jadro_vrati_objekty_s_id_na_vykreslenie() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 12);
        let (a, _) = core.track_and_count(&[det("person", 100, 100), det("car", 300, 300)], 0, 0.0);
        assert_eq!(a.iter().map(|i| i.id).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(a[0].group, Group::Person);
        assert_eq!(a[1].group, Group::Vehicle);
        // v ďalšej snímke majú rovnaké ID
        let (b, _) = core.track_and_count(&[det("person", 105, 100), det("car", 310, 300)], 1, 0.1);
        assert_eq!(b.iter().map(|i| i.id).collect::<Vec<_>>(), vec![1, 2]);
    }

    #[test]
    fn jadro_pocita_aj_ked_nikto_nic_nekresli_prazdne_snimky_su_ok() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 3);
        for f in 0..50 {
            let (items, ev) = core.track_and_count(&[], f, 0.0);
            assert!(items.is_empty() && ev.is_empty());
        }
    }

    // ---- relácia diváka ----

    fn cross(c: &mut LineCounter, track: u32, g: Group, dir: Direction) {
        let (a, b) = if dir == Direction::LeftToRight { (400.0, 600.0) } else { (600.0, 400.0) };
        c.observe(0, 0.0, track, g, a);
        c.observe(1, 0.0, track, g, b);
    }

    fn item(id: u32, group: Group) -> Counted {
        Counted { rect: Rect::new(0, 0, 10, 10), label: "x".into(), score: None, id, group, speed_kmh: None }
    }

    #[test]
    fn relacia_zacina_od_nuly_aj_ked_pred_nou_nieco_prejdeno() {
        let mut total = LineCounter::new(500.0, 10.0);
        cross(&mut total, 1, Group::Vehicle, Direction::LeftToRight);
        cross(&mut total, 2, Group::Vehicle, Direction::RightToLeft);
        cross(&mut total, 3, Group::Person, Direction::LeftToRight);
        assert_eq!(total.total(Group::Vehicle), 2);

        let session = Session::start(&total); // divák sa pripojil
        let v = session.view_at(&total, Duration::ZERO);
        assert_eq!((v.total(Group::Vehicle), v.total(Group::Person)), (0, 0), "v paneli nuly");
        assert_eq!(total.total(Group::Vehicle), 2, "celkové počty ostávajú");
    }

    #[test]
    fn relacia_pocita_len_to_co_prislo_po_pripojeni() {
        let mut total = LineCounter::new(500.0, 10.0);
        cross(&mut total, 1, Group::Vehicle, Direction::LeftToRight);
        let session = Session::start(&total);
        cross(&mut total, 2, Group::Vehicle, Direction::LeftToRight);
        cross(&mut total, 3, Group::Vehicle, Direction::RightToLeft);
        cross(&mut total, 4, Group::Person, Direction::RightToLeft);

        let v = session.view_at(&total, Duration::ZERO);
        assert_eq!(v.count(Group::Vehicle, Direction::LeftToRight), 1);
        assert_eq!(v.count(Group::Vehicle, Direction::RightToLeft), 1);
        assert_eq!(v.total(Group::Vehicle), 2);
        assert_eq!(v.total(Group::Person), 1);
        assert_eq!(total.total(Group::Vehicle), 3, "celkovo od štartu servera");
    }

    #[test]
    fn nova_relacia_po_odpojeni_zacina_znova_od_nuly() {
        let mut total = LineCounter::new(500.0, 10.0);
        let first = Session::start(&total);
        cross(&mut total, 1, Group::Vehicle, Direction::LeftToRight);
        assert_eq!(first.view_at(&total, Duration::ZERO).total(Group::Vehicle), 1);
        // divák odišiel a o chvíľu sa pripojil znova
        let second = Session::start(&total);
        assert_eq!(second.view_at(&total, Duration::ZERO).total(Group::Vehicle), 0);
    }

    #[test]
    fn cisla_objektov_od_jednotky_zvlast_pre_auta_a_ludi() {
        let total = LineCounter::new(500.0, 10.0);
        let mut s = Session::start(&total);
        // globálne ID sú veľké (server beží dlho)
        let out = s.relabel(vec![item(214, Group::Vehicle), item(215, Group::Person), item(230, Group::Vehicle)]);
        assert_eq!(out.iter().map(|i| i.id).collect::<Vec<_>>(), vec![1, 1, 2]);
    }

    #[test]
    fn objekt_si_drzi_cislo_v_dalsich_snimkach() {
        let total = LineCounter::new(500.0, 10.0);
        let mut s = Session::start(&total);
        s.relabel(vec![item(50, Group::Vehicle)]);
        let out = s.relabel(vec![item(51, Group::Vehicle), item(50, Group::Vehicle)]);
        assert_eq!(out.iter().map(|i| i.id).collect::<Vec<_>>(), vec![2, 1]);
    }

    #[test]
    fn pata_panela_ukazuje_cas_od_zapnutia() {
        let total = LineCounter::new(500.0, 10.0);
        let s = Session::start(&total);
        assert_eq!(s.view_at(&total, Duration::from_secs(0)).footer, "od zapnutia 0:00");
        assert_eq!(format_elapsed(Duration::from_secs(65)), "od zapnutia 1:05");
        assert_eq!(format_elapsed(Duration::from_secs(3600 + 59)), "od zapnutia 60:59");
    }

    #[test]
    fn view_zachova_ciaru_a_pasmo() {
        let total = LineCounter::new(321.0, 10.0).with_span(5.0, 50.0);
        let v = Session::start(&total).view_at(&total, Duration::ZERO);
        assert_eq!((v.line_x, v.span), (321.0, Some((5.0, 50.0))));
    }

    #[test]
    fn pocitanie_v_suradniciach_referencie_ignoruje_pohyb_kamery() {
        // Auto stojí na mieste, ale kamera sa postupne posúva doprava, takže sa v obraze zdanlivo
        // posúva doľava a prejde cez čiaru v x = 500. Bez korekcie by sa teda zrátalo len kvôli
        // pohybu kamery; s prepočtom do súradníc referencie nie.
        let mut stat = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 12);
        let mut anch = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 12);
        let mut ev_stat = 0;
        let mut ev_anch = 0;
        for f in 0..40 {
            let cam = f * 10; // posun kamery
            let d = det("car", 700 - cam, 100); // pozícia auta v obraze
            ev_stat += stat.track_and_count(&[d.clone()], f as u64, 0.0).1.len();
            // kotvené: do referencie sa vráti pripočítaním posunu kamery
            ev_anch += anch
                .track_and_count_with(&[d], f as u64, 0.0, |r| Rect::new(r.x + cam, r.y, r.width, r.height))
                .1
                .len();
        }
        assert_eq!(ev_stat, 1, "bez ukotvenia sa auto, ktoré stojí, zrátalo kvôli pohybu kamery");
        assert_eq!(ev_anch, 0, "s ukotvením stojace auto nič nezráta");
    }

    #[test]
    fn ukotvene_pocitanie_vrati_povodne_rameky_na_kreslenie() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 12);
        let d = det("car", 321, 100);
        let (items, _) = core.track_and_count_with(&[d], 0, 0.0, |r| Rect::new(r.x + 999, r.y, r.width, r.height));
        assert_eq!(items[0].rect.x, 321, "kreslí sa pôvodný rámček, nie ten v referencii");
    }

    #[test]
    fn auto_prechadza_ciaru_aj_ked_sa_kamera_hybe() {
        // auto ide zľava doprava rýchlosťou 20 px/snímku, kamera sa posúva rovnakým smerom o 8 px
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 12);
        let mut n = 0;
        for f in 0..40i32 {
            let cam = f * 8;
            let world_x = f * 20;
            let d = det("car", world_x - cam, 100); // v obraze sa posúva len o 12 px/snímku
            n += core
                .track_and_count_with(&[d], f as u64, 0.0, |r| Rect::new(r.x + cam, r.y, r.width, r.height))
                .1
                .len();
        }
        assert_eq!(n, 1);
        assert_eq!(core.counter().count(Group::Vehicle, Direction::LeftToRight), 1);
    }

    #[test]
    fn vyrez_zo_stvoruholnika_je_obmedzujuci_obdlznik_vo_velkej_snimke() {
        // štvoruholník vo výstupe (1280 px široký); veľká snímka je 2x väčšia (scale 0.5)
        let q = [(100.0, 200.0), (400.0, 190.0), (410.0, 300.0), (95.0, 310.0)];
        let r = roi_from_quad(&q, 0.5, 2560, 1440);
        assert_eq!((r.x, r.y), (190, 380));
        assert_eq!((r.x + r.width, r.y + r.height), (820, 620));
    }

    #[test]
    fn vyrez_sa_orezava_do_obrazu() {
        let q = [(-50.0, -20.0), (300.0, -20.0), (300.0, 100.0), (-50.0, 100.0)];
        let r = roi_from_quad(&q, 1.0, 1280, 720);
        assert_eq!((r.x, r.y), (0, 0));
        assert!(r.width <= 1280 && r.height <= 720);
        let q = [(1200.0, 650.0), (1500.0, 650.0), (1500.0, 900.0), (1200.0, 900.0)];
        let r = roi_from_quad(&q, 1.0, 1280, 720);
        assert!(r.x + r.width <= 1280 && r.y + r.height <= 720, "{r:?}");
        assert!(r.width >= 32 && r.height >= 32);
    }

    #[test]
    fn vyrez_celkom_mimo_obrazu_nepadne() {
        let q = [(5000.0, 5000.0), (5100.0, 5000.0), (5100.0, 5100.0), (5000.0, 5100.0)];
        let r = roi_from_quad(&q, 1.0, 1280, 720);
        assert!(r.x >= 0 && r.y >= 0 && r.x + r.width <= 1280 && r.y + r.height <= 720, "{r:?}");
    }

    fn ground_model() -> crate::speed::GroundModel {
        crate::speed::GroundModel {
            f: 3068.36, k1: 1.3747, k2: 0.0, pitch: 0.12417, roll: -0.01673, h: 3.7694,
            cx: 1920.0, cy: 1080.0, width: 3840.0, height: 2160.0, snapshot: None, points: vec![],
        }
    }

    #[test]
    fn jadro_meria_rychlost_pri_prechode() {
        // výstup 1280x720 (model je 3x väčší); auto ide po ceste (spodok rámčeka v y=480),
        // 12 px na snímku pri 20 fps
        let m = ground_model();
        let mapper = GroundMapper::new(m.clone(), 1280.0, 720.0).unwrap();
        let mut core = CountingCore::new(640.0, (0.0, 720.0), 12.0, 12).with_speed(mapper);
        let (mut ev, mut last_items) = (vec![], vec![]);
        for f in 0..60u64 {
            let x = 300 + 12 * f as i32;
            let (items, e) = core.track_and_count(&[Detection { rect: Rect::new(x, 420, 100, 60), label: "car".into(), score: None }], f, f as f64 / 20.0);
            ev.extend(e);
            last_items = items;
        }
        assert_eq!(ev.len(), 1);
        // očakávaná rýchlosť priamo z modelu: posun spodku rámčeka o 12 px za 1/20 s
        let a = m.to_ground(3.0 * 690.0, 3.0 * 480.0).unwrap();
        let b = m.to_ground(3.0 * 702.0, 3.0 * 480.0).unwrap();
        let expect = ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt() * 20.0 * 3.6;
        let v = ev[0].speed_kmh.expect("pri prechode sa má zmerať rýchlosť");
        assert!((v - expect).abs() / expect < 0.05, "{v} vs {expect}");
        assert_eq!(core.counter().events()[0].speed_kmh, Some(v), "rýchlosť je aj v uložených udalostiach");
        assert!(last_items[0].speed_kmh.is_some());
    }

    #[test]
    fn jadro_bez_modelu_nema_rychlost() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 12);
        for f in 0..30u64 {
            let (items, _) = core.track_and_count(&[det("car", 100 + 30 * f as i32, 100)], f, f as f64 / 20.0);
            assert_eq!(items[0].speed_kmh, None);
        }
        assert!(core.counter().events().iter().all(|e| e.speed_kmh.is_none()));
    }

    #[test]
    fn strateny_objekt_sa_chvilu_este_kresli() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 12);
        core.track_and_count(&[det("car", 100, 100)], 0, 0.0);
        // detektor auto na pár snímok nenájde: stále sa vráti (na poslednom mieste)
        for f in 1..=COAST_FRAMES as u64 {
            let (items, _) = core.track_and_count(&[], f, f as f64 / 20.0);
            assert_eq!(items.len(), 1, "snímka {f}");
            assert_eq!(items[0].rect, Rect::new(100, 100, 100, 60));
        }
        // potom už nie
        let (items, _) = core.track_and_count(&[], 99, 5.0);
        assert!(items.is_empty());
    }

    #[test]
    fn siroky_pas_sa_rozdeli_na_casti() {
        // pás 2560x504 (ulica cez celý záber): 3 časti po 1260 px s prekryvom >= 605 px
        let roi = Rect::new(0, 720, 2560, 504);
        let t = tiles(roi);
        assert_eq!(t.len(), 3, "{t:?}");
        assert_eq!(t[0].x, 0);
        assert_eq!(t.last().unwrap().x + t.last().unwrap().width, 2560, "pokryje celý pás");
        for w in t.windows(2) {
            assert!(w[0].x + w[0].width - w[1].x >= 605, "prekryv {w:?}");
        }
        assert!(t.iter().all(|r| r.y == 720 && r.height == 504));
        // úzky výrez sa nedelí
        assert_eq!(tiles(Rect::new(10, 20, 1200, 475)), vec![Rect::new(10, 20, 1200, 475)]);
    }

    #[test]
    fn spojenie_casti_necha_objekt_raz() {
        let d = |x, w, label: &str, cl, cr| TileDet {
            det: Detection { rect: Rect::new(x, 100, w, 80), label: label.into(), score: Some(0.8) },
            cut_left: cl,
            cut_right: cr,
        };
        // celé auto z jednej časti, odrezaná časť toho istého auta z druhej, rovnaké auto 2x, iný objekt
        let out = merge_tiles(vec![
            d(500, 300, "car", false, false),
            d(600, 200, "car", true, false),
            d(502, 298, "car", false, false),
            d(560, 40, "person", false, false),
        ]);
        assert_eq!(out.len(), 2, "{out:?}");
        assert!(out.iter().any(|x| x.label == "car" && x.rect.width == 300));
        assert!(out.iter().any(|x| x.label == "person"), "človek pred autom ostane");
    }

    #[test]
    fn odrezane_kusky_auta_sa_zlepia() {
        // auto je širšie než prekryv: ľavá časť ho odreže vpravo, pravá vľavo
        let a = TileDet { det: Detection { rect: Rect::new(400, 100, 250, 80), label: "car".into(), score: Some(0.7) }, cut_left: false, cut_right: true };
        let b = TileDet { det: Detection { rect: Rect::new(560, 102, 240, 78), label: "car".into(), score: Some(0.9) }, cut_left: true, cut_right: false };
        let out = merge_tiles(vec![a, b]);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].rect, Rect::new(400, 100, 400, 80));
        assert_eq!(out[0].score, Some(0.9));
    }

    #[test]
    fn dve_auta_vedla_seba_sa_nezlepia() {
        // dve celé (neodrezané) autá, ktoré sa dotýkajú, ostanú dve
        let d = |x| TileDet { det: Detection { rect: Rect::new(x, 100, 200, 80), label: "car".into(), score: Some(0.8) }, cut_left: false, cut_right: false };
        assert_eq!(merge_tiles(vec![d(100), d(290)]).len(), 2);
    }

    #[test]
    fn auto_s_nizkou_istotou_sa_ignoruje_clovek_nie() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 12);
        let d = |label: &str, x, s| Detection { rect: Rect::new(x, 100, 100, 60), label: label.into(), score: Some(s) };
        let (items, _) = core.track_and_count(&[d("car", 0, 0.3), d("person", 300, 0.3), d("car", 600, 0.5)], 0, 0.0);
        let labels: Vec<_> = items.iter().map(|i| (i.label.as_str(), i.rect.x)).collect();
        assert_eq!(labels, vec![("person", 300), ("car", 600)]);
    }

    #[test]
    fn nehybne_vozidlo_sa_po_chvili_nekresli() {
        let mut core = CountingCore::new(500.0, (0.0, 1000.0), 12.0, 12);
        let mut shown = vec![];
        for f in 0..60u64 {
            let t = f as f64 / 20.0;
            // skrinka (stojí, mierne sa chveje), stojaci človek, idúce auto
            let dets = [det("truck", 100 + (f % 2) as i32, 100), det("person", 300, 300), det("car", 20 * f as i32, 600)];
            let (items, _) = core.track_and_count(&dets, f, t);
            shown.push(items.iter().map(|i| i.label.clone()).collect::<Vec<_>>());
        }
        assert!(shown[10].contains(&"truck".to_string()), "prvé 2 s sa ešte kreslí");
        assert!(!shown[59].contains(&"truck".to_string()), "nehybné vozidlo zmizne: {:?}", shown[59]);
        assert!(shown[59].contains(&"person".to_string()), "stojaci človek ostáva");
        assert!(shown[59].contains(&"car".to_string()), "idúce auto ostáva");
    }
}
