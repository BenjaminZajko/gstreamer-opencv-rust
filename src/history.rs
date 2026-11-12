//! História detegovaných objektov: pre každý objekt jeden JPG (najlepší záber, rámček a
//! rýchlosť v okienku vľavo hore) a riadok v `historia.csv`.
//!
//! Objekt sa uloží, keď zmizne zo záberu (alebo pri `finish_all` na konci videa). Ukladajú sa
//! len objekty, ktoré sa kreslia (prešli filtrom: pohli sa, sieť ich našla viackrát).

use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::Write,
};

use chrono::{DateTime, Local};
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size, Vector},
    imgcodecs, imgproc,
    prelude::*,
};

use crate::{
    counter::{CrossEvent, Direction, Group},
    draw::{display_label, Counted, MIN_SHOWN_KMH},
    Result,
};

/// Ako sa zapisuje čas objektu.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Clock {
    /// skutočný čas (živá kamera)
    Wall,
    /// čas vo videu (s od začiatku)
    Video,
}

/// Najkratšie trvanie objektu (s), ktorý sa ešte uloží (kratšie sú väčšinou omyly).
pub const MIN_DURATION_S: f64 = 0.5;
/// Ako dlho (s) sa čaká s uložením zmiznutého objektu: ak sa za ten čas na očakávanom mieste
/// objaví "nový" objekt (ten istý, len ho chvíľu zakrýval oblúk / strom), spoja sa.
pub const JOIN_GAP_S: f64 = 1.5;
/// Objekt sa uloží len ak prešiel aspoň toľko šírok svojho rámčeka (alebo prešiel čiarou) –
/// inak je to stojaci objekt / omyl siete.
pub const MIN_TRAVEL_WIDTHS: f64 = 1.5;

struct Obj {
    group: Group,
    labels: HashMap<String, u32>,
    first: (f64, DateTime<Local>),
    last: (f64, DateTime<Local>),
    first_x: f64,
    last_x: f64,
    /// typická šírka rámčeka (px) a rýchlosť stredu (px/s) – na spojenie prerušenej stopy
    width: f64,
    vx: f64,
    speeds: Vec<f64>,
    cross: Option<(Direction, Option<f64>)>,
    /// najlepší záber: (skóre, zmenšená snímka, rámček, rýchlosť v tej chvíli)
    best: Option<(f64, Mat, Rect)>,
}

/// Záznam histórie.
pub struct ObjectLog {
    dir: String,
    prefix: String,
    clock: Clock,
    objs: HashMap<u32, Obj>,
    /// zmiznuté objekty čakajúce na uloženie (môžu sa ešte spojiť s pokračovaním)
    pending: Vec<(u32, Obj)>,
    /// nové ID -> pôvodné ID (pokračovanie prerušenej stopy)
    alias: HashMap<u32, u32>,
    saved: usize,
}

/// Jeden uložený objekt (na výpis).
#[derive(Debug, Clone)]
pub struct Saved {
    pub id: u32,
    pub label: String,
    pub speed_kmh: Option<f64>,
    pub image: String,
}

impl ObjectLog {
    /// `dir` – kam sa ukladá (vytvorí sa), `prefix` – začiatok názvov obrázkov (napr. názov videa).
    pub fn new(dir: &str, prefix: &str, clock: Clock) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let csv = format!("{dir}/historia.csv");
        if !std::path::Path::new(&csv).exists() {
            fs::write(&csv, "id,typ,skupina,zaciatok,koniec,trvanie_s,smer,rychlost_pri_ciare_kmh,rychlost_median_kmh,rychlost_max_kmh,obrazok\n")?;
        }
        Ok(Self { dir: dir.into(), prefix: prefix.into(), clock, objs: HashMap::new(), pending: vec![], alias: HashMap::new(), saved: 0 })
    }

    /// Koľko objektov sa už uložilo.
    pub fn saved(&self) -> usize {
        self.saved
    }

    /// Zapíše objekty z jednej snímky. `frame_hi` je veľká snímka (to, na čom beží detekcia),
    /// `out_size` rozmer zobrazeného záberu (v ňom sú rámčeky `items`).
    pub fn observe(&mut self, frame_hi: &Mat, out_size: Size, items: &[Counted], time_s: f64) -> Result<()> {
        let now = Local::now();
        let mut small: Option<Mat> = None;
        for it in items {
            let cx = it.rect.x as f64 + it.rect.width as f64 / 2.0;
            // nové ID: nie je to pokračovanie objektu, ktorý pred chvíľou zmizol?
            if !self.objs.contains_key(&it.id) && !self.alias.contains_key(&it.id) {
                let found = self.pending.iter().position(|(_, o)| {
                    let gap = time_s - o.last.0;
                    let expected = o.last_x + o.vx * gap;
                    o.group == it.group && gap <= JOIN_GAP_S && (cx - expected).abs() <= (0.6 * o.width).max(80.0)
                });
                if let Some(i) = found {
                    let (old_id, o) = self.pending.remove(i);
                    self.alias.insert(it.id, old_id);
                    self.objs.insert(old_id, o);
                }
            }
            let key = *self.alias.get(&it.id).unwrap_or(&it.id);
            let o = self.objs.entry(key).or_insert_with(|| Obj {
                group: it.group,
                labels: HashMap::new(),
                first: (time_s, now),
                last: (time_s, now),
                first_x: cx,
                last_x: cx,
                width: it.rect.width as f64,
                vx: 0.0,
                speeds: vec![],
                cross: None,
                best: None,
            });
            o.group = it.group;
            *o.labels.entry(it.label.clone()).or_default() += 1;
            let dt = time_s - o.last.0;
            if dt > 1e-3 {
                o.vx = 0.7 * o.vx + 0.3 * (cx - o.last_x) / dt;
            }
            o.width = 0.9 * o.width + 0.1 * it.rect.width as f64;
            o.last = (time_s, now);
            o.last_x = cx;
            if let Some(v) = it.speed_kmh.filter(|v| *v >= MIN_SHOWN_KMH) {
                o.speeds.push(v);
            }
            // najlepší záber: celý v obraze a čo najväčší
            let r = it.rect;
            let inside = r.x > 4 && r.y > 4 && r.x + r.width < out_size.width - 4 && r.y + r.height < out_size.height - 4;
            let score = r.area() as f64 * if inside { 1.0 } else { 0.3 };
            if o.best.as_ref().is_none_or(|b| score > b.0 * 1.05) {
                if small.is_none() {
                    let mut s = Mat::default();
                    imgproc::resize(frame_hi, &mut s, out_size, 0.0, 0.0, imgproc::INTER_AREA)?;
                    small = Some(s);
                }
                o.best = Some((score, small.as_ref().unwrap().clone(), r));
            }
        }
        Ok(())
    }

    /// Prechod cez čiaru (smer a rýchlosť pri čiare sa pridajú k objektu).
    pub fn record_cross(&mut self, ev: &CrossEvent) {
        let key = *self.alias.get(&ev.track).unwrap_or(&ev.track);
        let o = match self.objs.get_mut(&key) {
            Some(o) => Some(o),
            None => self.pending.iter_mut().find(|(id, _)| *id == key).map(|(_, o)| o),
        };
        if let Some(o) = o {
            // prvý prechod sa nechá (pokračovanie stopy môže prejsť čiarou znova)
            if o.cross.is_none() {
                o.cross = Some((ev.direction, ev.speed_kmh));
            }
        }
    }

    /// Objekty `ids` zmizli: po JOIN_GAP_S sa uloží ich obrázok a riadok histórie (ak sa
    /// dovtedy nespojili s pokračovaním). Vráti objekty uložené teraz. `time_s` je aktuálny čas.
    pub fn finish(&mut self, ids: &[u32], time_s: f64) -> Result<Vec<Saved>> {
        for id in ids {
            let key = self.alias.remove(id).unwrap_or(*id);
            // ak má objekt ešte iné živé ID (pokračovanie), nečaká sa
            if self.alias.values().any(|v| *v == key) {
                continue;
            }
            if let Some(mut o) = self.objs.remove(&key) {
                // pokračovanie už beží pod novým ID (tracker drží zmiznutý objekt ešte chvíľu,
                // takže nový sa mohol objaviť skôr): spojí sa s ním, ID ostane pôvodné
                let next = self.objs.iter().find(|(_, n)| {
                    let gap = n.first.0 - o.last.0;
                    let expected = o.last_x + o.vx * gap;
                    n.group == o.group && gap >= -0.5 && gap <= JOIN_GAP_S && (n.first_x - expected).abs() <= (0.6 * o.width).max(80.0)
                }).map(|(k, _)| *k);
                if let Some(nk) = next {
                    let n = self.objs.remove(&nk).unwrap();
                    o = merge(o, n);
                    for v in self.alias.values_mut() {
                        if *v == nk {
                            *v = key;
                        }
                    }
                    self.alias.insert(nk, key);
                    self.objs.insert(key, o);
                    continue;
                }
                self.pending.push((key, o));
            }
        }
        let (due, wait): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending).into_iter().partition(|(_, o)| time_s - o.last.0 > JOIN_GAP_S);
        self.pending = wait;
        let mut out = vec![];
        for (id, o) in due {
            if let Some(s) = self.save(id, o)? {
                out.push(s);
            }
        }
        Ok(out)
    }

    /// Uloží všetky ešte sledované aj čakajúce objekty (koniec videa).
    pub fn finish_all(&mut self) -> Result<Vec<Saved>> {
        let mut all: Vec<(u32, Obj)> = std::mem::take(&mut self.pending);
        all.extend(self.objs.drain());
        all.sort_by_key(|(id, _)| *id);
        self.alias.clear();
        let mut out = vec![];
        for (id, o) in all {
            if let Some(s) = self.save(id, o)? {
                out.push(s);
            }
        }
        Ok(out)
    }

    fn time_text(&self, t: (f64, DateTime<Local>)) -> String {
        match self.clock {
            Clock::Wall => t.1.format("%Y-%m-%d %H:%M:%S").to_string(),
            Clock::Video => format!("{:.1}", t.0),
        }
    }

    fn save(&mut self, id: u32, o: Obj) -> Result<Option<Saved>> {
        let duration = o.last.0 - o.first.0;
        let Some((_, mut img, rect)) = o.best else { return Ok(None) };
        if duration < MIN_DURATION_S || (o.cross.is_none() && (o.last_x - o.first_x).abs() < MIN_TRAVEL_WIDTHS * o.width) {
            return Ok(None);
        }
        let label = o.labels.iter().max_by_key(|(_, n)| **n).map(|(l, _)| l.clone()).unwrap_or_default();
        let mut sorted = o.speeds.clone();
        sorted.sort_by(f64::total_cmp);
        let median = (!sorted.is_empty()).then(|| sorted[sorted.len() / 2]);
        let max = sorted.last().copied();
        let cross_speed = o.cross.and_then(|c| c.1);
        // zobrazená rýchlosť: pri čiare, inak medián počas celej doby
        let speed = cross_speed.or(median);
        let dir = match o.cross {
            Some((d, _)) => d,
            None if o.last_x >= o.first_x => Direction::LeftToRight,
            None => Direction::RightToLeft,
        };

        // obrázok: rámček objektu + okienko vľavo hore
        let color = match o.group {
            Group::Vehicle => Scalar::new(0.0, 140.0, 255.0, 0.0),
            Group::Person => Scalar::new(0.0, 255.0, 0.0, 0.0),
        };
        imgproc::rectangle(&mut img, rect, color, 3, imgproc::LINE_8, 0)?;
        let when = match self.clock {
            Clock::Wall => o.first.1.format("%d.%m.%Y %H:%M:%S").to_string(),
            Clock::Video => format!("cas vo videu {:.1} s", o.first.0),
        };
        let lines = [
            (format!("{} #{}", display_label(&label).to_uppercase(), id), 0.8, Scalar::all(255.0)),
            (speed.map(|v| format!("{v:.0} km/h")).unwrap_or_else(|| "rychlost -".into()), 1.6, color),
            (if dir == Direction::LeftToRight { "smer: zlava doprava" } else { "smer: sprava dolava" }.to_string(), 0.6, Scalar::all(220.0)),
            (when, 0.6, Scalar::all(220.0)),
        ];
        draw_toast(&mut img, &lines)?;

        let stamp = match self.clock {
            Clock::Wall => o.first.1.format("%Y%m%d_%H%M%S").to_string(),
            Clock::Video => format!("{:07.1}s", o.first.0),
        };
        let speed_tag = speed.map(|v| format!("_{v:.0}kmh")).unwrap_or_default();
        let name = format!("{}{stamp}_{}_{id}{speed_tag}.jpg", self.prefix, display_label(&label));
        let path = format!("{}/{name}", self.dir);
        imgcodecs::imwrite(&path, &img, &Vector::from_slice(&[imgcodecs::IMWRITE_JPEG_QUALITY, 85]))?;

        let f = |v: Option<f64>| v.map(|v| format!("{v:.1}")).unwrap_or_default();
        let row = format!(
            "{id},{},{},{},{},{duration:.1},{},{},{},{},{name}\n",
            display_label(&label),
            o.group.name(),
            self.time_text(o.first),
            self.time_text(o.last),
            dir.name(),
            f(cross_speed),
            f(median),
            f(max)
        );
        OpenOptions::new().append(true).open(format!("{}/historia.csv", self.dir))?.write_all(row.as_bytes())?;
        self.saved += 1;
        Ok(Some(Saved { id, label: display_label(&label).into(), speed_kmh: speed, image: path }))
    }
}

/// Spojí starší objekt `a` s jeho pokračovaním `b`.
fn merge(mut a: Obj, b: Obj) -> Obj {
    for (l, n) in b.labels {
        *a.labels.entry(l).or_default() += n;
    }
    a.speeds.extend(b.speeds);
    if b.last.0 > a.last.0 {
        a.last = b.last;
        a.last_x = b.last_x;
        a.vx = b.vx;
    }
    if a.cross.is_none() {
        a.cross = b.cross;
    }
    if let Some(bb) = b.best {
        if a.best.as_ref().is_none_or(|ab| bb.0 > ab.0) {
            a.best = Some(bb);
        }
    }
    a
}

/// Tmavé okienko vľavo hore s riadkami textu (text, veľkosť, farba).
fn draw_toast(img: &mut Mat, lines: &[(String, f64, Scalar)]) -> Result<()> {
    let (pad, gap) = (14, 10);
    let mut sizes = vec![];
    for (t, s, _) in lines {
        let mut base = 0;
        sizes.push(imgproc::get_text_size(t, imgproc::FONT_HERSHEY_SIMPLEX, *s, 2, &mut base)?);
    }
    let w = sizes.iter().map(|s| s.width).max().unwrap_or(0) + 2 * pad;
    let h = sizes.iter().map(|s| s.height + gap).sum::<i32>() + 2 * pad;
    // polopriehľadné pozadie
    let area = Rect::new(12, 12, w.min(img.cols() - 24), h.min(img.rows() - 24));
    let mut roi = Mat::roi_mut(img, area)?;
    let mut dark = Mat::default();
    roi.convert_to(&mut dark, -1, 0.25, 0.0)?;
    dark.copy_to(&mut roi)?;
    imgproc::rectangle(img, area, Scalar::all(200.0), 1, imgproc::LINE_AA, 0)?;
    let mut y = 12 + pad;
    for ((t, s, c), sz) in lines.iter().zip(&sizes) {
        y += sz.height;
        imgproc::put_text(img, t, Point::new(12 + pad, y), imgproc::FONT_HERSHEY_SIMPLEX, *s, *c, 2, imgproc::LINE_AA, false)?;
        y += gap;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: u32, x: i32, speed: Option<f64>) -> Counted {
        Counted { rect: Rect::new(x, 100, 120, 60), label: "car".into(), score: Some(0.9), id, group: Group::Vehicle, speed_kmh: speed }
    }

    #[test]
    fn ulozi_obrazok_a_riadok_pre_kazdy_objekt() {
        let dir = std::env::temp_dir().join(format!("historia_test_{}", std::process::id()));
        let dir = dir.to_str().unwrap().to_string();
        let _ = fs::remove_dir_all(&dir);
        let mut log = ObjectLog::new(&dir, "v_", Clock::Video).unwrap();
        let frame = Mat::new_size_with_default(Size::new(1280, 720), opencv::core::CV_8UC3, Scalar::all(90.0)).unwrap();
        for f in 0..30 {
            let t = f as f64 * 0.05;
            let items = vec![item(1, 100 + 20 * f, Some(30.0 + (f % 3) as f64)), item(2, 900, None)];
            log.observe(&frame, Size::new(1280, 720), &items, t).unwrap();
        }
        log.record_cross(&CrossEvent { frame: 10, time_s: 0.5, track: 1, group: Group::Vehicle, direction: Direction::LeftToRight, speed_kmh: Some(31.0) });
        let saved = log.finish_all().unwrap();
        assert_eq!(saved.len(), 1, "stojaci objekt 2 sa neuloží");
        assert_eq!(saved[0].speed_kmh, Some(31.0), "rýchlosť pri čiare má prednosť");
        assert!(std::path::Path::new(&saved[0].image).exists());
        assert!(saved[0].image.ends_with("_auto_1_31kmh.jpg"), "{}", saved[0].image);
        let csv = fs::read_to_string(format!("{dir}/historia.csv")).unwrap();
        assert_eq!(csv.lines().count(), 2, "{csv}");
        assert!(csv.lines().nth(1).unwrap().contains(",zlava_doprava,31.0,31.0,32.0,"), "{csv}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn prerusena_stopa_sa_spoji_do_jedneho_objektu() {
        let dir = std::env::temp_dir().join(format!("historia_spoj_{}", std::process::id()));
        let dir = dir.to_str().unwrap().to_string();
        let _ = fs::remove_dir_all(&dir);
        let mut log = ObjectLog::new(&dir, "", Clock::Video).unwrap();
        let frame = Mat::new_size_with_default(Size::new(1280, 720), opencv::core::CV_8UC3, Scalar::all(90.0)).unwrap();
        let sz = Size::new(1280, 720);
        // objekt 1 ide doprava 20 px/snímku, zmizne za oblúkom, o 0,5 s sa objaví ako objekt 2
        for f in 0..20 {
            log.observe(&frame, sz, &[item(1, 100 + 20 * f, Some(25.0))], f as f64 * 0.05).unwrap();
        }
        assert!(log.finish(&[1], 1.0).unwrap().is_empty(), "čaká sa na pokračovanie");
        for f in 30..50 {
            log.observe(&frame, sz, &[item(2, 100 + 20 * f, Some(26.0))], f as f64 * 0.05).unwrap();
        }
        let saved = log.finish(&[2], 2.5).unwrap();
        assert!(saved.is_empty());
        let saved = log.finish(&[], 5.0).unwrap();
        assert_eq!(saved.len(), 1, "jeden objekt");
        assert_eq!(saved[0].id, 1);
        let csv = fs::read_to_string(format!("{dir}/historia.csv")).unwrap();
        assert_eq!(csv.lines().count(), 2, "{csv}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pokracovanie_ktore_zacalo_skor_nez_stary_zanikol_sa_spoji() {
        let dir = std::env::temp_dir().join(format!("historia_prekryv_{}", std::process::id()));
        let dir = dir.to_str().unwrap().to_string();
        let _ = fs::remove_dir_all(&dir);
        let mut log = ObjectLog::new(&dir, "", Clock::Video).unwrap();
        let frame = Mat::new_size_with_default(Size::new(1280, 720), opencv::core::CV_8UC3, Scalar::all(90.0)).unwrap();
        let sz = Size::new(1280, 720);
        for f in 0..20 {
            log.observe(&frame, sz, &[item(1, 100 + 20 * f, Some(25.0))], f as f64 * 0.05).unwrap();
        }
        // nová stopa 2 začne hneď (tracker ešte drží 1), 1 zanikne až neskôr
        for f in 22..45 {
            log.observe(&frame, sz, &[item(2, 100 + 20 * f, Some(25.0))], f as f64 * 0.05).unwrap();
            if f == 30 {
                log.finish(&[1], f as f64 * 0.05).unwrap();
            }
        }
        log.finish(&[2], 2.3).unwrap();
        let saved = log.finish(&[], 10.0).unwrap();
        assert_eq!(saved.len(), 1);
        let csv = fs::read_to_string(format!("{dir}/historia.csv")).unwrap();
        assert_eq!(csv.lines().count(), 2, "{csv}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn kratky_objekt_sa_neuklada() {
        let dir = std::env::temp_dir().join(format!("historia_kratke_{}", std::process::id()));
        let dir = dir.to_str().unwrap().to_string();
        let mut log = ObjectLog::new(&dir, "", Clock::Video).unwrap();
        let frame = Mat::new_size_with_default(Size::new(640, 360), opencv::core::CV_8UC3, Scalar::all(90.0)).unwrap();
        for f in 0..3 {
            log.observe(&frame, Size::new(640, 360), &[item(5, 100, None)], f as f64 * 0.05).unwrap();
        }
        assert!(log.finish(&[5], 10.0).unwrap().is_empty());
        // stojaci objekt (dlho, ale bez pohybu) sa tiež neuloží
        for f in 0..100 {
            log.observe(&frame, Size::new(640, 360), &[item(6, 100 + f % 3, None)], f as f64 * 0.05).unwrap();
        }
        assert!(log.finish_all().unwrap().is_empty());
        assert_eq!(log.saved(), 0);
        let _ = fs::remove_dir_all(&dir);
    }
}
