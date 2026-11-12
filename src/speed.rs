//! Rýchlosť objektov v km/h: prevod pixelu na metre na zemi podľa modelu kamery
//! (vypočítaného z meraní laserom, pozri `calib/calibrate.py`) a odhad rýchlosti z polohy
//! objektu na zemi za poslednú chvíľu.
//!
//! Model: kamera vo výške `h` nad zemou, sklonená o `pitch` nadol a naklonená o `roll`,
//! ohnisko `f` (px) a radiálne skreslenie širokouhlého objektívu `k1`, `k2`. Lúč z pixelu
//! sa pretne so zemou; výsledok je (X, Y) v metroch: X doprava, Y od kamery dopredu,
//! (0, 0) je miesto na zemi priamo pod kamerou.
//!
//! Model platí pre polohu kamery na snímke z merania (`calib/snapshot_4k.jpg`). Keď sa kamera
//! odvtedy pootočila, záber pri štarte sa so snímkou zarovná (`GroundMapper::with_alignment`)
//! a počas behu sa o pohyb kamery stará ukotvenie (objekty sa sledujú v súradniciach
//! referencie), takže model aj referenčné body 1–8 ostávajú na tom istom mieste ulice.

use std::{collections::HashMap, fs, path::Path};

use crate::{anchor::Homography, Result};

/// Model kamery: pixel -> bod na zemi (metre).
#[derive(Debug, Clone, PartialEq)]
pub struct GroundModel {
    pub f: f64,
    pub k1: f64,
    pub k2: f64,
    pub pitch: f64,
    pub roll: f64,
    pub h: f64,
    pub cx: f64,
    pub cy: f64,
    /// rozlíšenie snímky, z ktorej sa model počítal (px)
    pub width: f64,
    pub height: f64,
    /// snímka z merania (na zarovnanie so záberom pri štarte), ak existuje
    pub snapshot: Option<String>,
    /// referenčné body z merania: (číslo, x, y) v px snímky z merania
    pub points: Vec<(u32, f64, f64)>,
}

/// Číslo za kľúčom `"key":` v JSON texte (stačí na náš jednoduchý súbor, bez ďalšej knižnice).
pub(crate) fn json_number(text: &str, key: &str) -> Result<f64> {
    let pat = format!("\"{key}\":");
    let start = text.find(&pat).ok_or_else(|| format!("v modeli kamery chýba \"{key}\""))? + pat.len();
    let rest = text[start..].trim_start();
    let end = rest.find(|c: char| !(c.is_ascii_digit() || "+-.eE".contains(c))).unwrap_or(rest.len());
    rest[..end].parse::<f64>().map_err(|e| format!("\"{key}\" v modeli kamery: {e}").into())
}

impl GroundModel {
    /// Načíta model zo súboru `camera_model.json` (výstup `calib/calibrate.py`).
    /// Zo zložky modelu sa načíta aj snímka z merania (`snapshot_4k.jpg`) a body (`ref_points.csv`).
    pub fn load(path: &str) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(|e| format!("model kamery '{path}': {e}"))?;
        let mut m = Self::from_json(&text)?;
        let dir = Path::new(path).parent().unwrap_or(Path::new("."));
        let snap = dir.join("snapshot_4k.jpg");
        if snap.exists() {
            m.snapshot = Some(snap.to_string_lossy().into_owned());
        }
        if let Ok(csv) = fs::read_to_string(dir.join("ref_points.csv")) {
            m.points = parse_points(&csv);
        }
        Ok(m)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let n = |k| json_number(text, k);
        let m = GroundModel {
            f: n("f")?,
            k1: n("k1")?,
            k2: n("k2")?,
            pitch: n("pitch")?,
            roll: n("roll")?,
            h: n("h")?,
            cx: n("cx")?,
            cy: n("cy")?,
            width: n("width")?,
            height: n("height")?,
            snapshot: None,
            points: vec![],
        };
        if m.f <= 0.0 || m.h <= 0.0 || m.width <= 0.0 || m.height <= 0.0 {
            return Err("model kamery má nezmyselné hodnoty (f, h, width, height musia byť kladné)".into());
        }
        Ok(m)
    }

    /// Bod na zemi (X, Y v metroch) pre pixel (u, v) v rozlíšení modelu.
    /// `None`, ak lúč nemieri na zem (nad horizontom).
    pub fn to_ground(&self, u: f64, v: f64) -> Option<(f64, f64)> {
        let (mut x, mut y) = ((u - self.cx) / self.f, (v - self.cy) / self.f);
        // skreslenie objektívu: smerom od stredu sa obraz "stláča", tu ho naťahujeme späť
        let r2 = x * x + y * y;
        let s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        (x, y) = (x * s, y * s);
        let (c, sn) = (self.roll.cos(), self.roll.sin());
        (x, y) = (c * x - sn * y, sn * x + c * y);
        // smer lúča vo svete (Z hore): x doprava, dopredu a nadol podľa sklonu kamery
        let (sp, cp) = (self.pitch.sin(), self.pitch.cos());
        let dy = -y * sp + cp;
        let dz = -y * cp - sp;
        if dz > -1e-3 {
            return None;
        }
        let t = self.h / -dz;
        Some((t * x, t * dy))
    }
}

/// Body z `ref_points.csv` (id,x,y,popis); chybné riadky sa preskočia.
pub fn parse_points(csv: &str) -> Vec<(u32, f64, f64)> {
    csv.lines()
        .skip(1)
        .filter_map(|l| {
            let mut it = l.split(',');
            Some((it.next()?.trim().parse().ok()?, it.next()?.trim().parse().ok()?, it.next()?.trim().parse().ok()?))
        })
        .collect()
}

/// Za aké obdobie sa počíta rýchlosť (s). Kratšie = rýchlejšia reakcia, dlhšie = menej šumu.
pub const WINDOW_S: f64 = 1.2;
/// Pomalé objekty (chodec, pomalý bicykel): rýchlosť pod SLOW_KMH sa počíta z dlhšieho okna
/// LONG_WINDOW_S – pri malom posune za 1,2 s by chvenie rámčeka spravilo z 10 km/h aj 20.
pub const LONG_WINDOW_S: f64 = 2.4;
const SLOW_KMH: f64 = 15.0;
/// Bod ďalej od priamky než toto (m, a zároveň 3x typická odchýlka) sa pri výpočte vynechá –
/// skok rámčeka (zakrytie pilierom, spojenie s iným objektom).
const OUTLIER_M: f64 = 0.25;
/// Najkratší úsek, z ktorého sa rýchlosť ešte ukáže (s).
const MIN_SPAN_S: f64 = 0.5;
/// Najmenej vzoriek v okne.
const MIN_SAMPLES: usize = 4;
/// Body ďalej než toto (m) sa ignorujú (lúč tesne pod horizontom = nespoľahlivé).
pub const MAX_RANGE_M: f64 = 60.0;

/// O koľko (zlomok) sa smie výška rámčeka líšiť od mediánu v okne. Väčšia zmena = objekt je
/// čiastočne zakrytý (plot, pilier) a spodok rámčeka nie je na zemi – taká poloha sa nepoužije.
const SIZE_TOL: f64 = 0.2;

/// Jedna vzorka: čas, poloha ľavého a pravého spodného rohu na zemi, výška rámčeka (px).
type Sample = (f64, (f64, f64), (f64, f64), f64);

/// Odhad rýchlosti sledovaných objektov z ich polohy na zemi.
///
/// Rýchlosť sa počíta zvlášť z ľavého a z pravého okraja rámčeka a berie sa väčšia z nich:
/// keď objekt vychádza spoza piliera alebo zachádza za oblúk, jeden okraj rámčeka "stojí" na
/// prekážke (aj stred by išiel len polovičnou rýchlosťou), ale druhý sa hýbe s objektom.
#[derive(Debug, Default)]
pub struct SpeedTracker {
    history: HashMap<u32, Vec<Sample>>,
}

/// Rýchlosť (m/s) jednej súradnice v čase: sklon priamky metódou najmenších štvorcov.
fn slope(h: &[&Sample], f: impl Fn(&Sample) -> (f64, f64)) -> (f64, f64) {
    let n = h.len() as f64;
    let mt = h.iter().map(|s| s.0).sum::<f64>() / n;
    let mx = h.iter().map(|s| f(s).0).sum::<f64>() / n;
    let my = h.iter().map(|s| f(s).1).sum::<f64>() / n;
    let (mut stt, mut stx, mut sty) = (0.0, 0.0, 0.0);
    for s in h {
        let (x, y) = f(s);
        stt += (s.0 - mt) * (s.0 - mt);
        stx += (s.0 - mt) * (x - mx);
        sty += (s.0 - mt) * (y - my);
    }
    (stx / stt, sty / stt)
}

/// To isté, ale body, ktoré sú od priamky ďaleko (skok rámčeka), sa vynechajú a priamka sa
/// preloží znova.
fn robust_slope(h: &[&Sample], f: impl Fn(&Sample) -> (f64, f64) + Copy) -> (f64, f64) {
    let (vx, vy) = slope(h, f);
    let n = h.len() as f64;
    let mt = h.iter().map(|s| s.0).sum::<f64>() / n;
    let (mx, my) = (h.iter().map(|s| f(s).0).sum::<f64>() / n, h.iter().map(|s| f(s).1).sum::<f64>() / n);
    let res: Vec<f64> = h
        .iter()
        .map(|s| {
            let (x, y) = f(s);
            ((x - mx - vx * (s.0 - mt)).powi(2) + (y - my - vy * (s.0 - mt)).powi(2)).sqrt()
        })
        .collect();
    let mut sorted = res.clone();
    sorted.sort_by(f64::total_cmp);
    let limit = OUTLIER_M.max(3.0 * sorted[sorted.len() / 2]);
    let kept: Vec<&Sample> = h.iter().zip(&res).filter(|(_, r)| **r <= limit).map(|(s, _)| *s).collect();
    if kept.len() == h.len() || kept.len() < MIN_SAMPLES {
        return (vx, vy);
    }
    slope(&kept, f)
}

impl SpeedTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Zaznamená polohu objektu `id` (jeden bod na zemi) v čase `t` (s) a vráti rýchlosť (km/h).
    pub fn observe(&mut self, id: u32, t: f64, ground: (f64, f64)) -> Option<f64> {
        self.observe_box(id, t, ground, ground, 1.0)
    }

    /// Zaznamená ľavý a pravý spodný roh rámčeka na zemi a výšku rámčeka `size` (px);
    /// vzorky s náhle inou výškou (zakrytie zhora/zdola) sa vynechajú.
    pub fn observe_box(&mut self, id: u32, t: f64, left: (f64, f64), right: (f64, f64), size: f64) -> Option<f64> {
        let h = self.history.entry(id).or_default();
        if let Some(&(last, ..)) = h.last() {
            if t < last - 0.5 {
                h.clear(); // čas išiel výrazne dozadu (nové video / reštart) – začneme odznova
            } else if t <= last + 1e-3 {
                // snímka s rovnakým (alebo o kúsok starším) časom: záznamy z RTSP majú zhluky
                // snímok s jedným časom; posun "za nulový čas" by rýchlosť vyhnal do nebies
                return self.speed(id);
            }
        }
        h.push((t, left, right, size));
        h.retain(|s| t - s.0 <= LONG_WINDOW_S);
        self.speed(id)
    }

    /// Rýchlosť (km/h) z priamky preloženej polohami v okne (metóda najmenších štvorcov),
    /// takže jednotlivé roztrasené rámčeky ju veľmi neovplyvnia. Okno je WINDOW_S; ak je objekt
    /// pomalý (pod SLOW_KMH), použije sa dlhšie okno LONG_WINDOW_S.
    pub fn speed(&self, id: u32) -> Option<f64> {
        let all = self.history.get(&id)?;
        let t_last = all.last()?.0;
        let short = Self::speed_over(all, t_last - WINDOW_S);
        match short {
            Some(v) if v < SLOW_KMH => Self::speed_over(all, t_last - LONG_WINDOW_S).or(short),
            _ => short,
        }
    }

    fn speed_over(all: &[Sample], from: f64) -> Option<f64> {
        let win: Vec<&Sample> = all.iter().filter(|s| s.0 >= from - 1e-9).collect();
        if win.is_empty() {
            return None;
        }
        // len vzorky s "normálnou" výškou rámčeka (medián v okne)
        let mut sizes: Vec<f64> = win.iter().map(|s| s.3).collect();
        sizes.sort_by(f64::total_cmp);
        let med = sizes[sizes.len() / 2];
        let h: Vec<&Sample> = win.into_iter().filter(|s| (s.3 - med).abs() <= SIZE_TOL * med).collect();
        if h.len() < MIN_SAMPLES || h.last()?.0 - h.first()?.0 < MIN_SPAN_S {
            return None;
        }
        let (lx, ly) = robust_slope(&h, |s| s.1);
        let (rx, ry) = robust_slope(&h, |s| s.2);
        Some((lx * lx + ly * ly).sqrt().max((rx * rx + ry * ry).sqrt()) * 3.6)
    }

    /// Objekt zanikol.
    pub fn forget(&mut self, id: u32) {
        self.history.remove(&id);
    }
}

/// Prevod bodu zo súradníc počítania (referenčný záber) do rozlíšenia modelu a na zem.
#[derive(Debug, Clone)]
pub struct GroundMapper {
    pub model: GroundModel,
    /// px modelu na 1 px referenčného záberu
    pub scale: f64,
    /// referenčný záber -> snímka z merania (obe v rozlíšení referencie); identita, ak sa
    /// kamera medzi meraním a štartom nepohla
    pub calib_from_ref: Homography,
}

impl GroundMapper {
    /// `ref_w` x `ref_h` je rozlíšenie, v ktorom prebieha sledovanie (výstupný záber).
    /// Chyba, ak má záber iný pomer strán než snímka, z ktorej sa model počítal.
    pub fn new(model: GroundModel, ref_w: f64, ref_h: f64) -> Result<Self> {
        let (a, b) = (model.width / model.height, ref_w / ref_h);
        if (a - b).abs() > 0.02 {
            return Err(format!(
                "model kamery je pre pomer strán {:.3}, video má {:.3} (iná kamera alebo výrez?)",
                a, b
            )
            .into());
        }
        let scale = model.width / ref_w;
        Ok(Self { model, scale, calib_from_ref: Homography::IDENTITY })
    }

    /// Kamera sa od merania pootočila: `calib_from_ref` prevádza referenčný záber na snímku z merania.
    pub fn with_alignment(mut self, calib_from_ref: Homography) -> Self {
        self.calib_from_ref = calib_from_ref;
        self
    }

    /// Referenčné body z merania v súradniciach referenčného záberu (na vykreslenie).
    pub fn points_in_ref(&self) -> Vec<(u32, (f64, f64))> {
        let Some(inv) = self.calib_from_ref.inverse() else { return vec![] };
        self.model.points.iter().map(|&(id, x, y)| (id, inv.apply((x / self.scale, y / self.scale)))).collect()
    }

    /// Stred spodnej hrany rámčeka v px snímky z merania (vstup modelu, napr. na analýzu inými modelmi).
    pub fn foot_px(&self, x: f64, y: f64, w: f64, h: f64) -> (f64, f64) {
        let (u, v) = self.calib_from_ref.apply((x + w / 2.0, y + h));
        (u * self.scale, v * self.scale)
    }

    /// Ľavý a pravý spodný roh rámčeka na zemi.
    pub fn foot_edges(&self, x: f64, y: f64, w: f64, h: f64) -> Option<((f64, f64), (f64, f64))> {
        Some((self.foot(x, y, 0.0, h)?, self.foot(x + w, y, 0.0, h)?))
    }

    /// Bod na zemi pod objektom: stred spodnej hrany rámčeka (kolesá / chodidlá).
    pub fn foot(&self, x: f64, y: f64, w: f64, h: f64) -> Option<(f64, f64)> {
        let (u, v) = self.calib_from_ref.apply((x + w / 2.0, y + h));
        let g = self.model.to_ground(u * self.scale, v * self.scale)?;
        ((g.0 * g.0 + g.1 * g.1).sqrt() <= MAX_RANGE_M).then_some(g)
    }
}

/// Vyhladenie časov snímok. Kamera (Tapo cez RTSP) posiela snímky v zhlukoch s rovnakou časovou
/// značkou (napr. 8 snímok s jedným časom, potom skok o 0,4 s). Auto sa medzi nimi posunie, ale
/// "za nulový čas" – rýchlosť by vyskočila. Snímky v zhluku sa preto rozložia podľa priemerného
/// odstupu snímok (z posledných ~2 s); keď príde ďalšia značka, čas sa k nej znova priblíži.
pub struct FrameClock {
    last: Option<f64>,
    recent: std::collections::VecDeque<f64>,
}

impl Default for FrameClock {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameClock {
    pub fn new() -> Self {
        Self { last: None, recent: Default::default() }
    }

    /// Čas snímky (s) z časovej značky `pts` (s).
    pub fn next(&mut self, pts: f64) -> f64 {
        self.recent.push_back(pts);
        while self.recent.len() > 2 && pts - self.recent[0] > 2.0 {
            self.recent.pop_front();
        }
        // priemerný odstup snímok; na začiatku 20 fps
        let n = self.recent.len();
        let dt = if n >= 10 { (self.recent[n - 1] - self.recent[0]) / (n - 1) as f64 } else { 0.05 };
        let t = match self.last {
            // čas ide dozadu (nové video / reštart): prevezme sa
            Some(last) if pts < last - 1.0 => pts,
            Some(last) => pts.max(last + 0.9 * dt.clamp(0.02, 0.2)),
            None => pts,
        };
        self.last = Some(t);
        t
    }
}

#[cfg(test)]
mod tests_clock {
    use super::*;

    #[test]
    fn zhluky_s_rovnakym_casom_sa_rozlozia() {
        // skutočne 20 fps, ale značky chodia po 8 snímok s rovnakým časom
        let mut c = FrameClock::new();
        let ts: Vec<f64> = (0..80).map(|i| c.next((i / 8) as f64 * 0.4)).collect();
        // čas stále rastie a odstupy sú rozumné (nie nula ani 0,4 s)
        for w in ts.windows(2).skip(16) {
            let d = w[1] - w[0];
            assert!(d > 0.02 && d < 0.2, "{d}");
        }
        // a celkovo sedí so skutočným časom
        assert!((ts[79] - 79.0 * 0.05).abs() < 0.3, "{}", ts[79]);
    }

    #[test]
    fn pravidelne_znacky_ostanu_bez_zmeny() {
        let mut c = FrameClock::new();
        for i in 0..50 {
            let t = i as f64 * 0.05;
            assert!((c.next(t) - t).abs() < 1e-9);
        }
    }
}

#[cfg(test)]
mod tests_robust {
    use super::*;

    #[test]
    fn pomaly_objekt_so_skokmi_ramceka_ma_stabilnu_rychlost() {
        // 10 km/h (2,78 m/s), 20 fps; každá 7. snímka má rámček o 0,8 m vedľa (zakrytie)
        let mut st = SpeedTracker::new();
        let mut vals = vec![];
        for i in 0..80 {
            let t = i as f64 * 0.05;
            let mut x = 2.78 * t;
            if i % 7 == 3 {
                x += 0.8;
            }
            if let Some(v) = st.observe_box(1, t, (x, 10.0), (x + 1.8, 10.0), 60.0) {
                if t > 2.5 {
                    vals.push(v);
                }
            }
        }
        let (lo, hi) = vals.iter().fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(lo > 9.0 && hi < 11.0, "{lo} {hi}");
    }

    #[test]
    fn rychly_objekt_pouziva_kratke_okno() {
        // auto zrýchli z 30 na 50 km/h: po 1,2 s musí ukazovať blízko 50 (nie priemer 2,4 s)
        let mut st = SpeedTracker::new();
        let (mut x, mut v) = (0.0, None);
        for i in 0..80 {
            let t = i as f64 * 0.05;
            let speed = if t < 2.0 { 30.0 / 3.6 } else { 50.0 / 3.6 };
            x += speed * 0.05;
            v = st.observe_box(1, t, (x, 10.0), (x + 4.0, 10.0), 80.0).or(v);
        }
        let v = v.unwrap();
        assert!((v - 50.0).abs() < 2.0, "{v}");
    }
}

#[cfg(test)]
mod tests_duplicates {
    use super::*;

    #[test]
    fn snimky_s_rovnakym_casom_nezmenia_rychlost() {
        // auto ide 10 m/s (36 km/h); každá tretia snímka má rovnaký čas ako predchádzajúca
        // a polohu o kus ďalej (zhluk snímok v zázname)
        let mut st = SpeedTracker::new();
        let mut v = None;
        for i in 0..40 {
            let t = (i / 3) as f64 * 0.15;
            let x = i as f64 * 0.5;
            v = st.observe_box(1, t, (x, 10.0), (x + 4.0, 10.0), 50.0).or(v);
        }
        // bez ošetrenia by história mazala a rýchlosť by ušla; teraz ostane rozumná
        let v = v.unwrap();
        assert!(v > 20.0 && v < 60.0, "{v}");
    }

    #[test]
    fn navrat_casu_dozadu_zacne_odznova() {
        let mut st = SpeedTracker::new();
        for i in 0..20 {
            st.observe_box(1, i as f64 * 0.1, (i as f64, 0.0), (i as f64 + 4.0, 0.0), 50.0);
        }
        assert!(st.observe_box(1, 0.0, (0.0, 0.0), (4.0, 0.0), 50.0).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Model z meraní 5.11.2026 (calib/camera_model.json).
    const JSON: &str = r#"{
      "popis": "test",
      "params": { "f": 3068.363569853338, "k1": 1.3747005513703365, "k2": -9.99999998761368e-10,
                  "pitch": 0.12416688372729803, "roll": -0.01672659814828666, "h": 3.7694127709131457 },
      "cx": 1920.0, "cy": 1080.0,
      "points": { "1": [-9.231, 18.95] },
      "width": 3840, "height": 2160 }"#;

    fn model() -> GroundModel {
        GroundModel::from_json(JSON).unwrap()
    }

    fn close(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() < tol && (a.1 - b.1).abs() < tol
    }

    #[test]
    fn nacita_json() {
        let m = model();
        assert_eq!((m.width, m.height, m.cx), (3840.0, 2160.0, 1920.0));
        assert!((m.f - 3068.36).abs() < 0.01 && (m.h - 3.769).abs() < 0.001);
        assert!(m.k2.abs() < 1e-6, "záporný exponent sa musí načítať správne");
    }

    #[test]
    fn chybajuci_kluc_je_chyba() {
        let e = GroundModel::from_json(r#"{"f": 1.0}"#).unwrap_err().to_string();
        assert!(e.contains("k1"), "{e}");
    }

    #[test]
    fn body_sedia_s_pythonom() {
        // rovnaké body ako vypočítal calibrate.py
        let m = model();
        assert!(close(m.to_ground(1875.0, 2065.0).unwrap(), (-0.082, 7.316), 0.01));
        assert!(close(m.to_ground(710.0, 1242.0).unwrap(), (-9.231, 18.95), 0.01));
        assert!(close(m.to_ground(3385.0, 1250.0).unwrap(), (12.783, 19.971), 0.01));
    }

    #[test]
    fn nad_horizontom_nie_je_zem() {
        assert_eq!(model().to_ground(1920.0, 100.0), None);
    }

    #[test]
    fn nizsie_v_obraze_je_blizsie() {
        let m = model();
        let far = m.to_ground(1920.0, 1300.0).unwrap().1;
        let near = m.to_ground(1920.0, 2000.0).unwrap().1;
        assert!(near < far, "{near} {far}");
    }

    #[test]
    fn konstantna_rychlost() {
        // 10 m/s = 36 km/h pozdĺž X, 20 snímok za sekundu
        let mut s = SpeedTracker::new();
        let mut last = None;
        for i in 0..30 {
            let t = i as f64 / 20.0;
            last = s.observe(1, t, (10.0 * t, 15.0));
        }
        assert!((last.unwrap() - 36.0).abs() < 0.01, "{last:?}");
    }

    #[test]
    fn sum_sa_vyhladi() {
        // chodec 1,5 m/s (5,4 km/h) s roztraseným rámčekom ±20 cm
        let mut s = SpeedTracker::new();
        let mut last = None;
        for i in 0..40 {
            let t = i as f64 / 20.0;
            let noise = if i % 2 == 0 { 0.2 } else { -0.2 };
            last = s.observe(1, t, (1.5 * t, 18.0 + noise));
        }
        let v = last.unwrap();
        assert!((v - 5.4).abs() < 1.0, "{v}");
    }

    #[test]
    fn malo_dat_nema_rychlost() {
        let mut s = SpeedTracker::new();
        assert_eq!(s.observe(1, 0.0, (0.0, 0.0)), None);
        assert_eq!(s.observe(1, 0.1, (1.0, 0.0)), None);
        assert_eq!(s.observe(1, 0.2, (2.0, 0.0)), None);
        assert_eq!(s.observe(1, 0.3, (3.0, 0.0)), None, "úsek kratší než MIN_SPAN_S");
        assert!(s.observe(1, 0.6, (6.0, 0.0)).is_some());
    }

    #[test]
    fn stare_vzorky_vypadnu_z_okna() {
        // najprv stojí, potom ide 5 m/s: po uplynutí okna sa ukazuje len nová rýchlosť
        let mut s = SpeedTracker::new();
        for i in 0..40 {
            s.observe(1, i as f64 / 20.0, (0.0, 10.0));
        }
        let mut v = None;
        for i in 40..80 {
            let t = i as f64 / 20.0;
            v = s.observe(1, t, (5.0 * (t - 2.0), 10.0));
        }
        assert!((v.unwrap() - 18.0).abs() < 0.01, "{v:?}");
    }

    #[test]
    fn zakryty_objekt_nepokazi_rychlost() {
        // chodec 1,5 m/s; uprostred ho na 6 snímok zakryje pilier: rámček je o polovicu nižší
        // a jeho spodok (= "poloha na zemi") vyskočí o 4 m ďalej
        let mut s = SpeedTracker::new();
        let mut v = None;
        for i in 0..24 {
            let t = i as f64 / 20.0;
            let hidden = (9..15).contains(&i);
            let (y, size) = if hidden { (22.0, 60.0) } else { (18.0, 120.0) };
            v = s.observe_box(1, t, (1.5 * t, y), (1.5 * t, y), size);
        }
        let v = v.unwrap();
        assert!((v - 5.4).abs() < 0.3, "{v}");
    }

    #[test]
    fn okraj_na_prekazke_nespomali_auto() {
        // auto 10 m/s vychádza spoza piliera: ľavý okraj rámčeka stojí na pilieri, pravý ide s autom
        // (stred by ukázal len polovicu)
        let mut s = SpeedTracker::new();
        let mut v = None;
        for i in 0..20 {
            let t = i as f64 / 20.0;
            v = s.observe_box(1, t, (0.0, 12.0), (10.0 * t, 12.0), 100.0);
        }
        assert!((v.unwrap() - 36.0).abs() < 0.01, "{v:?}");
    }

    #[test]
    fn zabudnuty_objekt() {
        let mut s = SpeedTracker::new();
        for i in 0..20 {
            s.observe(3, i as f64 / 20.0, (i as f64, 0.0));
        }
        assert!(s.speed(3).is_some());
        s.forget(3);
        assert_eq!(s.speed(3), None);
    }

    #[test]
    fn mapper_preskaluje_a_kontroluje_pomer() {
        let g = GroundMapper::new(model(), 1280.0, 720.0).unwrap();
        assert!((g.scale - 3.0).abs() < 1e-9);
        // rámček so spodkom v bode 8 (1875, 2065) / 3
        let p = g.foot(1875.0 / 3.0 - 10.0, 2065.0 / 3.0 - 40.0, 20.0, 40.0).unwrap();
        assert!(close(p, (-0.082, 7.316), 0.05), "{p:?}");
        assert!(GroundMapper::new(model(), 1000.0, 1000.0).is_err());
    }

    #[test]
    fn zarovnanie_posunie_zem_aj_body() {
        // kamera sa od merania posunula: v referencii je všetko o 20 px vpravo a 10 px nižšie
        let mut m = model();
        m.points = vec![(8, 1875.0, 2065.0)];
        let back = Homography::translation(-20.0, -10.0); // referencia -> snímka z merania
        let g = GroundMapper::new(m, 1280.0, 720.0).unwrap().with_alignment(back);
        let p = g.foot(1875.0 / 3.0 + 20.0 - 10.0, 2065.0 / 3.0 + 10.0 - 40.0, 20.0, 40.0).unwrap();
        assert!(close(p, (-0.082, 7.316), 0.05), "zem sa má počítať z pôvodného miesta: {p:?}");
        let pts = g.points_in_ref();
        assert_eq!(pts[0].0, 8);
        assert!(close(pts[0].1, (1875.0 / 3.0 + 20.0, 2065.0 / 3.0 + 10.0), 1e-6), "{pts:?}");
    }

    #[test]
    fn body_z_csv() {
        let p = parse_points("id,x,y,popis\n1,710,1242,roh múru\nzly riadok\n8,1875,2065,pata\n");
        assert_eq!(p, vec![(1, 710.0, 1242.0), (8, 1875.0, 2065.0)]);
    }
}
