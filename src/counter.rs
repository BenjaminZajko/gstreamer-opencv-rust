//! Počítanie prechodov cez zvislú čiaru, zvlášť pre autá a ľudí a zvlášť podľa smeru.

use std::collections::HashMap;

use crate::labels::COCO;

/// Čo počítame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Group {
    /// auto, nákladiak, autobus, motorka
    Vehicle,
    /// chodec
    Person,
}

impl Group {
    pub fn name(self) -> &'static str {
        match self {
            Group::Vehicle => "auta",
            Group::Person => "ludia",
        }
    }
}

/// Smer prechodu cez zvislú čiaru.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    LeftToRight,
    RightToLeft,
}

impl Direction {
    pub fn name(self) -> &'static str {
        match self {
            Direction::LeftToRight => "zlava_doprava",
            Direction::RightToLeft => "sprava_dolava",
        }
    }
}

/// Čísla tried COCO, ktoré počítame: osoba, auto, motorka, autobus, nákladiak.
pub const COUNTED_CLASSES: [usize; 5] = [0, 2, 3, 5, 7];

/// Do ktorej skupiny patrí detekcia s týmto popisom (`None` = nepočítame).
pub fn group_of(label: &str) -> Option<Group> {
    match label {
        "person" | "clovek" => Some(Group::Person),
        "car" | "truck" | "bus" | "motorcycle" => Some(Group::Vehicle),
        _ => None,
    }
}

/// Skontroluje, že `COUNTED_CLASSES` sedia s názvami v `labels` (poistka proti preklepu).
pub fn counted_class_names() -> Vec<&'static str> {
    COUNTED_CLASSES.iter().map(|&c| COCO[c]).collect()
}

/// Počty na zobrazenie v paneli: poloha čiary, počty podľa skupiny a smeru a text v päte panela.
/// Oddelené od `LineCounter`, aby sa dali zobraziť aj iné počty než celkové (napr. od pripojenia diváka).
#[derive(Debug, Clone, Default)]
pub struct CountsView {
    pub line_x: f64,
    pub span: Option<(f64, f64)>,
    counts: HashMap<(Group, Direction), u32>,
    pub footer: String,
    /// Skutočné krajné body čiary v obraze, ak sa líšia od zvislej čiary `line_x` (napr. keď sa
    /// kamera otočila a čiara je naklonená). `None` = zvislá čiara na `line_x`.
    pub line_pts: Option<((f64, f64), (f64, f64))>,
    /// Zobraziť pätu panela červene (napr. stratené ukotvenie).
    pub warning: bool,
}

impl CountsView {
    pub fn new(line_x: f64, span: Option<(f64, f64)>, counts: HashMap<(Group, Direction), u32>) -> Self {
        Self { line_x, span, counts, footer: String::new(), line_pts: None, warning: false }
    }

    /// Nastaví text v päte panela (napr. "frame 5").
    pub fn with_footer(mut self, footer: impl Into<String>) -> Self {
        self.footer = footer.into();
        self
    }

    pub fn count(&self, group: Group, direction: Direction) -> u32 {
        self.counts.get(&(group, direction)).copied().unwrap_or(0)
    }

    pub fn total(&self, group: Group) -> u32 {
        self.count(group, Direction::LeftToRight) + self.count(group, Direction::RightToLeft)
    }
}

/// Jeden zaznamenaný prechod.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossEvent {
    pub frame: u64,
    pub time_s: f64,
    pub track: u32,
    pub group: Group,
    pub direction: Direction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Side {
    Left,
    Right,
}

/// Počítadlo prechodov cez zvislú čiaru na pozícii `line_x` (v pixeloch).
///
/// Strana objektu sa mení až keď jeho stred prejde čiaru o viac než `margin` pixelov (hysteréza),
/// takže objekt, ktorý sa chvíjou okolo čiary, sa nezapočíta viackrát.
pub struct LineCounter {
    line_x: f64,
    margin: f64,
    sides: HashMap<u32, Side>,
    counts: HashMap<(Group, Direction), u32>,
    events: Vec<CrossEvent>,
    /// Pásmo výšky (od, do) v pixeloch, v ktorom sa počíta; `None` = celá výška.
    span: Option<(f64, f64)>,
}

impl LineCounter {
    pub fn new(line_x: f64, margin: f64) -> Self {
        Self { line_x, margin, sides: HashMap::new(), counts: HashMap::new(), events: vec![], span: None }
    }

    /// Čiara platí len v pásme výšky `y0..y1` (px). Objekty mimo pásma sa nepočítajú
    /// (napr. aby sa nezapočítalo auto idúce po inej ulici nad cestou).
    pub fn with_span(mut self, y0: f64, y1: f64) -> Self {
        self.span = Some((y0, y1));
        self
    }

    pub fn span(&self) -> Option<(f64, f64)> {
        self.span
    }

    /// Ako `observe`, ale berie do úvahy aj výšku stredu objektu (`cy`) voči pásmu čiary.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_xy(&mut self, frame: u64, time_s: f64, track: u32, group: Group, cx: f64, cy: f64) -> Option<CrossEvent> {
        if let Some((y0, y1)) = self.span {
            if cy < y0 || cy > y1 {
                return None; // mimo pásma: strana sa nemení
            }
        }
        self.observe(frame, time_s, track, group, cx)
    }

    pub fn line_x(&self) -> f64 {
        self.line_x
    }

    /// Celkové počty od vzniku počítadla (kópia).
    pub fn counts_snapshot(&self) -> HashMap<(Group, Direction), u32> {
        self.counts.clone()
    }

    /// Celkové počty ako podklad pre panel.
    pub fn view(&self) -> CountsView {
        CountsView::new(self.line_x, self.span, self.counts_snapshot())
    }

    /// Spracuje polohu objektu (x súradnica stredu) v danej snímke. Vráti prechod, ak nastal.
    pub fn observe(&mut self, frame: u64, time_s: f64, track: u32, group: Group, cx: f64) -> Option<CrossEvent> {
        let now = if cx < self.line_x - self.margin {
            Side::Left
        } else if cx > self.line_x + self.margin {
            Side::Right
        } else {
            return None; // v pásme okolo čiary: strana sa nemení
        };
        let before = self.sides.insert(track, now);
        let direction = match (before, now) {
            (Some(Side::Left), Side::Right) => Direction::LeftToRight,
            (Some(Side::Right), Side::Left) => Direction::RightToLeft,
            _ => return None, // prvé zaradenie alebo žiadna zmena
        };
        *self.counts.entry((group, direction)).or_insert(0) += 1;
        let ev = CrossEvent { frame, time_s, track, group, direction };
        self.events.push(ev.clone());
        Some(ev)
    }

    /// Objekt zanikol – zabudneme jeho stranu.
    pub fn forget(&mut self, track: u32) {
        self.sides.remove(&track);
    }

    pub fn count(&self, group: Group, direction: Direction) -> u32 {
        self.counts.get(&(group, direction)).copied().unwrap_or(0)
    }

    /// Spolu oboma smermi.
    pub fn total(&self, group: Group) -> u32 {
        self.count(group, Direction::LeftToRight) + self.count(group, Direction::RightToLeft)
    }

    pub fn events(&self) -> &[CrossEvent] {
        &self.events
    }

    /// Udalosti ako CSV (hlavička + riadky).
    pub fn events_csv(&self) -> String {
        let mut s = String::from("time_s,frame,track_id,group,direction\n");
        for e in &self.events {
            s += &format!("{:.2},{},{},{},{}\n", e.time_s, e.frame, e.track, e.group.name(), e.direction.name());
        }
        s
    }

    /// Súhrn pre človeka.
    pub fn summary(&self) -> String {
        let line = |g: Group, label: &str| {
            format!(
                "{label}: spolu {}  (zľava doprava {}, sprava doľava {})\n",
                self.total(g),
                self.count(g, Direction::LeftToRight),
                self.count(g, Direction::RightToLeft)
            )
        };
        line(Group::Vehicle, "Autá") + &line(Group::Person, "Ľudia")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: Group = Group::Vehicle;
    const P: Group = Group::Person;

    /// Objekt sa posúva po osi x z `from` do `to` v krokoch; vráti počet zaznamenaných prechodov.
    fn walk(c: &mut LineCounter, track: u32, g: Group, from: f64, to: f64, step: f64) -> usize {
        let mut n = 0;
        let mut x = from;
        let mut frame = 0;
        while (step > 0.0 && x <= to) || (step < 0.0 && x >= to) {
            n += c.observe(frame, frame as f64 / 12.0, track, g, x).is_some() as usize;
            x += step;
            frame += 1;
        }
        n
    }

    #[test]
    fn prechod_zlava_doprava() {
        let mut c = LineCounter::new(500.0, 10.0);
        assert_eq!(walk(&mut c, 1, V, 100.0, 900.0, 20.0), 1);
        assert_eq!(c.count(V, Direction::LeftToRight), 1);
        assert_eq!(c.count(V, Direction::RightToLeft), 0);
        assert_eq!(c.total(V), 1);
    }

    #[test]
    fn prechod_sprava_dolava() {
        let mut c = LineCounter::new(500.0, 10.0);
        assert_eq!(walk(&mut c, 1, P, 900.0, 100.0, -20.0), 1);
        assert_eq!(c.count(P, Direction::RightToLeft), 1);
        assert_eq!(c.total(V), 0, "človek sa nepočíta medzi autá");
    }

    #[test]
    fn objekt_co_nepresiel_ciaru_sa_nepocita() {
        let mut c = LineCounter::new(500.0, 10.0);
        assert_eq!(walk(&mut c, 1, V, 100.0, 480.0, 20.0), 0);
        assert_eq!(c.total(V), 0);
    }

    #[test]
    fn objekt_ktory_zacal_za_ciarou_a_ostal_tam_sa_nepocita() {
        let mut c = LineCounter::new(500.0, 10.0);
        assert_eq!(walk(&mut c, 1, V, 600.0, 900.0, 20.0), 0);
    }

    #[test]
    fn chvenie_okolo_ciary_nepocita_viackrat() {
        let mut c = LineCounter::new(500.0, 15.0);
        c.observe(0, 0.0, 1, V, 400.0);
        // stred kmitá o ±10 px okolo čiary, to je v pásme (margin 15) a nič sa nezráta
        for (i, x) in [495.0, 505.0, 498.0, 507.0, 493.0, 504.0].iter().enumerate() {
            assert!(c.observe(i as u64 + 1, 0.0, 1, V, *x).is_none());
        }
        // až keď jasne prejde, zráta sa raz
        assert!(c.observe(20, 1.0, 1, V, 560.0).is_some());
        assert_eq!(c.total(V), 1);
    }

    #[test]
    fn tam_a_spat_su_dva_prechody() {
        let mut c = LineCounter::new(500.0, 10.0);
        walk(&mut c, 1, V, 100.0, 900.0, 20.0);
        walk(&mut c, 1, V, 900.0, 100.0, -20.0);
        assert_eq!(c.count(V, Direction::LeftToRight), 1);
        assert_eq!(c.count(V, Direction::RightToLeft), 1);
    }

    #[test]
    fn viac_objektov_sa_pocita_nezavisle() {
        let mut c = LineCounter::new(500.0, 10.0);
        walk(&mut c, 1, V, 100.0, 900.0, 20.0);
        walk(&mut c, 2, V, 100.0, 900.0, 20.0);
        walk(&mut c, 3, P, 900.0, 100.0, -20.0);
        assert_eq!((c.total(V), c.total(P)), (2, 1));
        assert_eq!(c.events().len(), 3);
    }

    #[test]
    fn zabudnute_id_sa_moze_pouzit_znova() {
        let mut c = LineCounter::new(500.0, 10.0);
        walk(&mut c, 1, V, 100.0, 900.0, 20.0);
        c.forget(1);
        // ID 1 už patrí inému objektu, ktorý začína vľavo – musí sa zrátať ako nový prechod
        walk(&mut c, 1, V, 100.0, 900.0, 20.0);
        assert_eq!(c.count(V, Direction::LeftToRight), 2);
    }

    #[test]
    fn csv_a_suhrn() {
        let mut c = LineCounter::new(500.0, 10.0);
        walk(&mut c, 7, V, 100.0, 900.0, 20.0);
        let csv = c.events_csv();
        let mut lines = csv.lines();
        assert_eq!(lines.next().unwrap(), "time_s,frame,track_id,group,direction");
        let row = lines.next().unwrap();
        assert!(row.ends_with(",7,auta,zlava_doprava"), "{row}");
        assert!(lines.next().is_none());
        let s = c.summary();
        assert!(s.contains("Autá: spolu 1") && s.contains("Ľudia: spolu 0"), "{s}");
    }

    #[test]
    fn skupiny_podla_popisu() {
        for l in ["car", "truck", "bus", "motorcycle"] {
            assert_eq!(group_of(l), Some(Group::Vehicle), "{l}");
        }
        assert_eq!(group_of("person"), Some(Group::Person));
        assert_eq!(group_of("clovek"), Some(Group::Person));
        for l in ["dog", "bicycle", "chair", ""] {
            assert_eq!(group_of(l), None, "{l}");
        }
    }

    #[test]
    fn pocitane_triedy_su_spravne_pomenovane() {
        assert_eq!(counted_class_names(), vec!["person", "car", "motorcycle", "bus", "truck"]);
        assert!(counted_class_names().iter().all(|n| group_of(n).is_some()));
    }

    #[test]
    fn pasmo_vysky_ignoruje_objekty_mimo() {
        let mut c = LineCounter::new(500.0, 10.0).with_span(100.0, 300.0);
        // auto ide po inej ulici vyššie (y = 50), mimo pásma: nesmie sa zrátať
        for (i, x) in [100.0, 400.0, 600.0, 900.0].iter().enumerate() {
            assert!(c.observe_xy(i as u64, 0.0, 1, V, *x, 50.0).is_none());
        }
        assert_eq!(c.total(V), 0);
        // auto v pásme (y = 200) sa zráta
        for (i, x) in [100.0, 400.0, 600.0].iter().enumerate() {
            c.observe_xy(i as u64, 0.0, 2, V, *x, 200.0);
        }
        assert_eq!(c.total(V), 1);
    }

    #[test]
    fn pasmo_hranice_su_vratane() {
        let mut c = LineCounter::new(500.0, 10.0).with_span(100.0, 300.0);
        c.observe_xy(0, 0.0, 1, V, 400.0, 100.0);
        assert!(c.observe_xy(1, 0.0, 1, V, 600.0, 300.0).is_some());
    }

    #[test]
    fn bez_pasma_sa_vyska_ignoruje() {
        let mut c = LineCounter::new(500.0, 10.0);
        assert_eq!(c.span(), None);
        c.observe_xy(0, 0.0, 1, V, 400.0, -5000.0);
        assert!(c.observe_xy(1, 0.0, 1, V, 600.0, 99999.0).is_some());
    }

    #[test]
    fn view_ma_rovnake_pocty_ako_pocitadlo() {
        let mut c = LineCounter::new(500.0, 10.0).with_span(10.0, 90.0);
        walk(&mut c, 1, V, 100.0, 900.0, 20.0);
        walk(&mut c, 2, P, 900.0, 100.0, -20.0);
        let v = c.view();
        assert_eq!(v.line_x, 500.0);
        assert_eq!(v.span, Some((10.0, 90.0)));
        assert_eq!((v.total(V), v.total(P)), (1, 1));
        assert_eq!(v.count(V, Direction::LeftToRight), 1);
        assert_eq!(v.count(P, Direction::RightToLeft), 1);
        assert_eq!(v.footer, "");
        assert_eq!(v.with_footer("pata").footer, "pata");
    }

    #[test]
    fn prazdny_view_ma_nuly() {
        let v = CountsView::default();
        assert_eq!(v.total(V) + v.total(P), 0);
    }
}
