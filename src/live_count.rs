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
    tracker::Tracker,
    Result,
};

/// Sledovanie + počítanie (čistá logika bez videa a modelu – dobre sa testuje).
pub struct CountingCore {
    tracker: Tracker,
    counter: LineCounter,
}

impl CountingCore {
    /// `line_x` a `span` sú v pixeloch výstupného záberu.
    pub fn new(line_x: f64, span: (f64, f64), margin: f64, max_missed: u32) -> Self {
        Self {
            tracker: Tracker::new(max_missed),
            counter: LineCounter::new(line_x, margin).with_span(span.0, span.1),
        }
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
        let counted: Vec<_> = dets
            .iter()
            .filter_map(|d| group_of(&d.label).map(|g| (d, g)))
            .map(|(d, g)| (d, g, to_ref(d.rect)))
            .collect();
        let update = self.tracker.update(&counted.iter().map(|(_, g, r)| (*r, *g)).collect::<Vec<_>>());

        let (mut items, mut events) = (Vec::with_capacity(counted.len()), vec![]);
        for ((d, group, rref), id) in counted.into_iter().zip(update.ids) {
            let cx = rref.x as f64 + rref.width as f64 / 2.0;
            let cy = rref.y as f64 + rref.height as f64 / 2.0;
            if let Some(ev) = self.counter.observe_xy(frame_idx, time_s, id, group, cx, cy) {
                events.push(ev);
            }
            items.push(Counted { rect: d.rect, label: d.label.clone(), score: d.score, id, group });
        }
        for id in update.ended {
            self.counter.forget(id);
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
    // výrez sa skopíruje (asi 1,7 MB), aby bol súvislý v pamäti
    let view = Mat::roi(frame_hi, roi)?.try_clone()?;
    Ok(detector
        .detect(&view)?
        .into_iter()
        .map(|d| Detection { rect: to_output_coords(d.rect, roi, scale), ..d })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counter::{Direction, Group};

    fn det(label: &str, x: i32, y: i32) -> Detection {
        Detection { rect: Rect::new(x, y, 100, 60), label: label.into(), score: Some(0.9) }
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
        Counted { rect: Rect::new(0, 0, 10, 10), label: "x".into(), score: None, id, group }
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
}
