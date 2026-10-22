//! Sledovanie objektov medzi snímkami: každému objektu pridelí stále číslo (ID), aby sa to isté
//! auto nepočítalo viackrát. Jednoduché párovanie "najbližší stred" v rámci tej istej skupiny.

use opencv::core::Rect;

use crate::counter::Group;

/// Jeden sledovaný objekt.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: u32,
    pub group: Group,
    pub rect: Rect,
    /// Koľko snímok za sebou sa objekt nenašiel (napr. ho zakryl plot).
    pub missed: u32,
}

/// Sledovač.
pub struct Tracker {
    tracks: Vec<Track>,
    next_id: u32,
    /// Po koľkých snímkach bez nálezu sa objekt zabudne.
    max_missed: u32,
}

/// Výsledok jednej aktualizácie.
#[derive(Debug, Default, PartialEq)]
pub struct Update {
    /// ID pre každú detekciu (rovnaké poradie ako vstup).
    pub ids: Vec<u32>,
    /// ID objektov, ktoré práve zanikli.
    pub ended: Vec<u32>,
}

fn center(r: &Rect) -> (f64, f64) {
    (r.x as f64 + r.width as f64 / 2.0, r.y as f64 + r.height as f64 / 2.0)
}

impl Tracker {
    pub fn new(max_missed: u32) -> Self {
        Self { tracks: vec![], next_id: 1, max_missed }
    }

    /// Aktuálne sledované objekty.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Spáruje nové detekcie s existujúcimi objektmi. Páruje sa len v rámci tej istej skupiny
    /// (auto s autom, človek s človekom). Vzdialenosť stredov musí byť v dosahu, ktorý rastie
    /// s počtom zmeškaných snímok (objekt sa medzitým mohol posunúť).
    pub fn update(&mut self, dets: &[(Rect, Group)]) -> Update {
        // všetky možné dvojice (objekt, detekcia) v dosahu, zoradené od najbližších
        let mut pairs: Vec<(f64, usize, usize)> = vec![];
        for (ti, t) in self.tracks.iter().enumerate() {
            let (tx, ty) = center(&t.rect);
            let reach = (t.rect.width.max(t.rect.height) as f64) * (1.0 + 0.5 * t.missed as f64) + 20.0;
            for (di, (r, g)) in dets.iter().enumerate() {
                if *g != t.group {
                    continue;
                }
                let (dx, dy) = center(r);
                let dist = ((tx - dx).powi(2) + (ty - dy).powi(2)).sqrt();
                if dist <= reach {
                    pairs.push((dist, ti, di));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut det_track: Vec<Option<usize>> = vec![None; dets.len()];
        let mut track_used = vec![false; self.tracks.len()];
        for (_, ti, di) in pairs {
            if !track_used[ti] && det_track[di].is_none() {
                track_used[ti] = true;
                det_track[di] = Some(ti);
            }
        }

        // aktualizácia nájdených, nové objekty pre nespárované detekcie
        let mut ids = Vec::with_capacity(dets.len());
        for (di, (rect, group)) in dets.iter().enumerate() {
            match det_track[di] {
                Some(ti) => {
                    self.tracks[ti].rect = *rect;
                    self.tracks[ti].missed = 0;
                    ids.push(self.tracks[ti].id);
                }
                None => {
                    let id = self.next_id;
                    self.next_id += 1;
                    self.tracks.push(Track { id, group: *group, rect: *rect, missed: 0 });
                    ids.push(id);
                }
            }
        }
        // nové objekty (pridané vyššie) nemajú záznam v track_used – počítajú sa ako použité
        track_used.resize(self.tracks.len(), true);

        // nenájdené objekty: zmeškaná snímka, po limite zaniknú
        let mut ended = vec![];
        for (ti, used) in track_used.iter().enumerate() {
            if !used {
                self.tracks[ti].missed += 1;
            }
        }
        let max = self.max_missed;
        self.tracks.retain(|t| {
            let keep = t.missed <= max;
            if !keep {
                ended.push(t.id);
            }
            keep
        });
        Update { ids, ended }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAR: Group = Group::Vehicle;
    const MAN: Group = Group::Person;

    fn r(x: i32, y: i32) -> Rect {
        Rect::new(x, y, 100, 60)
    }

    #[test]
    fn novy_objekt_dostane_id() {
        let mut t = Tracker::new(5);
        assert_eq!(t.update(&[(r(0, 0), CAR)]).ids, vec![1]);
    }

    #[test]
    fn pohybujuci_sa_objekt_si_drzi_id() {
        let mut t = Tracker::new(5);
        let mut ids = vec![];
        for i in 0..20 {
            ids.push(t.update(&[(r(i * 15, 100), CAR)]).ids[0]);
        }
        assert!(ids.iter().all(|&i| i == 1), "{ids:?}");
    }

    #[test]
    fn dva_objekty_maju_roznne_id_a_nezamenia_sa() {
        let mut t = Tracker::new(5);
        for i in 0..20 {
            let u = t.update(&[(r(i * 10, 0), CAR), (r(800 - i * 10, 300), CAR)]);
            assert_eq!(u.ids, vec![1, 2], "snímka {i}");
        }
    }

    #[test]
    fn rozne_skupiny_sa_neparuju() {
        let mut t = Tracker::new(5);
        t.update(&[(r(0, 0), CAR)]);
        // na tom istom mieste je teraz človek – má dostať nové ID
        assert_eq!(t.update(&[(r(0, 0), MAN)]).ids, vec![2]);
    }

    #[test]
    fn objekt_zakryty_na_chvilu_si_zachova_id() {
        let mut t = Tracker::new(10);
        t.update(&[(r(0, 0), CAR)]);
        for _ in 0..4 {
            t.update(&[]); // zakrytý plotom
        }
        assert_eq!(t.update(&[(r(120, 0), CAR)]).ids, vec![1], "po 4 chýbajúcich snímkach to je stále to isté auto");
    }

    #[test]
    fn objekt_zanikne_po_limite() {
        let mut t = Tracker::new(2);
        t.update(&[(r(0, 0), CAR)]);
        assert!(t.update(&[]).ended.is_empty());
        assert!(t.update(&[]).ended.is_empty());
        assert_eq!(t.update(&[]).ended, vec![1]);
        assert!(t.tracks().is_empty());
        // po zániku sa vráti ako nový objekt
        assert_eq!(t.update(&[(r(0, 0), CAR)]).ids, vec![2]);
    }

    #[test]
    fn velmi_vzdialena_detekcia_je_novy_objekt() {
        let mut t = Tracker::new(5);
        t.update(&[(r(0, 0), CAR)]);
        assert_eq!(t.update(&[(r(2000, 900), CAR)]).ids, vec![2]);
    }

    #[test]
    fn prazdny_vstup_nespadne() {
        let mut t = Tracker::new(3);
        assert_eq!(t.update(&[]), Update::default());
    }

    #[test]
    fn jedna_detekcia_nepatri_dvom_objektom() {
        let mut t = Tracker::new(5);
        t.update(&[(r(0, 0), CAR), (r(60, 0), CAR)]);
        // jedna detekcia medzi nimi: spáruje sa len s jedným, druhý ostane bez nálezu
        let u = t.update(&[(r(30, 0), CAR)]);
        assert_eq!(u.ids.len(), 1);
        assert_eq!(t.tracks().iter().filter(|x| x.missed == 1).count(), 1);
    }
}
