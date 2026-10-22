//! Testy celého reťazca počítania (tracker + počítadlo) na simulovaných detekciách,
//! bez videa a bez modelu: pohyb áut a ľudí cez čiaru, zakrytie plotom, stretnutie dvoch objektov.

use object_detect::{
    counter::{group_of, Direction, Group, LineCounter},
    tracker::Tracker,
};
use opencv::core::Rect;

const LINE_X: f64 = 640.0;

/// Jedna simulovaná scéna: v každej snímke zoznam (x ľavého okraja, šírka, popis triedy).
type Frame = Vec<(i32, i32, &'static str)>;

/// Prepustí scénu cez tracker a počítadlo tak ako `line_count` a vráti počítadlo.
fn run(frames: &[Frame]) -> LineCounter {
    let mut tracker = Tracker::new(12);
    let mut counter = LineCounter::new(LINE_X, 12.0);
    for (i, f) in frames.iter().enumerate() {
        let dets: Vec<(Rect, Group)> = f
            .iter()
            .filter_map(|&(x, w, label)| group_of(label).map(|g| (Rect::new(x, 300, w, 120), g)))
            .collect();
        let up = tracker.update(&dets);
        for ((rect, group), id) in dets.iter().zip(up.ids) {
            let cx = rect.x as f64 + rect.width as f64 / 2.0;
            counter.observe(i as u64, i as f64 / 12.0, id, *group, cx);
        }
        for id in up.ended {
            counter.forget(id);
        }
    }
    counter
}

/// Objekt (šírka `w`) ide z `from` do `to`; v snímkach `hidden` ho nevidno (zakrytý).
fn trip(label: &'static str, w: i32, from: i32, to: i32, frames: usize, hidden: std::ops::Range<usize>) -> Vec<Frame> {
    (0..frames)
        .map(|i| {
            if hidden.contains(&i) {
                vec![]
            } else {
                let x = from + (to - from) * i as i32 / (frames as i32 - 1);
                vec![(x, w, label)]
            }
        })
        .collect()
}

#[test]
fn auto_zlava_doprava() {
    let c = run(&trip("car", 300, -300, 1400, 40, 0..0));
    assert_eq!((c.count(Group::Vehicle, Direction::LeftToRight), c.total(Group::Vehicle)), (1, 1));
}

#[test]
fn auto_sprava_dolava() {
    let c = run(&trip("truck", 350, 1400, -350, 40, 0..0));
    assert_eq!(c.count(Group::Vehicle, Direction::RightToLeft), 1);
    assert_eq!(c.total(Group::Person), 0);
}

#[test]
fn clovek_sa_pocita_osobitne_od_auta() {
    let c = run(&trip("person", 60, 1300, -100, 60, 0..0));
    assert_eq!(c.count(Group::Person, Direction::RightToLeft), 1);
    assert_eq!(c.total(Group::Vehicle), 0);
}

#[test]
fn nepocitane_triedy_sa_ignoruju() {
    let c = run(&trip("dog", 80, -100, 1400, 40, 0..0));
    assert_eq!(c.total(Group::Person) + c.total(Group::Vehicle), 0);
}

#[test]
fn auto_zakryte_plotom_pred_ciarou_sa_pocita_raz() {
    // 5 snímok ho nevidno práve tesne pred čiarou
    let c = run(&trip("car", 300, -300, 1400, 40, 14..19));
    assert_eq!(c.total(Group::Vehicle), 1, "udalosti: {:?}", c.events());
}

#[test]
fn auto_zakryte_cez_ciaru_sa_pocita_raz() {
    // zakryté presne vtedy, keď prechádza cez čiaru (po skrytí je už na druhej strane)
    let c = run(&trip("car", 300, -300, 1400, 40, 16..22));
    assert_eq!(c.total(Group::Vehicle), 1, "udalosti: {:?}", c.events());
}

#[test]
fn dve_auta_po_sebe_su_dva() {
    let mut scene = trip("car", 300, -300, 1400, 40, 0..0);
    let second = trip("car", 300, -300, 1400, 40, 0..0);
    // druhé auto ide o 10 snímok za prvým
    let mut merged: Vec<Frame> = scene.drain(..).collect();
    merged.extend(std::iter::repeat(vec![]).take(10));
    for (i, f) in second.into_iter().enumerate() {
        merged[i + 10].extend(f);
    }
    let c = run(&merged);
    assert_eq!(c.count(Group::Vehicle, Direction::LeftToRight), 2);
}

#[test]
fn auto_a_clovek_proti_sebe() {
    let car = trip("car", 300, -300, 1400, 50, 0..0);
    let man = trip("person", 60, 1300, -100, 50, 0..0);
    let both: Vec<Frame> = car.into_iter().zip(man).map(|(mut a, b)| { a.extend(b); a }).collect();
    let c = run(&both);
    assert_eq!(c.count(Group::Vehicle, Direction::LeftToRight), 1);
    assert_eq!(c.count(Group::Person, Direction::RightToLeft), 1);
    assert_eq!(c.events().len(), 2);
}

#[test]
fn objekt_co_sa_otoci_pred_ciarou_sa_nepocita() {
    let mut scene = trip("person", 60, 100, 560, 30, 0..0); // dôjde k čiare
    scene.extend(trip("person", 60, 560, 100, 30, 0..0)); // a vráti sa
    let c = run(&scene);
    assert_eq!(c.total(Group::Person), 0);
}

#[test]
fn prazdne_video_ma_nulove_pocty() {
    let c = run(&vec![vec![]; 100]);
    assert_eq!(c.events().len(), 0);
    assert!(c.summary().contains("Autá: spolu 0") && c.summary().contains("Ľudia: spolu 0"));
}
