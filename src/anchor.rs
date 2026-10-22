//! Ukotvenie čiary a výrezu k scéne: keď sa kamera pootočí alebo posunie, čiara ostane na tom istom
//! mieste ULICE (nie obrazovky).
//!
//! Princíp: pri štarte si uložíme referenčný záber. Z neho sa vyberie malá PEVNÁ sada stabilných bodov
//! (predvolene 10). V každej ďalšej snímke sa sledujú práve tieto isté body (nie nové) voči
//! referenčnej fotke a z ich polohy sa vypočíta homografia = ako sa záber voči referencii posunul,
//! otočil alebo priblížil. Touto transformáciou sa čiara a výrez presunú zo "súradníc referencie" do
//! súradníc aktuálnej snímky; objekty sa naopak prepočítajú do referencie a počítajú sa tam.
//!
//! Tri fázy (pozri `AnchorMode`):
//!   1. Kalibrácia: pár prvých snímok sa poloha určuje cez ORB (globálne hľadanie bodov) a zároveň sa
//!      meria, ktoré z ~40 kandidátov sú stabilné (nehýbu sa, nemenia vzhľad – nie sú to ľudia, autá,
//!      časová pečiatka kamery...). Z nich sa vyberie `points` bodov rozložených po celom zábere.
//!   2. Jemné sledovanie: sleduje sa už len tých `points` bodov (Lucas-Kanade voči referencii,
//!      presnosť na zlomky pixela). Pri stojacej kamere sú výsledky prakticky rovnaké a mŕtve pásmo
//!      (`deadband`) zaručí, že čiara nevibruje.
//!   3. Záloha: ak sa body stratia (kamera sa prudko pohla), poloha sa nájde znova cez ORB.

use opencv::{
    core::{no_array, DMatch, KeyPoint, Mat, Point, Point2f, Ptr, Scalar, Size, TermCriteria, Vector, NORM_HAMMING},
    features::{self, BFMatcher, DescriptorMatcherTrait, Feature2DTrait, ORB},
    geometry, imgproc,
    prelude::*,
    video,
};

use crate::Result;

/// Homografia 3x3 (riadkovo): zobrazenie bodov z jednej snímky do druhej.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography(pub [f64; 9]);

impl Homography {
    pub const IDENTITY: Homography = Homography([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    /// Posun o (dx, dy).
    pub fn translation(dx: f64, dy: f64) -> Self {
        Homography([1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0])
    }

    /// Prevedie bod.
    pub fn apply(&self, (x, y): (f64, f64)) -> (f64, f64) {
        let m = &self.0;
        let w = m[6] * x + m[7] * y + m[8];
        ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)
    }

    /// Zloženie transformácií: najprv `other`, potom `self` (t. j. `self ∘ other`).
    pub fn compose(&self, other: &Homography) -> Homography {
        let (a, b) = (&self.0, &other.0);
        let mut m = [0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                m[r * 3 + c] = (0..3).map(|k| a[r * 3 + k] * b[k * 3 + c]).sum();
            }
        }
        let n = if m[8].abs() > 1e-12 { m[8] } else { 1.0 };
        Homography(m.map(|v| v / n))
    }

    /// Inverzná homografia (`None`, ak je matica singulárna).
    pub fn inverse(&self) -> Option<Homography> {
        let m = &self.0;
        let (a, b, c, d, e, f, g, h, i) = (m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = [
            (e * i - f * h) / det,
            (c * h - b * i) / det,
            (b * f - c * e) / det,
            (f * g - d * i) / det,
            (a * i - c * g) / det,
            (c * d - a * f) / det,
            (d * h - e * g) / det,
            (b * g - a * h) / det,
            (a * e - b * d) / det,
        ];
        // normalizácia (posledný prvok = 1), ak sa dá
        let n = if inv[8].abs() > 1e-12 { inv[8] } else { 1.0 };
        Some(Homography(inv.map(|v| v / n)))
    }

    /// Najväčší posun rohov obrazu `w` x `h` medzi dvoma transformáciami (v pixeloch).
    pub fn max_corner_shift(&self, other: &Homography, w: f64, h: f64) -> f64 {
        [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
            .iter()
            .map(|&p| {
                let (a, b) = (self.apply(p), other.apply(p));
                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
            })
            .fold(0.0, f64::max)
    }

    /// Dáva transformácia zmysel pre obraz `w` x `h`? Zamieta zdegenerované výsledky
    /// (obraz stlačený na čiaru, extrémne zväčšenie, "pretočenie").
    pub fn is_plausible(&self, w: f64, h: f64) -> bool {
        let corners: Vec<(f64, f64)> =
            [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].iter().map(|&p| self.apply(p)).collect();
        if corners.iter().any(|c| !c.0.is_finite() || !c.1.is_finite()) {
            return false;
        }
        // plocha štvoruholníka (shoelace); záporná = pretočenie
        let area: f64 = (0..4)
            .map(|i| {
                let (a, b) = (corners[i], corners[(i + 1) % 4]);
                a.0 * b.1 - b.0 * a.1
            })
            .sum::<f64>()
            / 2.0;
        let ratio = area / (w * h);
        // kamera sa nemôže zmeniť o viac než ~3x v ploche
        (0.3..3.0).contains(&ratio)
    }

    fn from_mat(m: &Mat) -> Option<Homography> {
        if m.rows() != 3 || m.cols() != 3 {
            return None;
        }
        let mut v = [0.0; 9];
        for r in 0..3 {
            for c in 0..3 {
                v[(r * 3 + c) as usize] = *m.at_2d::<f64>(r, c).ok()?;
            }
        }
        let n = if v[8].abs() > 1e-12 { v[8] } else { 1.0 };
        Some(Homography(v.map(|x| x / n)))
    }
}

/// Parametre ukotvenia.
#[derive(Debug, Clone, Copy)]
pub struct AnchorConfig {
    /// Najviac toľko bodov ORB hľadá v snímke (globálne hľadanie polohy).
    pub features: i32,
    /// Minimálny počet zhodných bodov pri globálnom hľadaní (ORB), aby sa poloha považovala za spoľahlivú.
    pub min_inliers: usize,
    /// Koľko stabilných bodov sa vyberie na jemné sledovanie (0 = vypnuté, len ORB).
    pub points: usize,
    /// Koľko aktualizácií trvá kalibrácia (výber stabilných bodov).
    pub calib_frames: usize,
    /// Mŕtve pásmo v px: poloha čiary sa nezmení, kým sa kamera neposunie aspoň o toľko.
    pub deadband: f64,
}

impl Default for AnchorConfig {
    fn default() -> Self {
        Self { features: 1500, min_inliers: 20, points: 10, calib_frames: 12, deadband: 1.0 }
    }
}

/// V akej fáze je ukotvenie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorMode {
    /// Vyberajú sa stabilné body (prvé snímky).
    Calibrating,
    /// Sleduje sa pevná sada bodov (normálny stav).
    Fine,
    /// Jemné sledovanie sa nepodarilo nastaviť (primálo stabilných bodov) – používa sa len ORB.
    OrbOnly,
}

/// Koľko kandidátov na stabilné body sa vyberie z referencie.
const CANDIDATES: usize = 48;
/// Okraj obrazu (px), v ktorom sa kandidáti nehľadajú (okno sledovania by vyšlo z obrazu).
const MARGIN: f32 = 28.0;
/// Bod je pri kalibrácii "dobrý", ak sa odchýli od modelu o menej než toľko px.
const CALIB_TOL: f64 = 1.5;
/// Maximálna odchýlka pri kontrole sledovania tam a späť (px).
const FB_TOL: f64 = 1.0;

/// Po koľkých aktualizáciách, počas ktorých kamera stojí, sa na novom mieste nastaví nový referenčný záber.
const RELOCK_FRAMES: usize = 8;
/// Kamera "stojí", ak sa medián pohybu bodov medzi dvoma aktualizáciami je menší než toľko px.
/// Meria sa priamo z obrazu (snímka voči predošlej), nezávisle od odhadu polohy.
const STATIC_TOL: f64 = 0.5;
/// Koľko aktualizácií po zastavení kamery sa ešte poloha dolaďuje; potom sa zmrazí.
const SETTLE_UPDATES: usize = 3;
/// Keď kamera stojí, poloha sa zmení len ak sa odhad líši o viac než toľko px (v strednej časti obrazu)...
const FREEZE_MAX: f64 = 8.0;
/// ...a to opakovane toľko aktualizácií za sebou (pomalé posúvanie kamery, nie kolísanie odhadu).
const DRIFT_UPDATES: usize = 5;
/// Koľko aktualizácií sa čaká pred ďalším pokusom o nový záber po neúspechu.
const RELOCK_COOLDOWN: usize = 20;
/// Dve polohy kamery sa považujú za rovnaké miesto, ak sa rohy líšia o menej než toľko px
/// (odhad polohy kolíše o niekoľko px, preto nie príliš málo).
const SAME_POSE: f64 = 15.0;
/// Najviac toľko referenčných záberov sa pamätá (pôvodný sa nikdy nevyhodí).
const MAX_KEYS: usize = 6;

/// Referenčný záber: obraz, v ňom vybrané stabilné body a poloha, v akej bol urobený voči pôvodnej referencii.
/// Sledovanie voči záberu urobenému v tej istej polohe nie je zaťažené skreslením objektívu.
struct KeyFrame {
    gray: Mat,
    /// vybrané stabilné body v súradniciach tohto záberu
    pts: Vec<Point2f>,
    /// pôvodná referencia -> tento záber
    key_from_ref: Homography,
    ref_from_key: Homography,
    last_used: u64,
}

/// Stav kalibrácie: ako sa správali kandidáti.
struct Calib {
    /// obraz, v ktorom sú kandidáti (referencia alebo nový záber)
    gray: Mat,
    cand: Vec<Point2f>,
    ok: Vec<u32>,
    sum: Vec<f64>,
    /// Koľko snímok sa použilo na meranie (len tie, v ktorých poloha bola známa).
    used: u32,
    /// Koľko aktualizácií už prebehlo.
    updates: usize,
    /// `None` = počiatočná kalibrácia (poloha z ORB); `Some(h)` = nový záber na pevnej polohe `h`
    /// (kamera stojí, kandidáti sa porovnávajú s identitou).
    fixed_pose: Option<Homography>,
}

impl Calib {
    fn new(gray: Mat, cand: Vec<Point2f>, fixed_pose: Option<Homography>) -> Self {
        let n = cand.len();
        Calib { gray, cand, ok: vec![0; n], sum: vec![0.0; n], used: 0, updates: 0, fixed_pose }
    }
}

/// Sledovač polohy kamery voči referenčnému záberu.
pub struct Anchor {
    cfg: AnchorConfig,
    orb: Ptr<ORB>,
    matcher: BFMatcher,
    ref_pts: Vec<Point2f>,
    size: (f64, f64),
    /// referencia -> aktuálna snímka
    cur_from_ref: Homography,
    ref_from_cur: Homography,
    tracking: bool,
    inliers: usize,
    /// body na vykreslenie (v aktuálnej snímke)
    shown: Vec<Point2f>,
    mode: AnchorMode,
    /// počiatočná kalibrácia
    calib: Option<Calib>,
    /// kalibrácia nového záberu po presune kamery
    pending: Option<Calib>,
    /// referenčné zábery (prvý je pôvodná referencia)
    keys: Vec<KeyFrame>,
    active: usize,
    /// jemné sledovanie sa nepodarilo, poloha sa hľadá globálne (ORB)
    fallback: bool,
    cooldown: usize,
    tick: u64,
    /// predošlá snímka (na meranie, či sa kamera hýbe)
    prev_gray: Option<Mat>,
    /// medián pohybu obrazu pri poslednej aktualizácii (px)
    motion: f64,
    /// koľko aktualizácií za sebou kamera stojí
    static_n: usize,
    /// koľko aktualizácií za sebou sa pri stojacej kamere odhad výrazne líši (pomalý posun)
    drift_n: usize,
}

/// Rozloží `k` bodov čo najrovnomernejšie: začne od prvého a stále pridáva ten, ktorý je najďalej
/// od už vybraných (maximin). Vráti indexy v poradí výberu.
pub fn pick_spread(points: &[(f64, f64)], k: usize) -> Vec<usize> {
    if points.is_empty() || k == 0 {
        return vec![];
    }
    let mut chosen = vec![0usize];
    let mut min_d: Vec<f64> = points.iter().map(|p| dist2(*p, points[0])).collect();
    while chosen.len() < k.min(points.len()) {
        let next = (0..points.len())
            .filter(|i| !chosen.contains(i))
            .max_by(|&a, &b| min_d[a].total_cmp(&min_d[b]));
        let Some(n) = next else { break };
        chosen.push(n);
        for (i, p) in points.iter().enumerate() {
            min_d[i] = min_d[i].min(dist2(*p, points[n]));
        }
    }
    chosen
}

fn dist2(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)
}

fn pt(p: &Point2f) -> (f64, f64) {
    (p.x as f64, p.y as f64)
}

/// Sleduje body z referenčnej snímky v aktuálnej (Lucas-Kanade, vždy voči referencii, takže sa
/// chyba nehromadí). Začína sa z odhadu `guess`. Výsledok sa overí sledovaním späť: bod sa prijme,
/// len ak sa vráti na pôvodné miesto (do `FB_TOL` px). Nesledovateľné body sú `None`.
pub fn track_points(
    ref_gray: &Mat,
    cur_gray: &Mat,
    pts_ref: &[Point2f],
    guess: &Homography,
) -> Result<Vec<Option<Point2f>>> {
    if pts_ref.is_empty() {
        return Ok(vec![]);
    }
    let prev: Vector<Point2f> = pts_ref.iter().copied().collect();
    let mut next: Vector<Point2f> = pts_ref
        .iter()
        .map(|p| {
            let (x, y) = guess.apply(pt(p));
            Point2f::new(x as f32, y as f32)
        })
        .collect();
    let (mut st, mut err) = (Vector::<u8>::new(), Vector::<f32>::new());
    let crit = TermCriteria::new(3, 30, 0.01)?; // COUNT + EPS
    let win = Size::new(31, 31);
    video::calc_optical_flow_pyr_lk(
        ref_gray, cur_gray, &prev, &mut next, &mut st, &mut err, win, 4, crit, video::OPTFLOW_USE_INITIAL_FLOW, 1e-4,
    )?;
    // kontrola späť: z aktuálnej snímky do referencie
    let mut back = prev.clone();
    let (mut st_b, mut err_b) = (Vector::<u8>::new(), Vector::<f32>::new());
    video::calc_optical_flow_pyr_lk(
        cur_gray, ref_gray, &next, &mut back, &mut st_b, &mut err_b, win, 4, crit, video::OPTFLOW_USE_INITIAL_FLOW, 1e-4,
    )?;
    Ok((0..pts_ref.len())
        .map(|i| {
            let (a, b) = (prev.get(i).ok()?, back.get(i).ok()?);
            let fb = ((a.x - b.x) as f64).hypot((a.y - b.y) as f64);
            let p = next.get(i).ok()?;
            (st.get(i).ok()? == 1 && st_b.get(i).ok()? == 1 && fb <= FB_TOL).then_some(p)
        })
        .collect())
}

impl Anchor {
    /// Vytvorí referenciu zo šedotónovej snímky. Zlyhá, ak má záber primálo detailov.
    pub fn new(reference_gray: &Mat, cfg: AnchorConfig) -> Result<Self> {
        let mut orb = ORB::create(cfg.features, 1.2, 8, 31, 0, 2, opencv::features::ORB_ScoreType::HARRIS_SCORE, 31, 20)?;
        let (mut kp, mut desc) = (Vector::<KeyPoint>::new(), Mat::default());
        orb.detect_and_compute(reference_gray, &no_array(), &mut kp, &mut desc, false)?;
        if kp.len() < cfg.min_inliers * 2 {
            return Err(format!(
                "v referenčnom zábere je primálo výrazných bodov ({}, treba aspoň {}); ukotvenie nie je možné",
                kp.len(),
                cfg.min_inliers * 2
            )
            .into());
        }
        let (w, h) = (reference_gray.cols() as f64, reference_gray.rows() as f64);
        let pts: Vec<(f64, f64)> = kp.iter().map(|k| (k.pt().x as f64, k.pt().y as f64)).collect();
        if !spread_ok(&pts, w, h) {
            return Err("referenčné body sú sústredené na malej ploche záberu (chýbajú detaily po celom \
                        zábere); ukotvenie nie je možné"
                .into());
        }
        let mut matcher = BFMatcher::new(NORM_HAMMING, false)?;
        let mut train = Vector::<Mat>::new();
        train.push(desc);
        matcher.add(&train)?;

        // kandidáti na stabilné body: silné rohy rozložené po celom zábere, nie pri okraji
        let (calib, mode) = if cfg.points > 0 {
            let cand = Self::candidates(reference_gray)?;
            if cand.len() >= cfg.points {
                (Some(Calib::new(reference_gray.try_clone()?, cand, None)), AnchorMode::Calibrating)
            } else {
                (None, AnchorMode::OrbOnly)
            }
        } else {
            (None, AnchorMode::OrbOnly)
        };

        Ok(Self {
            cfg,
            orb,
            matcher,
            ref_pts: kp.iter().map(|k| k.pt()).collect(),
            size: (w, h),
            cur_from_ref: Homography::IDENTITY,
            ref_from_cur: Homography::IDENTITY,
            tracking: true,
            inliers: kp.len(),
            shown: vec![],
            mode,
            calib,
            pending: None,
            keys: vec![],
            active: 0,
            fallback: false,
            cooldown: 0,
            tick: 0,
            prev_gray: None,
            motion: 0.0,
            static_n: 0,
            drift_n: 0,
        })
    }

    /// Silné rohy z obrazu, rozložené po celom zábere (maximin), bez okraja.
    fn candidates(gray: &Mat) -> Result<Vec<Point2f>> {
        let mut corners = Vector::<Point2f>::new();
        features::good_features_to_track_def(gray, &mut corners, 400, 0.01, 12.0)?;
        let (w, h) = (gray.cols() as f32, gray.rows() as f32);
        let inside: Vec<Point2f> = corners
            .iter()
            .filter(|p| p.x > MARGIN && p.y > MARGIN && p.x < w - MARGIN && p.y < h - MARGIN)
            .collect();
        let xy: Vec<(f64, f64)> = inside.iter().map(pt).collect();
        Ok(pick_spread(&xy, CANDIDATES).into_iter().map(|i| inside[i]).collect())
    }

    /// Aktuálna fáza ukotvenia.
    pub fn mode(&self) -> AnchorMode {
        self.mode
    }

    /// Vybrané stabilné body pôvodnej referencie (prázdne, kým kalibrácia neskončí).
    pub fn chosen_points(&self) -> &[Point2f] {
        self.keys.first().map(|k| k.pts.as_slice()).unwrap_or(&[])
    }

    /// Koľko referenčných záberov sa pamätá (1 = len pôvodný).
    pub fn keyframes(&self) -> usize {
        self.keys.len()
    }

    /// Ktorý referenčný záber sa práve používa (0 = pôvodný).
    pub fn active_key(&self) -> usize {
        self.active
    }

    /// Hľadá sa poloha globálne, lebo jemné sledovanie sa nepodarilo (kamera sa pohla)?
    pub fn in_fallback(&self) -> bool {
        self.fallback
    }

    /// Nastavuje sa práve nový referenčný záber (kamera sa ustálila na novom mieste)?
    pub fn relocking(&self) -> bool {
        self.pending.is_some()
    }

    /// Medián pohybu obrazu pri poslednej aktualizácii (px). Malý = kamera stojí.
    pub fn motion(&self) -> f64 {
        self.motion
    }

    /// Stojí kamera (podľa pohybu obrazu, nie podľa odhadu polohy)?
    pub fn camera_static(&self) -> bool {
        self.static_n > 0
    }

    /// Je poloha čiary zmrazená (kamera stojí dosť dlho)?
    pub fn frozen(&self) -> bool {
        self.static_n > SETTLE_UPDATES
    }

    /// Zmeria, o koľko sa obraz pohol oproti predošlej aktualizácii: medián posunu výrazných bodov
    /// (pohybujúce sa autá a ľudia medián neovplyvnia, ak netvoria väčšinu obrazu).
    fn measure_motion(&mut self, gray: &Mat) -> Result<f64> {
        let motion = match &self.prev_gray {
            None => 0.0,
            Some(prev) => {
                let mut pts = Vector::<Point2f>::new();
                features::good_features_to_track_def(prev, &mut pts, 150, 0.01, 10.0)?;
                if pts.len() < 10 {
                    0.0
                } else {
                    let mut next = pts.clone();
                    let (mut st, mut err) = (Vector::<u8>::new(), Vector::<f32>::new());
                    let crit = TermCriteria::new(3, 20, 0.03)?;
                    video::calc_optical_flow_pyr_lk(
                        prev, gray, &pts, &mut next, &mut st, &mut err, Size::new(21, 21), 3, crit, 0, 1e-4,
                    )?;
                    let mut d: Vec<f64> = (0..pts.len())
                        .filter(|&i| st.get(i).map(|v| v == 1).unwrap_or(false))
                        .filter_map(|i| {
                            let (a, b) = (pts.get(i).ok()?, next.get(i).ok()?);
                            Some(((a.x - b.x) as f64).hypot((a.y - b.y) as f64))
                        })
                        .collect();
                    if d.len() < 10 {
                        f64::INFINITY // väčšina bodov sa stratila: veľký pohyb
                    } else {
                        d.sort_by(|a, b| a.total_cmp(b));
                        d[d.len() / 2]
                    }
                }
            }
        };
        self.prev_gray = Some(gray.try_clone()?);
        Ok(motion)
    }

    /// Najväčší rozdiel dvoch polôh v strednej časti obrazu (tam, kde je skreslenie objektívu malé).
    fn inner_shift(&self, a: &Homography, b: &Homography) -> f64 {
        let (w, h) = self.size;
        let mut m: f64 = 0.0;
        for fx in [0.25, 0.5, 0.75] {
            for fy in [0.25, 0.5, 0.75] {
                let p = (w * fx, h * fy);
                let (x, y) = (a.apply(p), b.apply(p));
                m = m.max(((x.0 - y.0).powi(2) + (x.1 - y.1).powi(2)).sqrt());
            }
        }
        m
    }

    /// Prijme nový odhad polohy podľa pravidiel:
    ///   * kamera sa hýbe (alebo sa práve zastavila): zmena, ak je väčšia než mŕtve pásmo `band`,
    ///   * kamera stojí dlhšie: poloha je zmrazená; zmení sa len ak sa odhad líši o viac než
    ///     `FREEZE_MAX` opakovane `DRIFT_UPDATES`-krát (kamera sa veľmi pomaly posunula).
    /// Vráti, či sa poloha zmenila.
    fn publish(&mut self, new_h: Homography, inv: Homography, band: f64) -> bool {
        if self.frozen() {
            if self.inner_shift(&new_h, &self.cur_from_ref) < FREEZE_MAX {
                self.drift_n = 0;
                return false;
            }
            self.drift_n += 1;
            if self.drift_n < DRIFT_UPDATES {
                return false;
            }
        }
        self.drift_n = 0;
        if new_h.max_corner_shift(&self.cur_from_ref, self.size.0, self.size.1) >= band {
            self.cur_from_ref = new_h;
            self.ref_from_cur = inv;
            return true;
        }
        false
    }

    /// Krátky popis stavu do panela.
    pub fn status_text(&self) -> String {
        match self.mode {
            AnchorMode::Calibrating => "kalib.".into(),
            AnchorMode::OrbOnly => format!("{}", self.inliers),
            AnchorMode::Fine if self.pending.is_some() => "nastavujem".into(),
            AnchorMode::Fine if self.fallback => "hladam".into(),
            AnchorMode::Fine => format!("{}/{}", self.inliers, self.keys[self.active].pts.len()),
        }
    }

    /// Aktualizuje polohu z novej snímky (šedotónovej, rovnakého rozmeru ako referencia).
    /// Vráti, či sa sledovanie podarilo. Pri neúspechu ostáva posledná dobrá transformácia.
    pub fn update(&mut self, gray: &Mat) -> Result<bool> {
        self.tick += 1;
        // stojí kamera? (meria sa priamo z obrazu, nezávisle od odhadu polohy)
        self.motion = self.measure_motion(gray)?;
        if self.motion < STATIC_TOL {
            self.static_n += 1;
        } else {
            self.static_n = 0;
            self.drift_n = 0;
        }
        match self.mode {
            AnchorMode::Calibrating => {
                let ok = self.orb_update(gray)?;
                self.calibrate_step(gray, ok)?;
                Ok(ok)
            }
            AnchorMode::Fine => {
                let ok = if self.fine_try(gray)? {
                    self.fallback = false;
                    true
                } else {
                    // jemné sledovanie sa nepodarilo (kamera sa pohla): poloha sa nájde globálne
                    self.fallback = true;
                    self.orb_update(gray)?
                };
                if ok {
                    self.relock_step(gray)?;
                } else {
                    self.pending = None;
                }
                Ok(ok)
            }
            AnchorMode::OrbOnly => self.orb_update(gray),
        }
    }

    /// Zmeria, ktorí kandidáti sa držia modelu v tejto snímke.
    fn observe(cal: &mut Calib, gray: &Mat, guess: &Homography, model: &Homography) -> Result<()> {
        cal.used += 1;
        let tracked = track_points(&cal.gray, gray, &cal.cand, guess)?;
        for (i, t) in tracked.iter().enumerate() {
            if let Some(p) = t {
                let want = model.apply(pt(&cal.cand[i]));
                let r = ((p.x as f64 - want.0).powi(2) + (p.y as f64 - want.1).powi(2)).sqrt();
                if r <= CALIB_TOL {
                    cal.ok[i] += 1;
                    cal.sum[i] += r;
                }
            }
        }
        Ok(())
    }

    /// Z výsledkov kalibrácie vyberie `points` stabilných bodov rozložených po zábere (alebo `None`).
    fn select_stable(&self, cal: &Calib) -> Option<Vec<Point2f>> {
        // stabilní kandidáti = držali sa modelu aspoň v 80 % použitých snímok
        let mut stable: Vec<(f64, Point2f)> = vec![];
        if cal.used >= 3 {
            for i in 0..cal.cand.len() {
                if cal.ok[i] as f64 >= 0.8 * cal.used as f64 {
                    stable.push((cal.sum[i] / cal.ok[i].max(1) as f64, cal.cand[i]));
                }
            }
        }
        stable.sort_by(|a, b| a.0.total_cmp(&b.0)); // najpresnejší prví
        let pool: Vec<Point2f> = stable.iter().take(self.cfg.points * 3).map(|s| s.1).collect();
        let xy: Vec<(f64, f64)> = pool.iter().map(pt).collect();
        let picked: Vec<Point2f> = pick_spread(&xy, self.cfg.points).into_iter().map(|i| pool[i]).collect();
        let xy_picked: Vec<(f64, f64)> = picked.iter().map(pt).collect();
        let enough = picked.len() >= self.cfg.points.min(6);
        (enough && spread_ok_min(&xy_picked, self.size.0, self.size.1, FINE_SPREAD)).then_some(picked)
    }

    /// Jeden krok počiatočnej kalibrácie; po poslednom kroku vyberie body.
    fn calibrate_step(&mut self, gray: &Mat, pose_known: bool) -> Result<()> {
        let Some(mut cal) = self.calib.take() else { return Ok(()) };
        cal.updates += 1;
        if pose_known {
            let h = self.cur_from_ref;
            Self::observe(&mut cal, gray, &h, &h)?;
        }
        if cal.updates < self.cfg.calib_frames {
            self.calib = Some(cal);
            return Ok(());
        }
        match self.select_stable(&cal) {
            Some(picked) => {
                self.keys.push(KeyFrame {
                    gray: cal.gray,
                    pts: picked,
                    key_from_ref: Homography::IDENTITY,
                    ref_from_key: Homography::IDENTITY,
                    last_used: self.tick,
                });
                self.active = 0;
                self.mode = AnchorMode::Fine;
            }
            None => self.mode = AnchorMode::OrbOnly,
        }
        Ok(())
    }

    /// Po presune kamery: keď kamera stojí na novom mieste, nastaví sa tam nový referenčný záber
    /// (sledovanie voči záberu z toho istého miesta nie je zaťažené skreslením objektívu).
    fn relock_step(&mut self, gray: &Mat) -> Result<()> {
        // prebieha kalibrácia nového záberu
        if let Some(mut cal) = self.pending.take() {
            if !self.camera_static() {
                // kamera sa pohla: zrušiť a skúsiť, keď znova zastane
                self.cooldown = 0;
                return Ok(());
            }
            let fixed = cal.fixed_pose.unwrap_or(self.cur_from_ref);
            cal.updates += 1;
            Self::observe(&mut cal, gray, &Homography::IDENTITY, &Homography::IDENTITY)?;
            if cal.updates < self.cfg.calib_frames {
                self.pending = Some(cal);
                return Ok(());
            }
            match self.select_stable(&cal) {
                Some(picked) => self.add_key(cal.gray, picked, fixed),
                None => self.cooldown = RELOCK_COOLDOWN,
            }
            return Ok(());
        }

        // kamera je pri polohe aktívneho záberu: netreba nič robiť
        let h = self.cur_from_ref;
        let near_key = self
            .keys
            .get(self.active)
            .map(|k| self.inner_shift(&h, &k.key_from_ref) < SAME_POSE)
            .unwrap_or(true);
        if near_key {
            return Ok(());
        }
        if self.cooldown > 0 {
            self.cooldown -= 1;
            return Ok(());
        }
        // kamera stojí dosť dlho ďaleko od aktívneho záberu: nový záber v tejto polohe
        if self.static_n >= RELOCK_FRAMES {
            let cand = Self::candidates(gray)?;
            if cand.len() >= self.cfg.points {
                self.pending = Some(Calib::new(gray.try_clone()?, cand, Some(h)));
            } else {
                self.cooldown = RELOCK_COOLDOWN;
            }
        }
        Ok(())
    }

    /// Pridá nový referenčný záber (alebo ním nahradí záber z tej istej polohy).
    fn add_key(&mut self, gray: Mat, pts: Vec<Point2f>, pose: Homography) {
        let key = KeyFrame {
            gray,
            pts,
            key_from_ref: pose,
            ref_from_key: pose.inverse().unwrap_or(Homography::IDENTITY),
            last_used: self.tick,
        };
        // záber z tej istej polohy (okrem pôvodného) sa nahradí novším
        let same = (1..self.keys.len())
            .find(|&i| self.keys[i].key_from_ref.max_corner_shift(&pose, self.size.0, self.size.1) < SAME_POSE);
        let idx = if let Some(i) = same {
            self.keys[i] = key;
            i
        } else {
            if self.keys.len() >= MAX_KEYS {
                // vyhodí sa najdávnejšie použitý (pôvodný ostáva)
                if let Some(old) = (1..self.keys.len()).min_by_key(|&i| self.keys[i].last_used) {
                    self.keys.remove(old);
                }
            }
            self.keys.push(key);
            self.keys.len() - 1
        };
        self.active = idx;
    }

    /// Sledovanie bodov jedného referenčného záberu. Vráti novú polohu (pôvodná referencia ->
    /// aktuálna snímka) a počet zhodných bodov, alebo `None`.
    fn track_key(&self, idx: usize, gray: &Mat) -> Result<Option<(Homography, usize)>> {
        let key = &self.keys[idx];
        let min_good = self.cfg.points.min(6).max(5);
        // odhad: kam sa presunuli body záberu (záber -> referencia -> aktuálna snímka)
        let guess = self.cur_from_ref.compose(&key.ref_from_key);
        let tracked = track_points(&key.gray, gray, &key.pts, &guess)?;
        let (mut src, mut dst) = (Vector::<Point2f>::new(), Vector::<Point2f>::new());
        for (r, t) in key.pts.iter().zip(&tracked) {
            if let Some(p) = t {
                src.push(*r);
                dst.push(*p);
            }
        }
        if src.len() < min_good {
            return Ok(None);
        }
        let mut mask = Mat::default();
        let h = geometry::find_homography(&src, &dst, geometry::RANSAC, 2.0, &mut mask, 2000, 0.995)?;
        let Some(h_cur_from_key) = Homography::from_mat(&h) else { return Ok(None) };
        let flags = mask.data_bytes().map(|b| b.to_vec()).unwrap_or_default();
        let inlier_key: Vec<(f64, f64)> =
            src.iter().zip(&flags).filter(|(_, f)| **f != 0).map(|(p, _)| pt(&p)).collect();
        if inlier_key.len() < min_good || !spread_ok_min(&inlier_key, self.size.0, self.size.1, FINE_SPREAD) {
            return Ok(None);
        }
        let new_cur_from_ref = h_cur_from_key.compose(&key.key_from_ref);
        if !new_cur_from_ref.is_plausible(self.size.0, self.size.1) {
            return Ok(None);
        }
        Ok(Some((new_cur_from_ref, inlier_key.len())))
    }

    /// Jemné sledovanie: skúsi aktívny záber, potom ostatné. Vráti `false`, ak sa nepodarilo
    /// (vtedy sa použije globálne hľadanie).
    fn fine_try(&mut self, gray: &Mat) -> Result<bool> {
        self.tracking = false;
        self.shown.clear();
        // najprv záber s najbližšou polohou: sledovanie je najpresnejšie, keď je kamera blízko miesta,
        // z ktorého záber vznikol (podobné skreslenie obrazu)
        let (w, h, cur) = (self.size.0, self.size.1, self.cur_from_ref);
        let mut order: Vec<usize> = (0..self.keys.len()).collect();
        order.sort_by(|&a, &b| {
            let (da, db) = (self.keys[a].key_from_ref.max_corner_shift(&cur, w, h), self.keys[b].key_from_ref.max_corner_shift(&cur, w, h));
            da.total_cmp(&db)
        });
        for idx in order {
            let Some((new_h, inliers)) = self.track_key(idx, gray)? else { continue };
            let Some(inv) = new_h.inverse() else { continue };
            // mŕtve pásmo a zmrazenie pri stojacej kamere (čiara nevibruje)
            let band = self.cfg.deadband;
            self.publish(new_h, inv, band);
            self.active = idx;
            self.keys[idx].last_used = self.tick;
            self.inliers = inliers;
            self.shown = self.active_points();
            self.tracking = true;
            return Ok(true);
        }
        self.inliers = 0;
        Ok(false)
    }

    /// Body aktívneho záberu premietnuté do aktuálnej snímky (len tie, čo sú v obraze). Pri
    /// stojacej kamere sa nehýbu.
    fn active_points(&self) -> Vec<Point2f> {
        let Some(key) = self.keys.get(self.active) else { return vec![] };
        key.pts
            .iter()
            .map(|p| {
                let r = key.ref_from_key.apply(pt(p));
                self.cur_from_ref.apply(r)
            })
            .filter(|&(x, y)| x >= 0.0 && y >= 0.0 && x < self.size.0 && y < self.size.1)
            .map(|(x, y)| Point2f::new(x as f32, y as f32))
            .collect()
    }

    /// Globálne hľadanie polohy (ORB + RANSAC). Slúži na kalibráciu a ako záloha.
    fn orb_update(&mut self, gray: &Mat) -> Result<bool> {
        self.tracking = false;
        self.shown.clear();

        let (mut kp, mut desc) = (Vector::<KeyPoint>::new(), Mat::default());
        self.orb.detect_and_compute(gray, &no_array(), &mut kp, &mut desc, false)?;
        if kp.len() < self.cfg.min_inliers || desc.rows() < 2 {
            self.inliers = 0;
            return Ok(false);
        }

        // k najbližších susedov a Loweho test pomeru: berieme len jednoznačné zhody
        let mut knn = Vector::<Vector<DMatch>>::new();
        self.matcher.knn_match_def(&desc, &mut knn, 2)?;
        let (mut src, mut dst) = (Vector::<Point2f>::new(), Vector::<Point2f>::new());
        for pair in knn.iter() {
            if pair.len() == 2 && pair.get(0)?.distance < 0.75 * pair.get(1)?.distance {
                let m = pair.get(0)?;
                src.push(self.ref_pts[m.train_idx as usize]);
                dst.push(kp.get(m.query_idx as usize)?.pt());
            }
        }
        if src.len() < self.cfg.min_inliers {
            self.inliers = src.len();
            return Ok(false);
        }

        // homografia s RANSAC (body na pohybujúcich sa objektoch odpadnú ako odľahlé)
        let mut mask = Mat::default();
        let h = geometry::find_homography(&src, &dst, geometry::RANSAC, 3.0, &mut mask, 2000, 0.995)?;
        let Some(new_h) = Homography::from_mat(&h) else {
            self.inliers = 0;
            return Ok(false);
        };
        let flags = mask.data_bytes().map(|b| b.to_vec()).unwrap_or_default();
        let inlier_ref: Vec<(f64, f64)> = src
            .iter()
            .zip(&flags)
            .filter(|(_, f)| **f != 0)
            .map(|(p, _)| (p.x as f64, p.y as f64))
            .collect();
        let inliers = inlier_ref.len();
        self.inliers = inliers;
        if inliers < self.cfg.min_inliers
            || !spread_ok(&inlier_ref, self.size.0, self.size.1)
            || !new_h.is_plausible(self.size.0, self.size.1)
        {
            return Ok(false);
        }
        let Some(inv) = new_h.inverse() else { return Ok(false) };

        // mŕtve pásmo: globálne hľadanie je menej presné (skreslenie objektívu, pohybujúce sa
        // objekty), preto je pásmo väčšie, keď existujú presnejšie referenčné zábery
        let band = if self.mode == AnchorMode::OrbOnly { self.cfg.deadband } else { self.cfg.deadband.max(2.0) };
        self.publish(new_h, inv, band);
        // body na vykreslenie
        match self.mode {
            AnchorMode::Calibrating => {} // počas kalibrácie sa nekreslia
            AnchorMode::Fine => self.shown = self.active_points(), // stále tie isté body, nie blikajúce
            AnchorMode::OrbOnly => {
                self.shown = dst
                    .iter()
                    .zip(flags)
                    .filter(|(_, f)| *f != 0)
                    .map(|(p, _)| p)
                    .collect();
            }
        }
        self.tracking = true;
        Ok(true)
    }

    /// Je poloha kamery práve spoľahlivo známa?
    pub fn tracking(&self) -> bool {
        self.tracking
    }

    /// Počet zhodných bodov pri poslednej aktualizácii (pri jemnom sledovaní z vybraných bodov).
    pub fn inliers(&self) -> usize {
        self.inliers
    }

    /// Referencia -> aktuálna snímka.
    pub fn to_current(&self, p: (f64, f64)) -> (f64, f64) {
        self.cur_from_ref.apply(p)
    }

    /// Aktuálna snímka -> referencia.
    pub fn to_reference(&self, p: (f64, f64)) -> (f64, f64) {
        self.ref_from_cur.apply(p)
    }

    /// Rámček z aktuálnej snímky -> najmenší obdĺžnik v súradniciach referencie, ktorý ho obsahuje.
    pub fn rect_to_reference(&self, r: opencv::core::Rect) -> opencv::core::Rect {
        let corners = rect_corners(r.x as f64, r.y as f64, r.width as f64, r.height as f64)
            .map(|p| self.to_reference(p));
        let (x0, y0, x1, y1) = bounding_box(&corners);
        opencv::core::Rect::new(
            x0.round() as i32,
            y0.round() as i32,
            ((x1 - x0).round() as i32).max(1),
            ((y1 - y0).round() as i32).max(1),
        )
    }

    pub fn homography(&self) -> Homography {
        self.cur_from_ref
    }

    /// Body na vykreslenie v aktuálnej snímke, najviac `max`. Pri jemnom sledovaní sú to vybrané
    /// stabilné body (stále tie isté), počas kalibrácie nič.
    pub fn points(&self, max: usize) -> Vec<Point2f> {
        if self.shown.len() <= max || max == 0 {
            return self.shown.iter().take(max).copied().collect();
        }
        let step = self.shown.len() as f64 / max as f64;
        (0..max).map(|i| self.shown[(i as f64 * step) as usize]).collect()
    }
}

/// Aké pokrytie záberu sa vyžaduje od malej sady bodov pri jemnom sledovaní (menej prísne než ORB).
pub const FINE_SPREAD: f64 = 0.25;

/// Obdĺžnik (x, y, w, h) -> jeho štyri rohy.
pub fn rect_corners(x: f64, y: f64, w: f64, h: f64) -> [(f64, f64); 4] {
    [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
}

/// Najmenší obdĺžnik (x0, y0, x1, y1), ktorý obsahuje všetky body.
pub fn bounding_box(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let xs = points.iter().map(|p| p.0);
    let ys = points.iter().map(|p| p.1);
    (
        xs.clone().fold(f64::INFINITY, f64::min),
        ys.clone().fold(f64::INFINITY, f64::min),
        xs.fold(f64::NEG_INFINITY, f64::max),
        ys.fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Sú body rozložené po zábere? Posun odhadnutý z bodov nahromadených na malej ploche (napr. len
/// na hranách jedného objektu) je nespoľahlivý, preto sa vyžaduje, aby ich ohraničujúci obdĺžnik
/// pokrýval aspoň `MIN_SPREAD` šírky a výšky záberu.
pub fn spread_ok(points: &[(f64, f64)], w: f64, h: f64) -> bool {
    spread_ok_min(points, w, h, MIN_SPREAD)
}

/// To isté s vlastným minimom pokrytia (zlomok šírky a výšky záberu).
pub fn spread_ok_min(points: &[(f64, f64)], w: f64, h: f64, min: f64) -> bool {
    if points.len() < 2 {
        return false;
    }
    let (x0, y0, x1, y1) = bounding_box(points);
    (x1 - x0) >= min * w && (y1 - y0) >= min * h
}

/// Aká časť šírky a výšky záberu musí byť pokrytá bodmi (pozri `spread_ok`).
pub const MIN_SPREAD: f64 = 0.3;

/// Nakreslí malé body (zhodné referenčné body) do snímky.
pub fn draw_points(frame: &mut Mat, points: &[Point2f]) -> Result<()> {
    for p in points {
        imgproc::circle(
            frame,
            Point::new(p.x.round() as i32, p.y.round() as i32),
            3,
            Scalar::new(255.0, 255.0, 0.0, 0.0), // azúrová (BGR)
            -1,
            imgproc::LINE_AA,
            0,
        )?;
    }
    Ok(())
}

/// Referenčná fotka: záber, z ktorého sa pri štarte vychádza, s vybranými stabilnými bodmi
/// (očíslované), čiarou a výrezom. Slúži na kontrolu, čo sa ako referencia navrhlo.
pub fn render_reference(
    reference: &Mat,
    chosen: &[Point2f],
    line: ((f64, f64), (f64, f64)),
    roi: &[(f64, f64); 4],
) -> Result<Mat> {
    let mut img = reference.try_clone()?;
    // výrez (sivý) a čiara (žltá)
    let quad: Vector<Point> = roi.iter().map(|p| Point::new(p.0.round() as i32, p.1.round() as i32)).collect();
    imgproc::polylines(&mut img, &quad, true, Scalar::all(170.0), 1, imgproc::LINE_8, 0)?;
    imgproc::line(
        &mut img,
        Point::new(line.0 .0.round() as i32, line.0 .1.round() as i32),
        Point::new(line.1 .0.round() as i32, line.1 .1.round() as i32),
        Scalar::new(0.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        0,
    )?;
    // vybrané body s číslami
    for (i, p) in chosen.iter().enumerate() {
        let c = Point::new(p.x.round() as i32, p.y.round() as i32);
        imgproc::circle(&mut img, c, 6, Scalar::new(255.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_AA, 0)?;
        imgproc::put_text(
            &mut img,
            &(i + 1).to_string(),
            Point::new(c.x + 9, c.y - 9),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            Scalar::new(255.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Rect, Size, CV_8UC1};

    const W: i32 = 640;
    const H: i32 = 360;

    /// Deterministický "náhodný" generátor (rovnaký výsledok pri každom spustení).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, n: u64) -> i32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) % n) as i32
        }
    }

    /// Umelá scéna s množstvom tvarov, aby mala dosť výrazných bodov (rohov).
    fn scene(seed: u64) -> Mat {
        let mut img = Mat::new_rows_cols_with_default(H, W, CV_8UC1, Scalar::all(120.0)).unwrap();
        let mut r = Lcg(seed);
        for _ in 0..260 {
            let c = Scalar::all(30.0 + r.next(200) as f64);
            let (x, y) = (r.next(W as u64 - 60), r.next(H as u64 - 60));
            match r.next(3) {
                0 => {
                    let rect = Rect::new(x, y, 8 + r.next(50), 8 + r.next(50));
                    imgproc::rectangle(&mut img, rect, c, -1, imgproc::LINE_8, 0).unwrap();
                }
                1 => imgproc::circle(&mut img, Point::new(x + 30, y + 30), 5 + r.next(25), c, -1, imgproc::LINE_8, 0).unwrap(),
                _ => imgproc::line(&mut img, Point::new(x, y), Point::new(x + r.next(80), y + r.next(80)), c, 2, imgproc::LINE_8, 0).unwrap(),
            }
        }
        img
    }

    fn mat3(h: &Homography) -> Mat {
        Mat::from_slice(&h.0).unwrap().reshape(1, 3).unwrap().try_clone().unwrap()
    }

    /// Scéna videná kamerou, ktorá sa pohla: každý bod referencie sa presunie podľa `h`.
    fn moved(img: &Mat, h: &Homography) -> Mat {
        let mut out = Mat::default();
        imgproc::warp_perspective(
            img, &mut out, &mat3(h), Size::new(W, H), imgproc::INTER_LINEAR, opencv::core::BORDER_REPLICATE, Scalar::default(), opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap();
        out
    }

    fn close(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt() <= tol
    }

    const PROBES: [(f64, f64); 5] = [(60.0, 50.0), (320.0, 180.0), (580.0, 300.0), (100.0, 320.0), (500.0, 40.0)];

    fn assert_maps(a: &Anchor, truth: &Homography, tol: f64) {
        for p in PROBES {
            let got = a.to_current(p);
            let want = truth.apply(p);
            assert!(close(got, want, tol), "bod {p:?}: dostal som {got:?}, očakávam {want:?}");
            // a naspäť
            assert!(close(a.to_reference(want), p, tol), "spätné zobrazenie bodu {p:?}");
        }
    }

    // ---------- čistá matematika ----------

    #[test]
    fn identita_nemeni_body() {
        assert_eq!(Homography::IDENTITY.apply((12.5, -3.0)), (12.5, -3.0));
    }

    #[test]
    fn posun_a_inverzia() {
        let t = Homography::translation(40.0, -15.0);
        assert_eq!(t.apply((10.0, 10.0)), (50.0, -5.0));
        let inv = t.inverse().unwrap();
        assert!(close(inv.apply(t.apply((123.0, 45.0))), (123.0, 45.0), 1e-9));
    }

    #[test]
    fn inverzia_zlozitejsej_matice() {
        let h = Homography([1.02, 0.05, 12.0, -0.04, 0.98, -7.0, 0.0001, 0.00005, 1.0]);
        let inv = h.inverse().unwrap();
        for p in PROBES {
            assert!(close(inv.apply(h.apply(p)), p, 1e-6), "{p:?}");
        }
    }

    #[test]
    fn singularna_matica_nema_inverziu() {
        assert!(Homography([1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 0.0, 0.0, 1.0]).inverse().is_none());
        assert!(Homography([0.0; 9]).inverse().is_none());
    }

    #[test]
    fn posun_rohov_medzi_transformaciami() {
        let (a, b) = (Homography::IDENTITY, Homography::translation(3.0, 4.0));
        assert!((a.max_corner_shift(&b, 640.0, 360.0) - 5.0).abs() < 1e-9);
        assert_eq!(a.max_corner_shift(&a, 640.0, 360.0), 0.0);
    }

    #[test]
    fn rozumnost_transformacie() {
        assert!(Homography::IDENTITY.is_plausible(640.0, 360.0));
        assert!(Homography::translation(100.0, 50.0).is_plausible(640.0, 360.0));
        assert!(!Homography([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0]).is_plausible(640.0, 360.0), "10x zväčšenie");
        assert!(!Homography([0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0]).is_plausible(640.0, 360.0), "zmenšenie na bodku");
        assert!(!Homography([-1.0, 0.0, 640.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).is_plausible(640.0, 360.0), "zrkadlenie");
        assert!(!Homography([f64::NAN; 9]).is_plausible(640.0, 360.0));
    }

    #[test]
    fn rohy_a_ohranicenie() {
        let c = rect_corners(10.0, 20.0, 30.0, 40.0);
        assert_eq!(c, [(10.0, 20.0), (40.0, 20.0), (40.0, 60.0), (10.0, 60.0)]);
        assert_eq!(bounding_box(&[(5.0, 9.0), (1.0, 12.0), (7.0, -3.0)]), (1.0, -3.0, 7.0, 12.0));
    }

    // ---------- sledovanie bodov na umelých záberoch ----------

    #[test]
    fn nehybna_kamera_dava_identitu_bez_chvenia() {
        let s = scene(1);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        assert!(a.update(&s).unwrap());
        assert!(a.tracking() && a.inliers() > 50, "inliers: {}", a.inliers());
        assert_maps(&a, &Homography::IDENTITY, 0.6);
        let first = a.homography();
        a.update(&s).unwrap();
        assert_eq!(a.homography(), first, "rovnaký záber dvakrát = žiadna zmena (mŕtve pásmo)");
    }

    #[test]
    fn najde_posun_kamery() {
        let s = scene(2);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        let truth = Homography::translation(45.0, -20.0);
        assert!(a.update(&moved(&s, &truth)).unwrap());
        assert_maps(&a, &truth, 2.0);
    }

    #[test]
    fn najde_otocenie_a_priblizenie() {
        let s = scene(3);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        // otočenie o 6° okolo stredu, zväčšenie o 8 % a posun
        let (ang, k) = (6f64.to_radians(), 1.08);
        let (c, sn) = (ang.cos() * k, ang.sin() * k);
        let (cx, cy) = (W as f64 / 2.0, H as f64 / 2.0);
        let truth = Homography([c, -sn, cx - c * cx + sn * cy + 12.0, sn, c, cy - sn * cx - c * cy - 8.0, 0.0, 0.0, 1.0]);
        assert!(a.update(&moved(&s, &truth)).unwrap(), "inliers: {}", a.inliers());
        // odhad je mierne nepresnejší pri otočení + zväčšení (v krajnom rohu asi 3 px z 640)
        assert_maps(&a, &truth, 4.0);
    }

    #[test]
    fn najde_zmenu_perspektivy() {
        let s = scene(4);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        let truth = Homography([1.0, 0.03, 10.0, 0.02, 1.0, -8.0, 0.00003, 0.00002, 1.0]);
        assert!(a.update(&moved(&s, &truth)).unwrap());
        assert_maps(&a, &truth, 3.0);
    }

    #[test]
    fn pohybujuce_sa_objekty_nevadia() {
        let s = scene(5);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        let truth = Homography::translation(-30.0, 12.0);
        let mut cur = moved(&s, &truth);
        // "autá": veľké jednofarebné plochy, ktoré v referencii neboli (spolu asi 17 % obrazu)
        for (x, y, w, h, v) in [(80, 150, 130, 60, 240.0), (350, 220, 150, 70, 10.0), (440, 40, 90, 50, 250.0)] {
            imgproc::rectangle(&mut cur, Rect::new(x, y, w, h), Scalar::all(v), -1, imgproc::LINE_8, 0).unwrap();
        }
        assert!(a.update(&cur).unwrap(), "inliers: {}", a.inliers());
        assert_maps(&a, &truth, 3.0);
    }

    #[test]
    fn nepodobny_zaber_znamena_stratene_a_ponecha_poslednu_polohu() {
        let s = scene(6);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        let truth = Homography::translation(20.0, 10.0);
        assert!(a.update(&moved(&s, &truth)).unwrap());
        let last = a.homography();
        // úplne iná scéna (kamera sa otočila inam)
        assert!(!a.update(&scene(999)).unwrap());
        assert!(!a.tracking());
        assert_eq!(a.homography(), last, "posledná dobrá poloha ostáva");
        assert!(a.points(50).is_empty(), "stratené: žiadne body na kreslenie");
    }

    #[test]
    fn prazdny_alebo_jednofarebny_zaber_je_stratene() {
        let s = scene(7);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        let blank = Mat::new_rows_cols_with_default(H, W, CV_8UC1, Scalar::all(90.0)).unwrap();
        assert!(!a.update(&blank).unwrap());
        assert!(!a.tracking());
    }

    #[test]
    fn po_strate_sa_sledovanie_obnovi() {
        let s = scene(8);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        assert!(!a.update(&scene(1234)).unwrap());
        let truth = Homography::translation(-25.0, 5.0);
        assert!(a.update(&moved(&s, &truth)).unwrap(), "kamera sa vrátila");
        assert_maps(&a, &truth, 2.0);
    }

    #[test]
    fn referencia_bez_detailov_zlyha_s_napovedou() {
        let blank = Mat::new_rows_cols_with_default(H, W, CV_8UC1, Scalar::all(90.0)).unwrap();
        let e = Anchor::new(&blank, AnchorConfig::default()).err().expect("má zlyhať").to_string();
        assert!(e.contains("primálo výrazných bodov"), "{e}");
    }

    #[test]
    fn body_na_vykreslenie_su_obmedzene_a_v_obraze() {
        let s = scene(9);
        // len ORB (bez stabilných bodov): body sú zhodné body z poslednej snímky
        let mut a = Anchor::new(&s, AnchorConfig { points: 0, ..AnchorConfig::default() }).unwrap();
        a.update(&moved(&s, &Homography::translation(10.0, 10.0))).unwrap();
        let pts = a.points(25);
        assert!(!pts.is_empty() && pts.len() <= 25, "{}", pts.len());
        assert!(pts.iter().all(|p| p.x >= 0.0 && p.x <= W as f32 && p.y >= 0.0 && p.y <= H as f32));
        assert!(a.points(0).is_empty());
    }

    #[test]
    fn kreslenie_bodov_znaci_pixely() {
        let mut f = Mat::new_rows_cols_with_default(H, W, opencv::core::CV_8UC3, Scalar::all(0.0)).unwrap();
        draw_points(&mut f, &[Point2f::new(100.0, 100.0)]).unwrap();
        assert_ne!(f.at_2d::<opencv::core::Vec3b>(100, 100).unwrap().0, [0, 0, 0]);
        assert_eq!(f.at_2d::<opencv::core::Vec3b>(200, 200).unwrap().0, [0, 0, 0]);
    }

    #[test]
    fn ramcek_sa_prevedie_do_referencie() {
        let s = scene(10);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        a.update(&moved(&s, &Homography::translation(40.0, 20.0))).unwrap();
        // rámček v aktuálnej snímke (100,100,50,30) je v referencii o (40,20) späť
        let r = a.rect_to_reference(Rect::new(100, 100, 50, 30));
        assert!((r.x - 60).abs() <= 2 && (r.y - 80).abs() <= 2, "{r:?}");
        assert!((r.width - 50).abs() <= 2 && (r.height - 30).abs() <= 2, "{r:?}");
    }

    // ---- rozloženie bodov ----

    /// Scéna, v ktorej sú všetky detaily nahromadené v malom štvorci (zvyšok je prázdny).
    fn clustered_scene(seed: u64) -> Mat {
        let mut img = Mat::new_rows_cols_with_default(H, W, CV_8UC1, Scalar::all(120.0)).unwrap();
        let mut r = Lcg(seed);
        for _ in 0..120 {
            let c = Scalar::all(30.0 + r.next(200) as f64);
            let (x, y) = (250 + r.next(70), 130 + r.next(60));
            imgproc::rectangle(&mut img, Rect::new(x, y, 4 + r.next(14), 4 + r.next(14)), c, -1, imgproc::LINE_8, 0).unwrap();
        }
        img
    }

    #[test]
    fn spread_ok_vyzaduje_pokrytie_zaberu() {
        let wide = [(10.0, 10.0), (600.0, 20.0), (300.0, 340.0)];
        assert!(spread_ok(&wide, 640.0, 360.0));
        let tight = [(300.0, 150.0), (330.0, 170.0), (310.0, 160.0)];
        assert!(!spread_ok(&tight, 640.0, 360.0));
        let line = [(10.0, 100.0), (600.0, 102.0)]; // široké, ale nulová výška
        assert!(!spread_ok(&line, 640.0, 360.0));
        assert!(!spread_ok(&[], 640.0, 360.0));
        assert!(!spread_ok(&[(1.0, 1.0)], 640.0, 360.0));
    }

    #[test]
    fn referencia_so_sustredenymi_bodmi_sa_odmietne() {
        let e = Anchor::new(&clustered_scene(1), AnchorConfig::default()).err().expect("má zlyhať").to_string();
        assert!(e.contains("sústredené"), "{e}");
    }

    #[test]
    fn posun_z_sustredenych_bodov_sa_neprijme() {
        // dobrá referencia po celom zábere; v novej snímke ostane z textúry len malý zhluk
        let s = scene(31);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        let mut cur = Mat::new_rows_cols_with_default(H, W, CV_8UC1, Scalar::all(120.0)).unwrap();
        let patch = Mat::roi(&s, Rect::new(200, 100, 120, 80)).unwrap().try_clone().unwrap();
        let mut dst = Mat::roi_mut(&mut cur, Rect::new(220, 110, 120, 80)).unwrap();
        patch.copy_to(&mut dst).unwrap();
        let before = a.homography();
        assert!(!a.update(&cur).unwrap(), "z malého zhluku sa poloha odhadnúť nemá");
        assert!(!a.tracking());
        assert_eq!(a.homography(), before);
    }

    #[test]
    fn plna_scena_ma_body_roztiahnute_po_celom_zabere() {
        // poistka proti tomu, aby bol nový test spreadu príliš prísny pre bežné scény
        let s = scene(32);
        let mut a = Anchor::new(&s, AnchorConfig::default()).unwrap();
        assert!(a.update(&moved(&s, &Homography::translation(30.0, 10.0))).unwrap());
    }

    // =======================================================================
    // Stabilné referenčné body: pevná sada, jemné sledovanie, žiadne chvenie
    // =======================================================================

    /// Konfigurácia s krátkou kalibráciou, aby testy neboli zbytočne dlhé.
    fn fine_cfg() -> AnchorConfig {
        AnchorConfig { calib_frames: 4, ..AnchorConfig::default() }
    }

    /// Ukotvenie, ktoré už prešlo kalibráciou na nehybnej scéne.
    fn calibrated(seed: u64) -> (Anchor, Mat) {
        let s = scene(seed);
        let mut a = Anchor::new(&s, fine_cfg()).unwrap();
        assert_eq!(a.mode(), AnchorMode::Calibrating);
        for _ in 0..4 {
            assert!(a.update(&s).unwrap());
        }
        (a, s)
    }

    /// Pridá k snímke deterministický šum (simuluje šum senzora kamery).
    fn noisy(img: &Mat, seed: u64, amp: i32) -> Mat {
        let mut out = img.try_clone().unwrap();
        let mut r = Lcg(seed);
        let (w, h) = (out.cols(), out.rows());
        for y in 0..h {
            for x in 0..w {
                let v = *out.at_2d::<u8>(y, x).unwrap() as i32 + r.next(2 * amp as u64 + 1) - amp;
                *out.at_2d_mut::<u8>(y, x).unwrap() = v.clamp(0, 255) as u8;
            }
        }
        out
    }

    #[test]
    fn rozlozi_body_maximin() {
        let pts: Vec<(f64, f64)> = (0..10).flat_map(|x| (0..10).map(move |y| (x as f64 * 10.0, y as f64 * 10.0))).collect();
        let idx = pick_spread(&pts, 4);
        assert_eq!(idx.len(), 4);
        assert_eq!(idx[0], 0);
        // štyri body na mriežke 10x10 majú byť v štyroch rohoch (každý z nich vzdialený od ostatných)
        let chosen: Vec<_> = idx.iter().map(|&i| pts[i]).collect();
        for c in &chosen {
            assert!(c.0 <= 10.0 || c.0 >= 80.0, "{chosen:?}");
            assert!(c.1 <= 10.0 || c.1 >= 80.0, "{chosen:?}");
        }
    }

    #[test]
    fn pick_spread_okrajove_pripady() {
        assert!(pick_spread(&[], 5).is_empty());
        assert!(pick_spread(&[(1.0, 1.0)], 0).is_empty());
        assert_eq!(pick_spread(&[(0.0, 0.0), (5.0, 5.0)], 10).len(), 2, "menej bodov než k");
        let a = pick_spread(&[(0.0, 0.0), (9.0, 1.0), (4.0, 8.0), (7.0, 7.0)], 3);
        let b = pick_spread(&[(0.0, 0.0), (9.0, 1.0), (4.0, 8.0), (7.0, 7.0)], 3);
        assert_eq!(a, b, "deterministické");
    }

    #[test]
    fn po_kalibracii_je_pevna_sada_desiatich_bodov_rozlozena_po_zábere() {
        let (a, _) = calibrated(41);
        assert_eq!(a.mode(), AnchorMode::Fine);
        let pts = a.chosen_points();
        assert_eq!(pts.len(), 10);
        let xy: Vec<_> = pts.iter().map(|p| (p.x as f64, p.y as f64)).collect();
        assert!(spread_ok_min(&xy, W as f64, H as f64, FINE_SPREAD), "body sú príliš nahromadené: {xy:?}");
        // body nie sú tesne pri okraji
        assert!(xy.iter().all(|p| p.0 > 20.0 && p.1 > 20.0 && p.0 < W as f64 - 20.0 && p.1 < H as f64 - 20.0));
    }

    #[test]
    fn kalibracia_trva_zadany_pocet_aktualizacii() {
        let s = scene(42);
        let mut a = Anchor::new(&s, AnchorConfig { calib_frames: 6, ..AnchorConfig::default() }).unwrap();
        for i in 0..5 {
            a.update(&s).unwrap();
            assert_eq!(a.mode(), AnchorMode::Calibrating, "po {} aktualizáciách", i + 1);
            assert!(a.points(40).is_empty(), "počas kalibrácie sa body nekreslia");
        }
        a.update(&s).unwrap();
        assert_eq!(a.mode(), AnchorMode::Fine);
    }

    #[test]
    fn stojaca_kamera_s_sumom_nema_ziadne_chvenie_polohy_ani_bodov() {
        // Toto je presne sťažnosť: čiara a body sa pri nehybnej kamere nesmú hýbať.
        let (mut a, s) = calibrated(43);
        a.update(&s).unwrap();
        let (h0, pts0) = (a.homography(), a.points(40));
        assert_eq!(pts0.len(), 10);
        for i in 0..60 {
            let frame = noisy(&s, 1000 + i, 4); // každý snímok iný šum
            assert!(a.update(&frame).unwrap(), "snímok {i}");
            assert_eq!(a.homography(), h0, "poloha sa v snímku {i} zmenila (čiara by vibrovala)");
            assert_eq!(a.points(40), pts0, "body sa v snímku {i} pohli");
        }
    }

    #[test]
    fn stojaca_kamera_aj_ked_sa_v_zabere_nieco_hybe() {
        let (mut a, s) = calibrated(44);
        let h0 = a.homography();
        let mut r = Lcg(7);
        for i in 0..40 {
            let mut frame = noisy(&s, 500 + i, 3);
            // "chodec": malý tmavý obdĺžnik, ktorý sa každý snímok posunie inam
            let (x, y) = (r.next(500), r.next(260));
            imgproc::rectangle(&mut frame, Rect::new(x, y, 24, 70), Scalar::all(15.0), -1, imgproc::LINE_8, 0).unwrap();
            assert!(a.update(&frame).unwrap(), "snímok {i}");
            assert_eq!(a.homography(), h0, "pohyb v zábere nesmie pohnúť čiarou (snímok {i})");
        }
    }

    #[test]
    fn jemne_sledovanie_najde_posun_presnejsie_ako_orb() {
        let (mut a, s) = calibrated(45);
        let truth = Homography::translation(37.0, -14.0);
        assert!(a.update(&moved(&s, &truth)).unwrap());
        assert_eq!(a.mode(), AnchorMode::Fine);
        assert_maps(&a, &truth, 1.2);
    }

    #[test]
    fn plynula_jazda_kamery_cez_viac_snimok_sa_sleduje_bez_skokov() {
        let (mut a, s) = calibrated(46);
        let mut last = a.to_current((320.0, 180.0));
        for step in 1..=20 {
            let d = step as f64 * 5.0; // posun o 5 px za aktualizáciu, spolu 100 px
            let truth = Homography::translation(d, d * 0.3);
            assert!(a.update(&moved(&s, &truth)).unwrap(), "krok {step}");
            assert_maps(&a, &truth, 1.5);
            let now = a.to_current((320.0, 180.0));
            let jump = ((now.0 - last.0).powi(2) + (now.1 - last.1).powi(2)).sqrt();
            assert!(jump < 9.0, "skok čiary o {jump:.1} px v kroku {step} (očakávaných ~5)");
            last = now;
        }
    }

    #[test]
    fn rotacia_sa_sleduje_jemnym_sledovanim() {
        let (mut a, s) = calibrated(47);
        let (ang, k) = (3f64.to_radians(), 1.03);
        let (c, sn) = (ang.cos() * k, ang.sin() * k);
        let (cx, cy) = (W as f64 / 2.0, H as f64 / 2.0);
        let truth = Homography([c, -sn, cx - c * cx + sn * cy + 8.0, sn, c, cy - sn * cx - c * cy - 5.0, 0.0, 0.0, 1.0]);
        assert!(a.update(&moved(&s, &truth)).unwrap());
        assert_maps(&a, &truth, 2.5);
    }

    #[test]
    fn prudky_skok_kamery_sa_najde_zalohou_orb() {
        let (mut a, s) = calibrated(48);
        // 180 px naraz: jemné sledovanie to nemusí zvládnuť, záloha (ORB) áno
        let truth = Homography::translation(180.0, 40.0);
        assert!(a.update(&moved(&s, &truth)).unwrap());
        assert_maps(&a, &truth, 3.0);
        // a po návrate je zasa presné jemné sledovanie
        assert!(a.update(&moved(&s, &Homography::translation(182.0, 41.0))).unwrap());
        assert_maps(&a, &Homography::translation(182.0, 41.0), 1.5);
    }

    #[test]
    fn nepodobna_scena_je_stratene_a_po_navrate_sa_obnovi() {
        let (mut a, s) = calibrated(49);
        let last = a.homography();
        assert!(!a.update(&scene(777)).unwrap());
        assert!(!a.tracking());
        assert_eq!(a.homography(), last, "posledná poloha ostáva");
        assert!(a.points(40).is_empty(), "nič sa nekreslí, kým je poloha stratená");
        assert!(a.update(&moved(&s, &Homography::translation(10.0, 5.0))).unwrap(), "kamera sa vrátila");
        assert_maps(&a, &Homography::translation(10.0, 5.0), 1.5);
    }

    #[test]
    fn nestabilne_miesto_sa_pri_kalibracii_nevyberie() {
        // V ľavom hornom rohu "bliká" časový údaj (každý snímok iný): body tam nesmú byť vybrané.
        let s = scene(50);
        let mut a = Anchor::new(&s, fine_cfg()).unwrap();
        let mut r = Lcg(99);
        for _ in 0..4 {
            let mut frame = s.try_clone().unwrap();
            for _ in 0..40 {
                let (x, y) = (10 + r.next(200), 10 + r.next(60));
                imgproc::rectangle(&mut frame, Rect::new(x, y, 4 + r.next(10), 4 + r.next(10)), Scalar::all(r.next(255) as f64), -1, imgproc::LINE_8, 0).unwrap();
            }
            a.update(&frame).unwrap();
        }
        assert_eq!(a.mode(), AnchorMode::Fine);
        for p in a.chosen_points() {
            assert!(!(p.x < 230.0 && p.y < 90.0), "vybraný bod ({}, {}) leží v bliká­júcej oblasti", p.x, p.y);
        }
    }

    #[test]
    fn bez_stabilnych_bodov_sa_pouzije_len_orb() {
        // Každý snímok počas kalibrácie je iná scéna: nič sa nedá spoľahlivo sledovať.
        let s = scene(51);
        let mut a = Anchor::new(&s, fine_cfg()).unwrap();
        for i in 0..4 {
            a.update(&scene(2000 + i)).unwrap();
        }
        assert_eq!(a.mode(), AnchorMode::OrbOnly);
        assert!(a.chosen_points().is_empty());
        // ORB stále funguje
        let truth = Homography::translation(12.0, 3.0);
        assert!(a.update(&moved(&s, &truth)).unwrap());
        assert_maps(&a, &truth, 2.0);
    }

    #[test]
    fn points_nula_vypne_jemne_sledovanie() {
        let s = scene(52);
        let mut a = Anchor::new(&s, AnchorConfig { points: 0, ..AnchorConfig::default() }).unwrap();
        assert_eq!(a.mode(), AnchorMode::OrbOnly);
        assert!(a.update(&s).unwrap());
        assert!(a.chosen_points().is_empty());
    }

    #[test]
    fn mrtve_pasmo_sa_da_nastavit() {
        // veľké mŕtve pásmo (5 px): malý posun (3 px) čiaru nepohne, väčší (20 px) áno
        let s = scene(53);
        let mut a = Anchor::new(&s, AnchorConfig { calib_frames: 4, deadband: 5.0, ..AnchorConfig::default() }).unwrap();
        for _ in 0..4 {
            a.update(&s).unwrap();
        }
        let h0 = a.homography();
        a.update(&moved(&s, &Homography::translation(3.0, 0.0))).unwrap();
        assert_eq!(a.homography(), h0, "posun 3 px je v mŕtvom pásme");
        a.update(&moved(&s, &Homography::translation(20.0, 0.0))).unwrap();
        assert_ne!(a.homography(), h0);
        assert!(close(a.to_current((100.0, 100.0)), (120.0, 100.0), 1.5));
    }

    #[test]
    fn track_points_na_rovnakom_obraze_nehybe_bodmi() {
        let s = scene(54);
        let pts = Anchor::candidates(&s).unwrap();
        assert!(pts.len() >= 20, "{}", pts.len());
        let tracked = track_points(&s, &s, &pts, &Homography::IDENTITY).unwrap();
        let ok = tracked.iter().flatten().count();
        assert!(ok as f64 >= 0.9 * pts.len() as f64, "{ok}/{}", pts.len());
        for (p, t) in pts.iter().zip(&tracked) {
            if let Some(t) = t {
                assert!(((p.x - t.x).powi(2) + (p.y - t.y).powi(2)).sqrt() < 0.2, "bod sa pohol");
            }
        }
    }

    #[test]
    fn track_points_zvlada_posun_aj_zly_odhad() {
        let s = scene(55);
        let pts = Anchor::candidates(&s).unwrap();
        let cur = moved(&s, &Homography::translation(25.0, 10.0));
        // odhad je identita (kamera sa pohla o 25 px, ale my to ešte nevieme)
        let tracked = track_points(&s, &cur, &pts, &Homography::IDENTITY).unwrap();
        let (mut good, mut total) = (0, 0);
        for (p, t) in pts.iter().zip(&tracked) {
            total += 1;
            if let Some(t) = t {
                if ((t.x - p.x - 25.0).powi(2) + (t.y - p.y - 10.0).powi(2)).sqrt() < 1.0 {
                    good += 1;
                }
            }
        }
        assert!(good as f64 >= 0.7 * total as f64, "správne sledovaných {good}/{total}");
    }

    #[test]
    fn track_points_prazdny_vstup() {
        let s = scene(56);
        assert!(track_points(&s, &s, &[], &Homography::IDENTITY).unwrap().is_empty());
    }

    #[test]
    fn referencna_fotka_obsahuje_body_ciaru_a_vyrez() {
        let (a, s) = calibrated(57);
        let color = to_bgr(&s);
        let line = ((200.0, 100.0), (200.0, 300.0));
        let roi = [(50.0, 120.0), (350.0, 120.0), (350.0, 280.0), (50.0, 280.0)];
        let img = render_reference(&color, a.chosen_points(), line, &roi).unwrap();
        assert_eq!((img.cols(), img.rows()), (W, H));
        let yellow = |m: &Mat, x: i32, y: i32| m.at_2d::<opencv::core::Vec3b>(y, x).unwrap().0 == [0, 255, 255];
        assert!(yellow(&img, 200, 250) || yellow(&img, 201, 250) || yellow(&img, 199, 250), "žltá čiara");
        // pri vybranom bode je azúrový krúžok
        let p = a.chosen_points()[0];
        let cyan = (-8..=8).any(|d: i32| {
            let (x, y) = ((p.x as i32 + d).clamp(0, W - 1), (p.y as i32).clamp(0, H - 1));
            img.at_2d::<opencv::core::Vec3b>(y, x).unwrap().0 == [255, 255, 0]
        });
        assert!(cyan, "krúžok pri bode 1");
        // pôvodný obrázok sa nezmenil
        assert_ne!(to_bgr(&s).data_bytes().unwrap(), img.data_bytes().unwrap());
    }

    fn to_bgr(gray: &Mat) -> Mat {
        let mut out = Mat::default();
        imgproc::cvt_color_def(gray, &mut out, imgproc::COLOR_GRAY2BGR).unwrap();
        out
    }

    // =======================================================================
    // Posun kamery na nové miesto: nový referenčný záber, žiadne blikanie
    // =======================================================================

    /// Širokouhlé "sudovité" skreslenie objektívu (ako má skutočná kamera): homografia ho nevie opísať.
    fn barrel(img: &Mat, k: f64) -> Mat {
        let (w, h) = (img.cols(), img.rows());
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        let mut mx = Mat::new_rows_cols_with_default(h, w, opencv::core::CV_32FC1, Scalar::all(0.0)).unwrap();
        let mut my = Mat::new_rows_cols_with_default(h, w, opencv::core::CV_32FC1, Scalar::all(0.0)).unwrap();
        for y in 0..h {
            for x in 0..w {
                let (nx, ny) = ((x as f64 - cx) / cx, (y as f64 - cy) / cx);
                let f = 1.0 + k * (nx * nx + ny * ny);
                *mx.at_2d_mut::<f32>(y, x).unwrap() = (cx + nx * f * cx) as f32;
                *my.at_2d_mut::<f32>(y, x).unwrap() = (cy + ny * f * cx) as f32;
            }
        }
        let mut out = Mat::default();
        imgproc::remap(
            img, &mut out, &mx, &my, imgproc::INTER_LINEAR, opencv::core::BORDER_REPLICATE, Scalar::default(),
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap();
        out
    }

    /// Pohľad kamery z polohy `h` na scénu `s` cez objektív so skreslením `k`.
    fn view(s: &Mat, h: &Homography, k: f64) -> Mat {
        let v = moved(s, h);
        if k != 0.0 { barrel(&v, k) } else { v }
    }

    /// Kamera s daným skreslením, kalibrovaná v počiatočnej polohe.
    fn calibrated_lens(seed: u64, k: f64) -> (Anchor, Mat) {
        let s = scene(seed);
        let base = view(&s, &Homography::IDENTITY, k);
        let mut a = Anchor::new(&base, fine_cfg()).unwrap();
        for _ in 0..4 {
            a.update(&base).unwrap();
        }
        (a, s)
    }

    /// Prepustí `n` snímok z polohy `h` (každá s iným šumom); vráti, koľkokrát sa poloha zmenila
    /// a koľkokrát sa sledovanie podarilo, počítané len od snímku `from`.
    fn hold(a: &mut Anchor, s: &Mat, h: &Homography, k: f64, n: u64, from: u64, noise_seed: u64) -> (usize, usize) {
        let (mut changes, mut tracked, mut prev) = (0, 0, a.homography());
        for i in 0..n {
            let ok = a.update(&noisy(&view(s, h, k), noise_seed + i, 4)).unwrap();
            if i >= from {
                if a.homography() != prev { changes += 1; }
                if ok { tracked += 1; }
            }
            prev = a.homography();
        }
        (changes, tracked)
    }

    #[test]
    fn po_posune_kamery_so_skreslenim_objektivu_sa_poloha_ustali() {
        // PRESNE REPRODUKOVANÁ CHYBA: bez nového referenčného záberu sa poloha menila v KAŽDEJ snímke
        let (mut a, s) = calibrated_lens(70, 0.12);
        let h = Homography::translation(90.0, 10.0);
        let (changes, tracked) = hold(&mut a, &s, &h, 0.12, 100, 60, 900);
        assert_eq!(changes, 0, "poloha čiary sa po ustálení stále mení (preblikáva)");
        assert_eq!(tracked, 40, "sledovanie sa v ustálenom stave musí dariť stále");
        assert!(a.keyframes() >= 2, "po ustálení sa mal vytvoriť nový referenčný záber, je ich {}", a.keyframes());
        assert!(!a.in_fallback(), "po ustálení sa už nemá používať záloha");
    }

    #[test]
    fn po_ustaleni_sa_nehybu_ani_zobrazene_body() {
        let (mut a, s) = calibrated_lens(71, 0.12);
        let h = Homography::translation(70.0, -15.0);
        hold(&mut a, &s, &h, 0.12, 70, 70, 100);
        let pts = a.points(40);
        assert!(pts.len() >= 6, "{}", pts.len());
        for i in 0..30 {
            a.update(&noisy(&view(&s, &h, 0.12), 500 + i, 4)).unwrap();
            assert_eq!(a.points(40), pts, "body sa v snímku {i} pohli");
        }
    }

    #[test]
    fn velky_posun_kamery_so_skreslenim_sa_tiez_ustali() {
        let (mut a, s) = calibrated_lens(72, 0.12);
        let (changes, tracked) = hold(&mut a, &s, &Homography::translation(200.0, 25.0), 0.12, 100, 60, 800);
        assert_eq!((changes, tracked), (0, 40));
    }

    #[test]
    fn otocenie_kamery_so_skreslenim_sa_ustali() {
        let (mut a, s) = calibrated_lens(73, 0.1);
        let (ang, k) = (4f64.to_radians(), 1.02);
        let (c, sn) = (ang.cos() * k, ang.sin() * k);
        let (cx, cy) = (W as f64 / 2.0, H as f64 / 2.0);
        let h = Homography([c, -sn, cx - c * cx + sn * cy + 20.0, sn, c, cy - sn * cx - c * cy - 6.0, 0.0, 0.0, 1.0]);
        let (changes, tracked) = hold(&mut a, &s, &h, 0.1, 100, 60, 700);
        assert_eq!((changes, tracked), (0, 40));
    }

    #[test]
    fn poloha_v_novom_mieste_je_blizko_skutocnosti() {
        // čiara ostáva na tom istom mieste scény (v strede záberu, kde je skreslenie najmenšie)
        let (mut a, s) = calibrated_lens(74, 0.12);
        let h = Homography::translation(90.0, 10.0);
        hold(&mut a, &s, &h, 0.12, 100, 100, 600);
        let got = a.to_current((320.0, 180.0));
        assert!(close(got, (410.0, 190.0), 8.0), "stred scény sa premietol na {got:?}, očakávam ~(410, 190)");
    }

    #[test]
    fn navrat_do_povodnej_polohy_vrati_ciaru_na_povodne_miesto() {
        let (mut a, s) = calibrated_lens(75, 0.12);
        let h = Homography::translation(120.0, 15.0);
        hold(&mut a, &s, &h, 0.12, 80, 80, 300);
        assert!(a.keyframes() >= 2);
        // kamera sa vráti do pôvodnej polohy
        let (changes, tracked) = hold(&mut a, &s, &Homography::IDENTITY, 0.12, 80, 40, 400);
        assert_eq!(a.active_key(), 0, "po návrate sa má použiť pôvodný referenčný záber");
        assert!(a.homography().max_corner_shift(&Homography::IDENTITY, W as f64, H as f64) < 2.5, "{:?}", a.homography());
        assert_eq!((changes, tracked), (0, 40), "po návrate je poloha pokojná");
    }

    #[test]
    fn plynuly_pohyb_kamery_so_skreslenim_je_sledovany_bez_skokov() {
        let (mut a, s) = calibrated_lens(76, 0.12);
        let mut last = a.to_current((320.0, 180.0));
        for step in 1..=24 {
            let d = step as f64 * 5.0;
            let ok = a.update(&noisy(&view(&s, &Homography::translation(d, d * 0.2), 0.12), 50 + step, 3)).unwrap();
            assert!(ok, "krok {step}: sledovanie sa stratilo");
            let now = a.to_current((320.0, 180.0));
            let jump = ((now.0 - last.0).powi(2) + (now.1 - last.1).powi(2)).sqrt();
            assert!(jump < 14.0, "skok o {jump:.1} px v kroku {step}");
            last = now;
        }
    }

    #[test]
    fn kamera_co_sa_stale_hybe_nevytvara_zbytocne_zabery() {
        let (mut a, s) = calibrated_lens(77, 0.1);
        for step in 1..=40 {
            a.update(&noisy(&view(&s, &Homography::translation(step as f64 * 3.0, 0.0), 0.1), 20 + step, 3)).unwrap();
        }
        assert_eq!(a.keyframes(), 1, "kým sa kamera hýbe, nový záber sa nemá nastavovať");
    }

    #[test]
    fn viac_miest_nepresiahne_maximum_zaberov_a_povodny_ostane() {
        let (mut a, s) = calibrated_lens(78, 0.08);
        for (i, dx) in [60.0, -70.0, 110.0, -100.0, 140.0, -120.0, 90.0, -50.0].iter().enumerate() {
            hold(&mut a, &s, &Homography::translation(*dx, (i as f64) * 4.0), 0.08, 60, 60, 1000 * (i as u64 + 1));
        }
        assert!(a.keyframes() <= MAX_KEYS, "{}", a.keyframes());
        assert!(a.keyframes() >= 3, "{}", a.keyframes());
        // pôvodný záber stále funguje: návrat na začiatok
        hold(&mut a, &s, &Homography::IDENTITY, 0.08, 40, 40, 9000);
        assert_eq!(a.active_key(), 0);
        assert!(a.homography().max_corner_shift(&Homography::IDENTITY, W as f64, H as f64) < 3.0);
    }

    #[test]
    fn rovnake_miesto_nahradi_zaber_namiesto_pridania() {
        let (mut a, s) = calibrated_lens(79, 0.1);
        let h = Homography::translation(80.0, 5.0);
        hold(&mut a, &s, &h, 0.1, 80, 80, 100);
        let n = a.keyframes();
        // odíde a vráti sa na to isté miesto
        hold(&mut a, &s, &Homography::IDENTITY, 0.1, 40, 40, 200);
        hold(&mut a, &s, &h, 0.1, 80, 80, 300);
        assert_eq!(a.keyframes(), n, "na rovnakom mieste sa nemá hromadiť ďalší záber");
    }

    #[test]
    fn stav_do_panela() {
        let (mut a, s) = calibrated_lens(80, 0.1);
        a.update(&view(&s, &Homography::IDENTITY, 0.1)).unwrap();
        let txt = a.status_text();
        assert!(txt.contains('/'), "{txt}");
        let s2 = scene(80);
        let ca = Anchor::new(&s2, fine_cfg()).unwrap();
        assert_eq!(ca.status_text(), "kalib.");
    }

    #[test]
    fn compose_je_zlozenie_a_inverzia_ho_zrusi() {
        let a = Homography::translation(10.0, 5.0);
        let b = Homography([1.01, 0.02, 3.0, -0.02, 0.99, 4.0, 0.0001, 0.0, 1.0]);
        let ab = a.compose(&b); // najprv b, potom a
        for p in PROBES {
            let want = a.apply(b.apply(p));
            assert!(close(ab.apply(p), want, 1e-6), "{p:?}");
            assert!(close(ab.inverse().unwrap().apply(ab.apply(p)), p, 1e-6));
        }
        assert!(close(Homography::IDENTITY.compose(&b).apply((7.0, 9.0)), b.apply((7.0, 9.0)), 1e-9));
    }

    // =======================================================================
    // Zmrazenie polohy pri stojacej kamere (meranie pohybu priamo z obrazu)
    // =======================================================================

    /// Otočenie o `deg` stupňov okolo stredu + posun.
    fn rot(deg: f64, dx: f64, dy: f64) -> Homography {
        let a = deg.to_radians();
        let (c, sn) = (a.cos(), a.sin());
        let (cx, cy) = (W as f64 / 2.0, H as f64 / 2.0);
        Homography([c, -sn, cx - c * cx + sn * cy + dx, sn, c, cy - sn * cx - c * cy + dy, 0.0, 0.0, 1.0])
    }

    #[test]
    fn meranie_pohybu_obrazu() {
        let s = scene(90);
        let mut a = Anchor::new(&s, fine_cfg()).unwrap();
        a.update(&s).unwrap();
        a.update(&s).unwrap();
        assert!(a.motion() < 0.1, "rovnaká snímka: {}", a.motion());
        assert!(a.camera_static());
        a.update(&moved(&s, &Homography::translation(5.0, 0.0))).unwrap();
        assert!((a.motion() - 5.0).abs() < 0.6, "posun o 5 px: {}", a.motion());
        assert!(!a.camera_static());
    }

    #[test]
    fn otocenie_o_10_stupnov_so_silnym_skreslenim_ciara_stoji() {
        // Scenár z praxe: kamera sa otočí o ~10°, zastane, čiara musí stáť (silné skreslenie objektívu).
        let (mut a, s) = calibrated_lens(91, 0.25);
        let h = rot(10.0, 60.0, 0.0);
        let (changes, tracked) = hold(&mut a, &s, &h, 0.25, 100, 30, 3000);
        assert_eq!(changes, 0, "kamera stojí, čiara sa nesmie hýbať");
        assert!(tracked >= 60, "sledovanie: {tracked}/70");
    }

    #[test]
    fn aj_bez_jemnych_bodov_je_ciara_pri_stojacej_kamere_pokojna() {
        // Najhorší prípad: len globálne hľadanie (ORB), ktoré pri skreslení kolíše v každej snímke.
        let s = scene(92);
        let base = view(&s, &Homography::IDENTITY, 0.2);
        let mut a = Anchor::new(&base, AnchorConfig { points: 0, ..AnchorConfig::default() }).unwrap();
        a.update(&base).unwrap();
        let (changes, _) = hold(&mut a, &s, &Homography::translation(110.0, 12.0), 0.2, 80, 20, 4000);
        assert_eq!(changes, 0, "kamera stojí: poloha má byť zmrazená aj keď odhad kolíše");
    }

    #[test]
    fn meniaci_sa_jas_nepokladame_za_pohyb() {
        let (mut a, s) = calibrated_lens(93, 0.1);
        let h = Homography::translation(50.0, 0.0);
        hold(&mut a, &s, &h, 0.1, 30, 30, 10);
        let h0 = a.homography();
        let mut r = Lcg(5);
        for i in 0..40 {
            let mut f = noisy(&view(&s, &h, 0.1), 6000 + i, 6);
            // celkový jas kolíše (mraky, automatická expozícia)
            let delta = r.next(21) as f64 - 10.0;
            let mut g = Mat::default();
            f.convert_to(&mut g, -1, 1.0, delta).unwrap();
            f = g;
            a.update(&f).unwrap();
            assert!(a.camera_static(), "snímka {i}: zmena jasu nie je pohyb (motion {})", a.motion());
        }
        assert_eq!(a.homography(), h0);
    }

    #[test]
    fn velmi_pomaly_posun_kamery_sa_nakoniec_prejavi() {
        // 0,3 px za aktualizáciu je pod prahom "pohybu", ale po 80 krokoch je to 24 px:
        // poloha sa nesmie zaseknúť navždy
        let (mut a, s) = calibrated_lens(94, 0.0);
        let mut changed = 0;
        let mut prev = a.homography();
        for i in 1..=80 {
            let d = i as f64 * 0.3;
            a.update(&noisy(&view(&s, &Homography::translation(d, 0.0), 0.0), 7000 + i, 2)).unwrap();
            if a.homography() != prev { changed += 1; }
            prev = a.homography();
        }
        assert!(changed >= 1 && changed <= 10, "poloha sa zmenila {changed}-krát (má sa meniť zriedka, ale nie nikdy)");
        let got = a.to_current((320.0, 180.0));
        assert!(close(got, (344.0, 180.0), 12.0), "poloha zaostala príliš: {got:?}, skutočnosť (344, 180)");
    }

    #[test]
    fn skutocny_pohyb_zmrazenie_zrusi() {
        let (mut a, s) = calibrated_lens(95, 0.1);
        hold(&mut a, &s, &Homography::IDENTITY, 0.1, 20, 20, 10);
        assert!(a.frozen());
        // kamera sa pohne o 40 px a zastane
        let h = Homography::translation(40.0, 8.0);
        hold(&mut a, &s, &h, 0.1, 30, 30, 20);
        let got = a.to_current((320.0, 180.0));
        assert!(close(got, (360.0, 188.0), 5.0), "po pohybe sa poloha mala aktualizovať: {got:?}");
    }

    #[test]
    fn po_otoceni_a_navrate_je_ciara_pokojna_v_oboch_polohach() {
        let (mut a, s) = calibrated_lens(96, 0.2);
        let h = rot(-8.0, -50.0, 5.0);
        let (c1, _) = hold(&mut a, &s, &h, 0.2, 80, 30, 100);
        let (c2, _) = hold(&mut a, &s, &Homography::IDENTITY, 0.2, 80, 30, 200);
        let (c3, _) = hold(&mut a, &s, &h, 0.2, 80, 30, 300);
        assert_eq!((c1, c2, c3), (0, 0, 0), "otočená / pôvodná / znova otočená");
        assert!(a.homography().max_corner_shift(&h, W as f64, H as f64) < 40.0, "poloha zodpovedá otočeniu");
    }
}
