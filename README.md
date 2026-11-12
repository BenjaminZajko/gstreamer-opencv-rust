# gstreamer-opencv-rust

Detekcia objektov vo videu: **GStreamer** dekóduje a zakóduje video (HW enkodér VideoToolbox na macOS), **OpenCV** detekuje objekty. Napísané v Ruste.

## Rýchly štart (živá kamera -> VLC)
1. Nainštaluj požiadavky a stiahni model (pozri nižšie), potom `cargo build --release`.
2. Vytvor súbor s údajmi ku kamere `../GStream_configuration/camera.conf` (mimo repozitára, nikdy ho nenahrávaj):
   ```bash
   CAM_IP=192.168.0.100
   CAM_USER=pouzivatel
   CAM_PASS=heslo
   ```
   (alebo cestu zadaj premennou `CAM_CONF=/cesta/camera.conf`). Kamera musí mať zapnutý RTSP (Tapo: účet kamery v aplikácii).
3. `./live.sh count` a vo VLC *Médiá → Otvoriť sieťový stream* → adresa, ktorú vypíše (`rtsp://<ip-macu>:8554/detect`).
4. `./live.sh stop` zastaví.

Bez kalibrácie (`calib/`) beží detekcia a počítanie na pôvodnom (fisheye) obraze bez rýchlosti. S kalibráciou
(pozri „Kalibrácia kamery“) je obraz rovný a pri autách aj ľuďoch sa ukazuje rýchlosť v km/h.

## Požiadavky (macOS)
```bash
brew install gstreamer opencv
```
Rust (cargo) a Xcode Command Line Tools (kvôli libclang).

Pre režimy `people` a `objects` treba model YOLOX (35,9 MB, licencia Apache-2.0):
```bash
mkdir -p models
curl -L -o models/yolox_s.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
```
(Model sa do gitu nenahráva, je v `.gitignore`.)

## Použitie
```bash
./run.sh video.mp4 motion    # pohybujúce sa objekty (MOG2, bez modelu)
./run.sh video.mp4 people    # len ľudia (YOLOX)
./run.sh video.mp4 objects   # všetkých 80 tried COCO (YOLOX)
```
Výsledok sa uloží ako `motion.mp4` / `people.mp4` / `objects.mp4`.

Priamo (viac možností):
```bash
./target/release/object_detect video.mp4 --mode people --output vysledok.mp4 --width 1280 --conf 0.5
```
- `--conf` minimálna istota detekcie 0–1 (predvolené 0.4); vyššia = menej falošných detekcií
- `--model` cesta k ONNX modelu (predvolené `models/yolox_s.onnx`)

## Živá detekcia (kamera -> vlastný RTSP server -> VLC)
Program sa pripojí na kameru, detekuje ľudí naživo a výsledok vysiela ako RTSP stream. Nič sa neukladá.
Beží v pozadí; detekcia sa robí len kým sa niekto pozerá.
```bash
./live.sh start      # spustí (údaje ku kamere berie z ../GStream_configuration/camera.conf)
./live.sh status     # beží? + adresa
./live.sh logs       # log
./live.sh stop
```
Potom vo VLC: *Médiá → Otvoriť sieťový stream* → `rtsp://<ip-macu>:8554/detect`.
Prepínače sa dajú dať za `start`, napr. `./live.sh start --mode objects --conf 0.5`.
Bez kamery sa dá vyskúšať umelý zdroj: `object_detect_live --input test:ball --mode motion`.

## Živé počítanie áut a ľudí (kamera -> server -> VLC)
Server sa spustí jedným príkazom a stále počíta, koľko áut a ľudí prešlo cez zvislú čiaru. Vo VLC je
vidieť **celý záber** s čiarou, rámčekmi, ID a počtami. Počíta sa aj vtedy, keď sa nik nepozerá.
```bash
./live.sh count      # spustí počítanie (údaje ku kamere z ../GStream_configuration/camera.conf)
./live.sh counts     # vypíše aktuálne počty
./live.sh status     # beží? + adresa pre VLC
./live.sh stop
```
Vo VLC: *Médiá → Otvoriť sieťový stream* → `rtsp://localhost:8554/detect`.

Poloha čiary a výrez sa menia premennými (zlomky záberu):
`LINE=0.3 SPAN=0.5,0.9 ROI=0,0.5,0.5,0.4 ./live.sh count` (predvolene `LINE=0.3 SPAN=0.5,0.85 ROI=0,0.5,1,0.35`, `ENGINE=coreml`, `CONF=0.25`)
- `LINE` poloha zvislej čiary (zlomok šírky), `SPAN` pásmo výšky, v ktorom sa počíta,
- `ROI` výrez, na ktorom beží detekcia (zobrazuje sa celý záber, ale YOLOX vidí len výrez – vďaka tomu
  nestratí detaily, keď sa 4K zmenší na 640 px).

**Počty vo VLC sú „od zapnutia“:** panel aj čísla pri objektoch (`car #1`, `car #2`, `person #1`...) začínajú od nuly, keď sa pripojí
divák (a znova od nuly, keď odídu všetci a niekto sa pripojí znova). Celkové počty od spustenia servera sa neukazujú, ale
ukladajú sa na pozadí (`out/`).

**Ukotvenie čiary k scéne:** čiara a výrez sú pripnuté k ulici, nie k obrazovke. Keď sa kamera posunie alebo
pootočí, posunú sa spolu so scénou.
- Pri štarte sa uloží referenčný záber a z neho sa vyberie **pevná sada 10 stabilných bodov** (rohy okien, striech,
  plot...). Počas krátkej kalibrácie (~2 s) sa z ~48 kandidátov vyradia tie, ktoré sa hýbu alebo menia vzhľad
  (ľudia, autá, časová pečiatka kamery) a zvyšok sa rozloží po celom zábere. Referenčnú fotku s očíslovanými
  bodmi, čiarou a výrezom nájdeš v `out/anchor_reference.jpg`.
- Potom sa sledujú **stále tie isté body** voči referenčnej fotke (Lucas-Kanade, presnosť na zlomky pixela),
  nie nové v každej snímke. Pri stojacej kamere sa preto nehýbu ani body, ani čiara. Čiara sa posunie až keď sa
  kamera pohne aspoň o `--anchor-deadband` (1 px).
- **Keď kamera stojí, čiara stojí.** Program meria priamo z obrazu, či sa kamera hýbe (medián pohybu bodov
  medzi snímkami; chodci a autá ho neovplyvnia). Keď kamera stojí, poloha čiary sa po krátkom dorovnaní
  **zmrazí** a nezmení sa, nech odhad kolíše akokoľvek. Zmení sa až pri skutočnom pohybe kamery (alebo ak by sa
  kamera veľmi pomaly posúvala a rozdiel by pretrval).
- **Keď kameru pootočíš a nechaš stáť na novom mieste**, program to zistí (poloha sa ustáli ďaleko od aktívneho
  záberu) a urobí si **nový referenčný záber v tej polohe** s novými stabilnými bodmi. Dôvod: objektív kamery
  skresľuje obraz a jedna transformácia voči pôvodnej fotke to nevie opísať, takže sledovanie voči nej kolíše
  (čiara preblikáva). Voči záberu urobenému na tom istom mieste je sledovanie pokojné. Pamätá sa až 6 záberov;
  po návrate na pôvodné miesto sa zasa použije pôvodný. V konzole je správa o novom zábere a v paneli
  `REF nastavujem` počas nastavovania. Nový záber vznikne po asi 1–2 s státia.
- Ak sa kamera pohne príliš prudko a body sa stratia, poloha sa nájde znova cez ORB (globálne hľadanie).
- Vo videu je v paneli `REF 10/10` (sledované body); body sú malé azúrové krúžky. Keď sa poloha stratí úplne,
  ukáže sa červené `REF STRATENE` a počítanie sa pozastaví (aby sa nezrátalo nič nesprávne).
- Ak chceš kameru otočiť natrvalo, reštartuj server (`./live.sh stop`, `./live.sh count`): nový referenčný záber
  bude to, čo kamera vidí po štarte.
- Voľby: `--anchor on|off`, `--anchor-points 10` (0 = len ORB), `--anchor-calib 12` (dĺžka kalibrácie),
  `--anchor-deadband 1.0`, `--anchor-every 2`, `--anchor-features 1500`, `--anchor-min-inliers 20`, a
  `--anchor-log` (zapisuje polohu čiary do `out/anchor_log.csv`, na meranie stability).
- Obmedzenia: pri veľkej zmene svetla (noc, infračervený režim) sa obraz líši od dennej referencie, takže sa
  ukotvenie môže stratiť.

Prechody sa ukladajú do `out/`: `live_<čas>_events.csv` (každý prechod) a `live_counts.txt` (aktuálne počty).
Bez kamery sa dá vyskúšať na videu v reálnom tempe: `object_detect_live --input file:/cesta/video.mp4 --line 0.235 ...`.
Počítanie je náročné na CPU (beží stále, asi 4 jadrá).

## Počítanie zo súboru s ukotvenou čiarou (`video_count`)
To isté ako živé počítanie, ale vstupom je video a výsledok sa uloží. Celý záber, detekcia na výreze, čiara
ukotvená k scéne – ak sa v zázname zmení uhol kamery, čiara ostane na tom istom mieste ulice.
```bash
./target/release/video_count video.mp4 --line 0.235 --line-span 0.52,0.85 --roi 0,0.52,0.47,0.33
```
Výsledky v `out/`: `<názov>_counted.mp4` (video s čiarou, rámčekmi a počtami), `<názov>_events.csv`,
`<názov>_summary.txt` a `<názov>_reference.jpg` (vybrané referenčné body). Voľby sú rovnaké ako pri živom
počítaní (`--conf`, `--anchor on|off`, `--width`, `--detect-width`...), navyše `--output cesta/video.mp4`.

## Rýchlosť áut a ľudí (`--speed`)
Model kamery prevádza polohu v obraze na metre na zemi, takže z pohybu sledovaného objektu
sa dá vypočítať rýchlosť v km/h. Zobrazí sa pri rámčeku a pri každom prechode cez čiaru
sa zapíše do CSV (stĺpec `speed_kmh` / `rychlost_kmh`) a do súhrnu (priemer, min, max).
```bash
./target/release/video_count video.mp4 --line 0.5 --line-span 0.5,0.85 --roi 0,0.5,1,0.35 --speed calib/camera_model.json
./live.sh count      # live.sh pridá --speed sám, ak existuje calib/camera_model.json
```
Model sa počíta z meraní na mieste (`calib/calibrate.py`: vzdialenosti referenčných bodov od kamery,
výška kamery a pár vzdialeností medzi bodmi) a platí len pre tú polohu kamery, v ktorej sa merala.
Ak sa kamera od merania pootočila, záber sa pri štarte sám zarovná so snímkou z merania
(`calib/snapshot_4k.jpg`) a počas behu pohyb kamery vyrovnáva ukotvenie – čiara, referenčné body 1–8
(žlté krúžky s číslami) aj model rýchlosti ostávajú na tom istom mieste ulice.
Poloha objektu na zemi je stred spodnej hrany rámčeka (kolesá, chodidlá); rýchlosť je sklon priamky
cez polohy za posledných 1,2 s. Priečinok `calib/` je v `.gitignore` (obsahuje fotky okolia).

## Rovný obraz (`--dewarp`) a čistý pohľad (`--clean`)
`live.sh count` predvolene zapne obe, ak existuje `calib/dewarp/dewarp.json`:
- **Rovný obraz:** každá snímka sa hneď na vstupe prevedie z fisheye na rovný obraz (bez skreslenia, vyrovnaný
  náklon, orezaný tak, aby nemal čierne okraje). Detekcia, sledovanie, rýchlosť aj kreslenie bežia na ňom.
  Prevod je predpočítaná tabuľka (`cv::remap`), stojí pár ms na snímku.
- **Čistý pohľad:** v obraze sú len rámčeky áut a ľudí s rýchlosťou – bez čiary, panelu a bodov.
  Prechody sa ďalej zapisujú do `out/`.
- Vypnutie: `DEWARP= ./live.sh count` (pôvodný obraz), `CLEAN=0 ./live.sh count` (čiara a počty v obraze).
- Pri rovnom obraze sú predvolené `ROI=0,0.27,1,0.37` a pásmo čiary `0.33,0.62` (dá sa zmeniť premennými
  `ROI` a `SPAN_DW`).
- Obrázok sa dá previesť aj ručne: `cargo run --release --example dewarp_image -- calib/dewarp/dewarp.json vstup.jpg vystup.jpg`.

Čo sa kreslí: objekt až po 3 detekciách (jednorazové omyly siete sa neukážu), „vozidlo“, ktoré sa nepohlo,
sa po 2 s skryje (skrinka, zaparkované auto) a „človek“ sa ukáže až keď sa pohne (ozdoby brány).
Kúsky jedného vozidla z rôznych častí výrezu sa spájajú aj keď ich sieť nazve inak (bus/truck) a menšie rámčeky
vnútri väčšieho vozidla (okná autobusu ako „car“) sa zahodia.

Časy snímok: kamera posiela snímky v zhlukoch s rovnakou časovou značkou; časy sa vyhladia (`FrameClock`),
inak by rýchlosť pri zhluku vyskočila.

## Kalibrácia kamery (rýchlosť a rovný obraz)
Robí sa raz pre danú kameru a miesto; výsledok je v `calib/` (v `.gitignore` – fotky okolia a merania).
1. **Skreslenie objektívu z mriežky:** pred kameru sa dá mriežkovaný papier (čo najväčší, cez celý záber) a
   z priemeru ~10 snímok sa nájdu čiary mriežky. Hľadá sa korekcia (k1, k2, stred), po ktorej sú rovné
   (metóda rovných čiar, nezáleží na sklone papiera). Výsledok: čiary rovné na ~3 px (bez korekcie 23 px).
2. **Merania na mieste (laser):** vzdialenosti od päty kamery k 6–8 bodom na zemi (blízko aj na druhej strane
   ulice, vľavo aj vpravo), výška kamery a 1–2 vzdialenosti medzi bodmi. Z nich sa dopočíta ohnisko, sklon,
   náklon a výška kamery.
3. **Sklon automaticky z obrazu:** zo zvislých hrán (domy, stĺpy) v rovnom obraze sa nájde úbežník a z neho
   sklon a náklon kamery – zhoda s meraním laserom do 0,3°.
4. Z modelu sa pripraví rovný obraz (`calib/dewarp/`: `dewarp.json`, model rýchlosti, snímka z merania a body).

Overenie: moped s tachometrom na 26 km/h, 3 prejazdy → 25,2 / 26,6 / 26,0 km/h (chyba pod 1 % v priemere,
jednotlivé prejazdy ±3 %). Bez korekcie fisheye bola chyba na okrajoch záberu až −50 %, s korekciou len z meraní
laserom −6,5 %.

## Počítanie prechodov cez čiaru zo súboru s pevným orezaním (`line_count`)
Z videa sa orezá zadaná časť záberu (napr. len cesta), YOLOX nájde autá a ľudí, tracker im pridelí ID
a program počíta, koľkí prešli cez zvislú čiaru, zvlášť podľa smeru.
```bash
./target/release/line_count street.mp4 --crop 0,0.52,0.47,0.33 --line 0.5 --name street
```
- `--crop x,y,w,h` – ktorá časť pôvodného záberu (zlomky 0–1: ľavý horný roh a rozmery)
- `--line 0.5` – poloha čiary ako zlomok šírky orezaného záberu
- `--margin 12` – pásmo okolo čiary (px), v ktorom sa strana nemení (proti chveniu)
- `--every 2` – spracuje sa len každá 2. snímka (rýchlejšie; pozri nižšie)
- `--out-dir out` – kam sa uložia výsledky

Výstup: `<názov>_counted.mp4` (video s čiarou, rámčekmi a počtami), `<názov>_events.csv`
(každý prechod: čas, ID, skupina, smer) a `<názov>_summary.txt` (súhrn).

## Rýchlosť vs. presnosť pri počítaní zo súboru
Na videu `8xPresielSom.mp4` (osoba prešla cez čiaru 8-krát, 12 fps, 4K) som namerané (stroj bol pritom vyťažený,
takže čísla kolíšu):

| `--every` | spracovaných snímok za s videa | výsledok | rýchlosť spracovania |
|---|---|---|---|
| 1 (všetky) | 12 | 8 (správne) | ~9 fps zdroja |
| 2 | 6 | 8 (správne) | ~21 fps |
| 3 | 4 | 8 (správne) | ~30–36 fps |
| 4 | 3 | 8 (správne) | ~32–41 fps |
| 6 | 2 | 6 (zle) | |
| 8 a viac | ≤1,5 | 1–5 (zle) | |

Istota `--conf` od 0,25 do 0,85 aj šírka `--width` od 640 do 1920 dávali pri tomto videu stále 8, takže vplyv
na presnosť je malý. Hlavná záťaž je samotná sieť YOLOX (~84 ms na snímku na CPU, GPU cez OpenCL to nezrýchli):
`cargo run --release --example bench_dnn -- obrazok.jpg`. Rýchle objekty (autá) potrebujú menšie `--every` než chodec.

## Testy
```bash
cargo test --release
```
Unit testy (255), testy počítania (10) a e2e testy: súbory (8), živý server a počítanie (13), `line_count` (8), ukotvenie (5), `video_count` (5). Testy s modelom sa bez neho preskočia.

Poznámka: projekt používa OpenCV 5 (crate `opencv` 0.101), kde sú niektoré funkcie v iných moduloch než v OpenCV 4.
