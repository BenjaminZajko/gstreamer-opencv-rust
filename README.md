# gstreamer-opencv-rust

Detekcia objektov vo videu: **GStreamer** dekóduje a zakóduje video (HW enkodér VideoToolbox na macOS), **OpenCV** detekuje objekty. Napísané v Ruste.

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

## Testy
```bash
cargo test --release
```
Unit testy (57), e2e testy (8) a e2e testy živého servera (9). Testy s modelom sa bez neho preskočia.

Poznámka: projekt používa OpenCV 5 (crate `opencv` 0.101), kde sú niektoré funkcie v iných moduloch než v OpenCV 4.
