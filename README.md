# gstreamer-opencv-rust

Detekcia objektov vo videu: **GStreamer** dekóduje a zakóduje video (HW enkodér VideoToolbox na macOS), **OpenCV** detekuje objekty. Napísané v Ruste.

## Požiadavky (macOS)
```bash
brew install gstreamer opencv
```
Rust (cargo) a Xcode Command Line Tools (kvôli libclang).

## Použitie
```bash
./run.sh video.mp4 motion   # pohybujúce sa objekty (MOG2)
./run.sh video.mp4 people   # chodci (HOG)
```
Výsledok sa uloží ako `motion.mp4` / `people.mp4`.

Poznámka: projekt používa OpenCV 5 (crate `opencv` 0.101), kde sú niektoré funkcie v iných moduloch než v OpenCV 4.
