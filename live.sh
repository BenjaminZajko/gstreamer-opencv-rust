#!/bin/bash
# Živá detekcia z kamery cez vlastný RTSP server, beží v pozadí.
#   ./live.sh start [prepínače]   # spustí (napr. ./live.sh start --mode objects --conf 0.5)
#   ./live.sh stop                # zastaví
#   ./live.sh status              # beží? + adresa pre VLC
#   ./live.sh logs                # sleduje log (Ctrl+C = koniec sledovania, server beží ďalej)
#
# Kamera sa berie z CAM_CONF (predvolene ../GStream_configuration/camera.conf: CAM_IP, CAM_USER,
# CAM_PASS). Heslo sa posiela cez premennú prostredia, nie cez argumenty (nie je vidno v `ps`).

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
CAM_CONF="${CAM_CONF:-$DIR/../GStream_configuration/camera.conf}"
STREAM="${STREAM:-stream1}"
PORT="${PORT:-8554}"
MOUNT="${MOUNT:-/detect}"
PIDFILE="$DIR/live.pid"
LOG="$DIR/live.log"
BIN="$DIR/target/release/object_detect_live"
export DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
my_ip()   { ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<ip-macu>"; }
url()     { echo "rtsp://$(my_ip):$PORT$MOUNT"; }

case "$1" in
  start)
    shift
    if running; then echo "už beží (pid $(cat "$PIDFILE"))"; echo "VLC: $(url)"; exit 0; fi
    [ -f "$CAM_CONF" ] || { echo "chýba $CAM_CONF (CAM_IP, CAM_USER, CAM_PASS)"; exit 1; }
    source "$CAM_CONF"
    cargo build --release --bin object_detect_live 2>&1 | grep -E "^(error|warning)" -A6
    [ -x "$BIN" ] || { echo "build zlyhal"; exit 1; }
    LIVE_INPUT="rtsp://${CAM_USER}:${CAM_PASS}@${CAM_IP}:554/${STREAM}" \
      nohup "$BIN" --port "$PORT" --mount "$MOUNT" "$@" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    # počkáme, kým sa server rozbehne (alebo spadne)
    for _ in $(seq 1 20); do
      sleep 1
      if ! running; then echo "spadol pri štarte:"; grep -v -E "objc|GLib|NSApplication" "$LOG" | tail -5; rm -f "$PIDFILE"; exit 1; fi
      grep -q "RTSP server beží" "$LOG" && { echo "beží (pid $(cat "$PIDFILE"))"; echo "VLC: Médiá -> Otvoriť sieťový stream -> $(url)"; exit 0; }
    done
    echo "štartuje sa dlho, pozri ./live.sh logs"
    ;;
  stop)
    if running; then kill "$(cat "$PIDFILE")"; rm -f "$PIDFILE"; echo "zastavené"; else echo "nebeží"; rm -f "$PIDFILE"; fi
    ;;
  status)
    if running; then echo "beží (pid $(cat "$PIDFILE"))"; echo "VLC: $(url)"; else echo "nebeží"; fi
    ;;
  logs)
    tail -f "$LOG" | grep --line-buffered -v -E "objc|GLib|NSApplication"
    ;;
  *)
    echo "použitie: $0 start [prepínače] | stop | status | logs"
    ;;
esac
