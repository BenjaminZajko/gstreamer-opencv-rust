#!/bin/bash
# Živá detekcia z kamery cez vlastný RTSP server, beží v pozadí.
#   ./live.sh count [prepínače]   # spustí POČÍTANIE áut a ľudí cez čiaru (celý záber vo VLC)
#   ./live.sh counts              # vypíše aktuálne počty
#   ./live.sh start [prepínače]   # spustí detekciu s rámčekmi (napr. ./live.sh start --mode objects)
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
# nastavenie počítania pre aktuálny záber (zlomky obrazu); dá sa prepísať premennými prostredia
LINE="${LINE:-0.235}"           # poloha zvislej čiary (zlomok šírky záberu)
SPAN="${SPAN:-0.52,0.85}"       # pásmo výšky, v ktorom sa počíta
ROI="${ROI:-0,0.52,0.47,0.33}"  # výrez, na ktorom beží detekcia: x,y,w,h
PIDFILE="$DIR/live.pid"
LOG="$DIR/live.log"
BIN="$DIR/target/release/object_detect_live"
export DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
my_ip()   { ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<ip-macu>"; }
url()     { echo "rtsp://$(my_ip):$PORT$MOUNT"; }

# "count" = "start" so zapnutým počítaním
if [ "$1" = count ]; then
  shift
  set -- start --line "$LINE" --line-span "$SPAN" --roi "$ROI" "$@"
fi

case "$1" in
  counts)
    cat "$DIR/out/live_counts.txt" 2>/dev/null || echo "zatiaľ nič (spusti ./live.sh count)"
    ;;
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
    echo "použitie: $0 count | counts | start [prepínače] | stop | status | logs"
    ;;
esac
