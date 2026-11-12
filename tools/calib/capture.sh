#!/bin/bash
# Nasníma N snímok z kamery v plnom rozlíšení (na mriežku alebo snímku z merania).
#   tools/calib/capture.sh calib/grid 10        # 10 snímok mriežky -> calib/grid/g_01.png ...
#   tools/calib/capture.sh calib 1 snapshot     # snímka z merania -> calib/snapshot_01.png
#                                                 (premenuj na calib/snapshot_4k.jpg)
# Údaje ku kamere z CAM_CONF (predvolene ../GStream_configuration/camera.conf: CAM_IP, CAM_USER, CAM_PASS).
set -e
DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CAM_CONF="${CAM_CONF:-$DIR/../GStream_configuration/camera.conf}"
OUT="${1:?použitie: capture.sh <priečinok> [počet] [predpona]}"
N="${2:-10}"
PREFIX="${3:-g}"
[ -f "$CAM_CONF" ] || { echo "chýba $CAM_CONF (CAM_IP, CAM_USER, CAM_PASS)"; exit 1; }
source "$CAM_CONF"
mkdir -p "$OUT"
ffmpeg -v error -y -rtsp_transport tcp -i "rtsp://${CAM_USER}:${CAM_PASS}@${CAM_IP}:554/${STREAM:-stream1}" \
  -frames:v "$N" -q:v 1 "$OUT/${PREFIX}_%02d.png"
ls "$OUT"/${PREFIX}_*.png
