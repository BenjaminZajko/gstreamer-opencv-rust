#!/bin/bash
# Spustí detekciu: ./run.sh <video.mp4> [motion|people]
cd "$(dirname "$0")"
export DYLD_FALLBACK_LIBRARY_PATH=/Library/Developer/CommandLineTools/usr/lib
cargo run --release -- "$1" --mode "${2:-motion}" --output "${2:-motion}.mp4" 2>&1 | grep -v -E "objc|GLib|NSApplication"
