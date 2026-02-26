#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/vortex/deployments/Vortex"
BIN="$APP_DIR/target/release/dumapril-taglocalization"

cd "$APP_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "[vortex] release binary missing, building..."
  cargo build --release --bin dumapril-taglocalization
fi

# Runtime config
export VORTEX_NT_ENABLE=0
export VORTEX_UDP_ENABLE=1
export VORTEX_UDP_TARGET=192.168.1.24
export VORTEX_UDP_PORT=5809

exec "$BIN" 0,2
