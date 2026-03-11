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
export VORTEX_RUNTIME_CONFIG="${VORTEX_RUNTIME_CONFIG:-config/config.json}"
export VORTEX_NT_ENABLE=0
export VORTEX_UDP_ENABLE=1
export VORTEX_UDP_TARGET=10.5.9.2
export VORTEX_UDP_PORT=5091

exec "$BIN" 0,1
