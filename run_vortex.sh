#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$APP_DIR/tools/vortexctl" launch --source "$APP_DIR" "$@"
