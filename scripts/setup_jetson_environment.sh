#!/usr/bin/env bash
set -euo pipefail
# Optional terminal entry point for the same persistent setup job used by the app.
# Supply --root, --legacy-root, --user, --service, --job-id, --release-id, --sha256.
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONTROLLER="${VORTEXCTL:-$APP_DIR/tools/vortexctl}"
if [[ ! -x "$CONTROLLER" ]]; then
  echo "Missing native controller. Use Initialize Jetson in the deployment app, or install the matching Linux vortexctl in tools/." >&2
  exit 1
fi
exec "$CONTROLLER" initialize-submit "$@"
