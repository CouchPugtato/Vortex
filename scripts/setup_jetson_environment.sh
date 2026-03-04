#!/usr/bin/env bash
set -euo pipefail

# Idempotent bootstrap for a freshly deployed Jetson runtime environment.
# - Installs system packages needed by this project
# - Installs/updates Rust toolchain for the runtime user
# - Normalizes and validates run_vortex.sh
# - Builds release binaries
# - Installs and enables systemd startup service

APP_DIR_DEFAULT="/home/vortex/deployments/Vortex"
SERVICE_NAME_DEFAULT="vortex-vision.service"
RUN_USER_DEFAULT="vortex"
RUN_GROUP_DEFAULT="vortex"

APP_DIR="${APP_DIR_DEFAULT}"
SERVICE_NAME="${SERVICE_NAME_DEFAULT}"
RUN_USER="${RUN_USER_DEFAULT}"
RUN_GROUP="${RUN_GROUP_DEFAULT}"
SKIP_APT="0"
SKIP_RUST="0"
SKIP_BUILD="0"

usage() {
  cat <<EOF
Usage:
  sudo ./scripts/setup_jetson_environment.sh [options]

Options:
  --app-dir <path>          App directory (default: ${APP_DIR_DEFAULT})
  --service-name <name>     systemd service name (default: ${SERVICE_NAME_DEFAULT})
  --user <name>             Runtime user (default: ${RUN_USER_DEFAULT})
  --group <name>            Runtime group (default: ${RUN_GROUP_DEFAULT})
  --skip-apt                Skip apt package install
  --skip-rust               Skip rust toolchain setup
  --skip-build              Skip cargo release build
  -h, --help                Show this help

Example:
  sudo ./scripts/setup_jetson_environment.sh --app-dir /home/vortex/deployments/Vortex
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app-dir)
      APP_DIR="${2:-}"
      shift 2
      ;;
    --service-name)
      SERVICE_NAME="${2:-}"
      shift 2
      ;;
    --user)
      RUN_USER="${2:-}"
      shift 2
      ;;
    --group)
      RUN_GROUP="${2:-}"
      shift 2
      ;;
    --skip-apt)
      SKIP_APT="1"
      shift
      ;;
    --skip-rust)
      SKIP_RUST="1"
      shift
      ;;
    --skip-build)
      SKIP_BUILD="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script must be run with sudo/root."
  exit 1
fi

if ! id -u "${RUN_USER}" >/dev/null 2>&1; then
  echo "Runtime user does not exist: ${RUN_USER}"
  exit 1
fi

if [[ ! -d "${APP_DIR}" ]]; then
  echo "App directory not found: ${APP_DIR}"
  exit 1
fi

if [[ ! -f "${APP_DIR}/Cargo.toml" ]]; then
  echo "Cargo.toml not found in ${APP_DIR}. Deploy project first."
  exit 1
fi

if [[ ! -f "${APP_DIR}/run_vortex.sh" ]]; then
  echo "run_vortex.sh not found in ${APP_DIR}. Deploy project first."
  exit 1
fi

echo "==> Bootstrapping Jetson environment"
echo "    APP_DIR=${APP_DIR}"
echo "    SERVICE=${SERVICE_NAME}"
echo "    USER=${RUN_USER}:${RUN_GROUP}"

if [[ "${SKIP_APT}" != "1" ]]; then
  echo "==> Installing apt dependencies"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    cmake \
    clang \
    libclang-dev \
    libssl-dev \
    libudev-dev \
    libturbojpeg0-dev \
    libv4l-dev \
    v4l-utils
fi

if [[ "${SKIP_RUST}" != "1" ]]; then
  echo "==> Ensuring Rust toolchain for ${RUN_USER}"
  sudo -u "${RUN_USER}" -H bash -lc '
    set -euo pipefail
    if ! command -v rustup >/dev/null 2>&1; then
      curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
    fi
    source "$HOME/.cargo/env"
    rustup toolchain install stable --profile minimal
    rustup default stable
    rustup component add rustfmt || true
  '
fi

echo "==> Normalizing startup script"
sed -i 's/\r$//' "${APP_DIR}/run_vortex.sh"
chmod +x "${APP_DIR}/run_vortex.sh"
chown "${RUN_USER}:${RUN_GROUP}" "${APP_DIR}/run_vortex.sh"

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "==> Building release binaries"
  sudo -u "${RUN_USER}" -H bash -lc "
    set -euo pipefail
    source \"\$HOME/.cargo/env\"
    cd \"${APP_DIR}\"
    cargo build --release --bin dumapril-taglocalization
    cargo build --release --bin orin_bridge || true
  "
fi

SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
echo "==> Installing ${SERVICE_NAME}"
cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=Vortex Vision Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/run_vortex.sh
Restart=always
RestartSec=2
KillSignal=SIGINT
TimeoutStopSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

echo "==> Done"
echo "Check service:"
echo "  systemctl status ${SERVICE_NAME}"
echo "  journalctl -u ${SERVICE_NAME} -n 100 --no-pager"
