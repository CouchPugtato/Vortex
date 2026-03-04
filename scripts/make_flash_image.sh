#!/usr/bin/env bash
set -euo pipefail

# Creates a full raw image of the Jetson boot disk.
# Output .img can be flashed with Balena Etcher (for same target storage type/capacity).

usage() {
  cat <<'EOF'
Usage:
  sudo ./scripts/make_flash_image.sh [output_dir] [--compress]

Examples:
  sudo ./scripts/make_flash_image.sh /media/vortex/USB_DRIVE
  sudo ./scripts/make_flash_image.sh /media/vortex/USB_DRIVE --compress

Notes:
  - Run as root (or via sudo).
  - Output directory should be on a DIFFERENT disk than the source boot disk.
  - --compress creates .img.gz after imaging.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_USER="${SUDO_USER:-${USER:-vortex}}"
OUT_DIR="${1:-/media/${RUN_USER}}"
COMPRESS="0"
if [[ "${1:-}" == "--compress" || "${2:-}" == "--compress" ]]; then
  COMPRESS="1"
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script must run as root. Example:"
  echo "  sudo $0 ${OUT_DIR} --compress"
  exit 1
fi

if [[ ! -d "${OUT_DIR}" ]]; then
  echo "Output directory does not exist: ${OUT_DIR}"
  exit 1
fi

ROOT_SRC="$(findmnt -n -o SOURCE / || true)"
if [[ -z "${ROOT_SRC}" ]]; then
  echo "Could not determine root source device."
  exit 1
fi

SRC_PARENT="$(lsblk -no PKNAME "${ROOT_SRC}" 2>/dev/null || true)"
if [[ -n "${SRC_PARENT}" ]]; then
  SRC_DISK="/dev/${SRC_PARENT}"
else
  SRC_DISK="${ROOT_SRC}"
fi

if [[ ! -b "${SRC_DISK}" ]]; then
  echo "Resolved source disk is not a block device: ${SRC_DISK}"
  exit 1
fi

OUT_FS_DEV="$(df -P "${OUT_DIR}" | awk 'NR==2{print $1}')"
OUT_PARENT="$(lsblk -no PKNAME "${OUT_FS_DEV}" 2>/dev/null || true)"
if [[ -n "${OUT_PARENT}" ]]; then
  OUT_DISK="/dev/${OUT_PARENT}"
else
  OUT_DISK="${OUT_FS_DEV}"
fi

if [[ "${OUT_DISK}" == "${SRC_DISK}" ]]; then
  echo "Refusing to write image to the same physical disk being imaged."
  echo "Source disk: ${SRC_DISK}"
  echo "Output disk: ${OUT_DISK}"
  echo "Choose an output directory on an external USB/NVMe drive."
  exit 1
fi

SRC_SIZE_BYTES="$(blockdev --getsize64 "${SRC_DISK}")"
OUT_FREE_BYTES="$(df --output=avail -B1 "${OUT_DIR}" | tail -n1 | tr -d ' ')"

if [[ "${OUT_FREE_BYTES}" -lt "${SRC_SIZE_BYTES}" ]]; then
  echo "Not enough free space in ${OUT_DIR}."
  echo "Needed: ${SRC_SIZE_BYTES} bytes"
  echo "Free:   ${OUT_FREE_BYTES} bytes"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname -s 2>/dev/null || echo jetson)"
IMG_PATH="${OUT_DIR%/}/${HOST}_full_${STAMP}.img"

echo "Source disk: ${SRC_DISK}"
echo "Output path: ${IMG_PATH}"
echo "Starting imaging..."

dd if="${SRC_DISK}" of="${IMG_PATH}" bs=16M status=progress conv=fsync
sync

if [[ "${COMPRESS}" == "1" ]]; then
  echo "Compressing image..."
  if command -v pigz >/dev/null 2>&1; then
    pigz -1 "${IMG_PATH}"
    IMG_PATH="${IMG_PATH}.gz"
  else
    gzip -1 "${IMG_PATH}"
    IMG_PATH="${IMG_PATH}.gz"
  fi
fi

echo "Done."
echo "Image file: ${IMG_PATH}"
echo "You can flash this with Balena Etcher to a same-size-or-larger target device."
