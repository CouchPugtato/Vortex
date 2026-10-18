#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source "$HOME/.cargo/env"
export PATH="${VORTEX_ZIG_DIR:-$HOME/.cache/vortex-zig}:$PATH"
TOOLCHAIN="${VORTEXCTL_TOOLCHAIN:-1.96.0}"
rustup target add --toolchain "$TOOLCHAIN" aarch64-unknown-linux-musl x86_64-unknown-linux-musl
cargo "+$TOOLCHAIN" zigbuild --release --locked --target aarch64-unknown-linux-musl --target x86_64-unknown-linux-musl "${@}"
