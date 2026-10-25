# Offline initialization

Create a bundle on an internet-connected Ubuntu machine with the **same Ubuntu
version and CPU architecture** as the target. For ARM64 Jetsons, use an ARM64
preparation machine/image. Matching NVIDIA VPI/CUDA/TensorRT components must already
be present on the target; these bundles do not install or upgrade JetPack.

The preparation machine needs Rustup and the exact toolchain from the project's
`rust-toolchain.toml`. Build `vortexctl`, then prepare packages and export:

```sh
sudo /path/to/vortexctl offline-download --output /absolute/path/package-inputs
rustup toolchain install 1.96.0 --profile minimal
/path/to/vortexctl offline-create --source /absolute/path/Vortex \
  --packages /absolute/path/package-inputs/packages \
  --output /absolute/path/offline-bundle --profile cpu
```

Select the actual `cpu`, `gpu`, `tensorrt` or `gpu-tensorrt` profile. Package download
uses an empty installed-package database so dependencies already installed on the
preparation machine are included. Export fetches locked Cargo dependencies, copies
the native toolchain, Rustup and public crates.io caches, and writes
`offline.tar.gz` plus `offline.json`. Keep those two files together. Export supports
crates.io dependencies; Git/private registry dependencies are rejected. SSH keys,
Cargo credentials and the user's Cargo configuration are excluded.

In **Jetson Setup**, browse to `offline.tar.gz` and select **Initialize Jetson**.
The app checks its sidecar/hash, uploads it alongside the release and uses the normal
administrator-password and reconnect flow. Users do not need an SSH terminal.

The worker checks every bundled file and requires exact Ubuntu version,
architecture, selected profile, Rust pin and `Cargo.lock` agreement before installing
packages. It indexes only the bundled file repository inside a network namespace,
uses APT `--no-download --no-remove`, skips online APT/Rustup downloads, restores
the compiler/cache, and performs Cargo fetch/build with `--offline`. Missing inputs
fail setup; there is no online fallback. APT package changes that completed before
a later failure remain installed, as with online setup.

Allow room for the uploaded archive, its extracted contents, the restored toolchain,
Cargo caches and the vision build. The worker checks extraction size plus a 6 GiB
reserve; unusually large build profiles can need more. The normal release remains
unselected until **Activate**. Ordinary staging reuses the restored caches offline.

Recreate the bundle after changing the dependency lock, Rust version, profile or
target Ubuntu/architecture. Real clean-image/JetPack initialization remains part of
the [hardware acceptance run](deployment-validation.md).
