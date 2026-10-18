# vortexctl

Standalone Rust controller for packaging, staging, Jetson initialization,
activation, rollback, recovery and production diagnostics. It has no camera,
GPU or Python dependencies. The vision application remains a separate build.

```sh
cd deployment
cargo test --locked
cargo build --release --locked
```

The desktop app bundles Windows x64 and static Linux ARM64/x64 controllers.
On a build machine, install Rust 1.96.0, MinGW (Windows GNU target), Zig 0.13.0,
and `cargo install cargo-zigbuild --locked --version 0.23.4`, then run:

```sh
cd VortexDeploymentInterface
npm run build:native
npm test
npm run pack
```

On Windows, put MinGW and Zig on PATH. Alternatively, build the Linux targets
in WSL with `deployment/build-linux.sh`; set `VORTEX_ZIG_DIR` to the Zig directory.
Copy all three target/release outputs into one Cargo target directory, then use
`VORTEXCTL_ARTIFACTS=<directory> node scripts/build-native.js --import` from the
desktop folder. CI uploads the ready-to-bundle `native` directory, including its
SHA256 manifest. Extract that artifact into `VortexDeploymentInterface/native`
and run `npm run check:native` before packaging.

End users install neither Rust nor Python on their desktop. The app selects the
Linux controller using the device's architecture. Initialize Jetson installs the
vision toolchain and dependencies remotely; subsequent staging builds offline.

For source-tree CLI use, copy the matching controller to `tools/vortexctl`
(`tools/vortexctl.exe` on Windows), or invoke the built controller directly:

```sh
deployment/target/release/vortexctl launch --source . --validate-config
deployment/target/release/vortexctl launch --source . --build --online
```

See [deployment operations](../docs/deployment.md). Real camera and JetPack
validation still require a target device. The tests use isolated directories
and fake service managers; they do not install packages or modify host services.
