# Deployment

The deployment upgrade is being implemented in stages. The shared launcher and
build contract are implemented. Versioned releases, persistent deployment jobs,
health-based activation, automatic rollback, and service-attached monitoring are
still pending. The existing Deploy action still uploads into the existing project
directory; it is not yet a transactional release deployment.

## Runtime and configuration

Production runs `dumapril-taglocalization` through `scripts/run_vortex.py`.
`run_vortex.sh` is a location-independent compatibility wrapper around that Python
launcher. Startup never downloads dependencies or builds an executable.

`config/device.json` is the versioned launch configuration:

- Camera indices and explicit build profile.
- Runtime calibration/configuration path and field map path.
- NetworkTables enable flag, destination mode, and table.
- UDP enable flag, destination, and port.

The launcher removes inherited values for the settings it manages and applies this
file, so an old shell environment cannot override the selected network destination.
It validates device settings, camera calibration structure, and field-map geometry
before executing a verified binary. Camera availability and runtime library checks
are not yet implemented.

In the deployment interface, camera/network/profile settings are converted into
`config/device.json` on the Jetson when Deploy or Build Profile synchronizes the
launcher. The selected field map is copied to `config/apriltag_map.json`.
SSH credentials and desktop paths are not included in the device settings payload.
Other live runtime edits still use the existing remote config sync behavior;
production application of those edits may require a restart.

Desktop preferences are stored in Electron's user-data directory. Existing
repository preferences are migrated on first launch if no new preferences exist.

## Build profiles

| Profile | Cargo features |
| --- | --- |
| `cpu` | None |
| `gpu` | `gpu` |
| `tensorrt` | `tensorrt` |
| `gpu-tensorrt` | `gpu,tensorrt` |

The selected feature set is built explicitly; failures do not trigger a different
profile or removal of `Cargo.lock`. GPU/TensorRT profiles require the matching
NVIDIA libraries on the Jetson. Selecting a profile does not install those libraries.

Builds use the exact version in `rust-toolchain.toml`, `--locked`, and
`--no-default-features`. Dependency resolution is offline by default. Normal builds
also disable Rustup's automatic toolchain installation, so a missing pinned toolchain
fails instead of triggering a download. See [Rustup environment settings](https://rust-lang.github.io/rustup/environment-variables.html).

From the repository root on the Jetson:

```text
python3 scripts/run_vortex.py --validate-config
python3 scripts/run_vortex.py --build
python3 scripts/run_vortex.py --check
python3 scripts/run_vortex.py
```

For first-time dependency provisioning, explicitly allow downloads:

```text
python3 scripts/run_vortex.py --build --online --with-debug-bridge
```

Successful builds write `target/release/vortex-build.json`. Launch verifies the
binary checksum, source fingerprint, and selected profile. A legacy executable
without this fingerprint must be rebuilt. Changing cameras or networking does not
require recompilation; changing source, toolchain pin, or profile does.

## Migrating an existing Jetson

Deploy the updated source, select the intended profile/networking/cameras in the
interface, and run **Build Profile** before attempting to start the updated launcher.
Build Profile also builds the legacy diagnostic bridge. Missing offline dependencies
need the explicit online provisioning command above.

`scripts/setup_jetson_environment.sh` now installs Python, provisions the pinned
toolchain, grants the runtime user membership in `video`, builds through the shared
launcher, and verifies the build before installing/restarting the service. It still
requires a flashed Jetson and an existing runtime user; complete hardware/JetPack
compatibility detection and the Initialize Jetson interface are pending.

The service name defaults to `vortex-vision.service` in bootstrap. Set
`startup_service_name` to that name in the interface preferences if the service
should be stopped/resumed around the current tools. Stop failures abort the operation.
The service-control commands use noninteractive sudo and require suitable permissions.

## Legacy diagnostics

**Start Diagnostics** still temporarily replaces production capture with
`orin_bridge`, which provides the existing single-camera preview and diagnostics
but not the production telemetry. It now uses the same Python launcher and runtime
configuration, verifies its build before launching, and never installs packages,
deletes lockfiles, or silently chooses another build profile.

Service-attached preview will replace this interruption in a later stage.

## Local checks

These checks use temporary files and mocked remote operations; no Jetson is contacted:

```text
python -m unittest discover -s scripts -p "test_*.py"
```

From `VortexDeploymentInterface/`:

```text
npm test
```

Native builds, GPU libraries, actual camera access, and systemd behavior still need
Jetson validation.

## Remaining implementation order

1. Package only required files; upload checksummed archives into isolated staging;
   enforce a device operation lock.
2. Build and validate a complete release without changing the running installation.
3. Add release activation, application health, rollback, and interrupted-activation recovery.
4. Connect the interface to persistent jobs, status, rollback, and service-owned preview.
5. Add fresh-device compatibility checks, guided initialization, and offline provisioning.
