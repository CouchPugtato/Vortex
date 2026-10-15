# Deployment

**Stage Release** now packages, uploads, verifies, and builds a versioned release
without replacing files in the running installation or stopping its service.
Activation, camera/runtime health checks, rollback, and service-attached monitoring
are still pending. A ready release is explicitly **not activated**.

## Stage a release

Select the intended camera, networking, build profile, runtime configuration, and
field map in the interface, then click **Stage Release**. Python is required on the
desktop as well as the Jetson; set `python_executable` in desktop preferences if the
default `python` (Windows) or `python3` command is unavailable.

The app creates one compressed snapshot containing Rust sources, Cargo manifests,
the toolchain pin, launch scripts, and selected configuration/field resources.
Desktop credentials, virtual environments, captures, tests, documentation media,
and existing build outputs are excluded. When TensorRT object detection is enabled,
the currently supported canonical `models/rockpaperscizzors.engine` is required and
included. Engine compatibility still needs hardware validation.

The manifest records the source revision, worktree dirty flag when Git is available,
requested profile/features, snapshot hash, and every payload file's size, mode, and
SHA-256. If Git is unavailable, revision/dirty values are explicitly unknown; file
checksums still identify the exact snapshot.

On the Jetson the archive goes to a **sibling** of the existing deployment:

```text
/home/vortex/deployments/Vortex/              existing installation, unchanged
/home/vortex/deployments/Vortex.deployment/
  incoming/RELEASE_ID/                       archive and bootstrap controller
  staging/RELEASE_ID/                        incomplete or failed build
  releases/RELEASE_ID/                       verified ready release
  jobs/RELEASE_ID.json                       saved state and result
  jobs/RELEASE_ID.log                        validation and compiler output
  build-cache/PROFILE/                      reusable Cargo artifacts
```

The bootstrap controller is checksummed before execution. Its detached job verifies
the complete archive, rejects unsafe paths/links/duplicates and unexpected files,
checks free extraction space, and verifies each extracted file. A nonblocking OS
lock at `~/.local/state/vortex/deployment.lock` serializes verification and builds
across deployment roots for the runtime account. Another job gets a clear busy
failure. Uploads use unique incoming directories and can safely overlap.

The job validates configuration, builds the explicit profile offline, checks build
fingerprints, and rechecks the original payload before moving the completed tree
into `releases/`. Each release receives private executable copies; a later build
cannot replace its binaries through the shared cache. Builds use two Cargo jobs
and a lower process priority, but compilation still competes with live vision for
CPU and memory. Build duration and camera performance need Jetson measurement.

Closing the app or disconnecting SSH does not stop the detached build. The current
app reconnects to the same job after temporary connection loss. After reopening,
status and logs can be inspected over SSH; automatic job reattachment in the UI is
pending. Reboot/power loss can leave an incomplete staging directory and saved
in-progress status; it cannot activate that directory. Failed packages, staging
trees, and logs are retained for diagnosis. Automatic retention cleanup is pending.

No build failure, checksum failure, or lock rejection switches the running release.
The old independent **Build Profile** button is removed; its compatibility IPC
endpoint also stages a new snapshot instead of modifying the active tree.

To inspect a saved job on the Jetson:

```text
python3 /home/vortex/deployments/Vortex.deployment/incoming/RELEASE_ID/controller.py status --root /home/vortex/deployments/Vortex.deployment --release-id RELEASE_ID
```

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

The interface snapshots camera/network/profile settings into the package's
`config/device.json`; its selected field map uses `config/apriltag_map.json`.
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

Stage Release requires Python, the pinned Rust toolchain, native build libraries,
and cached Cargo dependencies on the Jetson. It does not provision dependencies or
install a service. A new device still needs the bootstrap script after JetPack is
installed and an initial project copy is available. Missing offline dependencies
need explicit online provisioning using the command above.

Existing source deployments continue running at their existing paths. Stage Release
does not migrate their systemd unit or select a newly prepared release. For now,
keep using the existing installation for production; activation and migration of
its service to a ready release are the next implementation stage.

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

1. Add native/runtime library and smoke-test validation, release activation,
   application health, rollback, and interrupted-activation recovery.
2. Manage jobs through systemd, add UI reattachment/status/rollback, and replace
   legacy diagnostics with service-owned preview.
3. Add fresh-device compatibility checks, guided initialization, offline provisioning,
   and release retention cleanup.
