# Deployment

**Stage Release** now packages, uploads, verifies, and builds a versioned release
without replacing files in the running installation or stopping its service.
**Activate** performs preflight checks, switches the systemd service, and verifies
camera health. **Rollback** restores the previous release/configuration. A ready
release remains **not activated** until Activate is selected.

## Stage a release

Select the intended camera, networking, build profile, runtime configuration, and
field map in the interface, then click **Stage Release**. Python is required on the
desktop as well as the Jetson; set `python_executable` in desktop preferences if the
default `python` (Windows) or `python3` command is unavailable.

The app creates one compressed snapshot containing Rust sources, Cargo manifests,
the toolchain pin, launch scripts, and selected configuration/field resources.
Desktop credentials, virtual environments, captures, test suites, documentation
media, and existing build outputs are excluded. One fixed saved-image smoke
fixture is included as a runtime validation resource. When TensorRT object detection is enabled,
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

The bootstrap controller is checksummed before execution. Its systemd job verifies
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

Staging and activation run as transient user-level systemd services, independent
of SSH and the desktop. Closing the app does not stop them. The interface saves the
job identity and device location before submitting, automatically reconnects on
reopening, and restores progress, the latest log output, and the result. It never
resubmits the saved operation. **Refresh Releases** retries attachment after a
longer outage; pending jobs block new deployment operations in that desktop.
Connection settings must still match the saved device and deployment folder.

The Jetson needs `dbus-user-session` and a persistent user manager. Bootstrap and
managed-service setup enable [user lingering](https://www.freedesktop.org/software/systemd/man/252/loginctl.html).
For an existing installation, run this once before staging:

```text
sudo loginctl enable-linger vortex
```

Jobs retain `jobs/ID.unit.json` supervisor metadata, `ID.controller.log`, and the
existing status/build logs. Staging has a two-hour limit; activation has a five-minute
limit. Workers do not restart automatically. Status reconciles unfinished jobs
against systemd and reports crashes, timeouts, or units lost during reboot as
**interrupted**, preserving activation recovery metadata. A temporary manager
connection failure remains an inspection error, rather than a claimed interruption.
Incomplete staging trees cannot be activated. Failed packages, trees, and logs are
retained; automatic retention cleanup is pending. Reattachment covers submitted
jobs; packaging and partial uploads must be retried after closing the app.

No build failure, checksum failure, or lock rejection switches the running release.
The old independent **Build Profile** button is removed; its compatibility IPC
endpoint also stages a new snapshot instead of modifying the active tree.

To inspect a saved job on the Jetson:

```text
python3 /home/vortex/deployments/Vortex.deployment/incoming/RELEASE_ID/controller.py status --root /home/vortex/deployments/Vortex.deployment --release-id RELEASE_ID
```

## Activate and roll back

After Stage Release completes, select its ID and click **Activate**. **Refresh
Releases** lists prepared releases and the committed selected/previous versions.
The desktop remembers its last prepared release across restarts.

Under the same device lock, activation:

1. Rechecks payload/build hashes and checks the ELF machine against the device.
2. Checks shared libraries with `ldd` and runs the production executable's
   `--deployment-smoke` mode on the fixed noisy tag-1 image. The CPU detector/pose
   solver must recover its known position within 20 cm; no camera is opened.
3. Flushes release files and records a durable activation journal before stopping
   the service.
4. Atomically replaces `current`, selects the release's own configuration snapshot,
   and starts the service through its stable management entry point.
5. Requires three seconds of advancing health reports within a 30-second startup
   deadline. Every configured camera must process fresh frames with successful
   required detector backends. A GPU profile falling back to CPU fails this check.
6. Commits the new/previous selections only after health verification.

Health is written atomically to `Vortex.deployment/health.json` and includes release
ID, an activation-specific launch token, service PID, monotonic uptime, and per-camera
frame counts/age/backend status. Verification checks the actual systemd MainPID,
requires report/frame counters to advance, and rejects stale data. AprilTags and
robot-network connectivity are not required for camera health. This is startup
verification, not continuous pose-accuracy monitoring or a runtime watchdog.

On startup failure, the controller stops the failed version, restores the previous
release and configuration, restarts it, and checks its health. **Rollback** manually
switches to the saved previous selection through the same transaction. A legacy
installation restored during the first migration can only be checked for a running
process; its old binary does not expose the new camera-health protocol. The result
explicitly says `legacy_process_only`. A failed first installation with no previous
selection remains stopped. If restoration also fails, the result is
`rollback_failed` and its journal/logs remain available; it never reports success.

Activation/rollback are systemd jobs, so closing the desktop does not abandon the
transaction. If power fails before confirmation, the systemd entry point restores
the previous selection before starting a runtime. A durable confirmed switch is
kept. The journal uses boot ID plus process start identity to distinguish an active
worker from an interrupted one. Recovery records whether camera health was actually
checked; recovery before service startup does not claim it was.

State files are `selection.json`, `active.json`, `previous.json`, `activation.json`
(only while pending), and `recovery.json`. Operation logs/status live in `jobs/`.
Manual recovery, using the deployed controller on the Jetson:

```text
python3 /home/vortex/deployments/Vortex.deployment/activate_release.py recover --root /home/vortex/deployments/Vortex.deployment
```

A failed confirmation or interrupted switch never edits either release's camera
calibration/mount configuration. Editing settings for the next deployment creates
another snapshot; it does not alter an already prepared release.

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
before executing a verified binary. Activation additionally checks ELF architecture,
runtime libraries, a saved-image CPU smoke test, and actual camera processing.

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

Existing source deployments continue running until Activate. First-time service
preparation requires root privileges; Activate attempts noninteractive sudo and
reports a concrete setup command if permission is unavailable. Run that command
once in an SSH terminal using sudo, then retry Activate. The updated bootstrap also
performs this preparation. Existing packages from before this feature must be
restaged because they lack the health/smoke modules.

The installer supports migration of the existing `run_vortex.sh` systemd unit.
It rejects unexpected launch commands and service drop-ins rather than discarding
custom behavior. Managed service/root paths must not contain spaces or systemd
specifier characters. It backs up the old unit, installs stable management scripts,
and grants only start/stop/reset-failed permission for that specific unit through
a validated sudoers entry. Installing/upgrading the controller itself still needs
root; ordinary Activate/Rollback operations run as the runtime user.

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

Before managed activation, **Start Diagnostics** can temporarily replace production capture with
`orin_bridge`, which provides the existing single-camera preview and diagnostics
but not the production telemetry. It now uses the same Python launcher and runtime
configuration, verifies its build before launching, and never installs packages,
deletes lockfiles, or silently chooses another build profile.

Legacy diagnostics is blocked when a managed release is selected so it cannot
kill/reconfigure production capture. Service-attached preview remains a later stage.

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

1. Add service-owned preview and a continuous frame-processing watchdog.
2. Add fresh-device compatibility checks, guided initialization, offline provisioning,
   and release retention cleanup.

Linux service installation, sudoers validation, real camera timing, GPU/TensorRT,
and boot recovery still require Jetson testing. Local transaction tests inject
service failures and power interruption; Windows cannot exercise the real Linux
systemd/symlink operations. The Rust smoke/health modules are compiled and tested
through the native CPU replay harness.
