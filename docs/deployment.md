# Deployment

**Stage Release** now packages, uploads, verifies, and builds a versioned release
without replacing files in the running installation or stopping its service.
**Activate** performs preflight checks, switches the systemd service, and verifies
camera health. **Rollback** restores the previous release/configuration. A ready
release remains **not activated** until Activate is selected.

## Initialize a Jetson from the app

Users do not need an SSH terminal. Open **Jetson Setup**, enter the device address,
SSH port/account/password, project source folder, deployment parent folder, and
startup service name. Enter the account's sudo password in **Administrator password**
(or leave it empty for passwordless sudo), select the intended build profile, and
click **Initialize Jetson**. The SSH account must already exist and have administrator
permission. Flashing JetPack/Ubuntu and enabling SSH remain image setup tasks.

Initialization supports Ubuntu 20.04/22.04/24.04 on aarch64 and x86_64, requires
systemd/sudo on the device and at least 6 GiB free after extraction, and
downloads packages/toolchains/dependencies using the Jetson's internet connection.
Architecture, OS, disk space and selected GPU/TensorRT prerequisites are checked
before package installation. NVIDIA components must already match the installed
JetPack; setup reports missing components and does not change the chosen profile.
VPI 2/3 layouts supported by this project's build are checked, including NVIDIA's
`lib/aarch64-linux-gnu` layout. These checks are not hardware certification.

The app uploads/checksums a source snapshot and native controller, then submits a
root-owned systemd job. The sudo password goes through SSH stdin without a PTY;
it is excluded from command arguments, files, logs, and saved preferences. SSH
passwords are kept only for the app session, so enter them again after reopening
before using **Refresh Releases**. Existing saved SSH credentials are read for
migration; saving preferences clears the stored password.

The initializer installs native build dependencies, the pinned Rust release and
locked Cargo dependencies, enables lingering/user-manager access, grants camera
group membership, and builds through the regular offline staging path as the
runtime account. Native library/image-smoke checks also run as that account. It
then installs or updates the stable service/controller and its restricted service
permissions. Initialization shares the deployment lock and app operation guard.

Progress and logs reconnect using the existing saved-job flow without resubmitting
setup or requesting the sudo password again. Setup has a three-hour limit and
never restarts itself. A failed or interrupted setup keeps its logs; after
inspecting its result, retry **Initialize Jetson** to perform a new attempt.
Package/toolchain changes already completed are retained. Existing live-release
files and selection are preserved. Initialization prepares a release and startup
configuration; click **Activate** afterward to start and check real cameras.

## Stage a release

Select the intended camera, networking, build profile, runtime configuration, and
field map in the interface, then click **Stage Release**. The app bundles native
Rust `vortexctl` executables for Windows x64 and static Linux ARM64/x64. Neither
desktop packaging nor remote deployment requires Python. The app reads the device
architecture, selects the matching controller and verifies its checksum before
execution. Source builds of the desktop app must first prepare the native artifacts;
see [controller builds](../deployment/README.md).

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

The Jetson needs `dbus-user-session` and a persistent user manager. **Initialize
Jetson**, bootstrap and managed-service setup enable
[user lingering](https://www.freedesktop.org/software/systemd/man/252/loginctl.html).
Use Initialize Jetson for an existing installation that has not enabled it yet.

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
/home/vortex/deployments/Vortex.deployment/incoming/RELEASE_ID/vortexctl status --root /home/vortex/deployments/Vortex.deployment --release-id RELEASE_ID
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
/home/vortex/deployments/Vortex.deployment/vortexctl recover --root /home/vortex/deployments/Vortex.deployment
```

A failed confirmation or interrupted switch never edits either release's camera
calibration/mount configuration. Editing settings for the next deployment creates
another snapshot; it does not alter an already prepared release.

## Runtime and configuration

Production runs `dumapril-taglocalization` through the Rust `vortexctl` launcher.
`run_vortex.sh` is a location-independent compatibility wrapper around that native
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
Runtime edits are saved locally; stage and activate another snapshot to apply them
to production.

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
tools/vortexctl launch --source . --validate-config
tools/vortexctl launch --source . --build
tools/vortexctl launch --source . --check
tools/vortexctl launch --source .
```

For first-time dependency provisioning, explicitly allow downloads:

```text
tools/vortexctl launch --source . --build --online --with-debug-bridge
```

Successful builds write `target/release/vortex-build.json`. Launch verifies the
binary checksum, source fingerprint, and selected profile. A legacy executable
without this fingerprint must be rebuilt. Changing cameras or networking does not
require recompilation; changing source, toolchain pin, or profile does.

## Migrating an existing Jetson

Stage Release requires the pinned Rust toolchain, native build libraries,
and cached Cargo dependencies on the Jetson. Use **Initialize Jetson** to prepare
these prerequisites and the managed service without a terminal. The standalone
bootstrap script remains available for operators who prefer it.

Existing source deployments continue running until Activate. First-time service
preparation requires root privileges; Activate attempts noninteractive sudo and
directs users to **Jetson Setup → Initialize Jetson** if permission is unavailable.
That flow also updates controllers when a new release needs privileged setup.
Existing packages from before this feature must be restaged to include the updated
controller. Existing manifests remain readable by the native controller, including
their original snapshot fingerprints. Older ready releases remain rollback targets;
a legacy source deployment uses its original startup script when restored.

The installer supports migration of the existing `run_vortex.sh` systemd unit.
It rejects unexpected launch commands and service drop-ins rather than discarding
custom behavior. Managed service/root paths must not contain spaces or systemd
specifier characters. It backs up the old unit, installs a stable native controller,
and grants only start/stop/reset-failed permission for that specific unit through
a validated sudoers entry. Installing/upgrading the controller itself still needs
root; ordinary Activate/Rollback operations run as the runtime user.

`scripts/setup_jetson_environment.sh` is an optional terminal wrapper around
`vortexctl initialize-submit`. It takes the same explicit package/job arguments as
the controller. The app prepares those arguments automatically and works without
an initial project copy on the device; initialization always defers activation.

The service name defaults to `vortex-vision.service` in bootstrap. Set
`startup_service_name` to that name in the interface preferences if the service
should be stopped/resumed around the current tools. Stop failures abort the operation.
The service-control commands use noninteractive sudo and require suitable permissions.

## Production diagnostics and watchdog

Use **Attach Diagnostics** after activating a release built with this feature.
The app observes the managed service over its existing SSH connection. It does not
launch a second pipeline, open cameras, stop the service, or change runtime config.
**Next Camera** switches the preview subscription; **Detach Diagnostics** releases
it and leaves production running. Attachments reconnect across connection loss or
service restart, including activation/rollback. Older binaries must be rebuilt and
activated; the app does not fall back to replacing production with `orin_bridge`.
Runtime form edits save locally; Stage and Activate applies the selected configuration
snapshot. Attaching diagnostics no longer enables automatic writes to the legacy
deployment's configuration.

The native detector offers its own processed grayscale pixels and corresponding
accepted detections to a bounded, nonblocking preview worker. Only the subscribed
camera is copied, at most five times per second. With no live subscription, frames
are not copied or encoded. JPEG previews fit within 640×480 at quality 65 without
changing detection resolution. One pending frame and three retained JPEGs bound
the queue and storage. A preview lease expires after three seconds without renewal;
another client waits for the owner to detach or expire.

Each immutable JPEG and its metadata share a frame ID, camera, process ID, release,
launch token and client ID. Metadata is published after the image. The controller
rejects stale/mismatched frames and the UI applies detections when that image loads.
Original image dimensions keep overlays aligned with downscaled previews. Stale
images are cleared during outages/restarts rather than appearing live. Preview
quality/rate and SSH overhead still need Jetson performance measurement.

Diagnostics shows processing FPS, detector-processing milliseconds, camera-dequeue
to processing-completion milliseconds, the remote read round trip, watchdog state,
service result and restart count. These timings do not measure sensor exposure or
an exact end-to-end camera-to-display delay. The detector-to-monitor statistics
queue is also bounded, so a slow telemetry consumer cannot build an unlimited backlog.

The managed unit accepts notifications from its main process. New runtimes opt in
to systemd's watchdog using the native main PID; older binaries and legacy rollback
targets do not opt in. Startup has a 30-second grace period with a 40-second initial
watchdog deadline. Once all cameras process frames, or the grace period ends, the
deadline becomes ten seconds. Heartbeats continue only while every configured
camera has processed a frame within three seconds, reports a working pipeline,
and has the required detector backends. Empty scenes are healthy; missing AprilTags
do not trigger restarts. A blocked main loop also stops heartbeats. Preview failures
alone do not restart vision.

Systemd restarts a failed/watchdog-stalled process after two seconds, subject to the
existing five-starts-per-minute limit. The watchdog does not automatically switch
releases; startup activation still owns verified rollback. Use **Initialize Jetson**
with this version to update the stable controller/unit, then **Activate** its prepared
release. This is necessary to install notification access and start the new native code.

## Local checks

These checks use temporary files and mocked remote operations; no Jetson is contacted:

```text
cargo test --manifest-path deployment/Cargo.toml --locked
```

From `VortexDeploymentInterface/`:

```text
npm run check:native
npm test
```

Native builds, GPU libraries, actual camera access, and systemd behavior still need
Jetson validation.

## Remaining implementation order

1. Complete the [Jetson acceptance run](deployment-validation.md), including real
   reboot/power interruption, preview performance under build load and GPU profiles.
2. Validate [offline initialization](offline-deployment.md) on each supported clean
   JetPack image. Native bundle creation and restore tests do not certify JetPack.

## Release maintenance

**Deployment maintenance** offers cleanup preview and explicit cleanup. It retains
the selected/previous releases, all referenced or nonterminal work, the newest three
releases and anything newer than 14 days. Unknown uploads/builds are conservatively
retained. Cleanup takes the deployment lock, refuses unfinished activation recovery,
rejects symlink trees and does not remove shared compiler caches.

The CLI defaults to preview; add `--apply` to delete the reported old files:

```sh
vortexctl cleanup --root /path/to/Vortex.deployment --keep-releases 3 --max-age-days 14
```

Offline initialization is available through Jetson Setup; see the
[bundle preparation guide](offline-deployment.md). Tag pushes matching `v*` build
the Windows installer and portable app with checksums and upload a **draft** GitHub
release. Workflow dispatch builds downloadable artifacts without publishing a release.
Release signing is not configured; the generated executables are unsigned.

Linux service installation, sudoers validation, real camera timing, GPU/TensorRT,
and boot recovery still require Jetson testing. Local transaction tests inject
service failures and power interruption; Windows cannot exercise the real Linux
systemd/symlink operations. The Rust smoke/health modules are compiled and tested
through the native CPU replay harness.
