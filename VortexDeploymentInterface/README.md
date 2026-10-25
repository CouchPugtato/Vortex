# Vortex Deployment Interface (Electron)

## Run

```powershell
cd VortexDeploymentInterface
npm install
npm run build:native
npm start
```

## Build Windows Installer/Exe

```powershell
cd VortexDeploymentInterface
npm install
npm run build:native
npm run dist:win
```

Outputs are written to:

- `VortexDeploymentInterface/dist/*.exe`

Available packaging scripts:

- `npm run pack` (unpacked app directory)
- `npm run dist` (default targets from `package.json`)
- `npm run dist:win` (Windows targets only)

`pack` creates an unsigned local app and skips executable editing, allowing the
packaging check to run without Windows signing-tool symlink privileges. Installer
builds retain the normal Electron Builder signing settings.

## Features

- Edit and save runtime `config/config.json`.
- Initialize a flashed Jetson from **Jetson Setup**, including dependencies, Rust,
  camera permissions and managed startup, without opening an SSH terminal.
- Initialize without internet using a matching [offline bundle](../docs/offline-deployment.md).
- Preview or apply old-release cleanup, and run camera/preview acceptance checks
  through **Deployment maintenance**.
- Stage a checksummed source release over SSH/SFTP and build it without replacing
  or restarting the running installation.
- Attach to production camera previews and detections without replacing the pipeline.
- Inspect FPS, processing/transfer timing, watchdog state and restart counts.

## Notes

- See [deployment architecture and migration](../docs/deployment.md) for the shared
  native Rust controller, device settings, build profiles, and staged deployment upgrade.
- **Stage Release** packages required files, verifies checksums, acquires the
  device deployment lock, and creates an isolated verified build.
- Deployment jobs run under systemd and survive closing the app. Reopening restores
  progress, logs, and results; **Refresh Releases** retries reconnection after an outage.
  Use **Initialize Jetson** first to prepare persistent jobs on existing devices.
- **Activate** checks the executable, libraries, saved-image smoke test, and fresh
  camera processing. A failed start restores the previous release/configuration.
  **Rollback** switches back manually. First-time service setup needs administrator
  credentials entered in **Jetson Setup**; passwords are not saved.
- Initialization needs an existing Ubuntu/JetPack image, SSH-enabled administrator
  account, systemd, internet and free disk space. It prepares a release;
  **Activate** starts it afterward. GPU/TensorRT need matching NVIDIA components.
- Packaged apps bundle Windows x64 and static Linux ARM64/x64 `vortexctl`
  executables. Deployment needs no Python. Source builds need Rust, Zig and MinGW;
  see [native build instructions](../deployment/README.md), including CI artifacts.
  The Jetson vision build needs the pinned Rust toolchain, native libraries and
  cached dependencies. Ordinary staging builds offline.
- **Attach Diagnostics** observes the active service. **Next Camera** changes the
  subscription; **Detach Diagnostics** leaves vision running. Previews are generated
  on demand, capped at 5 FPS and 640×480. Empty scenes do not trigger the processing watchdog.
- Run **Initialize Jetson** with this version, then **Activate** its prepared release
  to enable production diagnostics and watchdog notifications.

- Connection errors like `ECONNREFUSED` / `10061` mean host/port are unreachable.

Tag pushes matching `v*` build the installer and portable executable with SHA256
checksums and upload a draft GitHub release. Manual workflow dispatch produces
artifacts without publishing. Installers currently ship unsigned. Native files
and installers are generated artifacts rather than checked-in binaries.

Installer CI silently installs the NSIS executable, verifies all bundled
controllers and renderer files, checks the running page and preload API, then
uninstalls it. The artifact includes `installer-smoke.json`. For an existing
unpacked app, run `node scripts/smoke-installed.js --probe-only dist/win-unpacked`
to check packaged resources without installing or launching the UI.
