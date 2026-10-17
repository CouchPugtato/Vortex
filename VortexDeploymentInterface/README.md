# Vortex Deployment Interface (Electron)

## Run

```powershell
cd VortexDeploymentInterface
npm install
npm start
```

## Build Windows Installer/Exe

```powershell
cd VortexDeploymentInterface
npm install
npm run dist:win
```

Outputs are written to:

- `VortexDeploymentInterface/dist/*.exe`

Available packaging scripts:

- `npm run pack` (unpacked app directory)
- `npm run dist` (default targets from `package.json`)
- `npm run dist:win` (Windows targets only)

## Features

- Edit and save runtime `config/config.json`.
- Initialize a flashed Jetson from **Jetson Setup**, including dependencies, Rust,
  camera permissions and managed startup, without opening an SSH terminal.
- Stage a checksummed source release over SSH/SFTP and build it without replacing
  or restarting the running installation.
- Attach to production camera previews and detections without replacing the pipeline.
- Inspect FPS, processing/transfer timing, watchdog state and restart counts.

## Notes

- See [deployment architecture and migration](../docs/deployment.md) for the shared
  Python launcher, device settings, build profiles, and staged deployment upgrade.
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
  account, Python 3, systemd, internet and free disk space. It prepares a release;
  **Activate** starts it afterward. GPU/TensorRT need matching NVIDIA components.
- Python is needed on the desktop; the Jetson needs Python, the pinned Rust
  toolchain, native libraries, and cached dependencies. Builds are offline.
- **Attach Diagnostics** observes the active service. **Next Camera** changes the
  subscription; **Detach Diagnostics** leaves vision running. Previews are generated
  on demand, capped at 5 FPS and 640×480. Empty scenes do not trigger the processing watchdog.
- Run **Initialize Jetson** with this version, then **Activate** its prepared release
  to enable production diagnostics and watchdog notifications.

- Connection errors like `ECONNREFUSED` / `10061` mean host/port are unreachable.
