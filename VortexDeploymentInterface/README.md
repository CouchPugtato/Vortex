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
- Deploy project folder to Jetson over SSH/SFTP.
- Start/stop monitor command and parse AprilTag/Object output by camera.
- Live camera preview by running remote capture command and fetching image frames.
- Monitor command presets for main pipeline launch patterns.

## Notes

- Connection errors like `ECONNREFUSED` / `10061` mean host/port are unreachable.
- Preview command template supports:
  - `{cam}` for camera index
  - `{path}` for remote image path
