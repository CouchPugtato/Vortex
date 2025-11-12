# Vortex (AprilTag Localization)

This project detects AprilTags in images, writes overlay images, and generates a `results.json` summary. It is now decoupled from Visual Studio and vcpkg; builds and runs via Docker.

**Key Features**
- Generates `src/output/overlays` and `src/output/results.json`.
- Optional 3D translation `[tx, ty, tz]` in meters using camera intrinsics and tag size.
- Darkens input images via gamma correction for clearer overlays.

**Docker Build**
- Windows PowerShell:
  - `docker build -t dumapril-taglocalization .`
- Linux/macOS:
  - `docker build -t dumapril-taglocalization .`

**Docker Run**
- Default container args expect `/data/input` and `/data/output` (preconfigured via `CMD`). Mount your local folders and set envs if you want 3D pose:

- Windows PowerShell:
  - `docker run --rm \`
    `-v ${PWD}/src/input:/data/input \`
    `-v ${PWD}/src/output:/data/output \`
    `-e APRILTAG_CAM_FX=1000 -e APRILTAG_CAM_FY=1000 \`
    `-e APRILTAG_CAM_CX=640 -e APRILTAG_CAM_CY=360 \`
    `-e APRILTAG_TAG_SIZE_M=0.16 \`
    `-e APRILTAG_DARKEN_GAMMA=0.6 \`
    `dumapril-taglocalization`

- Linux/macOS:
  - `docker run --rm \`
    `-v $(pwd)/src/input:/data/input \`
    `-v $(pwd)/src/output:/data/output \`
    `-e APRILTAG_CAM_FX=1000 -e APRILTAG_CAM_FY=1000 \`
    `-e APRILTAG_CAM_CX=640 -e APRILTAG_CAM_CY=360 \`
    `-e APRILTAG_TAG_SIZE_M=0.16 \`
    `-e APRILTAG_DARKEN_GAMMA=0.6 \`
    `dumapril-taglocalization`

**Environment Variables**
- `APRILTAG_CAM_FX`, `APRILTAG_CAM_FY`, `APRILTAG_CAM_CX`, `APRILTAG_CAM_CY`: camera intrinsics in pixels.
- `APRILTAG_TAG_SIZE_M`: tag size in meters (outer black square size).
- `APRILTAG_DARKEN_GAMMA`: gamma for darkening input (e.g., `0.6`–`0.8`).
- `APRILTAG_STATIC=1` is set in the container to produce a portable binary.

**Outputs**
- `src/output/overlays/*.png`: input images with tag outlines and distance classification.
- `src/output/results.json`: per-image entries with pixel and normalized offsets and optional `translation_m`.

**Notes**
- On Windows, ensure your drive is shared in Docker Desktop for volume mounts.
- No vcpkg or Visual Studio is required; the container handles compilation and runtime.