# Vortex synthetic tests

Test CPU AprilTag detection and multi-tag camera localization using generated images—no robot or logs required.

## Run

From the repository root:

```text
python -m pip install -r simulation_tests/requirements.txt
python simulation_tests/run.py
```

Useful options:

```text
python simulation_tests/run.py --suite smoke
python simulation_tests/run.py --suite stress
python simulation_tests/run.py --generate-only
python simulation_tests/run.py --replay-executable path/to/replay.exe
python simulation_tests/run.py --backend blender --blender "path/to/blender.exe"
```

Image generation needs Python, NumPy, and OpenCV-contrib. Replay uses an optimized release build and needs Rust and a C compiler, or a prebuilt replay executable. Windows builds use MSYS2 MinGW at `C:\msys64\mingw64` (override with `--mingw-root`) and the Rust GNU target:

```text
rustup target add x86_64-pc-windows-gnu --toolchain stable
```

## Tests and results

- Extended suite: **256 scenarios** covering all 32 tags, distance, viewing angle, noise, blur, brightness, and blank images.
- Smoke suite: **7 scenarios**.
- Stress suite: **1,024 scenarios**—576 single-tag viewpoints, 256 single-tag image-quality cases, 96 repeated noisy field views, and 96 noisy blank controls.
- Generator checks: **11 automated tests**:

```text
python -m unittest discover -s simulation_tests -p test_*.py
```

Edit `scenarios_extended.json` or `scenarios_stress.json` to change the sweeps. Outputs go into `generated/`, with per-frame results in `report.json`, group counts in `summary.json`, and reproducible failures in `failures.json`. Generation-only runs produce images without testing Vortex.

Stress cases have explicit noise seeds. Reproduce one with `python simulation_tests/run.py --suite stress --case single_quality_0000`. Single-tag scenes render exactly one tag and reject detections of any other ID, including IDs present elsewhere in the field map. The position-error limit remains 20 cm.

The position-error limit remains **0.20 m** (`--max-error-meters`). Latest results: **256/256 extended** and **1,018/1,024 stress**. All false detections are eliminated and no required tags are missed. Six isolated-tag pose errors remain at 3 m. All 11 generator checks and nine Rust regression tests pass.

The CPU pipeline now checks decoding contrast, refines borders, and aligns small isolated tags against their pixel patterns. Reports include detection quality, pose candidates, and localization time. Image alignment is skipped for multi-tag views. Jetson performance remains unverified.

## Scope

Tests use ideal pinhole cameras, official tag orientation, the production CPU detector, and the production multi-tag pose solver. Replay scores the joint pose when available, with single-tag fallback; individual-tag errors remain in the report. Live capture, robot mounting transforms, telemetry, YOLO, and GPU performance are outside this harness. Blender rendering remains unverified on this machine.
