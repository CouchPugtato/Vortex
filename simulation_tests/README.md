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
python simulation_tests/run.py --generate-only
python simulation_tests/run.py --replay-executable path/to/replay.exe
python simulation_tests/run.py --backend blender --blender "path/to/blender.exe"
```

Image generation needs Python, NumPy, and OpenCV-contrib. Replay needs Rust and a C compiler, or a prebuilt replay executable. Windows builds use MSYS2 MinGW at `C:\msys64\mingw64` (override with `--mingw-root`) and the Rust GNU target:

```text
rustup target add x86_64-pc-windows-gnu --toolchain stable
```

## Tests and results

- Extended suite: **256 scenarios** covering all 32 tags, distance, viewing angle, noise, blur, brightness, and blank images.
- Smoke suite: **7 scenarios**.
- Generator checks: **8 automated tests**:

```text
python -m unittest discover -s simulation_tests -p test_*.py
```

Edit `scenarios_extended.json` to change the sweeps. Outputs go into `generated/`, with per-frame results in `report.json` and group counts in `summary.json`. Generation-only runs produce images without testing Vortex.

The default position-error limit is **0.20 m** (`--max-error-meters`). The corrected extended run passed **255/256** scenarios. All final camera position estimates meet the limit; the remaining case reports a false tag ID 46 under combined blur and noise. All eight generator checks and three Rust pose regression tests pass.

## Scope

Tests use ideal pinhole cameras, official tag orientation, the production CPU detector, and the production multi-tag pose solver. Replay scores the joint pose when available, with single-tag fallback; individual-tag errors remain in the report. Live capture, robot mounting transforms, telemetry, YOLO, and GPU performance are outside this harness. Blender rendering remains unverified on this machine.
