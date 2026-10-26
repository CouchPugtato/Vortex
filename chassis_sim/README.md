# Interactive chassis simulator

```text
python -m pip install -r chassis_sim/requirements.txt
python chassis_sim/drive.py
```

To install and launch AdvantageScope on Windows:

```text
winget install --exact --id MechanicalAdvantage.AdvantageScope --source winget
python chassis_sim/drive.py --launch-advantagescope
```

Use `--advantagescope PATH` for an executable installed elsewhere.

Requires Python with Tkinter and Rust. Windows native replay builds use MSYS2
MinGW64 at `C:/msys64/mingw64` (override `--mingw-root`). Alternatively pass
`--replay-executable PATH` to a replay built from this checkout.

In AdvantageScope, choose **File → Connect to Simulator → NetworkTables 4**.
Add a **3D Field**, select **2026 Field**, then drag:

- `/VortexSim/Actual` into Poses as **Robot**.
- `/VortexSim/Ghost` into Poses as **Ghost**.

The default **field** renderer loads the same 2026 GLB used by AdvantageScope.
Opaque geometry depth-tests against AprilTags, so field structures block detection.
It uses a separate persistent OpenGL process; it doesn't capture AdvantageScope's
window or modify that app. Install the Python requirements before running it.
If the model isn't installed, run `python chassis_sim/download_field.py`.
Use `--field-asset PATH` for a compatible wall-blue Field3d asset folder.
`--renderer fast` selects the old tag-only renderer without field occlusion.

For a camera view inside AdvantageScope, add `/VortexSim/Camera` as **Camera Override**
in the 3D Field, then right-click the field and select a fixed robot camera.
The published pose uses the **actual** mount and field-relative forward/left/up axes.
AdvantageScope keeps the selected robot camera's FOV and aspect ratio. The Python
preview matches the detector projection, including intrinsics and lens distortion.
With AdvantageScope closed, `python chassis_sim/setup_advantagescope.py` adds a
**Vortex Camera** tab and the camera override automatically, backing up its layout.

Both topics use WPILib Pose3d structs (meters, field X/Y, Z up). Error and timing
topics are under `/VortexSim`. Errors compare against the chassis at camera capture
time. `/VortexSim/RawVision` preserves the delayed, unmodified vision pose.
The displayed `/VortexSim/Ghost` advances that estimate using timestamped drive
commands at about 50 Hz, without using actual chassis positions. `PoseAgeMs` shows
raw frame age; `PredictionHorizonMs` shows the compensated time. Prediction stops
on missing poses, reset, or frames older than 250 ms (`PredictedValid=false`).
Errors and failure captures always use raw vision; prediction cannot hide failures.

Click the camera preview: **W/S** forward/back, **A/D** strafe, **Q/E** or **Left/Right arrows** turn,
**Space** stop. Losing focus stops movement. Reset returns to the starting pose.
The default spawn faces tag 18 at 1.5 m, clear of the trench structure.
Use `--start-tag ID` and `--start-distance METERS` to change it.

Edit **Camera mount** to change the rendered camera. **Calibration error** adds
offsets to that mount for Vortex's chassis estimate: meters for translation,
degrees for rotation. Zero means Vortex uses the correct mount. Errors always
start at zero on launch and are never saved to configuration files.
The camera mount defaults to the selected production camera profile (`--camera-index`).
Use `--demo-mount` for the upright demo camera, or `--mounts PATH` for saved mounts.
Offsets are robot-relative meters. Angles follow Vortex's current convention:
`Ry(yaw) Rx(pitch) Rz(roll)`, acting on camera optical axes (right/down/forward).
They are not conventional chassis yaw/pitch/roll. An upright forward camera uses
`yaw_deg=90`, `pitch_deg=0`, `roll_deg=-90` in that convention.
Introduce errors to test incorrect calibration; **Reset calibration error** clears
all offsets. `--mounts PATH` loads only the camera mount, ignoring any saved
assumed mount. Capture diagnostics retain the effective mounts for replay.
The production config is never overwritten.

Mount and error fields update live. **Live camera calibration** also lets you edit focal
lengths, principal point, distortion, tag size and output width/height. Changes
apply after 300 ms without typing; camera setup runs in the background and stale
poses are cleared. Incomplete or invalid fields pause sampling until corrected.
The selected camera profile in the config file is watched too: saving it updates
the matching fields without restarting or moving the chassis. Explicit mount
file overrides remain in effect. GUI edits stay local to the simulation.

The camera defaults to the full 1920×1080 calibration size, with the selected
profile's `fx`, `fy`, `cx`, `cy`, `k1`, `k2`, `k3`, `p1`, and `p2`. The current
config has no image-size fields; override `--calibration-width`/`--calibration-height`
if it was calibrated at another size. `--width` and optional `--height` resize
output and scale intrinsics per axis. The field and tags render with overscan,
then lens distortion is applied before sensor effects, detection and capture.
`--ideal-camera --demo-mount --width 960` restores the smaller idealized demo.
Full resolution increases rendering and detection cost.
`--camera-hz 40` controls sampling (default 40 FPS); actual throughput depends on
CPU localization. Chassis/Actual publishes continuously at about 50 Hz, including
while stationary. A second simulator using the same NT4 port is rejected.
The camera uses a persistent Vortex process and raw grayscale frames. `LatencyMs`
includes rendering and localization; `RenderMs`, `VortexMs`, `CameraFPS`, and
`LoopFPS` separate the costs. The fast renderer's floor grid uses lower resolution; tags
and detection retain the requested camera resolution. Some isolated small tags
still require expensive production image alignment, which can slow individual frames.

```text
python chassis_sim/drive.py --headless-frames 3
python -m unittest discover -s chassis_sim -p "test_*.py"
```

This is a kinematic chassis with field-boundary limits, not drivetrain physics or
collision response. The camera renders static CAD geometry and exact tag textures,
with simple lighting, calibrated lens distortion, and optional sensor effects.
It doesn't model moving opponents or shadows. The chassis can pass through geometry, putting the
camera inside an obstacle. Frames pass through production CPU detection/pose code;
ground truth only scores the output. Vortex's GPU detector and live camera capture
are not exercised.
Saved mounts and native error logs live in ignored `generated/`. Interactive frames
stay in memory except for automatically captured failures.

## Failure capture

**Capture frame**, beside the capture count, manually saves the displayed camera
frame even when automatic checks find no failure. It works during routines too.
Manual saves skip confirmation, cooldown and duplicate filtering, but respect
session/disk limits. Saves include pixels, poses, camera settings and diagnostics,
are marked `trigger: manual`, and announce completion in the banner. Frames with
no automatic failure stay in the analysis report's `manual_review` group; review
visibility and expectations before promoting them. The button is disabled until
a camera frame is available, during a request, or with `--no-capture`.

Enabled by default. Cases go to `chassis_sim/generated/captures/SESSION/regressions/`;
mount mismatches go to `experiments/`. Triggers are >20 cm position error, >5° heading
error, invalid output, invented IDs, or missing pose with clearly visible tags.
Captures stay active during **Run route** and benchmark routines, using the same
limits and deduplication. Saved diagnostics include the route name and elapsed time.
Visibility uses renderer depth, independent of Vortex. Ground truth is from image
capture time. Occluded or too-small tags alone do not trigger a missing-pose case.

Two bad frames confirm a failure; two good frames end it.
A two-second banner announces completed saves and worse-example updates, with
the failure reason and case path. **Open last capture** opens that folder even after
the banner disappears. A persistent counter shows total saved cases in the capture
directory (including experiments), plus this session's count. Replacements don't
increase the count; queued or failed writes don't announce saves.

A stationary episode saves
one case, with at most two substantially worse replacements. New cases require a
5-second cooldown and 0.5 m movement, 10° rotation, or changed mounts. Similar images
are deduplicated across sessions. Default limits: 50 cases/session, 1 GiB total, and
two pending background writes. Disk errors appear in the GUI; driving continues.

Each case contains the exact grayscale input PNG, color preview, replay manifest,
ground truth, both mounts, detection diagnostics, visibility, thresholds, field/config
snapshots, asset hash, and code hashes. Optional before/recovery frames are bounded.
Cases start unreviewed and are not automatically added to the permanent suite.

```text
python chassis_sim/replay_captures.py chassis_sim/generated/captures
python chassis_sim/analyze_captures.py chassis_sim/generated/captures
python chassis_sim/replay_captures.py PATH_TO_ONE_CASE
python chassis_sim/promote_capture.py PATH_TO_REVIEWED_CASE simulation_tests/captured_cases/CASE_NAME
python chassis_sim/replay_captures.py simulation_tests/captured_cases
```

`analyze_captures.py` writes `analysis/analysis.md` and `analysis.json` with groups
for missed tags, suspected ambiguity, rotation errors and other failures, plus
original capture latency indicators. It analyzes historical results; replay to
check current code. Replay also writes a companion `.analysis.md` with current
failure groups. Calibration experiments are counted separately. Similar-cost,
physically different pose candidates suggest ambiguity; timing alone doesn't
establish the cause of a bad pose.

Replay uses current Vortex on saved pixels and checks chassis position and heading;
no regeneration is needed. Experiments are skipped unless `--include-experiments`
is supplied and cannot be promoted as detector regressions. Use `--no-capture` to
disable saving. Limits and thresholds can be changed with `--capture-position-m`,
`--capture-heading-deg`, `--capture-cooldown`, `--capture-max-cases`,
`--capture-max-mb`, and `--capture-dir`. `CapturesSaved` and `CaptureDropped` are
published to AdvantageScope. Use the synthetic test runner for generated datasets.

Driving defaults to unconstrained 6-DoF localization. `--planar-pose` selects a
calibrated level-chassis control. Benchmarks retain their flat/planar default;
`--free-pose` also enables unconstrained localization in benchmarks and replay. The
level-chassis solver uses the assumed mount and observed pixels; ground truth
only scores results. Hardware keeps the unconstrained default. See the
[localization model](../docs/simulator-testing.md#localization-model) for limits.

## Routes, camera effects, and comparisons

The window now includes camera profiles, mount presets, and **Run route / Stop**.
Space stops a route. For repeatable reports and automatic capture replay:

```text
python chassis_sim/benchmark.py --routes approach --profiles clean --mount-presets matched
python chassis_sim/benchmark.py
```

Select **blue-alliance-survey** or **red-alliance-survey**, keep **Terrain: field**,
and click **Run route** for a roughly five-minute survey. Each makes two slow
circles in each half of its alliance zone while the chassis rotates at 7.5°/s,
crosses a bump into the neutral zone, then turns once more. The red route mirrors
the blue route. Starts reset to the appropriate alliance zone; **Stop** or Space
ends the routine. Capture is enabled unless launched with `--no-capture`.

For a repeatable full-field survey report (free 6-DoF, including bumps):

```text
python chassis_sim/benchmark.py --routes blue-alliance-survey red-alliance-survey --terrain field --free-pose --profiles clean --mount-presets matched --sample-hz 2 --skip-capture-replay
```

These long routes are opt-in and aren't included in the default benchmark matrix.

The full matrix runs 100 combinations (7,600 frames by default). Reports go to
`chassis_sim/generated/benchmark_TIMESTAMP/` as Markdown, JSON, and CSV.
See [simulator testing](../docs/simulator-testing.md) for profiles, extrinsic
experiments, custom routes, baseline comparisons, and scoring limits.

The simulator defaults to **field** terrain: both blue and both red bumps in
AdvantageScope's 2026 field are driveable at their actual locations and heights.
Choose **blue-lower-bump**, **blue-upper-bump**, **red-lower-bump**, or
**red-upper-bump**, then click **Run route**. Each route crosses 2.2 m and returns
in 10 seconds. Chassis contact blends entrances and crests over an 18 cm radius
to avoid sudden roll/pitch changes (`--bump-smoothing` adjusts it). Straight faces
retain their 15° slope; the existing CAD supplies the
blue/red visuals. Free 6-DoF localization is the interactive default. The
**Localization** selector switches between level-chassis and free 6-DoF estimates.
Actual and Ghost both publish full 3D poses, including height, roll and pitch.
Euler estimates are under `/VortexSim/Rotation/{Actual,RawVision,Ghost}/` as
`RollDeg`, `PitchDeg`, `YawDeg`, and `Valid`. These use WPILib Euler angles:
nose-up pitch is negative. Invalid estimates have `Valid=false` and NaN angles.
Compare planar and free-pose models with the
[ramp benchmark commands](../docs/simulator-testing.md#pitch-roll-and-ramps).
