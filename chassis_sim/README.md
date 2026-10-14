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
preview matches the detector projection exactly, including `fx`, `fy`, `cx`, and `cy`.
With AdvantageScope closed, `python chassis_sim/setup_advantagescope.py` adds a
**Vortex Camera** tab and the camera override automatically, backing up its layout.

Both topics use WPILib Pose3d structs (meters, field X/Y, Z up). Error and timing
topics are under `/VortexSim`. Errors compare against the chassis at camera capture
time; the live robot may move ahead of the delayed ghost. `PoseAgeMs` shows its age.
An empty ghost array and `Valid=false` mean localization failed.

Click the camera preview: **W/S** forward/back, **A/D** or **Left/Right arrows** strafe, **Q/E** turn,
**Space** stop. Losing focus stops movement. Reset returns to the starting pose.
The default spawn faces tag 18 at 1.5 m, clear of the trench structure.
Use `--start-tag ID` and `--start-distance METERS` to change it.

Edit **Actual mount** to change the rendered camera and **Vortex assumed mount**
to change conversion of Vortex's camera pose into the ghost chassis pose.
The default `demo_mounts.json` provides a matching upright forward-facing camera.
Use `--use-config-mount` to start with the production camera profile's mount instead.
Offsets are robot-relative meters. Angles follow Vortex's current convention:
`Ry(yaw) Rx(pitch) Rz(roll)`, acting on camera optical axes (right/down/forward).
They are not conventional chassis yaw/pitch/roll. An upright forward camera uses
`yaw_deg=90`, `pitch_deg=0`, `roll_deg=-90` in that convention.
Start with matching mounts, then introduce an offset or angle error in one column.
**Save mounts** writes a session file; reload with `--mounts PATH`.
The production config is never overwritten.

The camera defaults to 960×540, scaling intrinsics from 1920×1080. Override
`--calibration-width`/`--calibration-height` if the calibration uses another size.
`--camera-hz 20` controls sampling (default 20 FPS); actual throughput depends on
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
with simple lighting and an ideal pinhole lens. It doesn't model moving opponents,
lens distortion, or shadows. The chassis can pass through geometry, putting the
camera inside an obstacle. Frames pass through production CPU detection/pose code;
ground truth only scores the output. Vortex's GPU detector and live camera capture
are not exercised.
Saved mounts and native error logs live in ignored `generated/`. Interactive frames
stay in memory; use the synthetic test runner for image datasets and replay reports.
