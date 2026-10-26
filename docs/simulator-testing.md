# Repeatable simulator testing

```text
python chassis_sim/benchmark.py
```

Runs **100 combinations**: four routes × five camera profiles × five mount presets,
or **7,600 frames** at the default 10 samples/second. Uses the production CPU
detector and full-field renderer, including CAD occlusion. Dependencies and field
assets are the same as interactive driving. `--renderer fast` selects tag-only tests.

Camera output uses the selected production profile, its mount, and lens distortion
at the full calibration resolution (1920×1080 by default). `--width 960` scales
the intrinsics and output for a lighter run. To reproduce older idealized demo
baselines, add `--demo-mount --ideal-camera --width 960`. Camera changes make old
and new benchmark inputs incomparable; report signatures include output dimensions.

Interactive camera fields apply live after a 300 ms typing debounce. The selected
config profile is also reloaded after a stable file save. Camera projection,
renderer, and native replay swap together on the camera worker; failed setup
leaves the previous resources intact. Chassis state is preserved, and results
from before the edit are discarded. Saved failures snapshot each frame's effective
calibration and dimensions, including changes made during the session.
The GUI **Calibration error** fields are additive offsets from the camera mount
(meters/degrees), always zero on launch. They are session-only: there is no save
action, and `--mounts` ignores any saved assumed mount or calibration errors.
Changing the camera mount or reloading config preserves the current error offsets.

Start with a short run:

```text
python chassis_sim/benchmark.py --routes approach --profiles clean --mount-presets matched --sample-hz 2
```

Results go to `chassis_sim/generated/benchmark_TIMESTAMP/`:

- `report.md`: readable results and baseline changes.
- `report.json`: settings, per-frame errors, summaries, and capture replay results.
- `frames.csv`: samples for plotting errors and latency.
- `capture-replay-N.json`: results for each capture directory.

The console prints the exact directory. `--output NEW_DIRECTORY` chooses another;
existing output directories are rejected to preserve reports.

## Interactive controls

The Python window has camera-profile and mount-preset selectors, plus **Run route**
and **Stop**. Space stops a route; Reset also restores the spawn. Interactive
routes support inspection in AdvantageScope and automatic failure capture.
Use the benchmark command for route reports.

| Camera profile | Conditions |
|---|---|
| `clean` | Existing ideal camera; exact pixels preserved |
| `dim` | 45% exposure, periodic brightness changes, light noise |
| `noisy` | Stronger noise and periodic exposure changes |
| `motion` | Approximate horizontal motion blur; 20 ms exposure |
| `poor-radio` | 15% dropped frames, 100 ms added delay, ±40 ms jitter |

Interactive overrides:

```text
python chassis_sim/drive.py --camera-profile motion --exposure-ms 30 --noise-std 8 --exposure .7 --exposure-swing .2 --drop-rate .1 --latency-ms 80 --jitter-ms 20 --seed 509
```

Noise uses 8-bit pixel levels; delay and exposure time use milliseconds. Drop rate
and exposure swing are fractions. Changing the UI profile replaces overrides with
that preset. Dropped frames skip detection and keep the last ghost only until its
existing 250 ms age limit. Deliberate drops do not create failure captures.
Captures contain the exact degraded input pixels and sensor settings.

| Mount preset | Actual mount | Assumed mount |
|---|---|---|
| `matched` | Loaded mount | Same |
| `raised-tilted` | 20 cm higher; +10° Vortex mount yaw | Same |
| `rear-facing` | +180° Vortex mount pitch | Same |
| `translation-error` | Loaded mount | +15 cm forward offset |
| `rotation-error` | Loaded mount | +8° Vortex mount pitch |

Angles follow the Vortex optical-axis convention in the simulator README. Each
benchmark mount starts with its actual camera facing the spawn tag. Interactive
preset selection changes mounts without moving the chassis. Deliberate calibration
mismatches are labeled `experiment`; their errors do not gate detector acceptance.

## Routes and scoring

`approach` moves away from and back toward a tag. `slalom` combines translation
and rotation. `turn` tests turning in place. `obstacle-sweep` moves beside and
behind field structures. Fixed 20 ms physics steps and fixed camera timestamps
keep trajectories independent of processing speed.

Select subsets with `--routes`, `--profiles`, and `--mount-presets`.
`--route-file PATH` replaces built-in routes with custom JSON:

```json
{
  "name": "my-route",
  "segments": [[1, 0, 0, 0], [2, 0.5, 0, 0], [2, 0, 0, 0.4]]
}
```

Segments are `[seconds, forward m/s, left m/s, turn rad/s]`, relative to the robot.
`--publish` streams scripted samples to AdvantageScope; execution is accelerated
rather than synchronized to wall time.

Raw errors compare against truth **at capture time**. Delivery metrics compare
the delayed raw pose and command-compensated ghost against truth **at delivery
time**. Only drive commands and the vision estimate enter compensation. Reports
include mean, p95, and worst errors and timing. Delivery time uses measured CPU
processing plus injected delay; the benchmark does not model multiple queued
frames or transport reordering. Delivery metrics also score predictions retained
across drops, subject to the same 250 ms age limit as interactive driving.

Acceptance uses existing capture limits (20 cm, 5° by default), invalid poses,
unexpected tags, pipeline errors, and missing poses with clearly visible tags.
No-pose frames without clearly visible tags are counted without implying a
detector failure. No delivered visible-tag coverage is `inconclusive`, never a
pass. Exit code 1 means a matched-mount run failed, was inconclusive, or saved-case
replay failed. `--max-bad-frames N` permits an explicit outlier allowance.
This stress suite can expose existing localization failures.

### Localization model

Interactive driving defaults to unconstrained **free 6-DoF** localization.
`--planar-pose` selects the calibrated **level-chassis** solver; benchmarks retain
this model on flat ground by default.
The level-chassis solver fits robot X/Y/yaw from tag corners and the assumed camera mount, then jointly
aligns decoded tag patterns against the input pixels when possible. Camera height
and attitude come from configuration; ground truth is used only for scoring.
Multiple starting headings avoid relying on one ambiguous square-PnP seed.
Fits must satisfy front-face visibility and the existing 2-pixel reprojection
limit. An unsuccessful image fit falls back to the corner fit.

Use `--free-pose` with `drive.py`, `benchmark.py`, or `replay_captures.py` for
unconstrained 6-DoF localization. Saved simulator captures retain the assumed
mount and model selection; older simulator captures default to planar. Arbitrary
synthetic camera scenes retain their unconstrained model.

Hardware retains its unconstrained default. A robot that stays level and has
calibrated camera extrinsics can opt in with `processing.planar_localization: true`
in runtime config. Leave it false for ramps, tipping, or uncertain mounts. The
shared solver and production CPU build are tested; robot hardware validation
remains outstanding.

Sparse frames with small tags also use an interpolated detection pass, with
stricter decoding checks on recovered tags. This adds processing cost; inspect
timing metrics on the target device.

The chassis has no collision response. Routes can cross CAD geometry; obstacle
sweeps test visibility rather than safe navigation. Blur approximates optical flow
at 2 m depth rather than integrating rendered scenes throughout an exposure.
Noise and exposure are synthetic approximations. GPU detection and real camera
drivers are not exercised.

## Pitch, roll, and ramps

The interactive simulator defaults to **field** terrain and **Free 6-DoF**
localization. All four existing CAD bumps are driveable: two blue and two red.
Their contact surfaces use the installed AdvantageScope V2 field's transformed
bump-panel bounds: approximately 1.128 m deep, 1.856 m wide and 0.1654 m high,
with 15° faces. An 18 cm contact smoothing radius blends entrances, crests and side
edges so chassis height and attitude change continuously. Straight face sections
retain their 15° slope; the smoothed chassis path rounds about 16 mm below the CAD
crest and starts rising slightly before the panel lip. This approximates a finite
contact area, without modeling wheelbase, suspension, traction or collisions.
`--bump-smoothing .24` softens the transitions further; `--bump-smoothing 0` uses
the original unsmoothed contact planes, which match CAD faces within 2 mm.
The original CAD supplies the colors and camera occlusion; no orange overlay is
added. The `fast` renderer uses the same terrain contact but renders tags only.

Choose **blue-lower-bump**, **blue-upper-bump**, **red-lower-bump**, or
**red-upper-bump**, then click **Run route**. Lower/upper refer to field Y from the
blue-wall origin. Each route places the chassis on its alliance side, faces the hub,
strafes 2.2 m across the named bump at 0.55 m/s, pauses, and returns in 10 seconds.
Terrain stays fixed when resetting the chassis or switching routes.

```text
python chassis_sim/drive.py --launch-advantagescope
python chassis_sim/benchmark.py --routes blue-lower-bump blue-upper-bump red-lower-bump red-upper-bump --terrains flat field --models planar free --profiles clean --mount-presets matched --sample-hz 5 --no-capture --skip-capture-replay
```

The matrix has 16 combinations. Default benchmarks retain their existing flat,
planar baseline and original 100 combinations. Reports include height, tilt,
predicted tilt at delivery, visible-tag coverage, and inclination bands.

**flat**, **pitch**, and **roll** provide comparison controls. **custom-ramp** and
**custom-cross-ramp** retain the separate orange test ramp anchored behind or left
of the spawn; their routes are **ramp-crossing** and **cross-ramp-crossing**.
`--ramp-height`, `--ramp-length` (horizontal run of each face), `--ramp-width`, and
`--ramp-crest` affect only these custom ramps. Old `--terrain ramp` and
`--terrain cross-ramp` commands now select the real **field** terrain.
Positive chassis pitch raises the nose; positive roll raises the left side.
Camera translation and rotation use the complete chassis transform. Static
controls such as `--terrain flat --chassis-pitch 12 --chassis-roll 5` deliberately
impose attitude without a supporting surface.

New six-component captures score 3D position and surface-normal tilt as well as
heading. `--capture-tilt-deg` sets the tilt limit (default 5°). Legacy three-component
captures retain their existing pass criteria. Chassis arrays use
`[x, y, yaw, z, roll, pitch]`, with meters and radians. Capture replay preserves
the recorded model selection and scores the saved full chassis pose.

Use `--planar-pose` or the GUI level-chassis model to test planar limitations.
`--models planar free` compares it with unconstrained localization; terrain truth never chooses or corrects the
detector's estimate. The display predictor likewise has no terrain-height oracle,
so delivery errors can increase as the chassis changes slope.
It translates drive commands through the estimated 3D orientation and keeps the
last estimated surface normal fixed while turning. This compensates delay on a
steady slope; it cannot anticipate a crest or a change in slope.

The GUI **Localization** selector changes models without moving the chassis and
clears pending display estimates. Field mode selects **blue-lower-bump** initially;
choose any of the four named routes to traverse another bump. Static **pitch** and **roll** terrains are
useful for checking rotation without driving. Actual and Ghost publish Pose3d
rotation, so use 3D robot and ghost sources in AdvantageScope. A Pose2d source
cannot show roll or pitch.

Under `/VortexSim/Rotation/`, the `Actual`, `RawVision`, and `Ghost` subtables
contain `RollDeg`, `PitchDeg`, `YawDeg`, and `Valid`.
Values are WPILib Euler angles in degrees: `Rz(yaw) Ry(pitch) Rx(roll)`.
Nose-up pitch is negative here; the simulator's chassis controls and top-level
`ChassisPitchDeg` use nose-up positive. RawVision is the unmodified delayed
estimate, Ghost is the display prediction. Missing, invalid or cleared estimates
publish NaN angles and `Valid=false`; the actual chassis keeps its own attitude.

## Track fixes and regressions

Every benchmark automatically replays reviewed cases in
`simulation_tests/captured_cases/` and local captures in
`chassis_sim/generated/captures/`, including captures flushed by that run.
Add roots with repeated `--capture-replay-dir PATH`; disable with
`--skip-capture-replay`. Configuration experiments remain excluded from replay.
Existing capture budgets and deduplication still apply.

Compare after changing Vortex:

```text
python chassis_sim/benchmark.py --baseline PATH_TO_OLD/report.json
python chassis_sim/replay_captures.py chassis_sim/generated/captures --baseline OLD_REPLAY.json --report NEW_REPLAY.json
```

Use identical routes, seed, resolution, field assets, sensor and mount settings.
Changed inputs are incomparable. Capture comparisons show fixed, regressed, new,
removed, and improved/worsened metrics. Route comparisons show bad-frame,
invalid-frame, dropped-frame, and p95 metric deltas. Baselines are read-only;
copy a reviewed report somewhere persistent to establish one. Host load affects
latency even when pixels and routes are deterministic.

The simulator GitHub workflow builds current native replay, runs tests, and
publishes a short route report on relevant source changes. Local ignored captures
are replayed locally; GitHub sees only fixtures present in its checkout. The native
integration test also generates and replays a saved failure without robot hardware
or downloaded field assets.
