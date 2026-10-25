# Repeatable simulator testing

```text
python chassis_sim/benchmark.py
```

Runs **100 combinations**: four routes × five camera profiles × five mount presets,
or **7,600 frames** at the default 10 samples/second. Uses the production CPU
detector and full-field renderer, including CAD occlusion. Dependencies and field
assets are the same as interactive driving. `--renderer fast` selects tag-only tests.

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

The chassis has no collision response. Routes can cross CAD geometry; obstacle
sweeps test visibility rather than safe navigation. Blur approximates optical flow
at 2 m depth rather than integrating rendered scenes throughout an exposure.
Noise and exposure are synthetic approximations. GPU detection and real camera
drivers are not exercised.

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
