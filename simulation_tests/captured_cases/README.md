# Captured regressions

Four reviewed, frozen images from the full-field chassis simulator. The original
20 cm position and 5° heading limits are retained. Images are replayed directly;
rendering and the original field model are not required.

| Case | Original problem |
| --- | --- |
| small_oblique_missing_tags | No pose despite clearly visible tags |
| recover_extra_tags | Additional small tags missed, producing a large pose error |
| small_tags_ambiguous_pose | Sparse detections left the wrong planar pose solution |
| small_tags_joint_pose | Inaccurate multi-tag localization |

```text
python chassis_sim/replay_captures.py simulation_tests/captured_cases
```

Build the release replay first, or supply `--replay-executable PATH`. Each case
contains the unchanged grayscale input, manifest, capture metadata, and provenance.
Other unreviewed captures remain under `chassis_sim/generated/captures`.
The images were rendered from the [official AdvantageScope field asset](https://github.com/Mechanical-Advantage/AdvantageScopeAssets).
