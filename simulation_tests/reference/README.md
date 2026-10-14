`tag36h11_1.png` is the unmodified tag ID 1 reference from
[AprilRobotics/apriltag-imgs](https://github.com/AprilRobotics/apriltag-imgs/blob/master/tag36h11/tag36_11_00001.png).
The generator test compares its bit orientation against this independent reference.

`dim_noisy_tag.png` is the unchanged image from stress case
`single_quality_0059`: tag 1 at 2 m, brightness 0.6, blur sigma 0.8,
noise sigma 10, seed 83. It preserves the original pixels and calibration and
guards noisy single-tag pose accuracy when changing quad sampling.
