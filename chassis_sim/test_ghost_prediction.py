import math
import unittest

import numpy as np

from drive import chassis_rotation, step
from ghost_prediction import GhostPredictor
from chassis_geometry import transform


class GhostPredictionTests(unittest.TestCase):
    def setUp(self):
        self.predictor = GhostPredictor()
        self.raw = (np.array([3., 2., .1]), chassis_rotation(0))
        self.predictor.observe(self.raw, 10.)

    def test_delayed_forward_pose_catches_up_without_mutating_raw(self):
        self.predictor.record(10., 10.1, 2., 0., 0.)
        p, _ = self.predictor.predict(10.1)
        np.testing.assert_allclose(p, [3.2, 2., .1])
        np.testing.assert_array_equal(self.raw[0], [3., 2., .1])

    def test_command_history_handles_stop_and_strafe(self):
        self.predictor.record(10., 10.05, 2., 0., 0.)
        self.predictor.record(10.05, 10.1, 0., 0., 0.)
        self.predictor.record(10.1, 10.15, 0., -2., 0.)
        np.testing.assert_allclose(self.predictor.predict(10.15)[0], [3.1, 1.9, .1])

    def test_uses_estimated_heading_and_preserves_position_error(self):
        self.predictor.observe((np.array([4., 3., .7]), chassis_rotation(math.pi/2)), 10.)
        self.predictor.record(10., 10.1, 2., 0., 0.)
        np.testing.assert_allclose(self.predictor.predict(10.1)[0], [4., 3.2, .7])

    def test_matches_drive_ticks_through_translation_and_rotation(self):
        chassis = np.array([3., 2., 0.])
        for i, commands in enumerate([(2., 0., 1.5), (2., 0., 1.5), (0., 2., -1.5)]):
            self.predictor.record(10.+i*.02, 10.+(i+1)*.02, *commands)
            chassis = step(chassis, *commands, .02, dict(length=16., width=8.))
        p, r = self.predictor.predict(10.06)
        np.testing.assert_allclose(p[:2], chassis[:2])
        np.testing.assert_allclose(r, chassis_rotation(chassis[2]), atol=1e-12)

    def test_partial_interval_from_camera_timestamp(self):
        self.predictor.observe(self.raw, 10.05)
        self.predictor.record(10., 10.1, 2., 0., 0.)
        np.testing.assert_allclose(self.predictor.predict(10.1)[0], [3.1, 2., .1])

    def test_prediction_follows_estimated_slope_without_actual_pose(self):
        raw = transform([4., 3., .4, .8, .15, .3])
        self.predictor.observe(raw, 10.)
        self.predictor.record(10., 10.1, 2., .5, 0.)
        p, r = self.predictor.predict(10.1)
        np.testing.assert_allclose(p, raw[0] + raw[1] @ np.array([.2, .05, 0.]))
        np.testing.assert_allclose(r, raw[1], atol=1e-12)
        self.assertGreater(p[2], raw[0][2])

    def test_turning_keeps_estimated_surface_normal_fixed(self):
        raw = transform([4., 3., .4, .8, .15, .3])
        self.predictor.observe(raw, 10.)
        self.predictor.record(10., 10.1, 0., 0., 1.5)
        p, r = self.predictor.predict(10.1)
        np.testing.assert_allclose(p, raw[0])
        np.testing.assert_allclose(r[:, 2], raw[1][:, 2], atol=1e-12)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(math.atan2(r[1, 0], r[0, 0]), .55)

    def test_invalid_rotations_do_not_become_display_poses(self):
        for rotation in (np.zeros((3, 3)), -np.eye(3), np.full((3, 3), np.nan), np.eye(2)):
            self.predictor.observe((np.zeros(3), rotation), 10.)
            self.assertIsNone(self.predictor.predict(10.))

    def test_missing_invalid_and_stale_frames_hide_ghost(self):
        self.predictor.record(10., 10.3, 2., 0., 0.)
        self.assertIsNone(self.predictor.predict(10.3))
        self.predictor.observe(None, 10.3)
        self.assertIsNone(self.predictor.predict(10.3))
        self.predictor.observe((np.array([np.nan, 0., 0.]), np.eye(3)), 10.3)
        self.assertIsNone(self.predictor.predict(10.3))

    def test_reset_and_missing_history_do_not_invent_motion(self):
        self.predictor.record(10.05, 10.1, 2., 0., 0.)
        self.assertIsNone(self.predictor.predict(10.1))
        self.predictor.clear()
        self.assertIsNone(self.predictor.predict(10.))
        self.assertFalse(self.predictor.commands)

    def test_new_vision_reanchors_and_history_is_bounded(self):
        for i in range(100):
            self.predictor.record(10.+i*.02, 10.+(i+1)*.02, 2., 0., 0.)
        self.assertLess(len(self.predictor.commands), 15)
        self.predictor.observe((np.array([6., 5., 0.]), np.eye(3)), 11.9)
        np.testing.assert_allclose(self.predictor.predict(12.)[0], [6.2, 5., 0.])
