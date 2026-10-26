"""Sensor, timeline, metrics and baseline contracts plus native route integration."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmark import compare_runs, main, pose_errors, stats, summarize, write_reports
from camera_effects import CameraEffects, PROFILES, Sensor
from drive import HERE, Harness, build_parser, camera_pose, robot_pose
from failure_capture import FailureCapture
from replay_captures import compare_reports
from routes import MOUNT_PRESETS, ROUTES, Route, mounts_for


class SensorTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((32, 64, 3), 100, np.uint8)

    def test_clean_preserves_exact_pixels(self):
        result, report = Sensor(PROFILES['clean']).apply(self.image, 0, (0, 0, 0), 100)
        np.testing.assert_array_equal(result, self.image)
        self.assertFalse(report['dropped_frame'])
        self.assertEqual(report['injected_latency_ms'], 0)

    def test_seeded_noise_exposure_drops_and_jitter_repeat(self):
        settings = CameraEffects(noise_std=5, exposure=.6, exposure_swing=.3, drop_rate=.4,
                                 latency_ms=10, jitter_ms=30, exposure_ms=20)
        a, b = Sensor(settings, 12), Sensor(settings, 12)
        drops = []
        for t in range(20):
            x, xr = a.apply(self.image, t/4, (1, .2, .5), 200)
            y, yr = b.apply(self.image, t/4, (1, .2, .5), 200)
            np.testing.assert_array_equal(x, y)
            self.assertEqual(xr, yr)
            self.assertGreaterEqual(xr['injected_latency_ms'], 0)
            drops.append(xr['dropped_frame'])
        self.assertTrue(any(drops))
        self.assertFalse(all(drops))

    def test_timing_does_not_depend_on_resolution(self):
        a, b = Sensor(PROFILES['poor-radio']), Sensor(PROFILES['poor-radio'])
        for t in range(10):
            self.assertEqual(a.apply(self.image, t, (0, 0, 0), 100)[1],
                             b.apply(self.image[:4], t, (0, 0, 0), 100)[1])

    def test_different_seeds_change_noise(self):
        s = CameraEffects(noise_std=10)
        self.assertFalse(np.array_equal(Sensor(s, 1).apply(self.image, 0, (0, 0, 0), 100)[0],
                                       Sensor(s, 2).apply(self.image, 0, (0, 0, 0), 100)[0]))

    def test_motion_smears_edges_only_when_moving(self):
        image = self.image.copy()
        image[:, 32:] = 0
        sensor = Sensor(CameraEffects(exposure_ms=20))
        stationary, _ = sensor.apply(image, 0, (0, 0, 0), 1000)
        moving, report = sensor.apply(image, 0, (0, 0, 1), 1000)
        np.testing.assert_array_equal(stationary, image)
        self.assertGreater(report['blur_px'], 1)
        self.assertTrue(np.any((moving > 0) & (moving < 100)))

    def test_exposure_clips_without_uint8_wrapping(self):
        result, _ = Sensor(CameraEffects(exposure=10)).apply(self.image, 0, (0, 0, 0), 100)
        self.assertTrue(np.all(result == 255))

    def test_all_drop_profile(self):
        self.assertTrue(Sensor(CameraEffects(drop_rate=1)).apply(self.image, 0, (0, 0, 0), 100)[1]['dropped_frame'])

    def test_invalid_settings(self):
        for key in CameraEffects.__dataclass_fields__:
            for value in (-1, math.nan, math.inf):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    CameraEffects(**{key: value})
        for settings in ({'drop_rate': 1.1}, {'exposure': 0}, {'exposure_swing': 1.1},
                         {'latency_ms': 5001}, {'exposure_ms': 1001}):
            with self.assertRaises(ValueError):
                CameraEffects(**settings)


class RouteTests(unittest.TestCase):
    field = dict(length=16, width=8)

    def test_every_builtin_repeats_exactly_and_respects_bounds(self):
        for name, segments in ROUTES.items():
            route = Route.load(name, segments)
            a = route.timeline([3, 2, .3], self.field)
            b = route.timeline([3, 2, .3], self.field)
            np.testing.assert_array_equal(a.poses, b.poses)
            self.assertAlmostEqual(a.times[-1], route.duration)
            self.assertTrue(np.all(a.poses[:, :2] >= .4))
            self.assertTrue(np.all(a.poses[:, 0] <= 15.6))
            self.assertTrue(np.all(a.poses[:, 1] <= 7.6))

    def test_short_segment_and_boundary_commands(self):
        route = Route.load('short', [[.01, 1, 0, 0], [.03, -1, 0, 0]])
        path = route.timeline([3, 2, 0], self.field)
        self.assertEqual(route.command(0), (1, 0, 0))
        self.assertEqual(route.command(.01), (-1, 0, 0))
        self.assertEqual(path.command(.01), (-1, 0, 0))
        np.testing.assert_allclose(path.pose(.005), [3.005, 2, 0])
        np.testing.assert_allclose(path.pose(10), [2.98, 2, 0])

    def test_prediction_uses_commands_and_estimate_not_actual_pose(self):
        path = Route.load('drive', [[1, 1, 0, 0]]).timeline([3, 2, 0], self.field)
        raw = (np.array([2., 2., 0.]), np.eye(3))
        predicted = path.prediction(raw, .2, .3)
        np.testing.assert_allclose(predicted[0], [2.1, 2, 0])
        self.assertIsNone(path.prediction(raw, .2, .5))

    def test_prediction_after_route_end_stops(self):
        path = Route.load('drive', [[1, 1, 0, 0]]).timeline([3, 2, 0], self.field)
        predicted = path.prediction((np.array([3.9, 2, 0]), np.eye(3)), .9, 1.1)
        np.testing.assert_allclose(predicted[0], [4, 2, 0])

    def test_invalid_routes(self):
        for segments in ([], [[0, 1, 0, 0]], [[1, math.nan, 0, 0]], [[1, 2]], [[601, 0, 0, 0]]):
            with self.assertRaises(ValueError):
                Route.load('invalid', segments)

    def test_mount_presets_roundtrip_and_known_errors(self):
        base = json.loads((HERE/'demo_mounts.json').read_text())['actual']
        original = dict(base)
        for preset in MOUNT_PRESETS:
            actual, assumed = mounts_for(preset, base)
            ghost = robot_pose(*camera_pose([3, 2, 0], actual), assumed)
            position, heading = pose_errors(ghost, [3, 2, 0])
            if preset == 'translation-error':
                self.assertAlmostEqual(position, .15)
            elif preset == 'rotation-error':
                self.assertAlmostEqual(heading, 8)
            else:
                self.assertAlmostEqual(position, 0)
                self.assertAlmostEqual(heading, 0)
        self.assertEqual(base, original)


class ReportingTests(unittest.TestCase):
    def frame(self, **overrides):
        frame = dict(valid=True, dropped=False, expected_ids=[18], reasons=[], position_m=.1,
                     heading_deg=1., raw_delivery_position_m=.2, predicted_delivery_position_m=.1,
                     predicted_delivery_heading_deg=1., latency_ms=50, render_ms=5, vortex_ms=45)
        return dict(frame, **overrides)

    def test_drops_and_missing_tags_do_not_inflate_detector_failure_count(self):
        s = summarize([self.frame(), self.frame(valid=False, dropped=True),
                       self.frame(valid=False, expected_ids=[])])
        self.assertEqual((s['frames'], s['delivered'], s['dropped'], s['invalid'], s['bad_frames']), (3, 2, 1, 1, 0))
        self.assertEqual(s['visible_pose_rate'], 1)

    def test_reason_counts_and_metrics(self):
        s = summarize([self.frame(), self.frame(reasons=['position_error'], position_m=.3)], True)
        self.assertEqual(s['reasons'], {'position_error': 1})
        self.assertEqual(s['bad_frames'], 1)
        self.assertAlmostEqual(s['position_m']['mean'], .2)
        self.assertTrue(s['configuration_experiment'])

    def test_empty_metric_is_null_not_fake_zero(self):
        self.assertIsNone(stats([None, math.inf])['p95'])
        self.assertIsNone(summarize([])['visible_pose_rate'])

    def test_heading_error_wraps_at_pi(self):
        angle = math.radians(-179)
        r = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        self.assertAlmostEqual(pose_errors((np.zeros(3), r), [0, 0, math.radians(179)])[1], 2)

    def test_capture_baseline_fixed_regressed_new_removed_and_changed_inputs(self):
        def case(name, passed, error=.1, signature='x'):
            return dict(case=name, case_id=name, input_signature=signature, passed=passed,
                        evaluation=dict(position_error_m=error, heading_error_deg=1))
        before = dict(cases=[case('fixed', False, .3), case('regressed', True), case('removed', True), case('changed', True)])
        after = dict(cases=[case('fixed', True), case('regressed', False, .3), case('new', True), case('changed', True, signature='y')])
        changes = compare_reports(after, before)
        self.assertEqual(changes['fixed'], ['fixed'])
        self.assertEqual(changes['regressed'], ['regressed'])
        self.assertEqual(changes['new'], ['new'])
        self.assertEqual(changes['removed'], ['removed'])
        self.assertEqual(changes['incomparable'], ['changed'])
        self.assertEqual(len(changes['improved']), 1)
        self.assertEqual(len(changes['worsened']), 1)

    def test_route_baseline_checks_input_signature(self):
        def run(name, signature='x', bad=0):
            s = summarize([self.frame()])
            s['bad_frames'] = bad
            return dict(id=name, signature=signature, summary=s)
        changes = compare_runs([run('same', bad=1), run('changed', 'y'), run('new')],
                               dict(runs=[run('same'), run('changed'), run('removed')]))
        self.assertEqual([c['status'] for c in changes], ['regressed', 'incomparable', 'new', 'removed'])

    def test_reports_emit_valid_json_csv_and_human_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = summarize([self.frame()])
            summary['status'] = 'pass'
            report = dict(runs=[dict(id='test', summary=summary, frames=[self.frame(index=0, timestamp_s=0)])])
            write_reports(Path(tmp), report)
            self.assertEqual(json.loads((Path(tmp)/'report.json').read_text())['runs'][0]['id'], 'test')
            self.assertIn('test,0,0', (Path(tmp)/'frames.csv').read_text())
            self.assertIn('| test | pass |', (Path(tmp)/'report.md').read_text(encoding='utf-8'))


class NativeBenchmarkTests(unittest.TestCase):
    def test_native_route_and_automatic_capture_replay(self):
        executable = HERE.parent/'simulation_tests/replay/target'/(
            'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name == 'nt' else 'release/vortex-synthetic-replay')
        if not executable.is_file():
            self.skipTest('Build native replay before running route integration')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = build_parser().parse_args(['--renderer', 'fast', '--terrain', 'flat', '--planar-pose', '--no-capture', '--replay-executable', str(executable)])
            args.max_error_meters = .2
            harness = Harness(args)
            try:
                ghost, image, result, truth, _ = harness.localize(harness.initial, harness.actual, harness.assumed)
                import cv2
                eye, rotation = camera_pose(truth, harness.actual)
                capture = FailureCapture(root/'captures', 'fixture', dict(camera=harness.camera, tags=harness.tags, renderer='fast'))
                try:
                    # Historical bad estimate; actual pixels are solved by current Vortex.
                    record = dict(gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), image=image,
                        ghost=(np.array([truth[0]+1, truth[1], 0]), np.eye(3)), chassis=truth,
                        report=result, visibility=result['visibility'], actual_mount=harness.actual,
                        assumed_mount=harness.assumed, camera_position=eye, camera_rotation=rotation,
                        monotonic_s=0., captured_utc='2026-10-25T00:00:00+00:00')
                    capture.observe(record)
                    capture.observe(dict(record, monotonic_s=.1))
                finally:
                    capture.close()
                self.assertIsNone(capture.status()['error'])
                self.assertEqual(capture.status()['saved'], 1)
            finally:
                harness.close()
            route = root/'short.json'
            route.write_text(json.dumps(dict(name='short', segments=[[.2, 0, .1, 0]])))
            code = main(['--renderer', 'fast', '--replay-executable', str(executable), '--route-file', str(route),
                         '--profiles', 'clean', '--mount-presets', 'matched', '--sample-hz', '10',
                         '--no-capture', '--capture-dir', str(root/'captures'), '--output', str(root/'result')])
            report = json.loads((root/'result/report.json').read_text())
            self.assertEqual(code, 0)
            self.assertEqual(report['runs'][0]['summary']['frames'], 3)
            self.assertEqual(report['runs'][0]['summary']['valid'], 3)
            expected = len(list((HERE.parent/'simulation_tests/captured_cases').rglob('capture.json')))
            self.assertEqual(report['capture_replays'][0]['passed'], expected)
            self.assertEqual(report['capture_replays'][1]['passed'], 1)
            second = main(['--renderer', 'fast', '--replay-executable', str(executable), '--route-file', str(route),
                           '--profiles', 'clean', '--mount-presets', 'matched', '--sample-hz', '10',
                           '--no-capture', '--capture-dir', str(root/'captures'), '--output', str(root/'second'),
                           '--baseline', str(root/'result/report.json')])
            comparison = json.loads((root/'second/report.json').read_text())
            self.assertEqual(second, 0)
            self.assertEqual(comparison['changes'][0]['status'], 'unchanged')
            self.assertEqual(comparison['capture_replays'][1]['changes']['regressed'], [])
            self.assertEqual(comparison['capture_replays'][1]['changes']['fixed'], [])

    def test_dropped_frames_skip_native_detection_and_failure_capture(self):
        executable = HERE.parent/'simulation_tests/replay/target'/(
            'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name == 'nt' else 'release/vortex-synthetic-replay')
        if not executable.is_file():
            self.skipTest('Build native replay before running drop integration')
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(['--renderer', 'fast', '--drop-rate', '1', '--latency-ms', '100',
                '--capture-dir', tmp, '--replay-executable', str(executable)])
            args.max_error_meters = .2
            harness = Harness(args)
            try:
                with patch.object(harness.replay, 'request', wraps=harness.replay.request) as request:
                    for index in range(3):
                        result = harness.localize(harness.initial, harness.actual, harness.assumed, timestamp=index/10, realtime=False)
                        self.assertIsNone(result[0])
                        self.assertTrue(result[2]['sensor']['dropped_frame'])
                        self.assertGreaterEqual(result[4], .1)
                    request.assert_not_called()
            finally:
                harness.close()
            self.assertEqual(harness.capture.status()['saved'], 0)
            self.assertEqual(list(Path(tmp).rglob('capture.json')), [])


if __name__ == '__main__':
    unittest.main()
