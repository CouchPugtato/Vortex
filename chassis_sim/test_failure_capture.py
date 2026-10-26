"""Failure classification, flood prevention, bounded IO, and saved fixture checks."""
import json
from pathlib import Path
import tempfile
import threading
import unittest

import cv2
import numpy as np

from failure_capture import CapturePolicy, FailureCapture, classify


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.context = dict(camera=dict(fx=100, fy=100, cx=32, cy=32, tag_size_m=.16),
                            tags=[dict(id=1, position=[0, 0, 1], rotation=np.eye(3).tolist())], renderer='test')
        self.capture = FailureCapture(Path(self.temp.name), 'session', self.context)
        self.mount = dict(x_offset=0., y_offset=0., z_offset=.5, roll_deg=-90., pitch_deg=0., yaw_deg=90.)

    def tearDown(self):
        self.capture.close()
        self.temp.cleanup()

    def record(self, now=0, error=.7, x=3., gray=None, ghost='default', heading=0., assumed=None, visible=True):
        if gray is None:
            gray = np.full((64, 64), 180, np.uint8)
        if isinstance(ghost, str):
            c, s = np.cos(heading), np.sin(heading)
            ghost = (np.array([x+error, 2, 0]), np.array([[c,-s,0],[s,c,0],[0,0,1.]]))
        return dict(gray=gray, image=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), chassis=np.array([x,2,0]),
            ghost=ghost, report=dict(detected_ids=[1] if ghost is not None else [], poses=[], joint_pose=None),
            visibility=[dict(id=1, visible_fraction=1., clearly_visible=True)] if visible else [],
            actual_mount=self.mount.copy(), assumed_mount=assumed or self.mount.copy(),
            camera_position=[x,2,.5], camera_rotation=np.eye(3).tolist(), monotonic_s=now,
            captured_utc='2026-10-14T20:00:00+00:00')

    def cases(self):
        self.capture.jobs.join()
        return list(Path(self.temp.name).rglob('capture.json'))

    def test_thousand_stationary_failures_save_one_case(self):
        for i in range(1000):
            self.capture.observe(self.record(i*.05))
        paths = self.cases()
        self.assertEqual(len(paths), 1)
        case = paths[0].parent
        np.testing.assert_array_equal(cv2.imread(str(case/'image.png'), 0), self.record()['gray'])
        metadata = json.loads(paths[0].read_text())
        self.assertEqual(metadata['failure']['reasons'], ['position_error'])
        self.assertEqual(metadata['review_status'], 'unreviewed')
        self.assertEqual(metadata['frame']['chassis_pose'], [3,2,0])
        self.assertTrue((case/'manifest.json').exists())

    def test_live_calibration_is_saved_with_its_pixels_and_restarts_confirmation(self):
        self.capture.observe(dict(self.record(),camera=self.context['camera']))
        camera=dict(self.context['camera'],fx=140)
        self.capture.observe(dict(self.record(.1),camera=camera))
        self.assertEqual(self.cases(),[])
        self.capture.observe(dict(self.record(.2),camera=camera))
        path=self.cases()[0]
        manifest=json.loads((path.parent/'manifest.json').read_text())
        metadata=json.loads(path.read_text())
        self.assertEqual(manifest['camera'],camera)
        self.assertEqual(metadata['frame']['camera'],camera)
        self.assertEqual(metadata['context']['camera'],camera)

    def test_single_bad_frame_and_occluded_missing_pose_do_not_save(self):
        self.capture.observe(self.record())
        self.capture.observe(self.record(1,error=0))
        for i in range(20):
            self.capture.observe(self.record(i+2, ghost=None, visible=False))
        self.assertEqual(self.cases(), [])

    def test_visible_missing_pose_and_invalid_pose_are_failures(self):
        missing = self.record(ghost=None)
        result = classify(missing['ghost'], missing['chassis'], missing['report'], missing['visibility'], [1], CapturePolicy())
        self.assertIn('missing_pose_with_visible_tags', result['reasons'])
        bad = self.record(ghost=(np.array([np.nan,2,0]), np.eye(3)))
        result = classify(bad['ghost'], bad['chassis'], bad['report'], bad['visibility'], [1], CapturePolicy())
        self.assertIn('nonfinite_or_invalid_pose', result['reasons'])
        self.capture.observe(bad)
        self.capture.observe(dict(bad, monotonic_s=.1))
        metadata = json.loads(self.cases()[0].read_text())
        self.assertIsNone(metadata['frame']['predicted_robot_pose'][0][0])

    def test_heading_wrap_and_capture_time_ground_truth(self):
        record = self.record(error=0, heading=np.radians(179))
        record['chassis'][2] = np.radians(-179)
        result = classify(record['ghost'], record['chassis'], record['report'], record['visibility'], [1], CapturePolicy())
        self.assertAlmostEqual(result['heading_error_deg'], 2)
        self.assertEqual(result['reasons'], [])
        record = self.record(error=0, heading=np.radians(15))
        self.capture.observe(record)
        self.capture.observe(dict(record, monotonic_s=.1))
        self.assertEqual(json.loads(self.cases()[0].read_text())['failure']['reasons'], ['heading_error'])

    def test_cooldown_and_scene_change_required_for_new_case(self):
        self.capture.observe(self.record())
        self.capture.observe(self.record(.1))
        noisy = np.zeros((64,64), np.uint8)
        self.capture.observe(self.record(1,x=5,gray=noisy))
        self.assertEqual(len(self.cases()), 1)
        self.capture.observe(self.record(6,x=5,gray=noisy))
        self.assertEqual(len(self.cases()), 2)

    def test_recovery_then_identical_failure_is_deduplicated(self):
        self.capture.observe(self.record())
        self.capture.observe(self.record(.1))
        self.capture.observe(self.record(.2,error=0))
        self.capture.observe(self.record(.3,error=0))
        self.capture.jobs.join()
        self.capture.observe(self.record(10))
        self.capture.observe(self.record(10.1))
        self.assertEqual(len(self.cases()), 1)
        self.assertTrue((self.cases()[0].parent/'after.png').exists())

    def test_duplicate_is_remembered_across_sessions(self):
        self.capture.observe(self.record())
        self.capture.observe(self.record(.1))
        self.cases()
        other = FailureCapture(Path(self.temp.name), 'other_session', self.context)
        try:
            other.observe(self.record(10))
            other.observe(self.record(10.1))
            other.jobs.join()
            self.assertEqual(other.accepted, 0)
        finally:
            other.close()

    def test_worse_example_replaces_instead_of_creating_files(self):
        self.capture.observe(self.record())
        self.capture.observe(self.record(.1))
        self.capture.observe(self.record(6,error=2))
        paths = self.cases()
        self.assertEqual(len(paths), 1)
        self.assertAlmostEqual(json.loads(paths[0].read_text())['failure']['position_error_m'], 2)
        self.capture.observe(self.record(12,error=5))
        self.capture.observe(self.record(18,error=20))
        paths = self.cases()
        self.assertAlmostEqual(json.loads(paths[0].read_text())['failure']['position_error_m'], 5)

    def test_mount_mismatch_is_separate_experiment(self):
        assumed = dict(self.mount, x_offset=.3)
        self.capture.observe(self.record(assumed=assumed))
        self.capture.observe(self.record(.1,assumed=assumed))
        path = self.cases()[0]
        self.assertEqual(path.parent.parent.name, 'experiments')
        self.assertEqual(json.loads(path.read_text())['category'], 'configuration_experiment')

    def test_disk_and_case_limits(self):
        self.capture.policy.max_bytes = 1
        self.capture.observe(self.record())
        self.capture.observe(self.record(.1))
        self.assertEqual(self.cases(), [])
        self.assertIn('disk limit', self.capture.error)

    def test_case_limit_blocks_distinct_scenes(self):
        self.capture.policy.max_cases = 1
        self.capture.observe(self.record())
        self.capture.observe(self.record(.1))
        self.capture.observe(self.record(10,x=5,gray=np.zeros((64,64),np.uint8)))
        self.assertEqual(len(self.cases()), 1)

    def test_before_and_recovery_context_are_bounded(self):
        self.capture.observe(self.record(0,error=0))
        self.capture.observe(self.record(.1))
        self.capture.observe(self.record(.2))
        self.capture.observe(self.record(.3,error=0))
        self.capture.observe(self.record(.4,error=0))
        case = self.cases()[0].parent
        self.assertTrue((case/'before.png').exists())
        self.assertTrue((case/'after.png').exists())
        self.assertEqual(len(list(case.glob('*.png'))), 4)

    def test_slow_disk_drops_work_instead_of_growing_queue(self):
        entered, release = threading.Event(), threading.Event()
        original = self.capture._write
        def slow(job):
            entered.set()
            release.wait(3)
            original(job)
        self.capture._write = slow
        try:
            self.capture.observe(self.record())
            self.capture.observe(self.record(.1))
            self.assertTrue(entered.wait(1))
            for i in range(1,100):
                self.capture.observe(self.record(i*6,x=3+i,gray=np.full((64,64),i,np.uint8)))
            self.assertLessEqual(self.capture.jobs.qsize(), 2)
            self.assertGreater(self.capture.dropped, 0)
            self.assertLessEqual(self.capture.accepted, 3)
        finally:
            release.set()

    def test_write_errors_are_reported_and_do_not_crash_capture_loop(self):
        def broken(job):
            raise OSError('test disk failure')
        self.capture._write = broken
        self.capture.observe(self.record())
        self.capture.observe(self.record(.1))
        self.capture.jobs.join()
        self.assertIn('test disk failure', self.capture.status()['error'])
        self.capture.observe(self.record(10,x=7))
        self.assertEqual(self.capture.saved, 0)
        self.assertEqual(self.capture.take_notifications(), [])

    def test_notification_only_after_successful_save_and_once_per_write(self):
        entered, release = threading.Event(), threading.Event()
        original = self.capture._write
        def slow(job):
            entered.set()
            release.wait(3)
            original(job)
        self.capture._write = slow
        try:
            self.capture.observe(self.record())
            self.capture.observe(self.record(.1))
            self.assertTrue(entered.wait(1))
            self.assertEqual(self.capture.take_notifications(), [])
        finally:
            release.set()
        self.capture.jobs.join()
        events = self.capture.take_notifications()
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]['updated'])
        self.assertEqual(events[0]['reasons'], ['position_error'])
        self.assertTrue((Path(events[0]['path'])/'image.png').exists())
        self.assertEqual(self.capture.take_notifications(), [])
        self.capture.observe(self.record(6,error=2))
        self.capture.jobs.join()
        self.assertTrue(self.capture.take_notifications()[0]['updated'])

    def test_invented_tag_is_flagged_but_partially_visible_tags_are_allowed(self):
        record = self.record(error=0)
        record['visibility'][0].update(clearly_visible=False, visible_fraction=.3)
        result = classify(record['ghost'], record['chassis'], record['report'], record['visibility'], [1], CapturePolicy())
        self.assertEqual(result['reasons'], [])
        record['report']['detected_ids'] = [2]
        result = classify(record['ghost'], record['chassis'], record['report'], record['visibility'], [1,2], CapturePolicy())
        self.assertIn('unexpected_tag', result['reasons'])


if __name__ == '__main__':
    unittest.main()
