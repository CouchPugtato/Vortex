"""Coordinate convention and live telemetry integration regressions."""
import math
import json
import os
from pathlib import Path
import tempfile
import socket
import time
import unittest

import ntcore
import cv2
import numpy as np
from wpimath.geometry import Pose3d
from drive import camera_pose, robot_pose, ground_background, step, Telemetry, StreamReplay, HERE
from generate import camera_pose as tag_camera_pose, render_fast, tag_geometry, tag_texture
from chassis_geometry import transform


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.mount = dict(x_offset=.2, y_offset=-.1, z_offset=.5,
                          roll_deg=-90, pitch_deg=0, yaw_deg=90)

    def test_forward_camera_convention_and_roundtrip(self):
        chassis = np.array([3., 2., math.pi / 2])
        eye, r = camera_pose(chassis, self.mount)
        np.testing.assert_allclose(r[:, 2], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(r[:, 1], [0, 0, -1], atol=1e-12)
        p, rr = robot_pose(eye, r, self.mount)
        np.testing.assert_allclose(p, [3, 2, 0], atol=1e-12)
        np.testing.assert_allclose(rr[:, 0], [0, 1, 0], atol=1e-12)

    def test_assumed_mount_error_changes_ghost_only(self):
        eye, r = camera_pose([3, 2, 0], self.mount)
        incorrect = dict(self.mount, x_offset=.3)
        p, _ = robot_pose(eye, r, incorrect)
        np.testing.assert_allclose(p, [2.9, 2, 0], atol=1e-12)
        p, rr = robot_pose(eye, r, dict(self.mount, pitch_deg=5))
        self.assertAlmostEqual(math.degrees(math.atan2(rr[1, 0], rr[0, 0])), 5)

    def test_robot_relative_driving_and_boundaries(self):
        field = dict(length=16, width=8)
        p = step([3, 2, math.pi/2], 2, 0, 0, .5, field)
        np.testing.assert_allclose(p[:2], [3, 3])
        p = step([.4, .4, 0], -2, -2, 0, 1, field)
        np.testing.assert_allclose(p[:2], [.4, .4])

    def test_upright_preview_has_sky_above_floor(self):
        eye, r = camera_pose([3, 2, 0], self.mount)
        image = ground_background(eye, r, dict(fx=100, fy=100, cx=100, cy=50),
                                  200, 100, dict(length=16, width=8))
        self.assertEqual(image.shape, (100, 200, 3))
        np.testing.assert_array_equal(image[10, 100], [225, 205, 175])
        self.assertFalse(np.array_equal(image[90, 110], image[10, 100]))

    def test_nt4_structs_and_missing_pose(self):
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as p3, socket.socket() as p4:
                p3.bind(('127.0.0.1', 0))
                p4.bind(('127.0.0.1', 0))
                port3, port4 = p3.getsockname()[1], p4.getsockname()[1]
            server = Telemetry(Path(tmp), port3, port4)
            client = ntcore.NetworkTableInstance.create()
            actual = client.getStructTopic('/VortexSim/Actual', Pose3d).subscribe(
                Pose3d(), ntcore.PubSubOptions(keepDuplicates=True, sendAll=True, periodic=.02))
            ghost = client.getStructArrayTopic('/VortexSim/Ghost', Pose3d).subscribe([])
            raw = client.getStructArrayTopic('/VortexSim/RawVision', Pose3d).subscribe([])
            camera = client.getStructTopic('/VortexSim/Camera', Pose3d).subscribe(Pose3d())
            client.setServer('127.0.0.1', port4)
            client.startClient4('vortex-sim-test')
            try:
                result = ((np.array([3., 2., 0]), np.eye(3)), None, {}, np.array([3, 2, 0]), .1)
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    server.publish_actual([3, 2, 0], self.mount)
                    server.publish_result(result)
                    if actual.get().X() == 3 and len(ghost.get()) == 1:
                        break
                    time.sleep(.02)
                self.assertEqual(actual.get().X(), 3)
                self.assertEqual(len(ghost.get()), 1)
                np.testing.assert_allclose([camera.get().X(), camera.get().Y(), camera.get().Z()],
                                           [3.2, 1.9, .5], atol=1e-12)
                self.assertAlmostEqual(camera.get().rotation().Z(), 0)
                # A stationary chassis must still emit new samples.
                previous = actual.getAtomic().time
                deadline = time.monotonic() + 2
                while actual.getAtomic().time == previous and time.monotonic() < deadline:
                    server.publish_actual([3, 2, 0])
                    time.sleep(.025)
                self.assertGreater(actual.getAtomic().time, previous)
                deadline = time.monotonic()+2
                while time.monotonic() < deadline:
                    server.publish_result(result, display_raw=False)
                    server.publish_display((np.array([3.2, 2., 0]), np.eye(3)), .1)
                    if raw.get() and ghost.get() and ghost.get()[0].X() == 3.2:
                        break
                    time.sleep(.02)
                self.assertEqual(raw.get()[0].X(), 3.)
                self.assertEqual(ghost.get()[0].X(), 3.2)
                self.assertEqual(server.entries['PositionErrorM'].getDouble(-1), 0.)
                self.assertEqual(server.entries['PredictionHorizonMs'].getDouble(-1), 100.)
                # Full attitudes must reach the rendered structs, with separate
                # Euler values for truth, raw vision and display prediction.
                chassis = np.array([3., 2., .4, .3, .15, .25])
                raw_pose = transform([3.1, 2., .5, .4, .2, .3])
                display_pose = transform([3.2, 2., .6, .5, .25, .35])
                deadline = time.monotonic()+2
                while time.monotonic() < deadline:
                    server.publish_actual(chassis, self.mount)
                    server.publish_result((raw_pose, None, {'planar_localization': False,
                                          'terrain': {'kind': 'ramp'}}, chassis, .1), display_raw=False)
                    server.publish_display(display_pose, .1)
                    if abs(actual.get().rotation().Y()+.25)<1e-9 and raw.get() and ghost.get() and abs(ghost.get()[0].rotation().Y()+.35)<1e-9:
                        break
                    time.sleep(.02)
                for value, expected in ((actual.get(), transform(chassis)),
                                        (raw.get()[0], raw_pose), (ghost.get()[0], display_pose)):
                    np.testing.assert_allclose(value.rotation().toMatrix(), expected[1], atol=1e-12)
                    np.testing.assert_allclose([value.X(), value.Y(), value.Z()], expected[0])
                for kind, angles in (('Actual', (.15, -.25, .4)), ('RawVision', (.2, -.3, .5)),
                                     ('Ghost', (.25, -.35, .6))):
                    entries = server.rotation_entries[kind]
                    self.assertTrue(entries['Valid'].getBoolean(False))
                    for key, radians in zip(('RollDeg', 'PitchDeg', 'YawDeg'), angles):
                        self.assertAlmostEqual(entries[key].getDouble(float('nan')), math.degrees(radians))
                self.assertEqual(server.entries['LocalizationModel'].getString(''), 'free')
                self.assertEqual(server.entries['Terrain'].getString(''), 'ramp')
                server.clear_vision()
                server.publish_display(None, 0.)
                for kind in ('RawVision', 'Ghost'):
                    self.assertFalse(server.rotation_entries[kind]['Valid'].getBoolean(True))
                    self.assertTrue(math.isnan(server.rotation_entries[kind]['PitchDeg'].getDouble(0.)))
                self.assertTrue(server.rotation_entries['Actual']['Valid'].getBoolean(False))
                server.publish_result((None, None, {}, np.array([3, 2, 0]), .1))
                deadline = time.monotonic() + 2
                while ghost.get() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(ghost.get(), [])
                self.assertEqual(raw.get(), [])
            finally:
                client.stopClient()
                server.instance.stopServer()
                ntcore.NetworkTableInstance.destroy(client)
                ntcore.NetworkTableInstance.destroy(server.instance)

    def test_conflicting_server_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            with self.assertRaisesRegex(RuntimeError, 'already in use'):
                Telemetry(Path(tmp), port4=listener.getsockname()[1])

    def test_persistent_replay_handles_valid_blank_valid_without_restarting(self):
        executable = HERE.parent / 'simulation_tests/replay/target' / (
            'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name == 'nt'
            else 'release/vortex-synthetic-replay')
        if not executable.exists():
            self.skipTest('Build the release replay to run the stream integration test')
        camera = json.loads((HERE.parent/'config/config.json').read_text())['camera']
        camera.update(fx=520., fy=520., cx=320., cy=240., k1=0., k2=0., k3=0., p1=0., p2=0.)
        raw = json.loads((HERE.parent/'config/2026-rebuilt-welded.json').read_text())['tags']
        tags = [tag_geometry(tag) for tag in raw]
        tag = next(tag for tag in tags if tag['id'] == 18)
        eye, r = tag_camera_pose(tag, dict(distance_m=2))
        frame = dict(camera_position=eye.tolist(), camera_rotation=r.tolist(), case={})
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        textures = {tag['id']: tag_texture(dictionary, tag['id']) for tag in tags}
        image, _ = render_fast(frame, tags, camera, 640, 480, textures)
        optimized, _ = render_fast(frame, tags, camera, 640, 480, textures, roi=True)
        env = os.environ.copy()
        if os.name == 'nt':
            env['PATH'] = 'C:/msys64/mingw64/bin' + os.pathsep + env.get('PATH', '')
        manifest = dict(schema_version=1, width=640, height=480, camera=camera, tags=tags, frames=[])
        with tempfile.TemporaryDirectory() as tmp:
            replay = StreamReplay(executable, manifest, env, Path(tmp))
            try:
                pid = replay.process.pid
                first = replay.request(image)
                blank = replay.request(np.full_like(image, 180))
                final = replay.request(optimized)
                self.assertIn(18, first['detected_ids'])
                self.assertEqual(blank['detected_ids'], [])
                self.assertIsNone(blank['joint_pose'])
                self.assertEqual(first['detected_ids'], final['detected_ids'])
                self.assertNotIn('passed', final)  # no fake ground-truth scoring in live mode
                self.assertEqual(pid, replay.process.pid)
                self.assertIsNone(replay.process.poll())
                estimates = final['joint_pose'] or final['poses'][0]
                np.testing.assert_allclose(estimates['camera_position'], eye, atol=.2)
            finally:
                replay.close()
            self.assertEqual(replay.process.returncode, 0)


if __name__ == '__main__':
    unittest.main()
