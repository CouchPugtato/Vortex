"""Configured camera geometry must agree with rendered and native input pixels."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from drive import HERE, Harness, MOUNT_KEYS, build_parser
from camera_projection import CameraProjection, CameraConfigWatcher, DISTORTION_KEYS


class CameraProjectionTests(unittest.TestCase):
    def setUp(self):
        self.camera=json.loads((HERE.parent/'config/config.json').read_text())['camera']

    def test_default_keeps_full_calibration_and_config_mount(self):
        args=build_parser().parse_args([])
        self.assertIsNone(args.mounts)
        self.assertFalse(args.ideal_camera)
        projection=CameraProjection(self.camera)
        self.assertEqual((projection.width,projection.height),(1920,1080))
        self.assertEqual(projection.camera,self.camera)

    def test_resize_scales_each_intrinsic_axis_and_retains_distortion(self):
        projection=CameraProjection(self.camera,960,400)
        for key in ('fx','cx'):
            self.assertAlmostEqual(projection.camera[key],self.camera[key]/2)
        for key in ('fy','cy'):
            self.assertAlmostEqual(projection.camera[key],self.camera[key]*400/1080)
        for key in DISTORTION_KEYS:
            self.assertEqual(projection.camera[key],self.camera[key])

    def test_distorted_pixels_follow_opencv_projection_without_clipped_edges(self):
        projection=CameraProjection(self.camera,320)
        # Encode ideal ray coordinates as floating-point colors. After warping,
        # compare those rays' forward projections against their output pixels.
        yy,xx=np.indices((projection.render_height,projection.render_width),dtype=np.float32)
        render=projection.render_camera
        rays=np.stack(((xx-render['cx'])/render['fx'],(yy-render['cy'])/render['fy'],np.ones_like(xx)),axis=-1)
        output=projection.output(rays)
        np.testing.assert_allclose(output[:,:,2],1.)  # no black overscan borders
        sample=output[::17,::19].reshape(-1,3)
        pixels,_=cv2.projectPoints(sample,np.zeros(3),np.zeros(3),projection.matrix,projection.distortion)
        oy,ox=np.indices(output[::17,::19].shape[:2])
        np.testing.assert_allclose(pixels.reshape(-1,2),np.c_[ox.ravel()*19,oy.ravel()*17],atol=.06)

    def test_ideal_control_does_not_warp_or_copy_pixels(self):
        projection=CameraProjection(self.camera,320,ideal=True)
        image=np.full((180,320,3),123,np.uint8)
        self.assertIs(projection.output(image),image)
        self.assertTrue(all(projection.camera[k]==0 for k in DISTORTION_KEYS))

    def test_bad_dimensions_and_calibration_fail_early(self):
        for kwargs in ({'calibration_width':0},{'width':0},{'height':-1}):
            with self.assertRaises(ValueError):CameraProjection(self.camera,**kwargs)
        with self.assertRaises(ValueError):CameraProjection(dict(self.camera,fx=0))

    def test_file_reload_waits_for_stable_json_and_selected_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            path.write_text(json.dumps(dict(camera=self.camera,camera_profiles={'2':self.camera})))
            watcher=CameraConfigWatcher(path,2)
            path.write_text('{')
            self.assertIsNone(watcher.poll(1.))
            self.assertIsNone(watcher.poll(1.4))
            self.assertIsNotNone(watcher.error)
            selected=dict(self.camera,fx=self.camera['fx']+20)
            path.write_text(json.dumps(dict(camera=self.camera,camera_profiles={'2':selected})))
            self.assertIsNone(watcher.poll(2.))
            self.assertIsNone(watcher.poll(2.2))
            self.assertEqual(watcher.poll(2.4),selected)
            self.assertIsNone(watcher.error)
            self.assertIsNone(watcher.poll(3.))

    def test_file_reload_detects_same_size_changes_with_unchanged_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            path.write_text(json.dumps(dict(camera=self.camera)))
            stamp=path.stat().st_mtime_ns
            size=path.stat().st_size
            watcher=CameraConfigWatcher(path,0)
            updated=dict(self.camera,fx=self.camera['fx']+10)
            path.write_text(json.dumps(dict(camera=updated)))
            os.utime(path,ns=(stamp,stamp))
            self.assertEqual(path.stat().st_size,size)
            self.assertIsNone(watcher.poll(1.))
            self.assertEqual(watcher.poll(1.4),updated)


class NativeCameraConfigTests(unittest.TestCase):
    def test_selected_profile_pixels_and_calibration_reach_native_replay(self):
        executable=HERE.parent/'simulation_tests/replay/target'/(
            'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name=='nt' else 'release/vortex-synthetic-replay')
        if not executable.is_file():self.skipTest('Build native replay first')
        args=build_parser().parse_args(['--renderer','fast','--camera-index','2','--no-capture',
                                       '--replay-executable',str(executable)])
        args.max_error_meters=.2
        harness=Harness(args)
        try:
            profile=json.loads(args.config.read_text())['camera_profiles']['2']
            self.assertEqual(harness.camera,profile)
            self.assertEqual(harness.actual,{key:float(profile[key]) for key in MOUNT_KEYS})
            self.assertEqual(harness.assumed,harness.actual)
            self.assertEqual(harness.manifest['camera'],profile)
            with patch.object(harness.replay,'request',wraps=harness.replay.request) as request:
                _,image,report,_,_=harness.localize(harness.initial,harness.actual,harness.assumed)
                self.assertEqual(image.shape,(1080,1920,3))
                np.testing.assert_array_equal(request.call_args.args[0],cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
                self.assertTrue(report['detected_ids'])
            chassis=harness.chassis.copy()
            source=dict(profile,fx=profile['fx']*1.1,k1=.01)
            settings=dict(camera=source,width=480,height=270)
            old_replay=harness.replay
            _,image,report,_,_=harness.localize(chassis,harness.actual,harness.assumed,camera_settings=settings)
            self.assertEqual(old_replay.process.returncode,0)
            self.assertEqual(image.shape,(270,480,3))
            self.assertAlmostEqual(report['camera']['fx'],source['fx']/4)
            self.assertEqual(report['camera']['k1'],.01)
            self.assertEqual(harness.manifest['camera'],report['camera'])
            np.testing.assert_array_equal(harness.chassis,chassis)
            current=harness.replay
            with self.assertRaises(ValueError):
                harness.configure_camera(dict(settings,camera=dict(source,fx=0)),harness.assumed)
            self.assertIs(harness.replay,current)
            self.assertIsNone(current.process.poll())
            harness.configure_camera(settings,harness.assumed)
            self.assertIs(harness.replay,current)
        finally:
            harness.close()


if __name__=='__main__':unittest.main()
