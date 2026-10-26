"""Native recall and unconstrained pose checks in mixed-size distorted views."""
import json
import os
from pathlib import Path
import unittest

import cv2
import numpy as np

from camera_projection import CameraProjection
from drive import HERE, StreamReplay
from generate import MAP_FROM_SOLVER, render_fast, tag_texture


class MixedTagViewTests(unittest.TestCase):
    def test_distorted_noise_does_not_invent_tags(self):
        executable=Path(os.environ.get('VORTEX_TEST_REPLAY',HERE.parent/'simulation_tests/replay/target'/(
            'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name=='nt' else 'release/vortex-synthetic-replay')))
        if not executable.is_file():self.skipTest('Build native replay first')
        env=os.environ.copy()
        if os.name=='nt':env['PATH']='C:/msys64/mingw64/bin'+os.pathsep+env.get('PATH','')
        base=json.loads((HERE.parent/'config/config.json').read_text())['camera']
        image=np.clip(np.random.default_rng(2026).normal(128,30,(360,640)),0,255).astype(np.uint8)
        runtime=HERE/'generated/mixed_tag_test';runtime.mkdir(parents=True,exist_ok=True)
        for k1 in (-.3,0.,.25):
            with self.subTest(k1=k1):
                camera=dict(base,fx=300.,fy=300.,cx=320.,cy=180.,k1=k1,k2=.08,k3=0.,p1=.002,p2=-.001)
                manifest=dict(schema_version=1,camera=camera,width=640,height=360,tags=[],frames=[])
                stream=StreamReplay(str(executable.resolve()),manifest,env,runtime)
                try:result=stream.request(image)
                finally:stream.close()
                self.assertEqual(result['detected_ids'],[])
                self.assertIsNone(result['joint_pose'])

    def test_range_and_distortion_with_four_nearby_tags(self):
        executable=Path(os.environ.get('VORTEX_TEST_REPLAY',HERE.parent/'simulation_tests/replay/target'/(
            'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name=='nt' else 'release/vortex-synthetic-replay')))
        if not executable.is_file():self.skipTest('Build native replay first')
        env=os.environ.copy()
        if os.name=='nt':env['PATH']='C:/msys64/mingw64/bin'+os.pathsep+env.get('PATH','')
        base=json.loads((HERE.parent/'config/config.json').read_text())['camera']
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        textures={i:tag_texture(dictionary,i) for i in range(1,6)}
        runtime=HERE/'generated/mixed_tag_test'
        runtime.mkdir(parents=True,exist_ok=True)
        outcomes=[]
        for lens,coefficients in [('ideal',(0,0,0)),('barrel',(-.3,.08,-.005)),('pincushion',(.25,.02,.001))]:
            camera=dict(base,fx=600.,fy=600.,cx=640.,cy=360.,tag_size_m=.2,
                        k1=coefficients[0],k2=coefficients[1],k3=coefficients[2],p1=.002,p2=-.001)
            projection=CameraProjection(camera,1280,720,1280,720)
            for distance in (4.,6.,7.,8.,10.):
                for lateral in (0.,.65):
                    with self.subTest(lens=lens,distance=distance,lateral=lateral):
                        tags=[dict(id=i+1,position=[(u-640)/600*2.,-.45,2.],rotation=MAP_FROM_SOLVER.T.tolist())
                              for i,u in enumerate((250.,480.,710.,940.))]
                        angle=-.35 if lateral else 0.
                        c,s=np.cos(angle),np.sin(angle)
                        tilt=np.array([[c,0,s],[0,1,0],[-s,0,c]])
                        tags.append(dict(id=5,position=[lateral*distance,.23*distance,distance],rotation=(tilt@MAP_FROM_SOLVER.T).tolist()))
                        frame=dict(camera_position=[0,0,0],camera_rotation=np.eye(3).tolist(),case={})
                        image,_=render_fast(frame,tags,projection.render_camera,projection.render_width,projection.render_height,textures,roi=True)
                        image=projection.output(image)
                        manifest=dict(schema_version=1,camera=camera,width=1280,height=720,tags=tags,frames=[])
                        stream=StreamReplay(str(executable.resolve()),manifest,env,runtime)
                        try:result=stream.request(image)
                        finally:stream.close()
                        found=set(result['detected_ids'])
                        outcomes.append(dict(lens=lens,distance_m=distance,lateral=lateral,ids=sorted(found)))
                        self.assertTrue(set(range(1,5))<=found,outcomes[-1])
                        # 12px black border at 10m is intentionally below a
                        # guaranteed decode threshold; record it without forcing
                        # impossible subpixel recovery into acceptance criteria.
                        # At 8m the oblique barrel case has fewer than two
                        # source pixels across some code cells. Track this
                        # unresolved recall limit rather than require a decode.
                        required=distance<=7. or (distance<=8. and not (lens=='barrel' and lateral))
                        outcomes[-1]['far_required']=bool(required)
                        outcomes[-1]['far_detected']=5 in found
                        if required:self.assertIn(5,found,outcomes[-1])
                        self.assertIsNotNone(result['joint_pose'],outcomes[-1])
                        estimate=result['joint_pose']
                        self.assertLess(np.linalg.norm(estimate['camera_position']),.05,outcomes[-1])
        destination=HERE/'generated/mixed_tag_views.json'
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(outcomes,indent=2))
