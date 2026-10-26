"""Field alignment, projection, and actual pixel occlusion regressions."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from drive import HERE, StreamReplay, camera_pose
from failure_capture import FailureCapture
from replay_captures import replay_cases
from field_renderer import GPUScene, asset_transform, find_field_asset, projection
from generate import tag_geometry, tag_texture
from setup_advantagescope import configure


class FieldMathTests(unittest.TestCase):
    def test_contact_surfaces_align_with_all_four_cad_bumps(self):
        try:
            folder=find_field_asset()
        except FileNotFoundError as error:
            self.skipTest(str(error))
        import trimesh
        from terrain import FIELD_BUMPS, Terrain
        scene=trimesh.load_scene(folder/'model.glb',process=False)
        transform=asset_transform(json.loads((folder/'config.json').read_text(encoding='utf-8')))
        terrain=Terrain('field',bump_smoothing=0.)
        panels={b.name:0 for b in FIELD_BUMPS}
        for node in scene.graph.nodes_geometry:
            node_transform,geometry=scene.graph[node]
            if 'bump plastic' not in (str(node)+' '+str(geometry)).lower():continue
            mesh=scene.geometry[geometry]
            world=transform@node_transform
            vertices=mesh.vertices@world[:3,:3].T+world[:3,3]
            center=vertices.mean(axis=0)
            bump=next(b for b in FIELD_BUMPS if b.contains(*center[:2]))
            panels[bump.name]+=1
            # Sample the largest upward face, avoiding holes and thickness edges.
            triangles=vertices[mesh.faces]
            normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
            areas=np.linalg.norm(normals,axis=1)
            eligible=np.flatnonzero(normals[:,2]/areas>.9)
            triangle=triangles[eligible[np.argmax(areas[eligible])]]
            point=triangle.mean(axis=0)
            self.assertAlmostEqual(terrain.surface(*point[:2]),point[2],delta=.002)
            color=mesh.visual.material.baseColorFactor
            self.assertGreater(color[2] if bump.name.startswith('blue') else color[0],200)
        self.assertEqual(panels,{b.name:2 for b in FIELD_BUMPS})

    def test_camera_layout_setup_is_repeatable_and_preserves_field_tab(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state-26_0_2.json'
            field = dict(type=3, title='3D Field', controller=dict(sources=[], game='FRC:2026 Field'),
                         controllerUUID='original', renderer=dict(cameraIndex=-1), controlsHeight=200)
            path.write_text(json.dumps(dict(hubs=[dict(state=dict(tabs=dict(selected=0, tabs=[field])))])))
            _, backup = configure(Path(tmp))
            self.assertTrue(backup.exists())
            configure(Path(tmp))
            tabs = json.loads(path.read_text())['hubs'][0]['state']['tabs']['tabs']
            self.assertEqual(len(tabs), 2)
            self.assertEqual(tabs[0]['renderer']['cameraIndex'], -1)
            self.assertEqual(tabs[1]['renderer']['cameraIndex'], 0)
            self.assertEqual(tabs[1]['controller']['sources'][-1]['logKey'], 'NT:/VortexSim/Camera')

    def test_cad_rotation_and_blue_wall_origin(self):
        config = dict(coordinateSystem='wall-blue', widthInches=651.22,
                      heightInches=317.677, rotations=[dict(axis='x', degrees=90)])
        result = asset_transform(config) @ [3.645, 1.124, -.604, 1]
        np.testing.assert_allclose(result[:3], [4.625494, 3.4304979, 1.124], atol=.001)

    def test_asymmetric_intrinsics_project_to_correct_pixels(self):
        camera = dict(fx=520., fy=530., cx=311., cy=225.)
        optical = np.array([.2, -.1, 2.])
        gl = np.array([optical[0], -optical[1], -optical[2], 1])
        clip = projection(camera, 640, 480) @ gl
        ndc = clip[:3]/clip[3]
        pixels = [(ndc[0]+1)*640/2-.5, (1-ndc[1])*480/2-.5]
        np.testing.assert_allclose(pixels, [363, 198.5], atol=1e-4)


class FieldOcclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            folder = find_field_asset()
        except FileNotFoundError as error:
            raise unittest.SkipTest(str(error))
        import moderngl
        try:
            context = moderngl.create_standalone_context(require=330)
            context.release()
        except Exception as error:
            raise unittest.SkipTest(f'OpenGL unavailable: {error}')
        cls.camera = json.loads((HERE.parent/'config/config.json').read_text())['camera']
        for key in ('fx', 'fy', 'cx', 'cy'):
            cls.camera[key] *= .5
        for key in ('k1', 'k2', 'k3', 'p1', 'p2'):
            cls.camera[key] = 0.
        raw = json.loads((HERE.parent/'config/2026-rebuilt-welded.json').read_text())['tags']
        cls.tags = [tag_geometry(t) for t in raw]
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        textures = {t['id']: tag_texture(dictionary, t['id']) for t in cls.tags}
        cls.scene = GPUScene(folder, cls.camera, 960, 540, cls.tags, textures)

    @classmethod
    def tearDownClass(cls):
        cls.scene.ctx.release()

    def test_field_and_added_wall_really_occlude_vortex_detections(self):
        executable = HERE.parent/'simulation_tests/replay/target'/(
            'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name == 'nt'
            else 'release/vortex-synthetic-replay')
        if not executable.is_file():
            self.skipTest('Build the release replay first')
        mount = json.loads((HERE/'demo_mounts.json').read_text())['actual']
        tag = next(t for t in self.tags if t['id'] == 18)
        base = np.array([tag['position'][0], tag['position'][1]-1.5, np.pi/2])
        eye, r = camera_pose(base, mount)
        clear, clear_labels = self.scene.render(eye, r, with_visibility=True)
        blocked, blocked_labels = self.scene.render(eye, r, blockers=[(
            np.array([tag['position'][0], tag['position'][1]-.75, 1.]), [1., .15, 2.])], with_visibility=True)
        self.assertTrue(next(label for label in clear_labels if label['id'] == 18)['clearly_visible'])
        self.assertFalse(next(label for label in blocked_labels if label['id'] == 18)['clearly_visible'])
        farther = base.copy()
        farther[1] -= .5
        far_eye, far_r = camera_pose(farther, mount)
        field_blocked = self.scene.render(far_eye, far_r)
        tags_only = self.scene.render(far_eye, far_r, draw_field=False)
        env = os.environ.copy()
        if os.name == 'nt':
            env['PATH'] = 'C:/msys64/mingw64/bin' + os.pathsep + env.get('PATH', '')
        manifest = dict(schema_version=1, width=960, height=540, camera=self.camera, tags=self.tags, frames=[])
        with tempfile.TemporaryDirectory() as tmp:
            replay = StreamReplay(executable, manifest, env, Path(tmp))
            try:
                def detect(image):
                    return replay.request(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))['detected_ids']
                self.assertIn(18, detect(clear))
                self.assertNotIn(18, detect(blocked))
                self.assertIn(18, detect(tags_only))
                self.assertNotIn(18, detect(field_blocked))
                # Emulate an old bad prediction, then verify the stored pixels can
                # be replayed independently and pass with the current solver.
                report = replay.request(cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY))
                capture = FailureCapture(Path(tmp)/'captures', 'test', dict(
                    camera=self.camera, tags=self.tags, renderer='field'))
                record = dict(gray=cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY), image=clear,
                    ghost=(np.array([base[0]+1,base[1],0]), np.eye(3)), chassis=base.copy(),
                    report=report, visibility=clear_labels, actual_mount=mount, assumed_mount=mount,
                    camera_position=eye.tolist(), camera_rotation=r.tolist(),
                    monotonic_s=0., captured_utc='2026-10-14T20:00:00+00:00')
                try:
                    capture.observe(record)
                    capture.observe(dict(record,monotonic_s=.1))
                finally:
                    capture.close()
                case = next((Path(tmp)/'captures').rglob('capture.json')).parent
                outcome = replay_cases(case, executable)
                self.assertEqual(outcome['total'], 1)
                self.assertEqual(outcome['passed'], 1)
            finally:
                replay.close()

    def test_added_ramp_is_visible_geometry_and_blocks_tag_visibility(self):
        from terrain import Terrain
        mount=json.loads((HERE/'demo_mounts.json').read_text())['actual']
        tag=next(t for t in self.tags if t['id']==18)
        base=np.array([tag['position'][0],tag['position'][1]-1.5,np.pi/2])
        eye,rotation=camera_pose(base,mount)
        image,labels=self.scene.render(eye,rotation,with_visibility=True)
        self.assertTrue(next(t for t in labels if t['id']==18)['clearly_visible'])
        terrain=Terrain('ramp',anchor_x=base[0],anchor_y=base[1],heading=base[2]+np.pi,
                        height=2,length=.3,width=2)
        blocked,labels=self.scene.render(eye,rotation,with_visibility=True,terrain_vertices=terrain.mesh())
        self.assertFalse(next(t for t in labels if t['id']==18)['clearly_visible'])
        self.assertFalse(np.array_equal(image,blocked))
        _,labels=self.scene.render(eye,rotation,with_visibility=True)
        self.assertTrue(next(t for t in labels if t['id']==18)['clearly_visible'])


if __name__ == '__main__':
    unittest.main()
