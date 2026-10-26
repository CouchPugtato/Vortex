"""Surface contact, full chassis transforms, scoring, and native tilt experiments."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from chassis_geometry import transform, errors
from drive import HERE, camera_pose, robot_pose, step
from failure_capture import CapturePolicy, classify, same_scene
from routes import ROUTES, Route
from terrain import Terrain, FIELD_BUMPS, from_args


class GeometryTests(unittest.TestCase):
    def test_pitch_raises_nose_and_roll_raises_left(self):
        _,pitch=transform([2,3,0,.4,0,math.radians(15)])
        _,roll=transform([2,3,0,.4,math.radians(15),0])
        self.assertGreater(pitch[2,0],0)
        self.assertGreater(roll[2,1],0)

    def test_full_camera_mount_roundtrip(self):
        mount=json.loads((HERE/'demo_mounts.json').read_text())['actual']
        mount.update(x_offset=.2,y_offset=-.1)
        for yaw in (0,.7,3.1):
            state=[3,2,yaw,.3,.12,-.2]
            expected=transform(state)
            actual=robot_pose(*camera_pose(state,mount),mount)
            for a,b in zip(actual,expected):
                np.testing.assert_allclose(a,b,atol=1e-12)
            self.assertAlmostEqual(errors(actual,state)['position_m'],0)

    def test_level_estimate_is_not_correct_on_a_tilted_elevated_chassis(self):
        state=[3,2,0,.3,math.radians(10),0]
        result=classify((np.array([3.,2.,0.]),np.eye(3)),state,{'detected_ids':[]},[],[],CapturePolicy())
        self.assertIn('position_error',result['reasons'])
        self.assertIn('tilt_error',result['reasons'])
        self.assertAlmostEqual(result['height_error_m'],.3)
        self.assertAlmostEqual(result['tilt_error_deg'],10)

    def test_matching_tilt_does_not_trigger_and_legacy_floor_cases_still_work(self):
        state=[3,2,.2,.3,.1,.2]
        self.assertEqual(classify(transform(state),state,{},[],[],CapturePolicy())['reasons'],[])
        np.testing.assert_allclose(transform([3,2,0])[0],[3,2,0])

    def test_drive_speed_projects_onto_the_surface(self):
        state=[3,2,0,0,0,math.radians(30)]
        moved=step(state,1,0,0,1,dict(length=16,width=8))
        self.assertAlmostEqual(moved[0]-3,math.cos(math.radians(30)))
        self.assertEqual(len(moved),6)

    def test_advantagescope_pose_keeps_height_and_attitude(self):
        from drive import Telemetry
        pose=Telemetry.pose(*transform([3,2,.2,.3,math.radians(8),math.radians(12)]))
        self.assertAlmostEqual(pose.Z(),.3)
        self.assertAlmostEqual(pose.rotation().X(),math.radians(8))
        self.assertAlmostEqual(pose.rotation().Y(),-math.radians(12))
        self.assertAlmostEqual(pose.rotation().Z(),.2)

    def test_deduplication_notices_chassis_attitude(self):
        a=dict(signature='same',chassis=[3,2,0,0,0,0])
        b=dict(signature='same',chassis=[3,2,0,0,0,.3])
        self.assertFalse(same_scene(a,b,CapturePolicy()))


class TerrainTests(unittest.TestCase):
    def test_default_cli_and_legacy_ramp_commands_use_all_field_bumps(self):
        from drive import build_parser
        for argv in ([], ['--terrain','ramp'], ['--terrain','cross-ramp']):
            args=build_parser().parse_args(argv)
            self.assertTrue(args.free_pose)
            terrain=from_args(args,[0,0,0])
            self.assertEqual(terrain.kind,'field')
            self.assertEqual(len(terrain.describe()['bumps']),4)
            self.assertIsNone(terrain.mesh())  # CAD already renders all four.
            for bump in FIELD_BUMPS:
                expected=bump.height-math.tan(math.radians(15))*terrain.bump_smoothing/3
                self.assertAlmostEqual(terrain.surface(*bump.center),expected)

    def test_field_contacts_are_fixed_and_leave_hubs_and_trenches_clear(self):
        a=Terrain('field',anchor_x=0,anchor_y=0,heading=0)
        b=Terrain('field',anchor_x=10,anchor_y=7,heading=2)
        self.assertEqual(len(FIELD_BUMPS),4)
        for bump in FIELD_BUMPS:
            x,y=bump.center
            for dx in (-.3,.3):
                state=a.apply([x+dx,y,0])
                self.assertAlmostEqual(state[3],b.surface(x+dx,y))
                self.assertAlmostEqual(abs(math.degrees(state[5])),15,places=5)
                self.assertAlmostEqual(a.apply([x+dx,y,math.pi/2])[4],-state[5],places=5)
        for x,y in ((4.6256,4.0345),(11.9154,4.0345),(4.6256,.8),(11.9154,7.3),(8.27,2.5)):
            np.testing.assert_allclose(a.apply([x,y,0])[3:],[0,0,0],atol=1e-12)

    def test_each_field_route_crosses_its_named_bump_and_returns(self):
        field=dict(length=16.541,width=8.069)
        terrain=Terrain('field')
        for bump in FIELD_BUMPS:
            route=Route.load(bump.name+'-bump',ROUTES[bump.name+'-bump'])
            initial=route.prepare_initial([1,1,.8],field)
            path=route.timeline(terrain.apply(initial),field,terrain=terrain)
            self.assertGreater(path.poses[:,3].max(),.145,bump.name)
            self.assertAlmostEqual(path.pose(4.5)[3],0.,places=5)
            np.testing.assert_allclose(path.poses[-1,:2],initial[:2],atol=.02)
            self.assertGreater(abs(path.poses[:,4:6]).max(),.2)
            self.assertLessEqual(abs(path.poses[:,4:6]).max(),math.radians(15.001))
            self.assertLess(abs(np.diff(path.poses[:,4:6],axis=0)).max(),math.radians(2))

    def test_field_bump_height_and_attitude_are_continuous_at_transitions(self):
        terrain=Terrain('field')
        for bump in FIELD_BUMPS:
            x,y=bump.center
            half_run=bump.height/math.tan(math.radians(15))
            for boundary in (x-half_run-terrain.bump_smoothing,bump.x_min,x,x+half_run+terrain.bump_smoothing):
                before=terrain.apply([boundary-1e-6,y,0])
                after=terrain.apply([boundary+1e-6,y,0])
                np.testing.assert_allclose(before[3:],after[3:],atol=1e-4)
            for boundary in (bump.y_min,bump.y_max):
                before=terrain.apply([x,boundary-1e-6,0])
                after=terrain.apply([x,boundary+1e-6,0])
                np.testing.assert_allclose(before[3:],after[3:],atol=1e-4)
            exact=Terrain('field',bump_smoothing=0.)
            self.assertAlmostEqual(exact.surface(x,y),bump.height)

    def test_static_profiles_are_independent_of_mount(self):
        state=Terrain('pitch',tilt_deg=12).apply([3,2,.5])
        self.assertAlmostEqual(math.degrees(state[5]),12)
        self.assertAlmostEqual(Terrain('roll',tilt_deg=-7).apply([3,2,.5])[4],math.radians(-7))

    def test_ramp_is_flat_before_after_and_at_crest(self):
        t=Terrain('ramp',anchor_x=4,anchor_y=3,heading=math.pi,height=.3,length=1,crest=.5)
        self.assertAlmostEqual(t.surface(4,3),0)
        self.assertAlmostEqual(t.surface(5.5,3),.3)
        self.assertAlmostEqual(t.surface(8,3),0)
        self.assertAlmostEqual(t.surface(5.5,6),0)
        self.assertAlmostEqual(t.apply([5.5,3,0])[5],0,places=5)
        self.assertGreater(t.apply([4.75,3,0])[5],0)
        self.assertLess(t.apply([6.5,3,0])[5],0)

    def test_default_bump_has_2026_scale_and_fifteen_degree_faces(self):
        t=Terrain('ramp',anchor_x=4,anchor_y=3,heading=math.pi)
        self.assertAlmostEqual(t.width,73*.0254)
        self.assertAlmostEqual(t.height,6*.0254)
        self.assertAlmostEqual(2*t.length,1.1375,places=3)
        self.assertEqual(t.crest,0.)
        self.assertAlmostEqual(t.surface(4.25+t.length,3),t.height)
        self.assertAlmostEqual(math.degrees(t.apply([4.25+t.length/2,3,0])[5]),15,places=5)
        self.assertAlmostEqual(math.degrees(t.apply([4.25+1.5*t.length,3,0])[5]),-15,places=5)

    def test_cross_slope_produces_roll(self):
        t=Terrain('cross-ramp',anchor_x=3,anchor_y=2,heading=0)
        state=t.apply([3,2.85,0])
        self.assertGreater(abs(state[4]),.1)
        self.assertAlmostEqual(state[5],0,places=5)

    def test_ramp_route_repeats_and_returns_to_ground(self):
        terrain=Terrain('ramp',anchor_x=6,anchor_y=3,heading=0)
        initial=terrain.apply([6,3,0])
        route=Route.load('ramp-crossing',ROUTES['ramp-crossing'])
        field=dict(length=16,width=8)
        a=route.timeline(initial,field,terrain=terrain)
        b=route.timeline(initial,field,terrain=terrain)
        np.testing.assert_array_equal(a.poses,b.poses)
        self.assertGreater(a.poses[:,3].max(),.145)
        self.assertGreater(abs(a.poses[:,5]).max(),.2)
        self.assertLessEqual(abs(a.poses[:,5]).max(),math.radians(15.01))
        self.assertEqual(route.duration,10.)
        self.assertAlmostEqual(a.pose(4.5)[3],0.)
        self.assertAlmostEqual(a.poses[-1,3],0)
        for timestamp in (.51,1.123,4.432,9.7):
            state=a.pose(timestamp)
            self.assertAlmostEqual(state[3],terrain.surface(*state[:2]))

    def test_default_field_spawn_has_room_for_complete_crossing_and_return(self):
        field=dict(length=16.541,width=8.069)
        route=Route.load('ramp-crossing',ROUTES['ramp-crossing'])
        initial=route.prepare_initial([4.6256,1.9312,math.pi/2],field)
        terrain=Terrain('ramp',anchor_x=initial[0],anchor_y=initial[1],heading=initial[2])
        path=route.timeline(terrain.apply(initial),field,terrain=terrain)
        self.assertGreater(path.poses[:,3].max(),.145)
        self.assertAlmostEqual(path.pose(4.5)[3],0,places=5)
        np.testing.assert_allclose(path.poses[-1,:2],initial[:2],atol=.02)
        self.assertGreater(path.poses[:,1].min(),.4)

    def test_mesh_matches_height_function(self):
        terrain=Terrain('ramp',anchor_x=4,anchor_y=3,heading=.7)
        vertices=terrain.mesh()
        self.assertEqual(vertices.shape[1],10)
        for vertex in vertices[::100]:
            self.assertAlmostEqual(float(vertex[2]),terrain.surface(*vertex[:2])+.002,places=5)
        self.assertIsNone(Terrain().mesh())

    def test_invalid_terrain_parameters(self):
        for values in ({'kind':'missing'},{'height':float('nan')},{'length':0},{'width':0},
                       {'pitch_deg':90},{'elevation':-1},{'crest':-1},{'bump_smoothing':-.1},{'bump_smoothing':.5}):
            with self.assertRaises(ValueError):Terrain(**values)


class NativeTiltTests(unittest.TestCase):
    def test_native_planar_and_free_models_on_the_same_tilted_pixels(self):
        from benchmark import main
        executable=HERE.parent/'simulation_tests/replay/target'/(
            'x86_64-pc-windows-gnu/release/vortex-synthetic-replay.exe' if os.name=='nt' else 'release/vortex-synthetic-replay')
        if not executable.is_file():self.skipTest('Build native replay first')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            route=root/'route.json'
            route.write_text(json.dumps(dict(name='tilt-check',segments=[[.2,0,0,0]])))
            main(['--renderer','fast','--replay-executable',str(executable),'--route-file',str(route),
                  '--profiles','clean','--mount-presets','matched','--terrains','flat','pitch','roll',
                  '--models','planar','free','--sample-hz','10','--no-capture','--skip-capture-replay',
                  '--chassis-height','.3',
                  '--output',str(root/'report')])
            report=json.loads((root/'report/report.json').read_text())
            self.assertEqual(len(report['runs']),6)
            for run in report['runs']:
                if run['id'].endswith('/free'):
                    self.assertEqual(run['summary']['status'],'pass',run['summary'])
                    self.assertLess(run['summary']['tilt_deg']['p95'],CapturePolicy().tilt_deg)
                else:
                    self.assertEqual(run['summary']['status'],'fail',run['summary'])
                self.assertEqual(len(run['frames'][0]['chassis']),6)
                self.assertAlmostEqual(run['frames'][0]['chassis'][3],.3)
                bands=run['summary']['inclination_bands']
                self.assertEqual(sum(b['frames'] for b in bands),3)


if __name__=='__main__':unittest.main()
