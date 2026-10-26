"""Keyboard chassis, synthetic camera, production Vortex replay, and NT4 poses."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import math
import os
import queue
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import threading
import hashlib
from uuid import uuid4

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulation_tests"))
from generate import render_fast, tag_geometry, tag_texture
from run import replay_command
from ghost_prediction import GhostPredictor
from camera_effects import CameraEffects, PROFILES, Sensor
from camera_projection import CameraProjection, CameraConfigWatcher, CALIBRATION_KEYS
from dataclasses import asdict
from chassis_geometry import transform as chassis_transform, errors as chassis_errors, valid_pose
from terrain import TERRAINS, RAMP_HEIGHT, RAMP_LENGTH, RAMP_WIDTH, terrain_name, from_args as terrain_from_args

MOUNT_KEYS = ("x_offset", "y_offset", "z_offset", "roll_deg", "pitch_deg", "yaw_deg")


def find_advantagescope():
    executable = shutil.which("AdvantageScope")
    if executable:
        return Path(executable)
    for base in (Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
                 Path(os.environ.get("ProgramFiles", "C:/Program Files")),
                 Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))):
        for directory in ("AdvantageScope", "advantagescope"):
            candidate = base / directory / "AdvantageScope.exe"
            if candidate.is_file():
                return candidate
    return None


def mount_rotation(mount):
    """Exact src/main.rs convention: Ry(yaw) Rx(pitch) Rz(roll)."""
    roll, pitch, yaw = np.radians([mount[k] for k in MOUNT_KEYS[3:]])
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                             math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return (np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
            np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @
            np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]))


def chassis_rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def camera_pose(chassis, mount):
    p, r = chassis_transform(chassis)
    return (p + r @ np.array([mount[k] for k in MOUNT_KEYS[:3]]),
            r @ mount_rotation(mount))


def robot_pose(camera_position, camera_rotation, mount):
    r = np.asarray(camera_rotation) @ mount_rotation(mount).T
    p = np.asarray(camera_position) - r @ np.array([mount[k] for k in MOUNT_KEYS[:3]])
    return p, r


def apply_calibration_error(actual, error):
    assumed={key:float(actual[key])+float(error[key]) for key in MOUNT_KEYS}
    if not all(math.isfinite(value) for value in assumed.values()):
        raise ValueError('Mount and calibration error values must be finite')
    return assumed


def camera_rays(camera, width, height):
    yy, xx = np.indices((height, width), dtype=np.float32)
    return np.stack(((xx-camera["cx"])/camera["fx"],
                     (yy-camera["cy"])/camera["fy"], np.ones_like(xx)), axis=-1)


def ground_background(position, rotation, camera, width, height, field, rays=None):
    """Ray-plane floor grid for orientation; BGR pixels, no field obstacle model."""
    rays = (camera_rays(camera, width, height) if rays is None else rays) @ rotation.astype(np.float32).T
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = [225, 205, 175]  # sky, BGR
    ground = rays[:, :, 2] < -1e-6
    distance = np.divide(-position[2], rays[:, :, 2],
                         out=np.zeros((height, width)), where=ground)
    points = position + rays * distance[:, :, None]
    x, y = points[:, :, 0], points[:, :, 1]
    inside = ground & (distance > 0) & (x >= 0) & (x <= field["length"]) & (y >= 0) & (y <= field["width"])
    image[ground] = [100, 110, 115]
    checker = (np.floor(x).astype(int) + np.floor(y).astype(int)) % 2 == 0
    image[inside & checker] = [65, 95, 65]
    image[inside & ~checker] = [75, 108, 75]
    grid = (np.minimum(x % 1, 1-x % 1) < .012) | (np.minimum(y % 1, 1-y % 1) < .012)
    image[inside & grid] = [160, 175, 160]
    border = (x < .06) | (x > field["length"]-.06) | (y < .06) | (y > field["width"]-.06)
    image[inside & border] = [220, 220, 220]
    return image


class StreamReplay:
    """Keep the native detector alive; send grayscale bytes, receive pose JSON."""
    def __init__(self, executable, manifest, env, root):
        self.log = (root / "vortex-stderr.log").open("wb")
        self.process = subprocess.Popen([str(executable), "--stream"], env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        self.responses = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.process.stdin.write((json.dumps(manifest) + "\n").encode())
        self.process.stdin.flush()

    def _read(self):
        try:
            for line in self.process.stdout:
                # Native AprilTag occasionally writes a diagnostic to stdout.
                marker = line.find(b"VORTEX_FRAME ")
                if marker >= 0:
                    self.responses.put(json.loads(line[marker+len(b"VORTEX_FRAME "):]))
        except Exception as error:
            self.responses.put(error)
        finally:
            self.responses.put(RuntimeError("Vortex stream stopped; see vortex-stderr.log. "
                                            "Rebuild the replay if using an older executable."))

    def request(self, image):
        self.process.stdin.write(image.tobytes())
        self.process.stdin.flush()
        try:
            response = self.responses.get(timeout=30)
        except queue.Empty:
            self.process.terminate()
            raise TimeoutError("Vortex did not respond within 30 seconds")
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            self.process.terminate()
            self.process.wait(timeout=3)
        finally:
            self.process.stdout.close()
            self.log.close()


def step(chassis, forward, left, turn, dt, field):
    delta = chassis_transform(chassis)[1] @ np.array([forward, left, 0.]) * dt
    result = np.array([np.clip(chassis[0] + delta[0], 0.4, field["length"] - 0.4),
                     np.clip(chassis[1] + delta[1], 0.4, field["width"] - 0.4),
                     math.atan2(math.sin(chassis[2] + turn * dt),
                                math.cos(chassis[2] + turn * dt))])
    return np.concatenate((result,np.asarray(chassis[3:6]))) if len(chassis)>=6 else result


class Harness:
    def __init__(self, args):
        self.args = args
        settings = asdict(PROFILES[getattr(args, 'camera_profile', 'clean')])
        for key in settings:
            value = getattr(args, key, None)
            if value is not None:
                settings[key] = value
        self.sensor = Sensor(CameraEffects(**settings), getattr(args, 'seed', 509))
        self.sensor_started = time.perf_counter()
        raw = json.loads(args.config.read_text())
        self.camera = dict(raw.get("camera_profiles", {}).get(str(args.camera_index), raw["camera"]))
        self.source_camera = dict(self.camera)
        self.actual = {k: float(self.camera.get(k, 0)) for k in MOUNT_KEYS}
        if args.mounts:
            saved = json.loads(args.mounts.read_text())
            saved_actual=saved.get('actual',saved)
            self.actual = {k:float(saved_actual[k]) for k in MOUNT_KEYS}
            if not all(math.isfinite(v) for v in self.actual.values()):
                raise ValueError("Mount values must be finite")
        # Calibration errors are a transient experiment, never restored from files.
        self.calibration_error=dict.fromkeys(MOUNT_KEYS,0.)
        self.assumed = apply_calibration_error(self.actual,self.calibration_error)
        self.projection = CameraProjection(self.camera,args.width,args.height,args.calibration_width,
                                           args.calibration_height,args.ideal_camera)
        self.camera = self.projection.camera
        self.width,self.height = self.projection.width,self.projection.height
        self.render_camera = self.projection.render_camera
        self.render_width,self.render_height = self.projection.render_width,self.projection.render_height
        self.camera_settings=dict(camera=dict(self.source_camera),width=self.width,height=self.height)
        raw_field = json.loads(args.field.read_text())
        self.field = raw_field["field"]
        self.tags = [tag_geometry(t) for t in raw_field["tags"]]
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.textures = {t["id"]: tag_texture(dictionary, t["id"]) for t in self.tags}
        # Floor is only visual context; keep tags and Vortex at full resolution.
        self.floor_width, self.floor_height = max(2, self.render_width//4), max(2, self.render_height//4)
        self.floor_camera = dict(self.render_camera)
        for k in ("fx", "cx"):
            self.floor_camera[k] *= self.floor_width/self.render_width
        for k in ("fy", "cy"):
            self.floor_camera[k] *= self.floor_height/self.render_height
        self.rays = camera_rays(self.floor_camera, self.floor_width, self.floor_height)
        # Spawn in front of the hub, clear of the trench structure at the old 2 m spawn.
        tag = next(t for t in self.tags if t["id"] == args.start_tag)
        normal = np.array(tag["rotation"])[:, 0]
        forward = mount_rotation(self.actual)[:, 2]
        yaw = math.atan2(-normal[1], -normal[0]) - math.atan2(forward[1], forward[0])
        eye = np.array(tag["position"]) + getattr(args, 'start_distance', 1.5) * normal
        origin = eye - chassis_rotation(yaw) @ np.array([self.actual[k] for k in MOUNT_KEYS[:3]])
        self.spawn = np.array([origin[0], origin[1], yaw])
        self.configure_terrain(getattr(args,'terrain','flat'))
        self.chassis = self.initial.copy()
        self.root = HERE / "generated" / f"session_{time.time_ns()}_{uuid4().hex[:8]}"
        self.root.mkdir(parents=True)
        self.env = os.environ.copy()
        command = replay_command(args, self.root, self.env)
        if args.replay_executable:
            self.command = [str(x) for x in command]
        else:
            # Build once; invoking Cargo on every camera frame is unnecessary.
            split = command.index("--")
            build = command[:split]
            build[build.index("run")] = "build"
            subprocess.run([str(x) for x in build], env=self.env, check=True)
            target = "x86_64-pc-windows-gnu/" if "--target" in build else ""
            executable = HERE.parent / "simulation_tests/replay/target" / target / "release" / (
                "vortex-synthetic-replay.exe" if os.name == "nt" else "vortex-synthetic-replay")
            self.command = [str(executable.resolve()), str(self.root.resolve()), "0.20"]
        self.manifest = dict(schema_version=1, width=self.width, height=self.height,
                        camera=self.camera, tags=self.tags, frames=[],
                        robot_mount=None if getattr(args,'free_pose',False) else dict(self.assumed))
        self.replay = StreamReplay(self.command[0], self.manifest, self.env, self.root)
        self.field_renderer = None
        self.field_asset = None
        if getattr(args, 'renderer', 'field') == 'field':
            from field_renderer import FieldRenderer, find_field_asset
            try:
                self.field_asset = find_field_asset(getattr(args, 'field_asset', None))
                self.field_renderer = FieldRenderer(self.field_asset,
                    self.render_camera, self.render_width, self.render_height, self.tags, self.textures)
                print(f"Field camera GPU: {self.field_renderer.renderer}", flush=True)
            except Exception:
                self.replay.close()
                raise
        self.capture = None
        if not getattr(args, 'no_capture', False):
            from failure_capture import CapturePolicy, FailureCapture
            revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE.parent,
                                      capture_output=True, text=True).stdout.strip() if shutil.which('git') else 'unknown'
            source_hashes = {name: hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()
                for name in ('src/detector.rs', 'src/pose.rs', 'src/tag_refine.rs', 'src/config.rs',
                             'simulation_tests/replay/src/main.rs', 'simulation_tests/replay/Cargo.lock',
                             'chassis_sim/drive.py', 'chassis_sim/field_renderer.py', 'chassis_sim/failure_capture.py',
                             'chassis_sim/camera_effects.py', 'chassis_sim/camera_projection.py', 'chassis_sim/routes.py',
                             'chassis_sim/chassis_geometry.py', 'chassis_sim/terrain.py')}
            asset = None if self.field_asset is None else dict(path=str(self.field_asset),
                model_sha256=hashlib.sha256((self.field_asset/'model.glb').read_bytes()).hexdigest(),
                config=json.loads((self.field_asset/'config.json').read_text(encoding='utf-8')))
            context = dict(camera=self.camera, tags=self.tags, field_map=raw_field, runtime_config=raw,
                           renderer='field' if self.field_renderer else 'fast', asset=asset,
                           code_revision=revision, source_sha256=source_hashes,
                           camera_effects=asdict(self.sensor.settings), seed=getattr(args, 'seed', 509),
                           terrain=self.terrain.describe())
            context['planar_localization'] = not getattr(args,'free_pose',False)
            policy = CapturePolicy(position_m=getattr(args, 'capture_position_m', .2),
                heading_deg=getattr(args, 'capture_heading_deg', 5.),
                tilt_deg=getattr(args,'capture_tilt_deg',5.),
                cooldown_s=getattr(args, 'capture_cooldown', 5.), max_cases=getattr(args, 'capture_max_cases', 50),
                max_bytes=int(getattr(args, 'capture_max_mb', 1024)*1024*1024))
            self.capture = FailureCapture(getattr(args, 'capture_dir', HERE/'generated/captures'), self.root.name, context, policy)
            print(f"Failure captures: {self.capture.session}", flush=True)

    def configure_terrain(self, kind, spawn=None):
        spawn=self.spawn if spawn is None else spawn
        terrain = terrain_from_args(self.args,spawn,kind)
        mesh = terrain.mesh()
        self.terrain_configuration = (terrain,mesh)
        self.terrain,self.terrain_mesh = terrain,mesh
        self.initial = terrain.apply(spawn)

    def configure_camera(self, settings, assumed):
        """Run on the camera worker: prepare replacements before swapping them."""
        if settings==self.camera_settings:return
        projection=CameraProjection(settings['camera'],settings['width'],settings['height'],
            self.args.calibration_width,self.args.calibration_height,self.args.ideal_camera)
        floor_width,floor_height=max(2,projection.render_width//4),max(2,projection.render_height//4)
        floor_camera=dict(projection.render_camera)
        for key in ('fx','cx'):floor_camera[key]*=floor_width/projection.render_width
        for key in ('fy','cy'):floor_camera[key]*=floor_height/projection.render_height
        rays=camera_rays(floor_camera,floor_width,floor_height)
        manifest=dict(self.manifest,width=projection.width,height=projection.height,camera=projection.camera,
            robot_mount=None if self.args.free_pose else dict(assumed))
        renderer=None
        replay=None
        try:
            if self.field_renderer is not None:
                from field_renderer import FieldRenderer
                renderer=FieldRenderer(self.field_asset,projection.render_camera,projection.render_width,
                    projection.render_height,self.tags,self.textures)
            replay=StreamReplay(self.command[0],manifest,self.env,self.root)
        except Exception:
            if renderer is not None:renderer.close()
            if replay is not None:replay.close()
            raise
        old_replay,old_renderer=self.replay,self.field_renderer
        self.projection,self.camera,self.manifest=projection,projection.camera,manifest
        self.source_camera=dict(settings['camera'])
        self.width,self.height=projection.width,projection.height
        self.render_camera=projection.render_camera
        self.render_width,self.render_height=projection.render_width,projection.render_height
        self.floor_camera,self.floor_width,self.floor_height,self.rays=floor_camera,floor_width,floor_height,rays
        self.replay,self.field_renderer=replay,renderer
        self.camera_settings=dict(camera=dict(settings['camera']),width=self.width,height=self.height)
        old_replay.close()
        if old_renderer is not None:old_renderer.close()

    def close(self):
        if self.capture is not None:
            self.capture.close()
        self.replay.close()
        if self.field_renderer is not None:
            self.field_renderer.close()

    def localize(self, chassis, actual, assumed, timestamp=None, motion=(0., 0., 0.), realtime=True, camera_settings=None):
        if camera_settings is not None:self.configure_camera(camera_settings,assumed)
        started = time.perf_counter()
        terrain,terrain_mesh = self.terrain_configuration
        expected_mount = None if getattr(self.args,'free_pose',False) else assumed
        if self.manifest['robot_mount'] != expected_mount:
            self.replay.close()
            self.manifest['robot_mount'] = None if expected_mount is None else dict(expected_mount)
            self.replay = StreamReplay(self.command[0], self.manifest, self.env, self.root)
        from failure_capture import utc_now
        captured_utc = utc_now()
        position, rotation = camera_pose(chassis, actual)
        frame = {"name": "live", "image": "camera.png", "case": {},
                 "camera_position": position.tolist(), "camera_rotation": rotation.tolist(),
                 "expected_ids": []}
        if self.field_renderer is not None:
            image, visibility = self.field_renderer.render(position, rotation, with_visibility=True,
                                                           terrain_vertices=terrain_mesh)
        else:
            background = ground_background(position, rotation, self.floor_camera, self.floor_width,
                                           self.floor_height, self.field, self.rays)
            background = cv2.resize(background, (self.render_width, self.render_height), interpolation=cv2.INTER_LINEAR)
            image, expected = render_fast(frame, self.tags, self.render_camera, self.render_width, self.render_height,
                                   self.textures, background=background, roi=True)
            # Fast rendering has no opaque field structures. Test only the known
            # fully visible tags; retain all mapped IDs as allowed for cropped tags.
            visibility = [dict(id=t['id'], visible_fraction=1., clearly_visible=t['id'] in expected)
                          for t in self.tags]
        image = self.projection.output(image)
        visibility = self.projection.visibility(visibility,position,rotation,self.tags,self.camera['tag_size_m'])
        timestamp = started-self.sensor_started if timestamp is None else timestamp
        image, effects = self.sensor.apply(image, timestamp, motion, self.camera['fx'])
        rendered_at = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            report = (dict(joint_pose=None, poses=[], detected_ids=[]) if effects['dropped_frame']
                      else self.replay.request(gray))
        except Exception as error:
            if self.capture is None:
                raise
            report = dict(joint_pose=None, poses=[], detected_ids=[], pipeline_error=str(error))
        report["render_ms"] = (rendered_at-started)*1000
        report["vortex_ms"] = (time.perf_counter()-rendered_at)*1000
        report['sensor'] = effects
        report['sensor_settings'] = asdict(self.sensor.settings)
        report['terrain'] = terrain.describe()
        report['planar_localization'] = self.manifest['robot_mount'] is not None
        report['visibility'] = visibility
        report['camera'] = dict(self.camera)
        estimate = report["joint_pose"]
        if estimate is None and report["poses"]:
            estimate = min(report["poses"], key=lambda p: p["reprojection_rmse_px"])
        ghost = None if estimate is None else robot_pose(
            estimate["camera_position"], estimate["camera_rotation"], assumed)
        if self.capture is not None and not effects['dropped_frame']:
            self.capture.observe(dict(gray=gray, image=image, ghost=ghost, chassis=chassis.copy(),
                report=dict(report), actual_mount=dict(actual), assumed_mount=dict(assumed),
                camera_position=position.tolist(), camera_rotation=rotation.tolist(), visibility=visibility,
                camera=dict(self.camera),
                monotonic_s=started, captured_utc=captured_utc))
            report['capture'] = self.capture.status()
        if realtime and effects['injected_latency_ms'] > 0:
            time.sleep(effects['injected_latency_ms']/1000)
        return ghost, image, report, chassis.copy(), time.perf_counter() - started + (
            0 if realtime else effects['injected_latency_ms']/1000)


class Telemetry:
    def __init__(self, root, port3=1735, port4=5810):
        import ntcore
        from wpimath.geometry import Pose3d
        # A second NT server can silently fail to bind and leave AdvantageScope
        # connected to the other simulator, with competing-looking pose updates.
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port4))
            except OSError as error:
                raise RuntimeError(f"NT4 port {port4} is already in use. Close the other simulator first.") from error
        self.instance = ntcore.NetworkTableInstance.create()
        self.instance.startServer(str(root / "networktables.json"), "127.0.0.1", port3, port4)
        self.actual = self.instance.getStructTopic("/VortexSim/Actual", Pose3d).publish(
            ntcore.PubSubOptions(periodic=.02, keepDuplicates=True, sendAll=True))
        self.ghost = self.instance.getStructArrayTopic("/VortexSim/Ghost", Pose3d).publish(
            ntcore.PubSubOptions(periodic=.02, keepDuplicates=True, sendAll=True))
        self.raw_vision = self.instance.getStructArrayTopic("/VortexSim/RawVision", Pose3d).publish(
            ntcore.PubSubOptions(periodic=.02, keepDuplicates=True, sendAll=True))
        self.camera_pose = self.instance.getStructTopic("/VortexSim/Camera", Pose3d).publish(
            ntcore.PubSubOptions(periodic=.02, keepDuplicates=True, sendAll=True))
        self.entries = {name: self.instance.getEntry("/VortexSim/" + name) for name in
                        ("Valid", "PositionErrorM", "HeadingErrorDeg", "LatencyMs", "PoseAgeMs",
                         "RenderMs", "VortexMs", "CameraFPS", "LoopFPS", "CapturesSaved", "CaptureDropped",
                         "PredictedValid", "PredictionHorizonMs", "TiltErrorDeg", "HeightErrorM",
                         "ChassisPitchDeg", "ChassisRollDeg", "ChassisHeightM", "LocalizationModel", "Terrain")}
        self.result_times = deque(maxlen=100)
        self.entries["Valid"].setBoolean(False)
        self.ghost.set([])
        self.raw_vision.set([])
        self.entries["PredictedValid"].setBoolean(False)
        self.rotation_entries={kind:{key:self.instance.getEntry(f'/VortexSim/Rotation/{kind}/{key}')
            for key in ('RollDeg','PitchDeg','YawDeg','Valid')} for kind in ('Actual','RawVision','Ghost')}
        for kind in self.rotation_entries:self.publish_rotation(kind,None)

    @staticmethod
    def pose(position, rotation):
        from wpimath.geometry import Pose3d, Rotation3d, Translation3d
        return Pose3d(Translation3d(*map(float, position)), Rotation3d(np.asarray(rotation)))

    def publish_actual(self, chassis, mount=None):
        actual=chassis_transform(chassis)
        self.actual.set(self.pose(*actual))
        self.publish_rotation('Actual',actual)
        for name,value in zip(('ChassisHeightM','ChassisRollDeg','ChassisPitchDeg'),
                              (chassis[3],math.degrees(chassis[4]),math.degrees(chassis[5])) if len(chassis)>=6 else (0,0,0)):
            self.entries[name].setDouble(float(value))
        if mount is not None:
            position, optical_rotation = camera_pose(chassis, mount)
            # AdvantageScope's camera poses use forward/left/up, not optical axes.
            optical_from_nwu = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            self.camera_pose.set(self.pose(position, optical_rotation @ optical_from_nwu))
        self.instance.flush()

    def publish_result(self, result, display_raw=True):
        ghost, _, report, captured, latency = result
        ghost=ghost if valid_pose(ghost) else None
        self.publish_rotation('RawVision',ghost)
        self.entries["Valid"].setBoolean(ghost is not None)
        self.entries["LatencyMs"].setDouble(latency * 1000)
        model = 'unknown' if 'planar_localization' not in report else 'planar' if report['planar_localization'] else 'free'
        self.entries['LocalizationModel'].setString(model)
        self.entries['Terrain'].setString(report.get('terrain',{}).get('kind','unknown'))
        self.entries["RenderMs"].setDouble(report.get("render_ms", 0.))
        self.entries["VortexMs"].setDouble(report.get("vortex_ms", 0.))
        capture = report.get('capture', {})
        self.entries['CapturesSaved'].setDouble(capture.get('saved', 0))
        self.entries['CaptureDropped'].setDouble(capture.get('dropped', 0))
        now = time.perf_counter()
        self.result_times.append(now)
        while len(self.result_times) > 2 and self.result_times[0] < now-1:
            self.result_times.popleft()
        if len(self.result_times) > 1:
            self.entries["CameraFPS"].setDouble((len(self.result_times)-1) /
                                               max(now-self.result_times[0], 1e-6))
        self.raw_vision.set([] if ghost is None else [self.pose(*ghost)])
        # Headless callers display raw vision; the GUI updates this at drive rate.
        if display_raw:
            self.publish_display(ghost, 0.)
        if ghost is not None:
            p, r = ghost
            metrics=chassis_errors(ghost,captured)
            self.entries["PositionErrorM"].setDouble(metrics['position_m'])
            self.entries['TiltErrorDeg'].setDouble(metrics['tilt_deg'])
            self.entries['HeightErrorM'].setDouble(metrics['height_m'])
            heading = math.atan2(r[1, 0], r[0, 0]) - captured[2]
            self.entries["HeadingErrorDeg"].setDouble(math.degrees(math.atan2(math.sin(heading), math.cos(heading))))
        else:
            for name in ("PositionErrorM", "HeadingErrorDeg", "TiltErrorDeg", "HeightErrorM"):
                self.entries[name].setDouble(float("nan"))
        self.instance.flush()

    def publish_display(self, ghost, horizon):
        ghost=ghost if valid_pose(ghost) else None
        self.publish_rotation('Ghost',ghost)
        self.ghost.set([] if ghost is None else [self.pose(*ghost)])
        self.entries["PredictedValid"].setBoolean(ghost is not None)
        self.entries["PredictionHorizonMs"].setDouble(horizon*1000 if ghost is not None else float('nan'))
        self.instance.flush()

    def publish_rotation(self,kind,pose):
        entries=self.rotation_entries[kind]
        good=valid_pose(pose)
        entries['Valid'].setBoolean(good)
        if good:
            rotation=self.pose(*pose).rotation()
            values=[math.degrees(value) for value in (rotation.X(),rotation.Y(),rotation.Z())]
        else:
            values=[float('nan')]*3
        for name,value in zip(('RollDeg','PitchDeg','YawDeg'),values):
            entries[name].setDouble(value)

    def clear_vision(self):
        self.raw_vision.set([])
        self.publish_rotation('RawVision',None)
        self.entries['Valid'].setBoolean(False)
        for name in ('PositionErrorM','HeadingErrorDeg','TiltErrorDeg','HeightErrorM','PoseAgeMs'):
            self.entries[name].setDouble(float('nan'))


def gui(harness, telemetry):
    import tkinter as tk
    from tkinter import ttk
    timer = None
    if os.name == "nt":
        # Windows' default timer granularity can turn a 20 ms Tk callback into
        # a roughly 31 ms update. Restore the process timer request on exit.
        import ctypes
        timer = ctypes.WinDLL("winmm")
        if timer.timeBeginPeriod(1) != 0:
            timer = None
    root = tk.Tk()
    root.title("Vortex chassis simulator")
    root.geometry(f"{min(1000, root.winfo_screenwidth()-80)}x{min(850, root.winfo_screenheight()-100)}")
    root.minsize(600, 450)
    for column in range(4):
        root.columnconfigure(column, weight=1)
    root.rowconfigure(10, weight=1)
    camera_header=tk.StringVar(value=f"Camera {harness.args.camera_index}: {harness.width}×{harness.height}, "
              f"{'ideal lens' if harness.args.ideal_camera else 'configured lens'} — open AdvantageScope for the 3D field.\n"
              "Click camera to drive: W/S forward, A/D strafe, Q/E or Left/Right turn. Space stops.")
    ttk.Label(root,textvariable=camera_header).grid(columnspan=4)
    variables = {}
    for col, (title, mount) in enumerate((("Camera mount", harness.actual), ("Calibration error (+meters / +degrees)", harness.calibration_error))):
        ttk.Label(root, text=title).grid(row=1, column=col * 2, columnspan=2)
        for row, key in enumerate(MOUNT_KEYS, 2):
            ttk.Label(root, text=key).grid(row=row, column=col * 2)
            var = tk.StringVar(value=str(mount[key]))
            ttk.Entry(root, textvariable=var, width=12).grid(row=row, column=col * 2 + 1)
            variables[col, key] = var
    status = tk.StringVar(value="Connect AdvantageScope to 127.0.0.1 (NT4)")
    status_label = ttk.Label(root, textvariable=status, wraplength=900)
    status_label.grid(row=9, columnspan=4, sticky='ew')
    # Canvas image items don't set the widget's requested size. Camera resolution
    # therefore cannot force the window to grow or prevent it from shrinking.
    preview = tk.Canvas(root, takefocus=True, background='#181818',
                        width=1, height=1, highlightthickness=0)
    preview.grid(row=10, columnspan=4, sticky='nsew')
    preview_item = preview.create_image(0, 0, anchor='center')
    preview_frame = None
    def redraw_preview(event=None):
        if preview_frame is None:
            return
        width, height = max(1, preview.winfo_width()), max(1, preview.winfo_height())
        source_height, source_width = preview_frame.shape[:2]
        scale = min(width/source_width, height/source_height)
        size = (max(1, round(source_width*scale)), max(1, round(source_height*scale)))
        image = preview_frame if size == (source_width, source_height) else cv2.resize(
            preview_frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        # Tk accepts uncompressed PPM; avoid PNG compression in the drive loop.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ppm = f"P6\n{size[0]} {size[1]}\n255\n".encode() + rgb.tobytes()
        preview.image = tk.PhotoImage(data=ppm, format="PPM")
        preview.itemconfigure(preview_item, image=preview.image)
        preview.coords(preview_item, width/2, height/2)
    preview.bind('<Configure>', redraw_preview)
    notification = tk.StringVar(value='')
    banner = tk.Label(root, textvariable=notification, anchor='w', justify='left',
                      wraplength=min(harness.width, 960), padx=8, pady=5)
    banner.grid(row=11, columnspan=4, sticky='ew')
    banner.grid_remove()
    def resize_text(event):
        if event.widget is root:
            status_label.configure(wraplength=max(1, event.width-16))
            banner.configure(wraplength=max(1, event.width-16))
    root.bind('<Configure>', resize_text)
    last_capture = None
    notification_until = 0.
    def open_capture():
        if last_capture is not None:
            if os.name == 'nt':
                os.startfile(last_capture)
            else:
                subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', last_capture])
    open_button = ttk.Button(root, text='Open last capture', command=open_capture, state='disabled')
    open_button.grid(row=12, column=0, columnspan=2)
    capture_count = tk.StringVar(value='Captures: 0')
    ttk.Label(root, textvariable=capture_count).grid(row=12, column=2, columnspan=2)
    from routes import ROUTES, MOUNT_PRESETS, Route, mounts_for
    controls = ttk.Frame(root)
    controls.grid(row=13, columnspan=4, sticky='ew')
    ttk.Label(controls, text='Camera').pack(side='left')
    profile_var = tk.StringVar(value=harness.args.camera_profile)
    ttk.Combobox(controls, textvariable=profile_var, values=list(PROFILES), state='readonly', width=11).pack(side='left')
    ttk.Label(controls, text='Mount').pack(side='left')
    mount_var = tk.StringVar(value='matched')
    mount_box = ttk.Combobox(controls, textvariable=mount_var, values=MOUNT_PRESETS, state='readonly', width=17)
    mount_box.pack(side='left')
    preset_base=dict(harness.actual)
    def apply_mount_preset(event=None):
        actual,assumed=mounts_for(mount_var.get(),preset_base)
        for key in MOUNT_KEYS:
            variables[0,key].set(str(actual[key]))
            variables[1,key].set(str(assumed[key]-actual[key]))
    mount_box.bind('<<ComboboxSelected>>', apply_mount_preset)
    terrain_routes = {'field': 'blue-lower-bump', 'custom-ramp': 'ramp-crossing', 'custom-cross-ramp': 'cross-ramp-crossing'}
    route_var = tk.StringVar(value=terrain_routes.get(harness.terrain.kind, 'approach'))
    ttk.Combobox(controls, textvariable=route_var, values=list(ROUTES), state='readonly', width=20).pack(side='left')
    active_route = None
    route_started = 0.
    active_profile = harness.args.camera_profile
    preview.bind("<Button-1>", lambda _: preview.focus_set())
    keys = set()
    predictor = GhostPredictor()
    generation = 0
    pending_generation = 0
    preview.bind("<KeyPress>", lambda e: keys.add(e.keysym.lower()))
    preview.bind("<KeyRelease>", lambda e: keys.discard(e.keysym.lower()))
    preview.bind("<FocusOut>", lambda _: keys.clear())
    def invalidate_vision():
        nonlocal generation, result_captured_at
        generation += 1
        predictor.clear()
        result_captured_at = None
        telemetry.clear_vision()
        telemetry.publish_display(None, 0.)
    def reset():
        nonlocal active_route
        active_route=None
        keys.clear()
        harness.chassis=harness.initial.copy()
        invalidate_vision()
    ttk.Button(root, text="Reset chassis", command=reset).grid(row=8, column=0)
    def start_route():
        nonlocal active_route, route_started
        route = Route.load(route_var.get(), ROUTES[route_var.get()])
        harness.configure_terrain(harness.terrain.kind,route.prepare_initial(harness.spawn,harness.field))
        reset()
        active_route = route.timeline(harness.initial, harness.field, terrain=harness.terrain)
        route_started = time.perf_counter()
        preview.focus_set()
    ttk.Button(controls, text='Run route', command=start_route).pack(side='left')
    def stop_route():
        nonlocal active_route
        active_route = None
        keys.clear()
    ttk.Button(controls, text='Stop', command=stop_route).pack(side='left')
    terrain_controls=ttk.Frame(root)
    terrain_controls.grid(row=14,columnspan=4,sticky='ew')
    ttk.Label(terrain_controls,text='Terrain').pack(side='left')
    terrain_var=tk.StringVar(value=harness.terrain.kind)
    terrain_box=ttk.Combobox(terrain_controls,textvariable=terrain_var,values=TERRAINS,state='readonly',width=20)
    terrain_box.pack(side='left')
    attitude_label=tk.StringVar()
    ttk.Label(terrain_controls,textvariable=attitude_label).pack(side='left')
    def change_terrain(event=None):
        harness.configure_terrain(terrain_var.get())
        reset()
        if terrain_var.get() in terrain_routes:
            route_var.set(terrain_routes[terrain_var.get()])
            status.set('Click Run route to cross the selected bump and return. Field mode enables all four blue/red bumps.')
    terrain_box.bind('<<ComboboxSelected>>',change_terrain)
    model_controls=ttk.Frame(root)
    model_controls.grid(row=15,columnspan=4,sticky='ew')
    ttk.Label(model_controls,text='Localization').pack(side='left')
    model_var=tk.StringVar(value='Free 6-DoF' if harness.args.free_pose else 'Level chassis')
    model_box=ttk.Combobox(model_controls,textvariable=model_var,values=('Level chassis','Free 6-DoF'),state='readonly',width=15)
    model_box.pack(side='left')
    ttk.Label(model_controls,text='3D rotations: NT /VortexSim/Rotation (degrees)').pack(side='left')
    def change_model(event=None):
        harness.args.free_pose=model_var.get()=='Free 6-DoF'
        invalidate_vision()
    model_box.bind('<<ComboboxSelected>>',change_model)
    def reset_calibration_error():
        for key in MOUNT_KEYS:variables[1,key].set('0.0')
    ttk.Button(root, text="Reset calibration error", command=reset_calibration_error).grid(row=8, column=2)
    def read_mounts():
        try:
            actual,error = [{k: float(variables[col, k].get()) for k in MOUNT_KEYS} for col in range(2)]
            if not all(math.isfinite(v) for m in (actual,error) for v in m.values()):
                raise ValueError()
            return actual,apply_calibration_error(actual,error)
        except ValueError:
            status.set("Mount and calibration error values must be finite numbers; camera sampling paused.")
            return None
    calibration_controls=ttk.LabelFrame(root,text=f'Live camera calibration (intrinsics at {harness.args.calibration_width}×{harness.args.calibration_height}; config file watched)')
    calibration_controls.grid(row=16,columnspan=4,sticky='ew')
    source_profile=dict(harness.source_camera)
    calibration_vars={}
    for index,key in enumerate((*CALIBRATION_KEYS,'width','height')):
        value=source_profile[key] if key in CALIBRATION_KEYS else getattr(harness,key)
        variable=tk.StringVar(value=str(value))
        calibration_vars[key]=variable
        row,column=divmod(index,4)
        ttk.Label(calibration_controls,text=key).grid(row=row,column=column*2)
        ttk.Entry(calibration_controls,textvariable=variable,width=12).grid(row=row,column=column*2+1,sticky='ew')
    def read_camera_settings():
        try:
            camera=dict(source_profile)
            camera.update({key:float(calibration_vars[key].get()) for key in CALIBRATION_KEYS})
            width,height=(int(calibration_vars[key].get()) for key in ('width','height'))
            if not all(math.isfinite(camera[key]) for key in CALIBRATION_KEYS) or min(camera['fx'],camera['fy'],camera['tag_size_m'],width,height)<=0:
                raise ValueError()
            return dict(camera=camera,width=width,height=height)
        except ValueError:
            status.set('Enter finite calibration values and positive focal lengths, tag size, width and height; camera sampling paused.')
            return None
    fields_changed_at=-math.inf
    failed_generation=None
    def camera_fields_changed(*_):
        nonlocal fields_changed_at,preview_frame
        fields_changed_at=time.perf_counter()
        invalidate_vision()
        preview_frame=None
        preview.itemconfigure(preview_item,image='')
        status.set('Updating camera settings…')
    for variable in (*variables.values(),*calibration_vars.values()):
        variable.trace_add('write',camera_fields_changed)
    config_watcher=CameraConfigWatcher(harness.args.config,harness.args.camera_index)
    next_config_poll=0.
    pool = ThreadPoolExecutor(max_workers=1)
    pending = None
    last = time.perf_counter()
    physics_at = last+.02
    sample_at = 0.
    captured_at = None
    result_captured_at = None
    def tick():
        nonlocal pending, last, physics_at, sample_at, captured_at, result_captured_at, last_capture, notification_until, pending_generation, preview_frame, active_route, active_profile, next_config_poll, source_profile, failed_generation
        now = time.perf_counter()
        if now>=next_config_poll:
            profile=config_watcher.poll(now)
            next_config_poll=now+.2
            if profile is not None:
                try:
                    profile.update({key:float(profile.get(key,0.)) for key in (*CALIBRATION_KEYS,*MOUNT_KEYS)})
                    if not all(math.isfinite(profile[key]) for key in (*CALIBRATION_KEYS,*MOUNT_KEYS)) or min(profile['fx'],profile['fy'],profile['tag_size_m'])<=0:
                        raise ValueError('invalid camera calibration')
                    for key in CALIBRATION_KEYS:
                        if profile[key]!=source_profile.get(key):calibration_vars[key].set(str(profile[key]))
                    if harness.args.mounts is None:
                        for key in MOUNT_KEYS:
                            if profile[key]!=source_profile.get(key):
                                preset_base[key]=profile[key]
                                variables[0,key].set(str(profile[key]))
                    source_profile=profile
                except (ValueError,TypeError) as error:
                    config_watcher.applied=None
                    config_watcher.error=f'Camera config reload: {error}'
        if harness.capture is not None:
            counts = harness.capture.status()
            capture_count.set(f"Captures: {counts['total']} total ({counts['saved']} this session)")
            for event in harness.capture.take_notifications():
                last_capture = event['path']
                action = 'updated' if event['updated'] else 'captured'
                reasons = ', '.join(reason.replace('_', ' ') for reason in event['reasons'])
                category = ' [mount experiment]' if event['experiment'] else ''
                notification.set(f"Bad pose {action}{category}: {reasons}\n{last_capture}")
                banner.configure(background='#fff3cd', foreground='#664d03')
                banner.grid()
                open_button.configure(state='normal')
                notification_until = now+2
            if notification_until and now >= notification_until:
                notification.set('')
                banner.grid_remove()
                notification_until = 0.
        else:
            capture_count.set('Captures disabled')
        if "space" in keys:
            active_route = None
            keys.clear()
        commands = ((int("w" in keys)-int("s" in keys))*2,
                    (int("a" in keys)-int("d" in keys))*2,
                    (int(bool(keys & {"q", "left"}))-int(bool(keys & {"e", "right"})))*1.5)
        if active_route is not None:
            elapsed = now-route_started
            commands = active_route.command(elapsed)
            if elapsed >= active_route.times[-1]:
                harness.chassis = active_route.poses[-1].copy()
                active_route = None
        # Poll camera completion more often than physics: otherwise a 17 ms
        # worker takes two 20 ms callbacks and caps sampling at 25 FPS.
        if now >= physics_at:
            physics_at = max(physics_at+.02, now)
            previous = last
            dt, last = min(now-last, .1), now
            telemetry.entries["LoopFPS"].setDouble(1 / max(now-previous, 1e-6))
            if now-dt > previous:
                predictor.record(previous, now-dt, 0., 0., 0.)
            predictor.record(now-dt, now, *commands)
            harness.chassis = (active_route.pose(now-route_started) if active_route is not None else
                               harness.terrain.apply(step(harness.chassis, *commands, dt, harness.field)))
        live_mounts = read_mounts()
        telemetry.publish_actual(harness.chassis, live_mounts[0] if live_mounts else None)
        state=harness.chassis
        attitude_label.set(f" Height {state[3]:.2f} m | roll {math.degrees(state[4]):.1f}° | pitch {math.degrees(state[5]):.1f}°" if len(state)>=6 else ' Level ground')
        if pending is not None and pending.done():
            if pending_generation != generation:
                pending = None
                root.after(1, tick)
                return
            try:
                result = pending.result()
                ghost, image, report, _, latency = result
                if not report.get('sensor', {}).get('dropped_frame'):
                    telemetry.publish_result(result, display_raw=False)
                    result_captured_at = captured_at
                    predictor.observe(ghost, captured_at)
                preview_frame = image
                redraw_preview()
                camera_header.set(f"Camera {harness.args.camera_index}: {image.shape[1]}×{image.shape[0]}, "
                    f"{'ideal lens' if harness.args.ideal_camera else 'configured lens'} — open AdvantageScope for the 3D field.\n"
                    'Click camera to drive: W/S forward, A/D strafe, Q/E or Left/Right turn. Space stops.')
                state = 'frame dropped' if report.get('sensor', {}).get('dropped_frame') else 'pose valid' if ghost is not None else 'no pose'
                status.set(f"Tags: {report['detected_ids']} | {state} | "
                           f"frame {latency*1000:.0f} ms (render {report['render_ms']:.0f}, Vortex {report['vortex_ms']:.0f})")
                if 'capture' in report:
                    info = report['capture']
                    status.set(status.get()+f" | captures {info['saved']}"+(f" | {info['error']}" if info['error'] else ''))
                if report.get('pipeline_error'):
                    status.set(status.get()+f" | {report['pipeline_error']}")
            except Exception as error:
                predictor.observe(None, now)
                result_captured_at = None
                telemetry.clear_vision()
                telemetry.publish_display(None,0.)
                failed_generation=generation
                status.set(str(error))
            pending = None
        if result_captured_at is not None:
            telemetry.entries["PoseAgeMs"].setDouble((now-result_captured_at)*1000)
        # Commands cover physics ticks, not time spent displaying the preview.
        telemetry.publish_display(predictor.predict(last) if result_captured_at is not None and
                                  now-result_captured_at <= predictor.max_age else None,
                                  last-result_captured_at if result_captured_at is not None else 0.)
        if pending is None and now >= sample_at and now-fields_changed_at>=.3 and failed_generation!=generation:
            if profile_var.get() != active_profile:
                active_profile = profile_var.get()
                harness.sensor = Sensor(PROFILES[active_profile], harness.args.seed)
            mounts = live_mounts
            camera_settings=read_camera_settings()
            if mounts and camera_settings:
                pending = pool.submit(harness.localize, harness.chassis.copy(), *mounts, motion=commands,camera_settings=camera_settings)
                captured_at = last
                pending_generation = generation
            sample_at = max(sample_at + 1 / harness.args.camera_hz, now)
        if config_watcher.error:status.set(config_watcher.error)
        root.after(max(1, 5-round((time.perf_counter()-now)*1000)), tick)
    def close():
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    try:
        tick()
        root.mainloop()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        harness.close()
        telemetry.instance.stopServer()
        if timer is not None:
            timer.timeEndPeriod(1)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE.parent / "config/config.json")
    parser.add_argument("--field", type=Path, default=HERE.parent / "config/2026-rebuilt-welded.json")
    parser.add_argument("--camera-index", type=int, default=0)
    mounts = parser.add_mutually_exclusive_group()
    mounts.add_argument("--mounts", type=Path, help="Load the camera mount only; saved assumed mounts and calibration errors are ignored")
    mounts.add_argument('--demo-mount', action='store_const', dest='mounts', const=HERE/'demo_mounts.json', help='Use the upright demo mount')
    mounts.add_argument("--use-config-mount", action="store_true", help="Use mount from production camera config")
    parser.add_argument("--advantagescope", type=Path, help="Also launch an installed AdvantageScope executable")
    parser.add_argument("--launch-advantagescope", action="store_true", help="Find and launch AdvantageScope")
    parser.add_argument("--width", type=int, help='Output width (default: calibration width)')
    parser.add_argument('--height', type=int, help='Output height (default: preserve calibration aspect ratio)')
    parser.add_argument('--ideal-camera', action='store_true', help='Disable lens distortion for idealized regression controls')
    parser.add_argument("--calibration-width", type=int, default=1920)
    parser.add_argument("--calibration-height", type=int, default=1080)
    parser.add_argument("--start-tag", type=int, default=18)
    parser.add_argument("--start-distance", type=float, default=1.5, help="Initial horizontal distance from target tag, meters")
    parser.add_argument("--camera-hz", type=float, default=40)
    parser.add_argument("--renderer", choices=('field', 'fast'), default='field',
                        help="Depth-tested full field (default), or lightweight tag-only fixtures")
    parser.add_argument("--field-asset", type=Path, help="AdvantageScope Field3d asset folder or model.glb")
    parser.add_argument('--no-capture', action='store_true', help='Disable automatic regression capture')
    parser.add_argument('--capture-dir', type=Path, default=HERE/'generated/captures')
    parser.add_argument('--capture-position-m', type=float, default=.2)
    parser.add_argument('--capture-heading-deg', type=float, default=5.)
    parser.add_argument('--capture-tilt-deg', type=float, default=5.)
    parser.add_argument('--capture-cooldown', type=float, default=5., help='Minimum seconds between case writes')
    parser.add_argument('--capture-max-cases', type=int, default=50, help='Maximum new cases per session')
    parser.add_argument('--capture-max-mb', type=float, default=1024., help='Total capture directory size limit')
    parser.add_argument("--mingw-root", type=Path, default=Path("C:/msys64/mingw64"))
    parser.add_argument("--replay-executable", type=Path)
    parser.add_argument("--headless-frames", type=int, default=0, help="Finite integration check without GUI")
    parser.add_argument('--camera-profile', choices=PROFILES, default='clean')
    parser.add_argument('--seed', type=int, default=509)
    parser.add_argument('--free-pose', action='store_true', default=True, help='Use unconstrained 6-DoF pose (interactive default)')
    parser.add_argument('--planar-pose', action='store_false', dest='free_pose', help='Use the level-chassis model as a comparison control')
    parser.add_argument('--terrain', type=terrain_name, choices=TERRAINS, default='field', help='Field mode follows all four 2026 bumps; flat is a level-ground control')
    parser.add_argument('--bump-smoothing', type=float, default=.18, help='Field bump contact smoothing radius in meters (0 disables, default .18)')
    parser.add_argument('--chassis-pitch', type=float, default=0., help='Nose-up chassis pitch bias, degrees')
    parser.add_argument('--chassis-roll', type=float, default=0., help='Left-side-up chassis roll bias, degrees')
    parser.add_argument('--chassis-height', type=float, default=0., help='Base chassis elevation, meters')
    parser.add_argument('--tilt-deg', type=float, default=10., help='Static pitch/roll terrain preset angle')
    parser.add_argument('--ramp-height', type=float, default=RAMP_HEIGHT)
    parser.add_argument('--ramp-length', type=float, default=RAMP_LENGTH, help='Horizontal run of each ramp face, meters')
    parser.add_argument('--ramp-width', type=float, default=RAMP_WIDTH)
    parser.add_argument('--ramp-crest', type=float, default=0., help='Optional flat crest length, meters')
    for key in asdict(CameraEffects()):
        parser.add_argument('--'+key.replace('_', '-'), type=float,
                            help=f'Override camera profile {key}')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.use_config_mount:
        args.mounts = None
    if args.launch_advantagescope and not args.advantagescope:
        args.advantagescope = find_advantagescope()
        if args.advantagescope is None:
            parser.error("AdvantageScope not found. Install it with: winget install --exact "
                         "--id MechanicalAdvantage.AdvantageScope --source winget "
                         "(or provide --advantagescope PATH)")
    if args.advantagescope and not args.advantagescope.is_file():
        parser.error("--advantagescope must name an installed executable")
    args.max_error_meters = .20
    from failure_capture import CapturePolicy
    try:
        terrain_from_args(args,[0.,0.,0.])
        settings = asdict(PROFILES[args.camera_profile])
        settings.update({key: getattr(args, key) for key in settings if getattr(args, key) is not None})
        CameraEffects(**settings)
        if args.seed < 0:
            raise ValueError('Seed must be nonnegative')
        CapturePolicy(position_m=args.capture_position_m, heading_deg=args.capture_heading_deg,
                      tilt_deg=args.capture_tilt_deg,
                      cooldown_s=args.capture_cooldown, max_cases=args.capture_max_cases,
                      max_bytes=int(args.capture_max_mb*1024*1024))
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    if min(args.width if args.width is not None else 1,args.height if args.height is not None else 1, args.calibration_width, args.calibration_height) <= 0 or not math.isfinite(args.camera_hz) or args.camera_hz <= 0 or args.headless_frames < 0 or not math.isfinite(args.start_distance) or args.start_distance <= 0:
        parser.error("Dimensions and rate must be positive; headless frame count cannot be negative")
    harness = Harness(args)
    print(f'Camera {args.camera_index}: {harness.width}x{harness.height}, '
          f'{"ideal" if args.ideal_camera else "configured"} lens, '
          f'{"override" if args.mounts else "config"} mount',flush=True)
    try:
        telemetry = Telemetry(harness.root)
    except Exception:
        harness.close()
        raise
    print(f"NT4 server: 127.0.0.1 | Session: {harness.root}", flush=True)
    if args.advantagescope:
        subprocess.Popen([str(args.advantagescope.resolve())])
    if args.headless_frames:
        try:
            for _ in range(args.headless_frames):
                telemetry.publish_actual(harness.chassis, harness.actual)
                result = harness.localize(harness.chassis, harness.actual, harness.assumed)
                telemetry.publish_result(result)
                print(f"Tags {result[2]['detected_ids']}, valid={result[0] is not None}", flush=True)
        finally:
            harness.close()
            telemetry.instance.stopServer()
    else:
        gui(harness, telemetry)


if __name__ == "__main__":
    main()
