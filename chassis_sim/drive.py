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

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "simulation_tests"))
from generate import render_fast, tag_geometry, tag_texture
from run import replay_command

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
    r = chassis_rotation(chassis[2])
    return (np.array([*chassis[:2], 0.]) + r @ np.array([mount[k] for k in MOUNT_KEYS[:3]]),
            r @ mount_rotation(mount))


def robot_pose(camera_position, camera_rotation, mount):
    r = np.asarray(camera_rotation) @ mount_rotation(mount).T
    p = np.asarray(camera_position) - r @ np.array([mount[k] for k in MOUNT_KEYS[:3]])
    return p, r


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
    delta = chassis_rotation(chassis[2]) @ np.array([forward, left, 0.]) * dt
    return np.array([np.clip(chassis[0] + delta[0], 0.4, field["length"] - 0.4),
                     np.clip(chassis[1] + delta[1], 0.4, field["width"] - 0.4),
                     math.atan2(math.sin(chassis[2] + turn * dt),
                                math.cos(chassis[2] + turn * dt))])


class Harness:
    def __init__(self, args):
        self.args = args
        raw = json.loads(args.config.read_text())
        self.camera = dict(raw.get("camera_profiles", {}).get(str(args.camera_index), raw["camera"]))
        self.actual = {k: float(self.camera.get(k, 0)) for k in MOUNT_KEYS}
        self.assumed = self.actual.copy()
        if args.mounts:
            saved = json.loads(args.mounts.read_text())
            self.actual, self.assumed = ({k: float(saved[name][k]) for k in MOUNT_KEYS}
                                         for name in ("actual", "assumed"))
            if not all(math.isfinite(v) for m in (self.actual, self.assumed) for v in m.values()):
                raise ValueError("Mount values must be finite")
        # Ideal pinhole renderer. Scale intrinsics with the original calibration resolution.
        scale = args.width / args.calibration_width
        for k in ("fx", "fy", "cx", "cy"):
            self.camera[k] *= scale
        for k in ("k1", "k2", "k3", "p1", "p2"):
            self.camera[k] = 0.
        self.width = args.width
        self.height = round(args.calibration_height * scale)
        if self.height < 1 or min(self.camera["fx"], self.camera["fy"]) <= 0:
            raise ValueError("Invalid image size or camera focal lengths")
        raw_field = json.loads(args.field.read_text())
        self.field = raw_field["field"]
        self.tags = [tag_geometry(t) for t in raw_field["tags"]]
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.textures = {t["id"]: tag_texture(dictionary, t["id"]) for t in self.tags}
        # Floor is only visual context; keep tags and Vortex at full resolution.
        self.floor_width, self.floor_height = max(2, self.width//4), max(2, self.height//4)
        self.floor_camera = dict(self.camera)
        for k in ("fx", "cx"):
            self.floor_camera[k] *= self.floor_width/self.width
        for k in ("fy", "cy"):
            self.floor_camera[k] *= self.floor_height/self.height
        self.rays = camera_rays(self.floor_camera, self.floor_width, self.floor_height)
        # Spawn in front of the hub, clear of the trench structure at the old 2 m spawn.
        tag = next(t for t in self.tags if t["id"] == args.start_tag)
        normal = np.array(tag["rotation"])[:, 0]
        forward = mount_rotation(self.actual)[:, 2]
        yaw = math.atan2(-normal[1], -normal[0]) - math.atan2(forward[1], forward[0])
        eye = np.array(tag["position"]) + getattr(args, 'start_distance', 1.5) * normal
        origin = eye - chassis_rotation(yaw) @ np.array([self.actual[k] for k in MOUNT_KEYS[:3]])
        self.initial = np.array([origin[0], origin[1], yaw])
        self.chassis = self.initial.copy()
        self.root = HERE / "generated" / f"session_{time.time_ns()}"
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
        manifest = dict(schema_version=1, width=self.width, height=self.height,
                        camera=self.camera, tags=self.tags, frames=[])
        self.replay = StreamReplay(self.command[0], manifest, self.env, self.root)
        self.field_renderer = None
        self.field_asset = None
        if getattr(args, 'renderer', 'field') == 'field':
            from field_renderer import FieldRenderer, find_field_asset
            try:
                self.field_asset = find_field_asset(getattr(args, 'field_asset', None))
                self.field_renderer = FieldRenderer(self.field_asset,
                    self.camera, self.width, self.height, self.tags, self.textures)
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
                             'chassis_sim/drive.py', 'chassis_sim/field_renderer.py', 'chassis_sim/failure_capture.py')}
            asset = None if self.field_asset is None else dict(path=str(self.field_asset),
                model_sha256=hashlib.sha256((self.field_asset/'model.glb').read_bytes()).hexdigest(),
                config=json.loads((self.field_asset/'config.json').read_text(encoding='utf-8')))
            context = dict(camera=self.camera, tags=self.tags, field_map=raw_field, runtime_config=raw,
                           renderer='field' if self.field_renderer else 'fast', asset=asset,
                           code_revision=revision, source_sha256=source_hashes)
            policy = CapturePolicy(position_m=getattr(args, 'capture_position_m', .2),
                heading_deg=getattr(args, 'capture_heading_deg', 5.),
                cooldown_s=getattr(args, 'capture_cooldown', 5.), max_cases=getattr(args, 'capture_max_cases', 50),
                max_bytes=int(getattr(args, 'capture_max_mb', 1024)*1024*1024))
            self.capture = FailureCapture(getattr(args, 'capture_dir', HERE/'generated/captures'), self.root.name, context, policy)
            print(f"Failure captures: {self.capture.session}", flush=True)

    def close(self):
        if self.capture is not None:
            self.capture.close()
        self.replay.close()
        if self.field_renderer is not None:
            self.field_renderer.close()

    def localize(self, chassis, actual, assumed):
        started = time.perf_counter()
        from failure_capture import utc_now
        captured_utc = utc_now()
        position, rotation = camera_pose(chassis, actual)
        frame = {"name": "live", "image": "camera.png", "case": {},
                 "camera_position": position.tolist(), "camera_rotation": rotation.tolist(),
                 "expected_ids": []}
        if self.field_renderer is not None:
            if self.capture is not None:
                image, visibility = self.field_renderer.render(position, rotation, with_visibility=True)
            else:
                image, visibility = self.field_renderer.render(position, rotation), []
        else:
            background = ground_background(position, rotation, self.floor_camera, self.floor_width,
                                           self.floor_height, self.field, self.rays)
            background = cv2.resize(background, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            image, expected = render_fast(frame, self.tags, self.camera, self.width, self.height,
                                   self.textures, background=background, roi=True)
            # Fast rendering has no opaque field structures. Test only the known
            # fully visible tags; retain all mapped IDs as allowed for cropped tags.
            visibility = [dict(id=t['id'], visible_fraction=1., clearly_visible=t['id'] in expected)
                          for t in self.tags]
        rendered_at = time.perf_counter()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            report = self.replay.request(gray)
        except Exception as error:
            if self.capture is None:
                raise
            report = dict(joint_pose=None, poses=[], detected_ids=[], pipeline_error=str(error))
        report["render_ms"] = (rendered_at-started)*1000
        report["vortex_ms"] = (time.perf_counter()-rendered_at)*1000
        estimate = report["joint_pose"]
        if estimate is None and report["poses"]:
            estimate = min(report["poses"], key=lambda p: p["reprojection_rmse_px"])
        ghost = None if estimate is None else robot_pose(
            estimate["camera_position"], estimate["camera_rotation"], assumed)
        if self.capture is not None:
            self.capture.observe(dict(gray=gray, image=image, ghost=ghost, chassis=chassis.copy(),
                report=dict(report), actual_mount=dict(actual), assumed_mount=dict(assumed),
                camera_position=position.tolist(), camera_rotation=rotation.tolist(), visibility=visibility,
                monotonic_s=started, captured_utc=captured_utc))
            report['capture'] = self.capture.status()
        return ghost, image, report, chassis.copy(), time.perf_counter() - started


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
        self.camera_pose = self.instance.getStructTopic("/VortexSim/Camera", Pose3d).publish(
            ntcore.PubSubOptions(periodic=.02, keepDuplicates=True, sendAll=True))
        self.entries = {name: self.instance.getEntry("/VortexSim/" + name) for name in
                        ("Valid", "PositionErrorM", "HeadingErrorDeg", "LatencyMs", "PoseAgeMs",
                         "RenderMs", "VortexMs", "CameraFPS", "LoopFPS", "CapturesSaved", "CaptureDropped")}
        self.result_times = deque(maxlen=100)
        self.entries["Valid"].setBoolean(False)
        self.ghost.set([])

    @staticmethod
    def pose(position, rotation):
        from wpimath.geometry import Pose3d, Rotation3d, Translation3d
        return Pose3d(Translation3d(*map(float, position)), Rotation3d(np.asarray(rotation)))

    def publish_actual(self, chassis, mount=None):
        self.actual.set(self.pose([*chassis[:2], 0.], chassis_rotation(chassis[2])))
        if mount is not None:
            position, optical_rotation = camera_pose(chassis, mount)
            # AdvantageScope's camera poses use forward/left/up, not optical axes.
            optical_from_nwu = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            self.camera_pose.set(self.pose(position, optical_rotation @ optical_from_nwu))
        self.instance.flush()

    def publish_result(self, result):
        ghost, _, report, captured, latency = result
        self.entries["Valid"].setBoolean(ghost is not None)
        self.entries["LatencyMs"].setDouble(latency * 1000)
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
        self.ghost.set([] if ghost is None else [self.pose(*ghost)])
        if ghost is not None:
            p, r = ghost
            self.entries["PositionErrorM"].setDouble(float(np.linalg.norm(p - [*captured[:2], 0.])))
            heading = math.atan2(r[1, 0], r[0, 0]) - captured[2]
            self.entries["HeadingErrorDeg"].setDouble(math.degrees(math.atan2(math.sin(heading), math.cos(heading))))
        else:
            for name in ("PositionErrorM", "HeadingErrorDeg"):
                self.entries[name].setDouble(float("nan"))
        self.instance.flush()


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
    ttk.Label(root, text="Camera preview — open AdvantageScope for the 3D field and chassis.\n"
              "Click camera to drive: W/S forward, A/D strafe, Q/E or Left/Right turn. Space stops.").grid(columnspan=4)
    variables = {}
    for col, (title, mount) in enumerate((("Actual mount", harness.actual), ("Vortex assumed mount", harness.assumed))):
        ttk.Label(root, text=title).grid(row=1, column=col * 2, columnspan=2)
        for row, key in enumerate(MOUNT_KEYS, 2):
            ttk.Label(root, text=key).grid(row=row, column=col * 2)
            var = tk.StringVar(value=str(mount[key]))
            ttk.Entry(root, textvariable=var, width=12).grid(row=row, column=col * 2 + 1)
            variables[col, key] = var
    status = tk.StringVar(value="Connect AdvantageScope to 127.0.0.1 (NT4)")
    ttk.Label(root, textvariable=status).grid(row=9, columnspan=4)
    preview = tk.Label(root, takefocus=True)
    preview.grid(row=10, columnspan=4)
    notification = tk.StringVar(value='')
    banner = tk.Label(root, textvariable=notification, anchor='w', justify='left',
                      wraplength=min(harness.width, 960), padx=8, pady=5)
    banner.grid(row=11, columnspan=4, sticky='ew')
    banner.grid_remove()
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
    preview.bind("<Button-1>", lambda _: preview.focus_set())
    keys = set()
    preview.bind("<KeyPress>", lambda e: keys.add(e.keysym.lower()))
    preview.bind("<KeyRelease>", lambda e: keys.discard(e.keysym.lower()))
    preview.bind("<FocusOut>", lambda _: keys.clear())
    def reset():
        keys.clear()
        harness.chassis = harness.initial.copy()
    ttk.Button(root, text="Reset chassis", command=reset).grid(row=8, column=0)
    def save():
        mounts = read_mounts()
        if mounts:
            (harness.root / "mounts.json").write_text(json.dumps(dict(actual=mounts[0], assumed=mounts[1]), indent=2))
            status.set(f"Saved mounts to {harness.root / 'mounts.json'}")
    ttk.Button(root, text="Save mounts", command=save).grid(row=8, column=2)
    def read_mounts():
        try:
            mounts = [{k: float(variables[col, k].get()) for k in MOUNT_KEYS} for col in range(2)]
            if not all(math.isfinite(v) for m in mounts for v in m.values()):
                raise ValueError()
            return mounts
        except ValueError:
            status.set("Mount values must be finite numbers; camera sampling paused.")
            return None
    pool = ThreadPoolExecutor(max_workers=1)
    pending = None
    last = time.perf_counter()
    sample_at = 0.
    captured_at = None
    result_captured_at = None
    def tick():
        nonlocal pending, last, sample_at, captured_at, result_captured_at, last_capture, notification_until
        now = time.perf_counter()
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
        dt, last = min(now - last, .1), now
        telemetry.entries["LoopFPS"].setDouble(1 / max(dt, 1e-6))
        if "space" in keys:
            keys.clear()
        harness.chassis = step(harness.chassis, (int("w" in keys)-int("s" in keys))*2,
                               (int("a" in keys)-int("d" in keys))*2,
                               (int(bool(keys & {"q", "left"}))-int(bool(keys & {"e", "right"})))*1.5, dt, harness.field)
        live_mounts = read_mounts()
        telemetry.publish_actual(harness.chassis, live_mounts[0] if live_mounts else None)
        if pending is not None and pending.done():
            try:
                result = pending.result()
                telemetry.publish_result(result)
                result_captured_at = captured_at
                ghost, image, report, _, latency = result
                # Tk accepts uncompressed PPM; avoid PNG compression in the drive loop.
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ppm = f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode() + rgb.tobytes()
                preview.image = tk.PhotoImage(data=ppm, format="PPM")
                preview.configure(image=preview.image)
                status.set(f"Tags: {report['detected_ids']} | {'pose valid' if ghost is not None else 'no pose'} | "
                           f"frame {latency*1000:.0f} ms (render {report['render_ms']:.0f}, Vortex {report['vortex_ms']:.0f})")
                if 'capture' in report:
                    info = report['capture']
                    status.set(status.get()+f" | captures {info['saved']}"+(f" | {info['error']}" if info['error'] else ''))
                if report.get('pipeline_error'):
                    status.set(status.get()+f" | {report['pipeline_error']}")
            except Exception as error:
                telemetry.ghost.set([])
                telemetry.entries["Valid"].setBoolean(False)
                for name in ("PositionErrorM", "HeadingErrorDeg"):
                    telemetry.entries[name].setDouble(float("nan"))
                status.set(str(error))
            pending = None
        if result_captured_at is not None:
            telemetry.entries["PoseAgeMs"].setDouble((now-result_captured_at)*1000)
        if pending is None and now >= sample_at:
            mounts = live_mounts
            if mounts:
                pending = pool.submit(harness.localize, harness.chassis.copy(), *mounts)
                captured_at = now
            sample_at = max(sample_at + 1 / harness.args.camera_hz, now)
        root.after(max(1, 20-round((time.perf_counter()-now)*1000)), tick)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE.parent / "config/config.json")
    parser.add_argument("--field", type=Path, default=HERE.parent / "config/2026-rebuilt-welded.json")
    parser.add_argument("--camera-index", type=int, default=0)
    mounts = parser.add_mutually_exclusive_group()
    mounts.add_argument("--mounts", type=Path, default=HERE / "demo_mounts.json",
                        help="Load actual/assumed mounts (default: upright demo camera)")
    mounts.add_argument("--use-config-mount", action="store_true", help="Use mount from production camera config")
    parser.add_argument("--advantagescope", type=Path, help="Also launch an installed AdvantageScope executable")
    parser.add_argument("--launch-advantagescope", action="store_true", help="Find and launch AdvantageScope")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--calibration-width", type=int, default=1920)
    parser.add_argument("--calibration-height", type=int, default=1080)
    parser.add_argument("--start-tag", type=int, default=18)
    parser.add_argument("--start-distance", type=float, default=1.5, help="Initial horizontal distance from target tag, meters")
    parser.add_argument("--camera-hz", type=float, default=20)
    parser.add_argument("--renderer", choices=('field', 'fast'), default='field',
                        help="Depth-tested full field (default), or lightweight tag-only fixtures")
    parser.add_argument("--field-asset", type=Path, help="AdvantageScope Field3d asset folder or model.glb")
    parser.add_argument('--no-capture', action='store_true', help='Disable automatic regression capture')
    parser.add_argument('--capture-dir', type=Path, default=HERE/'generated/captures')
    parser.add_argument('--capture-position-m', type=float, default=.2)
    parser.add_argument('--capture-heading-deg', type=float, default=5.)
    parser.add_argument('--capture-cooldown', type=float, default=5., help='Minimum seconds between case writes')
    parser.add_argument('--capture-max-cases', type=int, default=50, help='Maximum new cases per session')
    parser.add_argument('--capture-max-mb', type=float, default=1024., help='Total capture directory size limit')
    parser.add_argument("--mingw-root", type=Path, default=Path("C:/msys64/mingw64"))
    parser.add_argument("--replay-executable", type=Path)
    parser.add_argument("--headless-frames", type=int, default=0, help="Finite integration check without GUI")
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
        CapturePolicy(position_m=args.capture_position_m, heading_deg=args.capture_heading_deg,
                      cooldown_s=args.capture_cooldown, max_cases=args.capture_max_cases,
                      max_bytes=int(args.capture_max_mb*1024*1024))
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    if min(args.width, args.calibration_width, args.calibration_height) <= 0 or not math.isfinite(args.camera_hz) or args.camera_hz <= 0 or args.headless_frames < 0 or not math.isfinite(args.start_distance) or args.start_distance <= 0:
        parser.error("Dimensions and rate must be positive; headless frame count cannot be negative")
    harness = Harness(args)
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
