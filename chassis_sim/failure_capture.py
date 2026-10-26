"""Bounded, asynchronous capture of reproducible simulation failures."""
from dataclasses import asdict, dataclass
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
from uuid import uuid4

import cv2
import numpy as np
from chassis_geometry import errors as chassis_errors


@dataclass
class CapturePolicy:
    position_m: float = .20
    heading_deg: float = 5.
    tilt_deg: float = 5.
    confirm_frames: int = 2
    recovery_frames: int = 2
    cooldown_s: float = 5.
    movement_m: float = .5
    turn_deg: float = 10.
    max_cases: int = 50
    max_bytes: int = 1024*1024*1024
    max_replacements: int = 2

    def __post_init__(self):
        for name in ('position_m', 'heading_deg', 'tilt_deg', 'movement_m', 'turn_deg'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not math.isfinite(self.cooldown_s) or self.cooldown_s < 0:
            raise ValueError('cooldown_s must be nonnegative and finite')
        for name in ('confirm_frames', 'recovery_frames', 'max_cases', 'max_bytes'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.max_replacements < 0:
            raise ValueError('max_replacements must be nonnegative')


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def classify(ghost, chassis, report, visibility, known_ids, policy):
    reasons = []
    labels = visibility or []
    expected = sorted(v['id'] for v in labels if v.get('clearly_visible'))
    # Very small or partly visible tags remain allowed. Only an entirely unseen
    # field tag, or an ID outside the map, counts as an invented detection.
    allowed = sorted(v['id'] for v in labels if v.get('visible_fraction', 0) > 0)
    detected = set(report.get('detected_ids', []))
    if report.get('pipeline_error'):
        reasons.append('pipeline_error')
    unexpected = sorted(detected-set(allowed)) if visibility is not None else sorted(detected-set(known_ids))
    if unexpected:
        reasons.append('unexpected_tag')
    position_error = heading_error = tilt_error = height_error = None
    if ghost is None:
        if expected:
            reasons.append('missing_pose_with_visible_tags')
    else:
        position, rotation = map(np.asarray, ghost)
        if (position.shape != (3,) or rotation.shape != (3, 3) or not np.all(np.isfinite(position))
                or not np.all(np.isfinite(rotation)) or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-3)
                or abs(np.linalg.det(rotation)-1) > 1e-3):
            reasons.append('nonfinite_or_invalid_pose')
        else:
            metrics=chassis_errors(ghost,chassis)
            position_error,tilt_error,height_error=metrics['position_m'],metrics['tilt_deg'],metrics['height_m']
            angle = math.atan2(rotation[1, 0], rotation[0, 0])-chassis[2]
            heading_error = abs(math.degrees(math.atan2(math.sin(angle), math.cos(angle))))
            if position_error > policy.position_m:
                reasons.append('position_error')
            if heading_error > policy.heading_deg:
                reasons.append('heading_error')
            if len(chassis)>=6 and tilt_error > policy.tilt_deg:
                reasons.append('tilt_error')
    severity = max(position_error/policy.position_m if position_error is not None else 0,
                   heading_error/policy.heading_deg if heading_error is not None else 0,
                   tilt_error/policy.tilt_deg if tilt_error is not None and len(chassis)>=6 else 0,
                   10 if any(r not in ('position_error', 'heading_error', 'tilt_error') for r in reasons) else 0)
    return dict(reasons=reasons, position_error_m=position_error, heading_error_deg=heading_error,
                tilt_error_deg=tilt_error,height_error_m=height_error,
                severity=severity, expected_ids=expected, allowed_ids=allowed, unexpected_ids=unexpected)


def fingerprint(record, result, context):
    signature = hashlib.sha256(json.dumps(json_safe(dict(camera=record.get('camera',context['camera']), size=record['gray'].shape[:2], tags=context['tags'],
        asset=context.get('asset'), actual=record['actual_mount'], assumed=record['assumed_mount'],
        terrain=record['report'].get('terrain'), planar=record['report'].get('planar_localization'))),
        sort_keys=True).encode()).hexdigest()
    thumbnail = cv2.resize(record['gray'], (32, 32), interpolation=cv2.INTER_AREA)
    return dict(signature=signature, chassis=list(map(float, record['chassis'])),
                reasons=sorted(result['reasons']), visible=result['allowed_ids'],
                thumbnail=thumbnail.tobytes().hex())


def same_scene(a, b, policy):
    if a['signature'] != b['signature']:
        return False
    p, q = a['chassis'], b['chassis']
    if len(p)>=6 and len(q)>=6 and (abs(p[3]-q[3]) >= policy.movement_m or
        max(abs(p[4]-q[4]),abs(p[5]-q[5])) >= math.radians(policy.turn_deg)):
        return False
    turn = abs(math.degrees(math.atan2(math.sin(p[2]-q[2]), math.cos(p[2]-q[2]))))
    return np.linalg.norm(np.asarray(p[:2])-q[:2]) < policy.movement_m and turn < policy.turn_deg


def duplicate(a, b, policy):
    if a['signature'] != b['signature'] or a['reasons'] != b['reasons'] or a['visible'] != b['visible']:
        return False
    if not same_scene(a, b, CapturePolicy(movement_m=policy.movement_m*2, turn_deg=policy.turn_deg*2)):
        return False
    aa = np.frombuffer(bytes.fromhex(a['thumbnail']), np.uint8).astype(float)
    bb = np.frombuffer(bytes.fromhex(b['thumbnail']), np.uint8).astype(float)
    return aa.shape == bb.shape and float(np.mean(np.abs(aa-bb))) <= 2.


class FailureCapture:
    def __init__(self, root, session, context, policy=None):
        self.root = Path(root).resolve()
        self.session = self.root/session
        self.context = json_safe(context)
        self.policy = policy or CapturePolicy()
        self.root.mkdir(parents=True, exist_ok=True)
        self.used_bytes = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        self.history = []
        for path in self.root.glob('*/**/capture.json'):
            try:
                self.history.append(json.loads(path.read_text(encoding='utf-8'))['fingerprint'])
            except (OSError, ValueError, KeyError):
                pass
        self.existing_count = len(self.history)
        self.jobs = queue.Queue(maxsize=2)
        self.lock = threading.Lock()
        self.notifications = deque(maxlen=16)
        self.accepted = self.saved = self.dropped = 0
        self.error = None
        self.last_save = -math.inf
        self.bad_streak = self.good_streak = 0
        self.episode = self.previous = None
        self.before_failure = None
        self.writer = threading.Thread(target=self._writer, name='failure-capture', daemon=True)
        self.writer.start()

    def status(self):
        with self.lock:
            return dict(saved=self.saved, total=self.existing_count+self.saved,
                        accepted=self.accepted, dropped=self.dropped, error=self.error,
                        path=str(self.session), pending=self.jobs.qsize())

    def take_notifications(self):
        """Only completed saves reach the UI; drain without blocking camera work."""
        with self.lock:
            events = list(self.notifications)
            self.notifications.clear()
        return events

    def observe(self, record):
        """Called by the camera worker, never by the drive/UI loop."""
        result = classify(record['ghost'], record['chassis'], record['report'], record['visibility'],
                          [t['id'] for t in self.context['tags']], self.policy)
        now = record['monotonic_s']
        previous = self.previous
        if previous is not None and record.get('camera',self.context['camera'])!=previous.get('camera',self.context['camera']):
            previous=None
            self.bad_streak=self.good_streak=0
            self.episode=None
        self.previous = record
        if not result['reasons']:
            self.bad_streak = 0
            self.good_streak += 1
            if self.good_streak >= self.policy.recovery_frames and self.episode is not None:
                self._queue(dict(kind='after', path=self.episode['path'], record=record))
                self.episode = None
            return result
        self.good_streak = 0
        if self.bad_streak == 0:
            self.before_failure = previous
        self.bad_streak += 1
        if self.bad_streak < self.policy.confirm_frames:
            return result
        mark = fingerprint(record, result, self.context)
        if self.episode is not None and same_scene(mark, self.episode['mark'], self.policy):
            episode = self.episode
            if (episode['replacements'] < self.policy.max_replacements and now-self.last_save >= self.policy.cooldown_s
                    and result['severity'] >= max(episode['severity']*1.5, episode['severity']+1)):
                if self._queue(dict(kind='replace', path=episode['path'], record=record, result=result,
                                    mark=mark, before=None)):
                    episode.update(mark=mark, severity=result['severity'], replacements=episode['replacements']+1)
                    self.last_save = now
            return result
        if now-self.last_save < self.policy.cooldown_s or self.accepted >= self.policy.max_cases:
            return result
        if any(duplicate(mark, saved, self.policy) for saved in self.history):
            return result
        experiment = any(abs(record['actual_mount'][k]-record['assumed_mount'][k]) > 1e-6 for k in record['actual_mount'])
        path = self.session/('experiments' if experiment else 'regressions')/f'case_{uuid4().hex[:12]}'
        before = self.before_failure
        if before is None or now-before['monotonic_s'] > 1:
            before = previous if previous and now-previous['monotonic_s'] <= 1 else None
        if self._queue(dict(kind='new', path=path, record=record, result=result, mark=mark,
                            before=before)):
            self.accepted += 1
            self.history.append(mark)
            self.episode = dict(path=path, mark=mark, severity=result['severity'], replacements=0)
            self.last_save = now
        return result

    def _queue(self, job):
        if self.error or self.used_bytes >= self.policy.max_bytes:
            return False
        try:
            job['context']=dict(self.context,camera=job['record'].get('camera',self.context['camera']))
            self.jobs.put_nowait(job)
            return True
        except queue.Full:
            with self.lock:
                self.dropped += 1
            return False

    @staticmethod
    def _png(image):
        success, png = cv2.imencode('.png', image)
        if not success:
            raise OSError('PNG encoding failed')
        return png.tobytes()

    def _write(self, job):
        path, record = job['path'], job['record']
        context=job.get('context',self.context)
        if job['kind'] == 'after':
            files = {'after.png': self._png(record['gray']),
                     'after.json': json.dumps(json_safe(self._frame_details(record)), indent=2).encode()}
        else:
            result, mark = job['result'], job['mark']
            experiment = path.parent.name == 'experiments'
            frame = dict(name=path.name, image='image.png', group='configuration_experiment' if experiment else 'captured_failure',
                case=dict(actual_mount=record['actual_mount'], assumed_mount=record['assumed_mount']),
                camera_position=record['camera_position'], camera_rotation=record['camera_rotation'],
                expected_ids=result['expected_ids'], allowed_ids=result['allowed_ids'])
            manifest = dict(schema_version=1, backend=context['renderer'], width=record['gray'].shape[1],
                            height=record['gray'].shape[0], camera=context['camera'], tags=context['tags'], frames=[frame])
            capture = dict(schema_version=1, review_status='unreviewed', category=frame['group'], policy=asdict(self.policy),
                           fingerprint=mark, failure=result, frame=self._frame_details(record), context=context)
            files = {'image.png': self._png(record['gray']), 'preview.png': self._png(record['image']),
                     'manifest.json': json.dumps(json_safe(manifest), indent=2).encode(),
                     'capture.json': json.dumps(json_safe(capture), indent=2).encode()}
            if job['before'] is not None:
                files['before.png'] = self._png(job['before']['gray'])
                files['before.json'] = json.dumps(json_safe(self._frame_details(job['before'])), indent=2).encode()
        old = sum((path/name).stat().st_size for name in files if (path/name).exists())
        added = sum(len(value) for value in files.values())-old
        if self.used_bytes+added > self.policy.max_bytes:
            with self.lock:
                self.error = 'Capture disk limit reached; saving paused'
            return
        path.mkdir(parents=True, exist_ok=True)
        for name, value in files.items():
            temporary = path/(name+'.tmp')
            temporary.write_bytes(value)
            temporary.replace(path/name)
        with self.lock:
            self.used_bytes += added
            if job['kind'] == 'new':
                self.saved += 1
            if job['kind'] in ('new', 'replace'):
                self.notifications.append(dict(path=str(path), updated=job['kind'] == 'replace',
                    reasons=job['result']['reasons'], experiment=path.parent.name == 'experiments'))

    @staticmethod
    def _frame_details(record):
        return dict(captured_utc=record['captured_utc'], monotonic_s=record['monotonic_s'],
                    chassis_pose=record['chassis'], camera_position=record['camera_position'],
                    camera_rotation=record['camera_rotation'], actual_mount=record['actual_mount'],
                    assumed_mount=record['assumed_mount'], predicted_robot_pose=record['ghost'],
                    terrain=record['report'].get('terrain'),
                    camera=record.get('camera'),
                    visibility=record['visibility'], result=record['report'])

    def _writer(self):
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                self._write(job)
            except Exception as error:
                with self.lock:
                    self.error = f'Capture write failed: {error}'
            finally:
                self.jobs.task_done()

    def close(self):
        self.jobs.put(None)
        self.writer.join()


def utc_now():
    return datetime.now(timezone.utc).isoformat()
