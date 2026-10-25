"""Fixed-step command routes; all coordinates start at the selected tag spawn."""
from dataclasses import dataclass
import math

import numpy as np


ROUTES = {
    'approach': [(1, 0, 0, 0), (2, -.35, 0, 0), (2, .35, 0, 0), (1, 0, 0, 0)],
    'slalom': [(1, 0, 0, 0), (2, 0, .5, .18), (2, 0, -.5, -.18), (1, 0, 0, 0)],
    'turn': [(1, 0, 0, 0), (2, 0, 0, .7), (2, 0, 0, -.7), (1, 0, 0, 0)],
    # Sweeps the camera behind/beside hub/trench geometry using the full renderer.
    # Kinematic paths can intersect CAD: these are visibility stress tests, not
    # collision-free autonomous robot trajectories.
    'obstacle-sweep': [(1, 0, 0, 0), (3, 0, .8, 0), (2, .6, 0, -.5),
                       (3, 0, -.8, 0), (2, -.6, 0, .5), (1, 0, 0, 0)],
}


@dataclass(frozen=True)
class Route:
    name: str
    segments: tuple

    @classmethod
    def load(cls, name, segments):
        if not segments:
            raise ValueError('Route must contain segments')
        checked = []
        for segment in segments:
            if len(segment) != 4 or not all(math.isfinite(v) for v in segment):
                raise ValueError('Each segment is [seconds, forward m/s, left m/s, turn rad/s]')
            if segment[0] <= 0:
                raise ValueError('Segment duration must be positive')
            checked.append(tuple(segment))
        if sum(s[0] for s in checked) > 600:
            raise ValueError('Route duration cannot exceed 600 seconds')
        return cls(name, tuple(checked))

    @property
    def duration(self):
        return sum(segment[0] for segment in self.segments)

    def command(self, timestamp):
        start = 0.
        for duration, *motion in self.segments:
            if start <= timestamp < start+duration:
                return tuple(motion)
            start += duration
        return (0., 0., 0.)

    def timeline(self, initial, field, dt=.02):
        # Split at segment boundaries so short custom segments aren't skipped.
        from drive import step
        times, poses, commands = [0.], [np.array(initial, dtype=float)], []
        start = 0.
        for duration, *motion in self.segments:
            end = start+duration
            while times[-1] < end-1e-10:
                t = min(times[-1]+dt, end)
                commands.append(tuple(motion))
                poses.append(step(poses[-1], *motion, t-times[-1], field))
                times.append(t)
            start = end
        return Trajectory(np.array(times), np.array(poses), commands, field)


class Trajectory:
    def __init__(self, times, poses, commands, field):
        self.times, self.poses, self.commands, self.field = times, poses, commands, field

    def command(self, timestamp):
        if not 0 <= timestamp < self.times[-1]:
            return (0., 0., 0.)
        return self.commands[int(np.searchsorted(self.times, timestamp, side='right')-1)]

    def pose(self, timestamp):
        from drive import step
        if timestamp <= 0:
            return self.poses[0].copy()
        if timestamp >= self.times[-1]:
            return self.poses[-1].copy()
        i = int(np.searchsorted(self.times, timestamp, side='right')-1)
        return step(self.poses[i], *self.commands[i], timestamp-self.times[i], self.field)

    def prediction(self, raw, captured_at, delivered_at):
        from ghost_prediction import GhostPredictor
        predictor = GhostPredictor()
        predictor.observe(raw, captured_at)
        # Only command history goes into prediction; truth is reserved for scoring.
        for i, motion in enumerate(self.commands):
            if self.times[i+1] <= captured_at:
                continue
            if self.times[i] >= delivered_at:
                break
            predictor.record(max(captured_at, self.times[i]), min(delivered_at, self.times[i+1]), *motion)
        if delivered_at > self.times[-1]:
            predictor.record(max(captured_at, self.times[-1]), delivered_at, 0, 0, 0)
        return predictor.predict(delivered_at)


MOUNT_PRESETS = ('matched', 'raised-tilted', 'rear-facing', 'translation-error', 'rotation-error')


def mounts_for(name, base):
    actual = dict(base)
    if name == 'raised-tilted':
        actual['z_offset'] += .2
        actual['yaw_deg'] += 10
    elif name == 'rear-facing':
        actual['pitch_deg'] += 180
    elif name not in MOUNT_PRESETS:
        raise ValueError(f'Unknown mount preset: {name}')
    assumed = dict(actual)
    if name == 'translation-error':
        assumed['x_offset'] += .15
    elif name == 'rotation-error':
        assumed['pitch_deg'] += 8
    return actual, assumed
