"""Fixed-step command routes; all coordinates start at the selected tag spawn."""
from dataclasses import dataclass
import math

import numpy as np
from terrain import FIELD_BUMPS


def alliance_survey():
    """Slow rotating loops in both halves of the blue zone, then a bump crossing.

    Red uses the same robot-relative commands from a mirrored starting pose.
    Sample world paths into body commands without giving truth to localization.
    """
    segments=[]
    point=np.array([3.15,2.])
    yaw=0.
    spin=2*math.pi/48

    def follow(seconds, target, turn=spin):
        nonlocal point,yaw
        count=math.ceil(seconds/.1)
        dt=seconds/count
        for i in range(count):
            next_point=np.asarray(target((i+1)/count))
            velocity=(next_point-point)/dt
            mid=yaw+turn*dt/2
            c,s=math.cos(mid),math.sin(mid)
            segments.append((dt,c*velocity[0]+s*velocity[1],-s*velocity[0]+c*velocity[1],turn))
            point=next_point;yaw+=turn*dt

    def transfer(destination, speed=.18):
        start=point.copy();destination=np.asarray(destination)
        follow(float(np.linalg.norm(destination-start))/speed,lambda t:start+(destination-start)*t)

    for y in (2.,6.):
        if y==6.:transfer([3.15,y])
        follow(96,lambda t:np.array([2.35+.8*math.cos(4*math.pi*t),y+.85*math.sin(4*math.pi*t)]))
    bump=next(b for b in FIELD_BUMPS if b.name=='blue-upper')
    transfer([bump.center[0]-1.1,bump.center[1]])
    delta=math.atan2(math.sin(-math.pi/2-yaw),math.cos(-math.pi/2-yaw))
    if abs(delta)>1e-8:
        stationary=point.copy()
        follow(abs(delta)/spin,lambda t:stationary,math.copysign(spin,delta))
    # Face the hub and strafe across the bump into the neutral zone.
    segments.extend([(5.5,0,.4,0),(48,0,0,spin),(2,0,0,0)])
    return segments


ROUTES = {
    **{side+'-alliance-survey':alliance_survey() for side in ('blue','red')},
    # Face the hub while strafing across each bump, keeping its nearby tags in
    # view during the tilted portion instead of looking downfield at tiny tags.
    **{b.name+'-bump': [(.5,0,0,0),(4,0,(-.55 if b.name in ('blue-lower','red-upper') else .55),0),
                       (.5,0,0,0),(4,0,(.55 if b.name in ('blue-lower','red-upper') else -.55),0),
                       (1,0,0,0)] for b in FIELD_BUMPS},
    # A 2.2 m crossing clears the default 2026-sized bump before reversing.
    'ramp-crossing': [(.5,0,0,0),(4,-.55,0,0),(.5,0,0,0),(4,.55,0,0),(1,0,0,0)],
    'cross-ramp-crossing': [(.5,0,0,0),(4,0,.55,0),(.5,0,0,0),(4,0,-.55,0),(1,0,0,0)],
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

    def prepare_initial(self, initial, field):
        """Reserve room for built-in terrain routes; preserve other spawns."""
        result=np.array(initial,dtype=float)
        if self.name in ('blue-alliance-survey','red-alliance-survey'):
            if field['length']<16 or field['width']<7.8:
                raise ValueError('Alliance survey requires a full-size 2026 field')
            result[:3]=([3.15,2.,0.] if self.name.startswith('blue') else
                        [field['length']-3.15,field['width']-2.,math.pi])
            return result
        bump=next((b for b in FIELD_BUMPS if self.name==b.name+'-bump'),None)
        if bump is not None:
            direction=1 if bump.name.startswith('blue') else -1
            result[:3]=[bump.center[0]-direction*1.1,bump.center[1],
                        math.pi/2 if bump.name.endswith('lower') else -math.pi/2]
            if not (.4 <= result[0] <= field['length']-.4 and .4 <= result[1] <= field['width']-.4):
                raise ValueError('2026 bump route does not fit inside this field')
            return result
        if self.name not in ('ramp-crossing','cross-ramp-crossing'):
            return result
        _,forward,left,_=next(s for s in self.segments if s[1] or s[2])
        duration=next(s[0] for s in self.segments if s[1] or s[2])
        c,s=math.cos(result[2]),math.sin(result[2])
        displacement=duration*np.array([c*forward-s*left,s*forward+c*left])
        for axis,size in enumerate((field['length'],field['width'])):
            lo=.4-min(0.,displacement[axis]);hi=size-.4-max(0.,displacement[axis])
            if lo>hi:raise ValueError('Terrain route does not fit inside the field')
            result[axis]=np.clip(result[axis],lo,hi)
        return result

    def command(self, timestamp):
        start = 0.
        for duration, *motion in self.segments:
            if start <= timestamp < start+duration:
                return tuple(motion)
            start += duration
        return (0., 0., 0.)

    def timeline(self, initial, field, dt=.02, terrain=None):
        # Split at segment boundaries so short custom segments aren't skipped.
        from drive import step
        times, poses, commands = [0.], [np.array(initial, dtype=float)], []
        start = 0.
        for duration, *motion in self.segments:
            end = start+duration
            while times[-1] < end-1e-10:
                t = min(times[-1]+dt, end)
                commands.append(tuple(motion))
                pose = step(poses[-1], *motion, t-times[-1], field)
                poses.append(terrain.apply(pose) if terrain is not None else pose)
                times.append(t)
            start = end
        return Trajectory(np.array(times), np.array(poses), commands, field, terrain)


class Trajectory:
    def __init__(self, times, poses, commands, field, terrain=None):
        self.times, self.poses, self.commands, self.field = times, poses, commands, field
        self.terrain = terrain

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
        pose=step(self.poses[i], *self.commands[i], timestamp-self.times[i], self.field)
        return self.terrain.apply(pose) if self.terrain is not None else pose

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
