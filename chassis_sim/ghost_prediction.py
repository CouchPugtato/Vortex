"""Short display-only dead reckoning from vision and timestamped drive commands."""
from collections import deque
import math

import numpy as np
from chassis_geometry import valid_pose


class GhostPredictor:
    def __init__(self, max_age=.25):
        self.max_age = max_age
        self.commands = deque()
        self.clear()

    def clear(self):
        self.commands.clear()
        self.raw = None
        self.captured_at = None

    def record(self, start, end, forward, left, turn):
        self.commands.append((start, end, forward, left, turn))
        while self.commands and self.commands[0][1] < end-self.max_age:
            self.commands.popleft()

    def observe(self, raw, captured_at):
        self.raw = tuple(np.array(v,dtype=float,copy=True) for v in raw) if valid_pose(raw) else None
        self.captured_at = captured_at

    def predict(self, now):
        if self.raw is None or self.captured_at is None:
            return None
        age = now-self.captured_at
        if age < 0 or age > self.max_age or not all(np.isfinite(v).all() for v in self.raw):
            return None
        position, rotation = (v.copy() for v in self.raw)
        covered = self.captured_at
        for start, end, forward, left, turn in self.commands:
            if end <= covered or start >= now:
                continue
            # Missing history cannot be filled using ground truth or current keys.
            if start > covered+1e-8:
                return None
            dt = min(end, now)-max(start, covered)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            # Project drive commands through the estimated attitude, including
            # vertical travel. Assume the last estimated surface normal stays
            # fixed; no simulator terrain or actual attitude enters prediction.
            position += rotation @ np.array([forward,left,0.])*dt
            normal=rotation[:,2]
            if abs(normal[2])>1e-6:
                yaw+=turn*dt
                f=np.array([math.cos(yaw),math.sin(yaw),0.])
                f[2]=-(normal[:2]@f[:2])/normal[2]
                f/=np.linalg.norm(f)
                rotation=np.column_stack((f,np.cross(normal,f),normal))
            else:
                c,s=math.cos(turn*dt),math.sin(turn*dt)
                rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])@rotation
            covered = min(end, now)
        return (position, rotation) if covered >= now-1e-8 else None
