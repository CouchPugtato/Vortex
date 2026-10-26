"""Chassis state: legacy [x,y,yaw] or [x,y,yaw,z,roll,pitch], radians/meters.

Positive pitch raises the nose; positive roll raises the left side.
"""
import math
import numpy as np


def valid_pose(pose):
    if pose is None:
        return False
    p,r=map(np.asarray,pose)
    return (p.shape==(3,) and r.shape==(3,3) and np.isfinite(p).all() and np.isfinite(r).all()
            and np.allclose(r.T@r,np.eye(3),atol=1e-3) and abs(np.linalg.det(r)-1)<1e-3)


def transform(state):
    x, y, yaw = state[:3]
    z, roll, pitch = state[3:6] if len(state) >= 6 else (0., 0., 0.)
    c, s = math.cos(yaw), math.sin(yaw)
    cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    rotation = (np.array([[c,-s,0],[s,c,0],[0,0,1.]]) @
                np.array([[cp,0,-sp],[0,1,0],[sp,0,cp]]) @
                np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]))
    return np.array([x,y,z]), rotation


def errors(estimate, truth):
    if estimate is None:
        return dict(position_m=None, heading_deg=None, tilt_deg=None, height_m=None)
    p, r = map(np.asarray, estimate)
    if p.shape != (3,) or r.shape != (3,3) or not np.isfinite(p).all() or not np.isfinite(r).all():
        return dict(position_m=None, heading_deg=None, tilt_deg=None, height_m=None)
    target, rotation = transform(truth)
    angle = math.atan2(r[1,0],r[0,0])-truth[2]
    return dict(position_m=float(np.linalg.norm(p-target)),
                heading_deg=abs(math.degrees(math.atan2(math.sin(angle),math.cos(angle)))),
                tilt_deg=math.degrees(math.acos(float(np.clip(r[:,2]@rotation[:,2],-1,1)))),
                height_m=abs(float(p[2]-target[2])))
