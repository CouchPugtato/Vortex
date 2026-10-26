"""Calibrated lens output from an overscanned, ideal pinhole render."""
import math
import json
import hashlib

import cv2
import numpy as np


DISTORTION_KEYS = ('k1','k2','p1','p2','k3')
CALIBRATION_KEYS = ('fx','fy','cx','cy',*DISTORTION_KEYS,'tag_size_m')


class CameraConfigWatcher:
    """Debounced file changes; an incomplete save never replaces a good profile."""
    def __init__(self,path,index):
        self.path,self.index=path,str(index)
        try:self.seen=self.applied=self.signature()
        except OSError:self.seen=self.applied=None
        self.changed_at=0.
        self.error=None

    def signature(self):
        stat=self.path.stat()
        return stat.st_mtime_ns,stat.st_size,hashlib.sha256(self.path.read_bytes()).digest()

    def poll(self,now):
        try:
            signature=self.signature()
            if signature!=self.seen:
                self.seen,self.changed_at=signature,now
                return None
            if signature==self.applied:
                self.error=None
                return None
            if now-self.changed_at<.3:return None
            raw=json.loads(self.path.read_text(encoding='utf-8'))
            profile=dict(raw.get('camera_profiles',{}).get(self.index,raw['camera']))
            self.applied=signature
            self.error=None
            return profile
        except (OSError,ValueError,KeyError,TypeError) as error:
            self.error=f'Camera config reload: {error}'
            return None


class CameraProjection:
    def __init__(self, camera, width=None, height=None, calibration_width=1920,
                 calibration_height=1080, ideal=False):
        if min(calibration_width,calibration_height)<=0:
            raise ValueError('Camera calibration dimensions must be positive')
        self.width = calibration_width if width is None else width
        self.height = round(calibration_height*self.width/calibration_width) if height is None else height
        if min(self.width,self.height,calibration_width,calibration_height)<=0:
            raise ValueError('Camera dimensions must be positive')
        self.camera = dict(camera)
        for key in ('fx','cx'):
            self.camera[key] *= self.width/calibration_width
        for key in ('fy','cy'):
            self.camera[key] *= self.height/calibration_height
        for key in DISTORTION_KEYS:
            self.camera[key] = 0. if ideal else float(camera.get(key,0.))
        if not all(math.isfinite(self.camera[k]) for k in CALIBRATION_KEYS) or min(self.camera['fx'],self.camera['fy'],self.camera['tag_size_m'])<=0:
            raise ValueError('Camera calibration must be finite with positive focal lengths')
        self.matrix = np.array([[self.camera['fx'],0.,self.camera['cx']],
                                [0.,self.camera['fy'],self.camera['cy']],[0.,0.,1.]])
        self.distortion = np.array([self.camera[k] for k in DISTORTION_KEYS])
        self.render_camera = dict(self.camera,**dict.fromkeys(DISTORTION_KEYS,0.))
        self.render_width,self.render_height = self.width,self.height
        self.maps = None
        if not np.any(self.distortion):
            return
        yy,xx = np.indices((self.height,self.width),dtype=np.float32)
        pixels = np.stack((xx,yy),axis=-1).reshape(-1,1,2)
        rays = cv2.undistortPointsIter(pixels,self.matrix,self.distortion,None,self.matrix,
            (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-7)).reshape(self.height,self.width,2)
        if not np.isfinite(rays).all():
            raise ValueError('Lens calibration produced nonfinite camera rays')
        low = np.minimum(0,np.floor(rays.min(axis=(0,1)))-2).astype(int)
        high = np.maximum([self.width-1,self.height-1],np.ceil(rays.max(axis=(0,1)))+2).astype(int)
        self.render_width,self.render_height = map(int,high-low+1)
        if self.render_width*self.render_height>16*self.width*self.height:
            raise ValueError('Lens calibration requires excessive overscan; check distortion and calibration dimensions')
        self.render_camera['cx'] -= int(low[0])
        self.render_camera['cy'] -= int(low[1])
        rays -= low.astype(np.float32)
        self.maps = cv2.convertMaps(rays[:,:,0],rays[:,:,1],cv2.CV_16SC2)

    def output(self, image):
        if image.shape[:2] != (self.render_height,self.render_width):
            raise ValueError('Rendered image does not match calibrated overscan size')
        return image if self.maps is None else cv2.remap(image,*self.maps,cv2.INTER_LINEAR)

    def visibility(self, labels, position, rotation, tags, tag_size):
        from generate import corners
        by_id = {tag['id']:tag for tag in tags}
        output = []
        for label in labels:
            tag = by_id[label['id']]
            points = (corners(tag,tag_size)-position)@rotation
            uv,_ = cv2.projectPoints(points,np.zeros(3),np.zeros(3),self.matrix,self.distortion)
            uv = uv.reshape(4,2).astype(np.float32)
            full = bool(np.all(points[:,2]>.02) and np.isfinite(uv).all() and
                np.all(uv[:,0]>8) and np.all(uv[:,0]<self.width-8) and
                np.all(uv[:,1]>8) and np.all(uv[:,1]<self.height-8))
            area = abs(cv2.contourArea(uv))
            output.append(dict(label,area_px=area,fully_in_frame=full,
                clearly_visible=full and area>=225 and label['visible_fraction']>=1.))
        return output
