"""Deterministic surface contact for kinematic ramp and tilt experiments."""
from dataclasses import dataclass, asdict
import math
import numpy as np

TERRAINS = ('field', 'flat', 'pitch', 'roll', 'custom-ramp', 'custom-cross-ramp')
# Simplified REBUILT bump: 6-inch rise, 15-degree faces, 73-inch width.
# Omits the HDPE thickness/lips included in the official 6.513-inch envelope.
RAMP_HEIGHT = 6 * .0254
RAMP_LENGTH = RAMP_HEIGHT / math.tan(math.radians(15))
RAMP_WIDTH = 73 * .0254


@dataclass(frozen=True)
class FieldBump:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    height: float = .165423

    @property
    def center(self):
        return ((self.x_min+self.x_max)/2, (self.y_min+self.y_max)/2)

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def surface(self, x, y):
        return max(0., self.height-abs(x-self.center[0])*math.tan(math.radians(15))) if self.contains(x,y) else 0.

    def contact_surface(self, x, y, radius):
        if radius == 0:
            return self.surface(x,y)
        slope=math.tan(math.radians(15))
        half_run=self.height/slope
        dx=x-self.center[0]
        if abs(dx)>=half_run+radius or y<=self.y_min-radius or y>=self.y_max+radius:
            return 0.
        # Convolve a triangular face profile with a triangular contact window.
        # This preserves the straight 15-degree faces but rounds their joins.
        def ramp(t):
            if t<=-radius:return 0.
            if t>=radius:return t
            return (t+radius)**3/(6*radius**2) if t<0 else t+(radius-t)**3/(6*radius**2)
        def edge(t):
            if t<=-radius:return 0.
            if t>=radius:return 1.
            return (max(t+radius,0.)**2-2*max(t,0.)**2+max(t-radius,0.)**2)/(2*radius**2)
        height=slope*(ramp(dx+half_run)-2*ramp(dx)+ramp(dx-half_run))
        return max(0.,height)*(edge(y-self.y_min)-edge(y-self.y_max))


# GE-26101 Bump Plastic bounds after the same wall-blue transform as the
# renderer, from AdvantageScope Field3d_2026FRCFieldV2. The contact plane omits
# screw holes and the 1.6 mm seam; the existing CAD remains the rendered surface.
FIELD_BUMPS = (
    FieldBump('blue-lower',4.061776,5.189412,1.584160,3.440423),
    FieldBump('blue-upper',4.061776,5.189412,4.628573,6.484836),
    FieldBump('red-lower',11.351576,12.479212,1.584160,3.440423),
    FieldBump('red-upper',11.351576,12.479212,4.628573,6.484836),
)


def terrain_name(value):
    # Older launch commands now select the real field bumps.
    return 'field' if value in ('ramp','cross-ramp') else value


@dataclass
class Terrain:
    kind: str = 'flat'
    anchor_x: float = 0.
    anchor_y: float = 0.
    heading: float = 0.
    height: float = RAMP_HEIGHT
    length: float = RAMP_LENGTH
    width: float = RAMP_WIDTH
    crest: float = 0.
    tilt_deg: float = 10.
    pitch_deg: float = 0.
    roll_deg: float = 0.
    elevation: float = 0.
    bump_smoothing: float = .18

    def __post_init__(self):
        self.kind={'ramp':'custom-ramp','cross-ramp':'custom-cross-ramp'}.get(self.kind,self.kind)
        if self.kind not in TERRAINS:
            raise ValueError('Unknown terrain')
        if not all(math.isfinite(v) for k,v in asdict(self).items() if k != 'kind'):
            raise ValueError('Terrain parameters must be finite')
        if not 0 <= self.height <= 2 or self.length < .2 or self.width < .5:
            raise ValueError('Ramp height must be 0..2 m, slope length >= .2 m, width >= .5 m')
        if not 0 <= self.crest <= 2:
            raise ValueError('Ramp crest length must be 0..2 m')
        if not 0 <= self.bump_smoothing <= .3:
            raise ValueError('Bump smoothing radius must be 0..0.3 m')
        if max(abs(self.tilt_deg),abs(self.pitch_deg),abs(self.roll_deg)) > 45 or self.elevation < 0:
            raise ValueError('Chassis tilt must be within 45 degrees; elevation nonnegative')

    def axes(self):
        angle = self.heading + (math.pi/2 if self.kind == 'custom-cross-ramp' else math.pi)
        axis = np.array([math.cos(angle),math.sin(angle)])
        return axis, np.array([-axis[1],axis[0]])

    def describe(self):
        if self.kind == 'field':
            return dict(kind=self.kind,layout='AdvantageScope-2026FRCFieldV2',
                        bumps=[asdict(b) for b in FIELD_BUMPS],elevation=self.elevation,
                        pitch_deg=self.pitch_deg,roll_deg=self.roll_deg,bump_smoothing=self.bump_smoothing)
        return asdict(self)

    def surface(self, x, y):
        if self.kind == 'field':
            return self.elevation+max(b.contact_surface(x,y,self.bump_smoothing) for b in FIELD_BUMPS)
        if self.kind not in ('custom-ramp','custom-cross-ramp'):
            return self.elevation
        axis, left = self.axes()
        delta = np.array([x-self.anchor_x,y-self.anchor_y])
        u, v = float(delta@axis)-.25, abs(float(delta@left))
        # Straight bump faces with an optional flat crest. Unlike smoothstep,
        # linear faces do not increase the peak slope above height / length.
        def smooth(t):
            t = np.clip(t,0.,1.)
            return t*t*(3-2*t)
        longitudinal = float(np.clip(min(u/self.length,(2*self.length+self.crest-u)/self.length),0.,1.))
        lateral = smooth((self.width/2-v)/.25)
        return self.elevation+float(self.height*longitudinal*lateral)

    def apply(self, state):
        if self.kind == 'flat' and not (self.pitch_deg or self.roll_deg or self.elevation) and len(state) == 3:
            return np.array(state,dtype=float)
        x,y,yaw = state[:3]
        h=self.surface(x,y)
        epsilon=1e-4
        gx=(self.surface(x+epsilon,y)-self.surface(x-epsilon,y))/(2*epsilon)
        gy=(self.surface(x,y+epsilon)-self.surface(x,y-epsilon))/(2*epsilon)
        if self.kind == 'field' and self.bump_smoothing == 0:
            # Panel edge lips are steps, not near-vertical driveable faces.
            # Use the 15-degree contact normal, rather than differentiating the lip.
            bump=next((b for b in FIELD_BUMPS if b.contains(x,y)),None)
            gx=0. if bump is None or abs(x-bump.center[0])<epsilon else math.copysign(math.tan(math.radians(15)),bump.center[0]-x)
            gy=0.
        up=np.array([-gx,-gy,1.]);up/=np.linalg.norm(up)
        forward=np.array([math.cos(yaw),math.sin(yaw),gx*math.cos(yaw)+gy*math.sin(yaw)])
        forward/=np.linalg.norm(forward)
        left=np.cross(up,forward)
        rotation=np.column_stack((forward,left,up))
        pitch=math.atan2(rotation[2,0],math.hypot(rotation[0,0],rotation[1,0]))
        roll=math.atan2(rotation[2,1],rotation[2,2])
        pitch+=math.radians(self.pitch_deg+(self.tilt_deg if self.kind=='pitch' else 0))
        roll+=math.radians(self.roll_deg+(self.tilt_deg if self.kind=='roll' else 0))
        return np.array([x,y,yaw,h,roll,pitch])

    def mesh(self):
        if self.kind not in ('custom-ramp','custom-cross-ramp') or not self.height:
            return None
        axis,left=self.axes()
        vertices=[]
        us=np.unique(np.r_[np.linspace(.25,2*self.length+self.crest+.25,65),
                           .25+self.length,.25+self.length+self.crest])
        vs=np.linspace(-self.width/2,self.width/2,13)
        for i in range(len(us)-1):
            for j in range(len(vs)-1):
                quad=[]
                for u,v in ((us[i],vs[j]),(us[i+1],vs[j]),(us[i+1],vs[j+1]),(us[i],vs[j+1])):
                    xy=np.array([self.anchor_x,self.anchor_y])+u*axis+v*left
                    quad.append([*xy,self.surface(*xy)+.002,0,0,1,.8,.48,.12,1])
                vertices.extend(quad[k] for k in (0,1,2,0,2,3))
        return np.asarray(vertices,dtype='f4')


def from_args(args, initial, kind=None):
    return Terrain(kind=terrain_name(kind or getattr(args,'terrain','field')),anchor_x=float(initial[0]),anchor_y=float(initial[1]),
        heading=float(initial[2]),height=getattr(args,'ramp_height',RAMP_HEIGHT),length=getattr(args,'ramp_length',RAMP_LENGTH),
        width=getattr(args,'ramp_width',RAMP_WIDTH),crest=getattr(args,'ramp_crest',0.),tilt_deg=getattr(args,'tilt_deg',10.),
        pitch_deg=getattr(args,'chassis_pitch',0.),roll_deg=getattr(args,'chassis_roll',0.),
        elevation=getattr(args,'chassis_height',0.),bump_smoothing=getattr(args,'bump_smoothing',.18))
