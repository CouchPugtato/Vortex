"""Seeded sensor approximations shared by live driving and route benchmarks."""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraEffects:
    exposure: float = 1.
    exposure_swing: float = 0.
    noise_std: float = 0.
    exposure_ms: float = 0.
    drop_rate: float = 0.
    latency_ms: float = 0.
    jitter_ms: float = 0.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.exposure <= 0 or self.exposure_swing > 1 or self.drop_rate > 1:
            raise ValueError('Exposure must be positive; swing and drop rate must be <= 1')
        if self.latency_ms + self.jitter_ms > 5000 or self.exposure_ms > 1000:
            raise ValueError('Latency plus jitter must be <= 5000 ms; exposure time <= 1000 ms')


PROFILES = {
    'clean': CameraEffects(),
    'dim': CameraEffects(exposure=.45, exposure_swing=.25, noise_std=5),
    'noisy': CameraEffects(noise_std=15, exposure_swing=.3),
    'motion': CameraEffects(exposure_ms=20, noise_std=2),
    'poor-radio': CameraEffects(drop_rate=.15, latency_ms=100, jitter_ms=40),
}


class Sensor:
    def __init__(self, settings, seed=509):
        self.settings = settings
        # Independent streams keep drops/delays stable when resolution changes.
        seeds = np.random.SeedSequence(seed).spawn(2)
        self.timing = np.random.default_rng(seeds[0])
        self.pixels = np.random.default_rng(seeds[1])

    def apply(self, image, timestamp, motion, fx):
        s = self.settings
        dropped = bool(self.timing.random() < s.drop_rate)
        delay = max(0., s.latency_ms + self.timing.uniform(-s.jitter_ms, s.jitter_ms)) / 1000
        result = image
        # Optical flow approximation at a nominal 2 m scene depth. Translation
        # along the optical axis is omitted; this is not temporal supersampling.
        forward, left, turn = motion
        smear = abs(fx * (turn + left / 2) * s.exposure_ms / 1000)
        length = min(101, max(1, round(smear)))
        if length > 1:
            kernel = np.ones((1, length), np.float32) / length
            result = cv2.filter2D(result, -1, kernel)
        gain = s.exposure * (1 + s.exposure_swing * math.sin(2 * math.pi * timestamp / 3))
        if gain != 1 or s.noise_std:
            values = result.astype(np.float32) * gain
            if s.noise_std:
                # Luminance-like noise shared by B/G/R, before grayscale conversion.
                values += self.pixels.normal(0, s.noise_std, (*image.shape[:2], 1))
            result = np.clip(values, 0, 255).astype(np.uint8)
        return result, dict(dropped_frame=dropped, injected_latency_ms=delay*1000,
                            blur_px=length, exposure_gain=gain)
