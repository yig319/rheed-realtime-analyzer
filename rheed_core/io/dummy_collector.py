from __future__ import annotations

import math
import time

import numpy as np

from rheed_core.core.types import FramePacket, SignalPacket
from rheed_core.io.collector_base import CollectorOutput, DataCollector


class DummyCollector(DataCollector):
    """Simulated intensity stream and synthetic image frames for pipeline testing."""

    def __init__(
        self,
        sample_rate_hz: float = 20.0,
        frame_rate_hz: float = 2.0,
        osc_period_s: float = 4.0,
        noise_std: float = 0.15,
        seed: int = 123,
    ):
        """Configure synthetic stream parameters for local development."""
        self.sample_rate_hz = sample_rate_hz
        self.frame_rate_hz = frame_rate_hz
        self.osc_period_s = osc_period_s
        self.noise_std = noise_std

        self._rng = np.random.default_rng(seed)
        self._start = time.monotonic()
        self._next_signal_ts = self._start
        self._next_frame_ts = self._start

    def poll(self) -> CollectorOutput:
        """Generate all due signal/frame events at current wall-clock time."""
        out = CollectorOutput()
        now = time.monotonic()

        while now >= self._next_signal_ts:
            t_rel = self._next_signal_ts - self._start
            intensity = self._gen_intensity(t_rel)
            out.signals.append(
                SignalPacket(ts=t_rel, name="intensity", value=float(intensity), meta={"source": "dummy"})
            )
            self._next_signal_ts += 1.0 / self.sample_rate_hz

        while now >= self._next_frame_ts:
            t_rel = self._next_frame_ts - self._start
            out.frames.append(FramePacket(ts=t_rel, img=self._gen_frame(t_rel), meta={"source": "dummy"}))
            self._next_frame_ts += 1.0 / self.frame_rate_hz

        return out

    def _gen_intensity(self, t: float) -> float:
        """Generate one synthetic oscillatory intensity sample."""
        envelope = 1.0 - 0.15 * (t / 120.0)
        oscillation = math.sin(2.0 * math.pi * (t / self.osc_period_s))
        noise = self._rng.normal(0.0, self.noise_std)
        baseline = 2.0
        return baseline + envelope * oscillation + noise

    def _gen_frame(self, t: float) -> np.ndarray:
        """Generate one synthetic elongated spot image with slow drift."""
        h, w = 96, 128
        yy, xx = np.indices((h, w))

        cx = w / 2 + 4.0 * math.sin(2.0 * math.pi * t / 40.0)
        cy = h / 2 + 2.0 * math.cos(2.0 * math.pi * t / 35.0)

        sigma_x = 9.0
        sigma_y = 2.0
        spot = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x**2) + ((yy - cy) ** 2) / (2 * sigma_y**2)))
        background = 0.18 + 0.03 * self._rng.standard_normal((h, w))
        img = 100.0 * spot + 15.0 * background
        return np.clip(img, 0.0, None).astype(np.float32)
