from __future__ import annotations

from collections import deque

import numpy as np

from .draft_bridge import estimate_latest_cycle_tau
from .types import FeatureVector, State


class OscillationTracker:
    """Tracks period, phase, and amplitude from a rolling intensity stream."""

    def __init__(
        self,
        max_points: int = 400,
        min_period_s: float = 0.2,
        max_period_s: float = 30.0,
        tau_peak_prominence: float = 0.05,
        tau_min_points: int = 8,
    ):
        """Initialize oscillation and per-cycle relaxation trackers."""
        self._t = deque(maxlen=max_points)
        self._x = deque(maxlen=max_points)
        self._min_period_s = min_period_s
        self._max_period_s = max_period_s
        self._tau_peak_prominence = tau_peak_prominence
        self._tau_min_points = tau_min_points
        self.latest_tau: float | None = None

    def update(self, ts: float, intensity: float) -> tuple[float | None, float | None, float | None, float]:
        """Ingest one sample and return `(period, phase, amp, confidence)`."""
        self._t.append(ts)
        self._x.append(intensity)

        if len(self._x) < 20:
            return None, None, None, 0.0

        t = np.asarray(self._t, dtype=float)
        x = np.asarray(self._x, dtype=float)
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        if dt <= 0:
            return None, None, None, 0.0

        x0 = x - np.mean(x)
        amp = float(0.5 * (np.percentile(x0, 95) - np.percentile(x0, 5)))

        ac = np.correlate(x0, x0, mode="full")
        ac = ac[len(ac) // 2 :]
        ac0 = float(ac[0]) + 1e-9
        acn = ac / ac0

        min_lag = max(1, int(self._min_period_s / dt))
        max_lag = min(len(acn) - 1, int(self._max_period_s / dt))
        if max_lag <= min_lag:
            return None, None, amp, 0.0

        window = acn[min_lag : max_lag + 1]
        lag_rel = int(np.argmax(window))
        lag = min_lag + lag_rel
        period = lag * dt

        conf = float(np.clip(window[lag_rel], 0.0, 1.0))
        phase = float((ts % period) / period) if period > 0 else None

        tau_min_distance = max(3, int(0.6 * period / dt)) if period > 0 else 3
        tau_est = estimate_latest_cycle_tau(
            t,
            x0,
            min_distance=tau_min_distance,
            prominence=self._tau_peak_prominence,
            min_points=self._tau_min_points,
        )
        if tau_est is not None and tau_est.tau_s is not None:
            self.latest_tau = float(tau_est.tau_s)

        return period, phase, amp, conf


class StateEstimator:
    def __init__(self, tracker: OscillationTracker):
        """Bind a tracker instance to a state-estimation facade."""
        self._tracker = tracker

    def update(self, feat: FeatureVector) -> State:
        """Convert one feature vector into a high-level growth state."""
        period, phase, amp, conf = self._tracker.update(feat.ts, feat.I_spec)

        if conf < 0.25:
            mode = "unknown"
        elif amp is not None and amp > 0.5:
            mode = "oscillatory"
        else:
            mode = "weak_oscillation"

        return State(
            ts=feat.ts,
            osc_period=period,
            osc_phase=phase,
            osc_amp=amp,
            mode=mode,
            confidence=conf,
            relax_tau=self._tracker.latest_tau,
        )
