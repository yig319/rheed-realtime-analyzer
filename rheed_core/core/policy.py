from __future__ import annotations

from collections import deque

import numpy as np

from .types import ActionRequest, FeatureVector, State


class PolicyEngine:
    """Advisory-first policy rules for process monitoring."""

    def __init__(
        self,
        drift_threshold: float = 8.0,
        bad_frame_limit: int = 5,
        period_cv_threshold: float = 0.2,
        tau_cv_threshold: float = 0.35,
        advisory_mode: bool = True,
    ):
        """Configure policy thresholds and action mode."""
        self._drift_threshold = drift_threshold
        self._bad_frame_limit = bad_frame_limit
        self._period_cv_threshold = period_cv_threshold
        self._tau_cv_threshold = tau_cv_threshold
        self._advisory_mode = advisory_mode

        self._bad_frame_hist = deque(maxlen=20)
        self._period_hist = deque(maxlen=20)
        self._amp_hist = deque(maxlen=30)
        self._tau_hist = deque(maxlen=20)

    def evaluate(self, feat: FeatureVector, state: State) -> list[ActionRequest]:
        """Evaluate policy rules for one `(feature, state)` pair."""
        actions: list[ActionRequest] = []
        kind = "recommend" if self._advisory_mode else "command"

        self._bad_frame_hist.append(1 if feat.bad_frame else 0)
        if state.osc_period is not None:
            self._period_hist.append(state.osc_period)
        if state.osc_amp is not None:
            self._amp_hist.append(state.osc_amp)
        if state.relax_tau is not None:
            self._tau_hist.append(state.relax_tau)

        if abs(feat.drift_x) > self._drift_threshold or abs(feat.drift_y) > self._drift_threshold:
            actions.append(
                ActionRequest(
                    ts=feat.ts,
                    kind=kind,
                    message="RHEED beam drift exceeds threshold; check alignment.",
                )
            )

        if sum(self._bad_frame_hist) >= self._bad_frame_limit:
            actions.append(
                ActionRequest(
                    ts=feat.ts,
                    kind=kind,
                    message="Persistent bad frames detected; inspect camera and screen conditions.",
                )
            )

        if len(self._period_hist) >= 6:
            arr = np.asarray(self._period_hist, dtype=float)
            cv = float(np.std(arr) / (np.mean(arr) + 1e-9))
            if cv > self._period_cv_threshold:
                actions.append(
                    ActionRequest(
                        ts=feat.ts,
                        kind=kind,
                        message="Oscillation period instability detected; consider tuning process parameters.",
                    )
                )

        if len(self._amp_hist) >= 10:
            arr = np.asarray(self._amp_hist, dtype=float)
            early = float(np.mean(arr[: len(arr) // 2]))
            late = float(np.mean(arr[len(arr) // 2 :]))
            if late < 0.7 * early:
                actions.append(
                    ActionRequest(
                        ts=feat.ts,
                        kind=kind,
                        message="Oscillation amplitude decay observed; growth may be transitioning.",
                    )
                )

        if len(self._tau_hist) >= 6:
            arr = np.asarray(self._tau_hist, dtype=float)
            cv = float(np.std(arr) / (np.mean(arr) + 1e-9))
            if cv > self._tau_cv_threshold:
                actions.append(
                    ActionRequest(
                        ts=feat.ts,
                        kind=kind,
                        message="Relaxation-time instability detected across cycles; review plume/flux conditions.",
                    )
                )

        return actions
