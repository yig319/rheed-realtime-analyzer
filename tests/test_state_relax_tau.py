import numpy as np

from rheed_core.core.features import FeatureExtractor
from rheed_core.core.state import OscillationTracker, StateEstimator


def test_state_estimator_produces_relax_tau_from_cycles() -> None:
    tracker = OscillationTracker(max_points=600, min_period_s=0.2, max_period_s=5.0)
    estimator = StateEstimator(tracker)
    features = FeatureExtractor()

    dt = 0.05
    period = 2.0
    tau_true = 0.5
    state = None

    for i in range(500):
        ts = i * dt
        phase = np.mod(ts, period)
        signal = 1.0 - np.exp(-phase / tau_true)
        feat = features.update_signal(ts, signal)
        state = estimator.update(feat)

    assert state is not None
    assert state.relax_tau is not None
    assert state.relax_tau > 0
