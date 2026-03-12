import math

from rheed_core.core.state import OscillationTracker


def test_oscillation_tracker_estimates_period_close_to_truth() -> None:
    tracker = OscillationTracker(max_points=500, min_period_s=0.5, max_period_s=10.0)
    truth_period = 3.0

    period = None
    for i in range(400):
        t = i * 0.1
        x = math.sin(2.0 * math.pi * t / truth_period)
        period, _, _, _ = tracker.update(t, x)

    assert period is not None
    assert abs(period - truth_period) < 0.4
