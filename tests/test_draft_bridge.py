import numpy as np

from rheed_core.core.draft_bridge import (
    detect_peaks_1d,
    estimate_latest_cycle_tau,
    median_filter_1d,
)


def test_median_filter_reduces_spike() -> None:
    x = np.array([0.0, 0.1, 0.2, 10.0, 0.3, 0.2, 0.1], dtype=float)
    y = median_filter_1d(x, kernel_size=3)
    assert y[3] < 1.0


def test_detect_peaks_and_tau_estimate() -> None:
    dt = 0.05
    period = 2.0
    tau_true = 0.45
    t = np.arange(0.0, 20.0, dt)
    phase = np.mod(t, period)

    # Sawtooth-like growth cycle with exponential rise, inspired by draft fitting flow.
    y = 1.0 - np.exp(-phase / tau_true)
    y += 0.02 * np.sin(2.0 * np.pi * t / period)

    peaks = detect_peaks_1d(y, min_distance=int(0.6 * period / dt), prominence=0.05)
    assert peaks.size >= 4

    est = estimate_latest_cycle_tau(
        t,
        y,
        min_distance=int(0.6 * period / dt),
        prominence=0.05,
        mode="growth",
        min_points=8,
    )
    assert est is not None
    assert est.tau_s is not None
    # Single-cycle estimation without explicit denoise is approximate by design.
    assert abs(est.tau_s - tau_true) < 0.5
