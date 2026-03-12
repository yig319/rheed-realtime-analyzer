import numpy as np

from rheed_core.core.offline_analysis import (
    analyze_rheed_signal,
    detect_cycle_boundaries,
    preprocess_signal,
    select_range,
)


def _synthetic_growth_series(
    sample_rate_hz: float = 20.0, period_s: float = 2.0, tau_s: float = 0.45, duration_s: float = 40.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic peak-to-peak growth-style cycle series."""
    dt = 1.0 / sample_rate_hz
    t = np.arange(0.0, duration_s, dt)
    phase = np.mod(t, period_s)
    y = 1.0 - np.exp(-phase / tau_s)
    y += 0.02 * np.sin(2.0 * np.pi * t / period_s)
    return t, y


def test_select_range_extracts_time_window() -> None:
    t, y = _synthetic_growth_series(duration_s=5.0)
    data = np.stack([t, y], axis=1)
    win = select_range(data, 1.0, 2.5, y_col=1)
    assert win.shape[1] == 2
    assert np.all(win[:, 0] > 1.0)
    assert np.all(win[:, 0] < 2.5)


def test_detect_cycle_boundaries_finds_peaks() -> None:
    t, y = _synthetic_growth_series(duration_s=30.0)
    _, y_proc = preprocess_signal(t, y, sample_rate_hz=20.0, median_kernel_size=5, fft_band=(0.05, 4.0))
    peaks = detect_cycle_boundaries(y_proc, camera_freq=20.0, laser_freq=0.5, convolve_step=5, prominence=0.03)
    assert peaks.size >= 8


def test_analyze_rheed_signal_returns_tau_estimates() -> None:
    tau_true = 0.45
    t, y = _synthetic_growth_series(tau_s=tau_true, duration_s=60.0)
    _, y_proc = preprocess_signal(t, y, sample_rate_hz=20.0, median_kernel_size=5, fft_band=(0.05, 4.0))

    results = analyze_rheed_signal(
        t,
        y_proc,
        camera_freq=20.0,
        laser_freq=0.5,
        convolve_step=5,
        prominence=0.03,
        tune_tail=True,
        trim_first=0,
        linear_ratio=0.8,
        fit_mode="growth",
    )
    taus = [r.tau.tau_s for r in results if r.tau.tau_s is not None]

    assert len(taus) >= 6
    assert abs(float(np.median(taus)) - tau_true) < 0.3
