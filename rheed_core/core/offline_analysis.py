from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .draft_bridge import (
    TauEstimate,
    bandpass_filter_fft,
    detect_peaks_step_1d,
    fit_relaxation_tau,
    median_filter_1d,
    remove_linear_background,
    trim_cycle_tail,
)


@dataclass(slots=True)
class CycleFit:
    """Container for one peak-to-peak cycle and its fit output."""

    cycle_index: int
    start_ts: float
    end_ts: float
    tau: TauEstimate
    x: np.ndarray
    y: np.ndarray
    y_processed: np.ndarray


def select_range(data: np.ndarray, start: float, end: float, y_col: int = 1) -> np.ndarray:
    """Select a time window from a 2D array with time in column 0."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] <= y_col:
        raise ValueError("data must be a 2D array with enough columns")
    mask = (arr[:, 0] > start) & (arr[:, 0] < end)
    return np.stack([arr[mask, 0], arr[mask, y_col]], axis=1)


def preprocess_signal(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_rate_hz: float,
    median_kernel_size: int | None = 5,
    fft_band: tuple[float, float] | None = (0.05, 5.0),
    smooth_window: int | None = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply draft-style denoise passes to an offline signal series."""
    x = np.asarray(sample_x, dtype=float)
    y = np.asarray(sample_y, dtype=float)
    if x.size != y.size:
        raise ValueError("sample_x and sample_y must have equal length")

    y_proc = y.copy()
    if median_kernel_size is not None:
        y_proc = median_filter_1d(y_proc, median_kernel_size)
    if fft_band is not None:
        y_proc = bandpass_filter_fft(y_proc, fft_band[0], fft_band[1], sample_rate_hz)
    if smooth_window is not None and smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
        y_proc = np.convolve(y_proc, kernel, mode="same")
    return x, y_proc


def detect_cycle_boundaries(
    sample_y: np.ndarray,
    camera_freq: float,
    laser_freq: float,
    convolve_step: int = 5,
    prominence: float = 0.1,
) -> np.ndarray:
    """Detect cycle-peak boundaries using camera/laser frequency prior."""
    if camera_freq <= 0 or laser_freq <= 0:
        raise ValueError("camera_freq and laser_freq must be > 0")
    min_distance = max(1, int(camera_freq / laser_freq * 0.6))
    return detect_peaks_step_1d(
        np.asarray(sample_y, dtype=float),
        min_distance=min_distance,
        convolve_step=convolve_step,
        prominence=prominence,
        mode="same",
    )


def split_cycles(
    sample_x: np.ndarray, sample_y: np.ndarray, peak_indices: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert peak indices into a list of peak-to-peak cycle arrays."""
    x = np.asarray(sample_x, dtype=float)
    y = np.asarray(sample_y, dtype=float)
    peaks = np.asarray(peak_indices, dtype=int)
    if x.size != y.size:
        raise ValueError("sample_x and sample_y must have equal length")
    if peaks.size < 2:
        return []

    cycles: list[tuple[np.ndarray, np.ndarray]] = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        if right - left < 3:
            continue
        cycles.append((x[left:right], y[left:right]))
    return cycles


def process_cycle_curve(
    x: np.ndarray,
    y: np.ndarray,
    tune_tail: bool = True,
    trim_first: int = 0,
    linear_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply draft-like per-cycle cleanup before fitting."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    y_proc = ys.copy()

    if tune_tail:
        y_proc = trim_cycle_tail(y_proc, ratio=0.1)

    if trim_first > 0 and trim_first < y_proc.size:
        y_proc = y_proc[trim_first:]
        xs = np.linspace(xs[0], xs[-1], y_proc.size)

    if linear_ratio is not None and linear_ratio > 0:
        y_proc = remove_linear_background(xs, y_proc, linear_ratio=linear_ratio)
    return xs, y_proc


def analyze_rheed_signal(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    camera_freq: float,
    laser_freq: float,
    convolve_step: int = 5,
    prominence: float = 0.1,
    tune_tail: bool = True,
    trim_first: int = 0,
    linear_ratio: float = 0.8,
    fit_mode: str = "auto",
) -> list[CycleFit]:
    """Full offline analysis pipeline adapted from legacy draft workflow."""
    peaks = detect_cycle_boundaries(
        sample_y=sample_y,
        camera_freq=camera_freq,
        laser_freq=laser_freq,
        convolve_step=convolve_step,
        prominence=prominence,
    )
    cycles = split_cycles(sample_x, sample_y, peaks)

    results: list[CycleFit] = []
    for i, (x_cycle, y_cycle) in enumerate(cycles):
        x_proc, y_proc = process_cycle_curve(
            x_cycle,
            y_cycle,
            tune_tail=tune_tail,
            trim_first=trim_first,
            linear_ratio=linear_ratio,
        )
        tau = fit_relaxation_tau(x_proc, y_proc, mode=fit_mode, min_points=8)
        results.append(
            CycleFit(
                cycle_index=i,
                start_ts=float(x_cycle[0]),
                end_ts=float(x_cycle[-1]),
                tau=tau,
                x=x_cycle,
                y=y_cycle,
                y_processed=y_proc,
            )
        )
    return results
