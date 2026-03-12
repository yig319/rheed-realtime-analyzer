from __future__ import annotations

from collections import deque

import numpy as np

from .draft_bridge import bandpass_filter_fft, median_filter_1d

class SignalPreprocessor:
    """Rolling denoise + detrend for real-time intensity streams.

    Draft integration:
    - Optional median filter (from legacy analysis workflow)
    - Optional FFT band-pass filtering
    """

    def __init__(
        self,
        trend_window: int = 64,
        clip_sigma: float = 4.0,
        median_kernel_size: int | None = None,
        fft_band: tuple[float, float] | None = None,
        sample_rate_hz: float | None = None,
        fft_window: int = 128,
    ):
        """Initialize rolling denoise/detrend state.

        Args:
            trend_window: Window size for rolling baseline subtraction.
            clip_sigma: Sigma threshold for outlier clipping before detrending.
            median_kernel_size: Optional kernel for median denoise.
            fft_band: Optional `(low, high)` passband in Hz.
            sample_rate_hz: Sampling rate required for FFT filtering.
            fft_window: Number of recent points used by FFT filter.
        """
        self._history = deque(maxlen=trend_window)
        raw_maxlen = max(trend_window, fft_window, median_kernel_size or 1)
        self._raw_history = deque(maxlen=raw_maxlen)
        self._clip_sigma = clip_sigma
        self._median_kernel_size = median_kernel_size
        self._fft_band = fft_band
        self._sample_rate_hz = sample_rate_hz
        self._fft_window = fft_window

    def update(self, value: float) -> float:
        """Process one scalar sample and return detrended output."""
        self._raw_history.append(float(value))
        raw = np.asarray(self._raw_history, dtype=float)

        filtered = float(value)
        if self._median_kernel_size is not None and raw.size >= self._median_kernel_size:
            filtered = float(median_filter_1d(raw, self._median_kernel_size)[-1])

        if (
            self._fft_band is not None
            and self._sample_rate_hz is not None
            and raw.size >= max(16, self._fft_window // 2)
        ):
            win = raw[-min(raw.size, self._fft_window) :]
            filtered_win = bandpass_filter_fft(
                win,
                low_cutoff=self._fft_band[0],
                high_cutoff=self._fft_band[1],
                sample_frequency=self._sample_rate_hz,
            )
            filtered = float(filtered_win[-1])

        self._history.append(filtered)
        arr = np.asarray(self._history, dtype=float)
        if arr.size < 4:
            return filtered

        mean = float(np.mean(arr))
        std = float(np.std(arr)) + 1e-9
        clipped = float(np.clip(filtered, mean - self._clip_sigma * std, mean + self._clip_sigma * std))
        return clipped - mean
