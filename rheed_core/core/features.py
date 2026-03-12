from __future__ import annotations

from collections import deque

import numpy as np

from .types import FeatureVector, FramePacket


class FeatureExtractor:
    """Extracts signal and image-derived RHEED features."""

    def __init__(self, amp_window: int = 40):
        """Initialize rolling signal/image feature state."""
        self._signal_hist = deque(maxlen=amp_window)
        self._last_drift_x = 0.0
        self._last_drift_y = 0.0
        self._last_streak_width = 0.0
        self._last_quality = 1.0
        self._last_bad_frame = False

    def update_frame(self, packet: FramePacket) -> None:
        """Update image-derived features from a new frame packet."""
        img = packet.img.astype(float)
        if img.size == 0:
            self._last_bad_frame = True
            self._last_quality = 0.0
            return

        h, w = img.shape
        cx, cy = w // 2, h // 2
        roi = img[max(cy - 4, 0) : min(cy + 5, h), max(cx - 4, 0) : min(cx + 5, w)]
        bg = img[: max(h // 8, 1), : max(w // 8, 1)]

        roi_mean = float(np.mean(roi))
        bg_mean = float(np.mean(bg))
        contrast = (roi_mean - bg_mean) / (abs(bg_mean) + 1e-6)

        x_cm, y_cm, width_x, width_y = _gaussian_moment_like_features(img)
        streak_width = float(0.5 * (width_x + width_y))

        self._last_drift_x = x_cm - cx
        self._last_drift_y = y_cm - cy
        self._last_streak_width = streak_width
        self._last_bad_frame = bool(np.isnan(contrast) or contrast < -0.5)
        self._last_quality = float(np.clip(1.0 - abs(self._last_drift_x) / max(w, 1), 0.0, 1.0))

    def update_signal(self, ts: float, value: float) -> FeatureVector:
        """Merge latest signal value with most recent image features."""
        self._signal_hist.append(value)
        arr = np.asarray(self._signal_hist, dtype=float)

        i_spec = float(value)
        i_bg = float(np.median(arr)) if arr.size else 0.0
        contrast = (i_spec - i_bg) / (abs(i_bg) + 1e-6)

        return FeatureVector(
            ts=ts,
            I_spec=i_spec,
            I_bg=i_bg,
            contrast=contrast,
            streak_width=self._last_streak_width,
            drift_x=self._last_drift_x,
            drift_y=self._last_drift_y,
            bad_frame=self._last_bad_frame,
            quality_score=self._last_quality,
        )


def _gaussian_moment_like_features(img: np.ndarray) -> tuple[float, float, float, float]:
    """Draft-derived Gaussian-moment estimator for spot center and spread."""
    h, w = img.shape
    yy, xx = np.indices(img.shape)
    total = float(np.sum(img))
    if total <= 1e-9:
        return w / 2.0, h / 2.0, 0.0, 0.0

    x_cm = float(np.sum(xx * img) / total)
    y_cm = float(np.sum(yy * img) / total)

    x_idx = int(np.clip(round(x_cm), 0, w - 1))
    y_idx = int(np.clip(round(y_cm), 0, h - 1))

    col = img[:, x_idx]
    row = img[y_idx, :]
    width_y = float(np.sqrt(np.abs(np.sum(((np.arange(h) - y_cm) ** 2) * col) / (np.sum(col) + 1e-9))))
    width_x = float(np.sqrt(np.abs(np.sum(((np.arange(w) - x_cm) ** 2) * row) / (np.sum(row) + 1e-9))))
    return x_cm, y_cm, width_x, width_y
