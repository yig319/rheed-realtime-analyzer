from __future__ import annotations

"""Canonical data contracts shared across collectors, analytics, and operators."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(slots=True)
class FramePacket:
    """One image frame event from a camera or replay stream."""

    ts: float
    img: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SignalPacket:
    """One scalar signal event, e.g., specular intensity vs time."""

    ts: float
    name: str
    value: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureVector:
    """Aggregated image + signal features for one pipeline timestamp."""

    ts: float
    I_spec: float
    I_bg: float
    contrast: float
    streak_width: float
    drift_x: float
    drift_y: float
    bad_frame: bool
    quality_score: float


@dataclass(slots=True)
class State:
    """Estimated growth/oscillation state derived from recent features."""

    ts: float
    osc_period: float | None
    osc_phase: float | None
    osc_amp: float | None
    mode: str
    confidence: float
    relax_tau: float | None = None


@dataclass(slots=True)
class ActionRequest:
    """Policy output that can be advisory or bounded-command style."""

    ts: float
    kind: Literal["recommend", "command"]
    message: str
    command: dict[str, Any] | None = None
