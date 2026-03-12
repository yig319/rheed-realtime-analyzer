from __future__ import annotations

"""Collector interfaces for ingesting frame and scalar streams."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rheed_core.core.types import FramePacket, SignalPacket


@dataclass(slots=True)
class CollectorOutput:
    """Batch of new events returned by one collector poll."""

    signals: list[SignalPacket] = field(default_factory=list)
    frames: list[FramePacket] = field(default_factory=list)


class DataCollector(ABC):
    """Abstract collector interface for live or replay data sources."""

    @abstractmethod
    def poll(self) -> CollectorOutput:
        """Return currently available frame/signal events without blocking long."""
        raise NotImplementedError
