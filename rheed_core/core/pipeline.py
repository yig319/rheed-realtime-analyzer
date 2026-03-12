from __future__ import annotations

"""Streaming orchestration layer for collector -> analytics -> operator."""

import time

from rheed_core.core.features import FeatureExtractor
from rheed_core.core.logging import JsonlEventLogger
from rheed_core.core.policy import PolicyEngine
from rheed_core.core.preprocess import SignalPreprocessor
from rheed_core.core.state import StateEstimator
from rheed_core.io.collector_base import DataCollector
from rheed_core.io.operator_base import Operator


class RHEEDPipeline:
    """Coordinates collection, feature/state inference, policy, and logging."""

    def __init__(
        self,
        collector: DataCollector,
        operator: Operator,
        preprocessor: SignalPreprocessor,
        features: FeatureExtractor,
        state: StateEstimator,
        policy: PolicyEngine,
        logger: JsonlEventLogger | None = None,
    ):
        """Bind modular pipeline components into one runnable object."""
        self.collector = collector
        self.operator = operator
        self.preprocessor = preprocessor
        self.features = features
        self.state = state
        self.policy = policy
        self.logger = logger

    def step(self) -> int:
        """Run one poll/process cycle and return number of signal samples handled."""
        out = self.collector.poll()
        produced = 0

        for frame in out.frames:
            self.features.update_frame(frame)
            if self.logger:
                self.logger.log("frame", frame)

        for sig in out.signals:
            if sig.name != "intensity":
                continue

            x = self.preprocessor.update(sig.value)
            feat = self.features.update_signal(sig.ts, x)
            st = self.state.update(feat)
            actions = self.policy.evaluate(feat, st)

            if self.logger:
                self.logger.log("signal", sig)
                self.logger.log("feature", feat)
                self.logger.log("state", st)

            for action in actions:
                self.operator.submit(action)
                if self.logger:
                    self.logger.log("action", action)

            produced += 1

        return produced

    def run(self, duration_s: float = 30.0, sleep_s: float = 0.01) -> None:
        """Run the pipeline loop for a wall-clock duration."""
        start = time.monotonic()
        while time.monotonic() - start <= duration_s:
            self.step()
            time.sleep(sleep_s)

        if self.logger:
            self.logger.close()
