from __future__ import annotations

"""CLI entrypoint for running the sidecar in local dummy mode."""

import argparse
from pathlib import Path

import yaml

from rheed_core.core.features import FeatureExtractor
from rheed_core.core.logging import JsonlEventLogger
from rheed_core.core.pipeline import RHEEDPipeline
from rheed_core.core.policy import PolicyEngine
from rheed_core.core.preprocess import SignalPreprocessor
from rheed_core.core.state import OscillationTracker, StateEstimator
from rheed_core.io.dummy_collector import DummyCollector
from rheed_core.io.dummy_operator import DummyOperator


def load_config(path: Path) -> dict:
    """Load YAML config file into a dictionary."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_pipeline(cfg: dict) -> tuple[RHEEDPipeline, DummyOperator]:
    """Construct a pipeline and in-memory operator from config."""
    dummy_cfg = cfg.get("dummy", {})
    policy_cfg = cfg.get("policy", {})
    preprocess_cfg = dict(cfg.get("preprocess", {}))
    state_cfg = cfg.get("state", {})
    log_cfg = cfg.get("logging", {})

    if preprocess_cfg.get("sample_rate_hz") is None and dummy_cfg.get("sample_rate_hz") is not None:
        preprocess_cfg["sample_rate_hz"] = dummy_cfg["sample_rate_hz"]

    collector = DummyCollector(**dummy_cfg)
    operator = DummyOperator()
    preprocessor = SignalPreprocessor(**preprocess_cfg)
    features = FeatureExtractor()
    tracker = OscillationTracker(**state_cfg)
    state = StateEstimator(tracker)
    policy = PolicyEngine(**policy_cfg)
    logger = JsonlEventLogger(log_cfg.get("path", "logs/events.jsonl"))

    pipeline = RHEEDPipeline(
        collector=collector,
        operator=operator,
        preprocessor=preprocessor,
        features=features,
        state=state,
        policy=policy,
        logger=logger,
    )
    return pipeline, operator


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(description="RHEED real-time analytics sidecar (dummy mode)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--duration", type=float, default=None, help="Override run duration in seconds")
    return parser.parse_args()


def main() -> None:
    """Run pipeline for configured duration and print action summary."""
    args = parse_args()
    cfg = load_config(args.config)
    pipeline, operator = build_pipeline(cfg)

    runtime_cfg = cfg.get("runtime", {})
    duration = args.duration if args.duration is not None else runtime_cfg.get("duration_s", 20.0)
    sleep_s = runtime_cfg.get("loop_sleep_s", 0.01)

    pipeline.run(duration_s=duration, sleep_s=sleep_s)
    print(f"Finished run. Advisory actions emitted: {len(operator.actions)}")


if __name__ == "__main__":
    main()
