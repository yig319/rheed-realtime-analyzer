from rheed_core.main import build_pipeline


def test_pipeline_smoke_runs_steps() -> None:
    cfg = {
        "dummy": {
            "sample_rate_hz": 50.0,
            "frame_rate_hz": 5.0,
            "osc_period_s": 2.0,
            "noise_std": 0.05,
            "seed": 7,
        },
        "policy": {
            "advisory_mode": True,
            "bad_frame_limit": 50,
        },
        "preprocess": {},
        "state": {
            "max_points": 300,
            "min_period_s": 0.2,
            "max_period_s": 8.0,
        },
        "logging": {
            "path": "logs/test_events.jsonl",
        },
    }

    pipeline, _ = build_pipeline(cfg)
    for _ in range(30):
        pipeline.step()

    assert True
