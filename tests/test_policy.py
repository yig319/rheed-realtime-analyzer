from rheed_core.core.policy import PolicyEngine
from rheed_core.core.types import FeatureVector, State


def test_policy_emits_bad_frame_advisory() -> None:
    policy = PolicyEngine(bad_frame_limit=3)

    actions = []
    for i in range(5):
        feat = FeatureVector(
            ts=float(i),
            I_spec=1.0,
            I_bg=1.0,
            contrast=0.0,
            streak_width=1.0,
            drift_x=0.0,
            drift_y=0.0,
            bad_frame=True,
            quality_score=0.1,
        )
        state = State(ts=float(i), osc_period=2.0, osc_phase=0.2, osc_amp=1.0, mode="oscillatory", confidence=0.8)
        actions = policy.evaluate(feat, state)

    assert actions
    assert any("bad frames" in a.message.lower() for a in actions)
