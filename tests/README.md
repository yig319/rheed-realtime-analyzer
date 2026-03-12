# Tests And Demos

This folder includes both automated tests (`pytest`) and instructional notebooks (`.ipynb`).

## Automated Tests

Run:

```bash
pytest -q
```

Coverage focus:

- `test_state.py`
  - Verifies oscillation period estimation is near ground truth.
- `test_state_relax_tau.py`
  - Verifies state estimator surfaces cycle relaxation time (`relax_tau`).
- `test_policy.py`
  - Verifies advisory emission for persistent bad frames.
- `test_pipeline.py`
  - Smoke test for end-to-end pipeline wiring.
- `test_draft_bridge.py`
  - Verifies draft-derived utilities (median filter, peak detect, tau estimate).
- `test_offline_analysis.py`
  - Verifies second-pass offline analysis flow adapted from draft code.

## Notebook Demos

- `demo_draft_bridge.ipynb`
  - Focused examples for `draft_bridge` utility functions.
- `demo_pipeline_integration.ipynb`
  - End-to-end sidecar simulation with draft-aware preprocessing/state/policy settings.
- `demo_offline_analysis.ipynb`
  - Full cycle workflow similar to your legacy analysis path.

## Recommended Order

1. Run `pytest -q` first to confirm baseline integrity.
2. Run `demo_draft_bridge.ipynb` to understand signal primitives.
3. Run `demo_offline_analysis.ipynb` to inspect cycle-wise fitting behavior.
4. Run `demo_pipeline_integration.ipynb` to see runtime integration.
