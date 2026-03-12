# RHEED Real-Time Analyzer

Python sidecar framework for real-time RHEED analytics in a TSST-centered PLD workflow.

## Scope

- Keeps TSST as authoritative control and safety gate.
- Runs independent analytics and policy logic.
- Starts in advisory mode, with closed-loop hooks for later bounded control.

## Installation

### Option A: Conda environment (recommended)

```bash
conda create -n rheed-rt python=3.11 -y
conda activate rheed-rt
pip install -r requirements.txt
pip install -e .
```

### Option B: Dev environment (tests + notebook + publishing tools)

```bash
conda create -n rheed-rt-dev python=3.11 -y
conda activate rheed-rt-dev
pip install -r requirements-dev.txt
pip install -e .[dev]
```

## Quick Start

Run a short dummy sidecar session:

```bash
rheed-sidecar --duration 10
```

This writes JSONL events to `logs/events.jsonl` by default.

## Test

```bash
pytest -q
```

Detailed test and demo notes are in `tests/README.md`.

## Build And Publish

Detailed release instructions are in `docs/PUBLISHING.md`.

### Automated release via GitHub Actions

This repo includes `.github/workflows/main.yml` with keyword-triggered release on pushes to `main`:

- `#major` -> bump `X.0.0`
- `#minor` -> bump `x.Y.0`
- `#patch` -> bump `x.y.Z`

Example commit message:

```text
feat: add TSST adapter #major
```

When triggered, the workflow will:

1. bump versions in `pyproject.toml` and `rheed_core/__init__.py`
2. create/push a release commit and `vX.Y.Z` tag
3. build and publish to PyPI using `pypa/gh-action-pypi-publish`

Note: configure PyPI Trusted Publishing for this repository/workflow before first automated release.

### Local package build check

```bash
python -m build
twine check dist/*
```

## Current Capabilities

- Dummy real-time collector with noisy oscillatory intensity and synthetic image frames.
- Rolling preprocessing + feature extraction (signal + image proxies).
- Oscillation tracker with period/phase/amplitude/confidence estimation.
- Draft-integrated cycle analysis (peak segmentation + per-cycle relaxation `tau` estimate).
- Offline reusable analysis pipeline adapted from draft code:
  - range selection
  - denoise chain
  - peak-to-peak cycle splitting
  - tail/background cleanup
  - per-cycle relaxation fitting
- Advisory-first policy engine for drift, frame quality, period instability, and amplitude decay.
- Replayable JSONL event logging.

## Draft Integration

Legacy analysis ideas from `rheed_core/core/draft codes/` are bridged into the runtime path through:

- `rheed_core/core/draft_bridge.py`: FFT band-pass, median filtering, peak detection, cycle segmentation, and linearized `tau` fitting.
- `rheed_core/core/preprocess.py`: optional median + FFT filtering in streaming mode.
- `rheed_core/core/features.py`: Gaussian-moment-like image spread and drift estimation.
- `rheed_core/core/state.py`: latest cycle relaxation-time tracking.
- `rheed_core/core/policy.py`: advisory on unstable relaxation-time behavior.

## Notebook Demos

- `tests/demo_draft_bridge.ipynb`: draft-derived signal processing + `tau` fitting utilities.
- `tests/demo_pipeline_integration.ipynb`: end-to-end sidecar usage with draft-aware config.
- `tests/demo_offline_analysis.ipynb`: second-pass offline flow mirroring real draft workflow.

## Project Layout

```text
rheed_core/
  main.py
  config.yaml
  core/
  io/
tests/
docs/
```

## Roadmap

1. Replace dummy collector with TSST stream adapter.
2. Validate feature/state outputs against recorded experimental datasets.
3. Introduce bounded command outputs behind explicit safety constraints.
