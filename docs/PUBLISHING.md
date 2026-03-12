# Publishing Guide

This guide covers both automated and manual publishing for `rheed-realtime-analyzer`.

## Automated Release (Recommended)

The workflow `.github/workflows/main.yml` is configured to release from `main` when the head commit message contains:

- `#major`
- `#minor`
- `#patch`

### Trigger behavior

- `#major`: `X.Y.Z -> (X+1).0.0`
- `#minor`: `X.Y.Z -> X.(Y+1).0`
- `#patch`: `X.Y.Z -> X.Y.(Z+1)`

### Example trigger commit

```bash
git commit -m "feat: improve cycle fit stability #minor"
git push origin main
```

### What the action does

1. Runs CI build checks.
2. Bumps version in:
   - `pyproject.toml`
   - `rheed_core/__init__.py`
3. Creates a release commit on `main`.
4. Tags release as `vX.Y.Z`.
5. Builds and publishes to PyPI.

### One-time PyPI setup

Use PyPI Trusted Publishing and add this GitHub workflow as a trusted publisher in your PyPI project settings.

## 1. Prepare Environment

```bash
conda create -n rheed-release python=3.11 -y
conda activate rheed-release
pip install -r requirements-dev.txt
pip install -e .[dev]
```

## 2. Run Quality Checks

```bash
pytest -q
```

## 3. Build Distribution Artifacts

```bash
rm -rf dist build *.egg-info
python -m build
```

Expected outputs in `dist/`:

- `rheed_realtime_analyzer-<version>-py3-none-any.whl`
- `rheed_realtime_analyzer-<version>.tar.gz`

## 4. Validate Package Metadata

```bash
twine check dist/*
```

## 5. Upload To TestPyPI First

1. Create a TestPyPI token.
2. Export credentials:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<testpypi-token>
```

3. Upload:

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

4. Verify install:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple rheed-realtime-analyzer
```

## 6. Upload To PyPI

1. Create a PyPI token.
2. Export credentials:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<pypi-token>
```

3. Upload:

```bash
twine upload dist/*
```

## 7. Post-Release Check

```bash
pip install --upgrade rheed-realtime-analyzer
python -c "import rheed_core; print(rheed_core.__version__)"
rheed-sidecar --duration 2
```

## Versioning

Before publishing a new release, update version in both:

- `pyproject.toml`
- `rheed_core/__init__.py`

If using automated release, the workflow updates these files for you.
