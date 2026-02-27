# CobraBox

CobraBox is a lightweight toolbox for feature extraction from EEG and fMRI
time-series data.

## Quick Start

For setup, use the onboarding:
[`docs`](docs)

Short version:

```bash
uv sync && uv run pre-commit install
```

## Repository Layout

- `src/cobrabox/` - Core package code (data model, features, loaders)
- `tests/` - Unit tests
- `examples/` - Minimal runnable examples
- `docs/` - Setup and contribution guides
- `data/` - Dummy/example data files

## Python Version

- Minimum supported: Python 3.11
- Development target: Python 3.14

## Minimal Runnable Usage

```python
import cobrabox as cb
import numpy as np

# Create synthetic 2D input data: [time, space]
my_array = np.random.default_rng(seed=0).normal(size=(100, 4))

data = cb.from_numpy(arr=my_array, dims=["time", "space"], sampling_rate=100.0)
feat = cb.feature.line_length(data)

print(feat)
```

## Core Concepts

- Data container: `cobrabox.Data` (with `EEG` and `FMRI` subclasses)
- Features are functions under `cb.feature.*`
- Features return new `Data` objects and append to `history`

## Built-in Dummy Datasets

Use `cb.dataset(name)` with:

- `dummy_chain`
- `dummy_random`
- `dummy_star`
- `dummy_noise`

## Coverage

- Test coverage is measured with `pytest-cov`.
- Coverage output is shown by default in test runs (configured in `pyproject.toml`).
- Run tests with:

```bash
uv run pytest -q
```

## Documentation

- Setup repo: [`docs/setup_repo.md`](docs/setup_repo.md)
- Contribute a feature: [`docs/contributing_feature.md`](docs/contributing_feature.md)
- Make a pull request: [`docs/how_to_make_a_pr.md`](docs/how_to_make_a_pr.md)
- Set up GitHub SSH key: [`docs/setup_github_ssh_key.md`](docs/setup_github_ssh_key.md)
- Docs index: [`docs/README.md`](docs/README.md)
