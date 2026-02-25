# CobraBox

A lightweight toolbox for extracting time-series features from EEG and fMRI
data. This repository provides a clean, reproducible starting point for feature
engineering across modalities.

## Repository layout

- `src/cobrabox/` - Core library code (feature extractors, pipelines).
- `examples/` - Minimal end-to-end example scripts and notebooks.
- `scripts/` - CLI utilities for batch processing and dataset prep.
- `tests/` - Unit and integration tests.
- `docs/` - User guides and developer notes.
- `data/`
  - `raw/` - Unmodified input data (not tracked).
  - `processed/` - Derived outputs (not tracked).
  - `synthetic/` - Generated datasets for testing (not tracked).

## Getting started

1. Create a virtual environment and install dependencies.
2. Add a small EEG or fMRI example to `data/raw/`.
3. Run an example from `examples/` once available.

### `uv` basics

We use `uv` for package and project management. Docs can be found [here](https://docs.astral.sh/uv/).

#### Installation

```bash
brew install uv
```

or

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Basic usage

Instead of using `requirements.txt`, all dependencies are part of `pyproject.toml`.
The pinned Python version is inside a `.python-version` file.

To start using `uv`, first, install a Python:

```bash
uv python install
```

This will install a pinned Python version in the virtual environment.

`uv` relies on lock files (`uv.lock`) that contain package versions to be installed. Sync your local environment using:

```bash
uv sync
```

Running scripts or any commands that rely on the `uv` virtual environment is done by prepending `uv run` before the command, so e.g.,

```bash
# old
python main_run.py
# `uv`
uv run python main_run.py
# or you can omit python
uv run main_run.py

# old
alembic upgrade head
# `uv`
uv run alembic upgrade head
```

#### Package operations

Adding package:

```bash
uv add numpy
uv add "celery[sqs]"
uv add "netcdf==1.6.5"
...
```

Adding package as a `dev` dependency only:

```bash
uv add --dev pytest
```

(Development dependencies are synced by default)

Adding a custom group of dependencies:

```bash
uv add --group plot seaborn
```

(Custom group dependencies are not synced by default; you need to `uv sync --group plot` or `uv sync --all-groups` to sync all dependencies.)

Removing package:

```bash
uv remove numpy
```

Manually locking package versions:

```bash
uv lock
```

Checking the environment up-to-date with the lock file:

```bash
uv lock --check
```

Upgrading all packages to the newest version (with regards to dependencies):

```bash
uv lock --upgrade
uv sync
```

Locking and syncing by default work on default and `dev` groups. To operate on dependency groups, do not forget to use `--group ...` or `--all-groups`.

### Tools

Helping with keeping the code clean: linter, imports sorter, and formatter. For both, we use [`ruff`](https://astral.sh/ruff), which is much much faster than `black` / `flake8` / `sort`. Also, there is a neat way to this all of this automatically and that is the concept of pre-commit hooks (below).

Set it up:

```bash
uv tool install ruff
uv tool install pre-commit
```

There is no need to set up anything else since `ruff` listens to all settings that are part of `project. tool`. There is a VS Code extension `ruff` that works flawlessly and also listens to `pyproject.toml`, just ensure the configuration is read from the project config file.

#### Linting

```bash
# just lint
uvx ruff check
# also fix, if possible
uvx ruff check --fix
```

#### Formatting

The best way is to set up the "format on save" in your [IDE](https://docs.astral.sh/ruff/editors/setup/). For manual formatting (including import sorting), use:

```bash
# actually format
uvx ruff format
# just check what would be formatted
uvx ruff format --check
uvx ruff format --diff
```

#### Pre-commit hooks

To utilize pre-commit hooks, we use the [pre-commit](https://pre-commit.com) python library. Although written in Python, it supports many programming languages. The definition of a pre-commit hook(s) is given in the `.pre-commit-config.yaml` file in each respective repository. In other to use the pre-commit hooks, you can just:

```bash
uvx pre-commit install
```

and you are good to go. To test the hooks, you can invoke hook run on all files using `uvx pre-commit run --all-files`.

That's it. Now, any time you would commit, these hooks are run automatically, so you do not need to worry about failing linting on the CI or forgetting to update packages anymore.

## Feature API (sample docs)

### Core ideas

- Data is represented by `cobrabox.Data` (internally wraps an xarray `DataArray`).
- Feature functions are called from `cb.feature.*`.
- Features return a new `Data` object and append to `history`.

### Minimal feature pipeline

```python
import cobrabox as cb

data = cb.from_numpy(
    arr=my_array,                     # shape: (time, space, ...)
    dims=["time", "space"],           # must include "time"
    sampling_rate=100.0,              # optional
)
wdata = cb.feature.sliding_window(data, window_size=20, step_size=10)
feat = cb.feature.line_length(wdata)

print(feat.data.shape)
print(feat.history)  # ['sliding_window', 'line_length']
```

### Built-in feature functions

- `cb.feature.sliding_window(data, window_size=10, step_size=5)`
  - Adds a `window_index` dimension from overlapping windows on `time`.
- `cb.feature.line_length(data)`
  - Computes line-length feature from temporal differences.

### Loading dummy datasets

`cb.dataset(name)` currently supports dummy identifiers and returns a list of
`Data` parts loaded from compressed CSV files:

- `dummy_chain`
- `dummy_random`
- `dummy_star`
- `dummy_noise`

Example:

```python
parts = cb.dataset("dummy_random")
print(len(parts))
print(parts[0].data.dims)
```

## Planned scope

- Time-domain features: basic statistics, complexity, entropy.
- Frequency-domain features: band power, spectral entropy.
- Connectivity features: coherence, correlation, graph metrics.
- Windowing utilities for multichannel time series.
- Modality helpers for EEG and fMRI I/O and preprocessing hooks.

## Contributing

Contributions are welcome. Open an issue to propose new features or datasets.

## License

Add a license before distributing the toolbox publicly.
