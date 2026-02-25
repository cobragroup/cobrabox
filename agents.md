# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for package and environment management.

```bash
# Install/sync dependencies
uv sync

# Run tests (coverage reported automatically)
uv run pytest

# Run a single test
uv run pytest tests/test_from_constructors.py::test_from_numpy_basic

# Enforce 95% coverage threshold
uv run pytest --cov-fail-under=95

# HTML coverage report → htmlcov/index.html
uv run pytest --cov-report=html

# Lint (check only)
uvx ruff check

# Lint (auto-fix)
uvx ruff check --fix

# Format
uvx ruff format

# Run any script
uv run python main.py
```

Pre-commit hooks (ruff) run automatically on commit. Install once with `uvx pre-commit install`.

## Architecture

**Core data container** (`src/cobrabox/data.py`): `Data` is an immutable wrapper around `xarray.DataArray`. It requires `time` and `space` dimensions and stores metadata (`subjectID`, `groupID`, `condition`, `sampling_rate`, `history`, `extra`) in xarray attrs. `EEG` and `FMRI` are empty subclasses for type distinction. Construct via `cb.from_numpy(arr, dims, ...)` or `cb.from_xarray(ar, ...)`.

**Feature system** (`src/cobrabox/function_wrapper.py`, `src/cobrabox/feature.py`, `src/cobrabox/features/`): Features are plain functions decorated with `@feature` from `function_wrapper.py`. The decorator automatically repacks the return value (`xr.DataArray` or `Data`) into a new `Data` object via `_copy_with_new_data`, appending the function name to `history`. Feature discovery is automatic: `feature.py` scans all modules in the `features/` subpackage and registers any callable marked `_is_cobrabox_feature = True`. Adding a new feature means creating a new file in `features/` with a `@feature`-decorated function — no manual registration needed.

**Datasets** (`src/cobrabox/datasets.py`, `src/cobrabox/dataset_loader.py`): `cb.dataset(name)` returns a `list[Data]`. Built-in dummy datasets (`dummy_chain`, `dummy_random`, `dummy_star`, `dummy_noise`) are loaded from compressed CSV files in `data/dummy/`.

**Public API** (`src/cobrabox/__init__.py`): Top-level imports expose `Data`, `EEG`, `FMRI`, `dataset`, `from_numpy`, `from_xarray`, plus all auto-discovered feature functions. The `feature` submodule is also accessible as `cb.feature.*`.

## Key conventions

- Feature functions live in `src/cobrabox/features/` as individual files, decorated with `@feature`.
- Feature functions take `Data` as first argument, return `xr.DataArray` or `Data`.
- `Data` is immutable — features always produce new instances; never mutate in-place.
- `history` is automatically maintained by the `@feature` decorator.
- Ruff line length is 100; target Python 3.14+.
- `src/cobrabox/features/dummy.py` is a negative reference (bad docstring, has `print`, no validation) — do not model new features after it.

## Agent skills

Two project-level Claude Code skills are in `.claude/skills/`:

- `/review-feature <path>` — audits a feature file for code quality; writes report to `docs/agent-reviews/<feature>.md`.
- `/review-feature-tests <path>` — reviews or generates tests for a feature; writes plan to `docs/agent-reviews/<feature>-tests.md`.

`CLAUDE.md` is a symlink to `agents.md` — edit `agents.md` directly.
