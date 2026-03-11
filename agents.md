# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses `uv` for package and environment management.

```bash
# First-time setup (installs git-lfs hooks, syncs deps, installs pre-commit)
make setup

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

**Core data containers** (`src/cobrabox/data.py`):

- `Data` — General immutable wrapper around `xarray.DataArray` with no dimension requirements. Stores metadata (`subjectID`, `groupID`, `condition`, `sampling_rate`, `history`, `extra`) in xarray attrs. `sampling_rate` is `None` if no time dimension. Construct via `cb.Data.from_numpy()` or `cb.Data.from_xarray()`.
- `SignalData` — Time-series data container that requires a 'time' dimension. Automatically transposes data to put time last for performance. Use for EEG/fMRI and any time-series analysis. Inherits from `Data`.
- `EEG` / `FMRI` — Type markers inheriting from `SignalData` for EEG and fMRI data respectively.
- `Dataset[T]` — Generic, immutable sequence of `Data` objects with `.filter()`, `.groupby()`, `.describe()`, indexing, slicing, and `+` for concatenation.

Class hierarchy: `Data` ← `SignalData` ← (`EEG`, `FMRI`)

**Feature system** (`src/cobrabox/base_feature.py`, `src/cobrabox/feature.py`, `src/cobrabox/features/`): Features are `@dataclass` classes that inherit one of three base classes:

- `BaseFeature` (`Data → Data`): standard feature; implement `__call__`; call `.apply(data)` which wraps the result via `_copy_with_new_data` and appends the class name to `history`. Supports pipe syntax: `Feature1() | Feature2()` produces a `Pipeline`.
  - Use `output_type: ClassVar[type[Data]] = Data` when the output removes the time dimension — scalars, spatial matrices, frequency-only arrays. Without this, `apply()` tries to preserve the input container type and may fail.
  - Default (`output_type = None`) preserves input container type.
- `SplitterFeature` (`Data → Iterator[Data]`): yields one `Data` per split (e.g. `SlidingWindow`). Lazy generator — does not materialise all splits in memory.
- `AggregatorFeature` (`(Data, Iterator[Data]) → Data`): folds a stream back into one `Data` (e.g. `MeanAggregate`); responsible for merging per-window history into the result.
- `Chord(BaseFeature)`: composes a `SplitterFeature` + `BaseFeature`/`Pipeline` + `AggregatorFeature` into a single `BaseFeature` (fan-out → map → fan-in). Itself composable with `|`.

Feature discovery is automatic: `feature.py` scans all modules in `features/` (including subdirectories) and registers any callable with `_is_cobrabox_feature = True` **and** `__module__ == <that module>` (the `__module__` filter prevents base classes imported into feature files from being registered as duplicates). Adding a new feature means creating a new file in `features/<domain>/` with a class inheriting the appropriate base — no manual registration needed. Domain directories: `time_domain/`, `frequency_domain/`, `time_frequency/`, `connectivity/`, `decomposition/`, `windowing/`, `reductions/`.

**Datasets** (`src/cobrabox/datasets.py`, `src/cobrabox/dataset_loader.py`, `src/cobrabox/dataset.py`): `cb.dataset(name)` returns a `Dataset[SignalData]`. `Dataset[T]` is a generic, immutable sequence of `Data` objects (`src/cobrabox/dataset.py`) with helpers: `.filter(subjectID=..., groupID=..., condition=...)` → `Dataset[T]`, `.groupby("groupID")` → `dict[str, Dataset[T]]`, `.describe()` prints a summary. Supports indexing, slicing, iteration, and `+` for concatenation. Built-in dummy datasets (`dummy_chain`, `dummy_random`, `dummy_star`, `dummy_noise`) are loaded from compressed CSV files in `data/dummy/`. The datasets differ in structure:
- `dummy_chain`: Sequential data with known ground truth
- `dummy_random`: Random Gaussian noise
- `dummy_star`: Star-shaped pattern with one central channel
- `dummy_noise`: High-dimensional noise for stress testing

**Public API** (`src/cobrabox/__init__.py`): Top-level imports expose `Data`, `SignalData`, `EEG`, `FMRI`, `dataset`, `from_numpy`, `from_xarray`, the base classes `BaseFeature`, `SplitterFeature`, `AggregatorFeature`, `Pipeline`, and `Chord`, plus hardcoded imports of key feature classes (`LineLength`, `SlidingWindow`, `MeanAggregate`, `Nonreversibility`, `RecurrenceMatrix`, `ConcatAggregate`). The `feature` submodule is also accessible as `cb.feature.*` (auto-discovered). **Note:** `globals().update()` in `features/__init__.py` is opaque to IDEs/type-checkers; the `# noqa: PLE0604` there is intentional. See `docs/plans/2026-02-27-feature-autodiscovery-static-analysis.md` for the open decision on a permanent fix.

**Feature D&D Alignments**: Every feature has a D&D alignment capturing its "moral character" — how it treats your data. Law axis: +1 = Lawful (imposes structure), 0 = Neutral (passive), -1 = Chaotic (disruptive). Good axis: +1 = Good (preserves meaning), 0 = Neutral (indifferent), -1 = Evil (discards). Use `uv run python -m cobrabox.dnd_alignment --roster` to see the full grid, or `uv run python -m cobrabox.dnd_alignment F1 F2 F3` to compute a pipeline's aggregate alignment.

**Public API** (`src/cobrabox/__init__.py`): Top-level imports expose `Data`, `SignalData`, `EEG`, `FMRI`, `dataset`, `from_numpy`, `from_xarray`, the base classes `BaseFeature`, `SplitterFeature`, `AggregatorFeature`, `Pipeline`, and `Chord`, plus hardcoded imports of key feature classes (`LineLength`, `SlidingWindow`, `MeanAggregate`, `Nonreversibility`, `RecurrenceMatrix`, `ConcatAggregate`). The `feature` submodule is also accessible as `cb.feature.*` (auto-discovered). **Note:** `globals().update()` in `features/__init__.py` is opaque to IDEs/type-checkers; the `# noqa: PLE0604` there is intentional. See `docs/plans/2026-02-27-feature-autodiscovery-static-analysis.md` for the open decision on a permanent fix.

**Serialization** (`src/cobrabox/serialization.py`): `cb.save(obj, path)` / `cb.load(path)` for file I/O; `cb.serialize(obj)` / `cb.deserialize(content)` for string I/O. Also accessible as `cb.serialization.*`. All objects serialize as a `pipeline:` YAML/JSON list — single features and chords are wrapped as one-element lists. `BaseFeature` and `Pipeline` expose `.to_yaml()` / `.from_yaml()` / `.to_dict()` / `.from_dict()` convenience methods. Callable parameters use `dill` (mandatory dependency). Schema version is tracked in every file; major version bump raises `SchemaVersionError`.

## Key conventions

- Feature classes live in `src/cobrabox/features/<domain>/` as individual files, one class per file.
- Each file defines a `@dataclass` class inheriting `BaseFeature`, `SplitterFeature`, or `AggregatorFeature` from `src/cobrabox/base_feature.py`.
- Base classes are generic: `BaseFeature[DataT]` and `SplitterFeature[DataT]`. Features that require time dimension should use `BaseFeature[SignalData]` or `SplitterFeature[SignalData]`; generic features use `BaseFeature[Data]`.
  - Use `[SignalData]` when the algorithm **inherently** needs time-series structure (uses `sampling_rate`, applies Hilbert/FFT on the time axis, etc.). Use `[Data]` when `dim` is a user-configurable parameter — even if callers always pass time-series data.
- `BaseFeature.__call__` takes `DataT`, returns `xr.DataArray | Data`. Use `.apply(data)` externally — it handles wrapping and history.
- `SplitterFeature.__call__` takes `DataT`, yields `Data` (generator). No `.apply()` — used inside `Chord`.
- `AggregatorFeature.__call__` takes `(Data, Iterator[Data])`, returns `Data`. Must propagate per-window history manually.
- `Data` is immutable — features always produce new instances via `_copy_with_new_data`; never mutate in-place.
- `history` is automatically maintained by `BaseFeature.apply`; `AggregatorFeature` subclasses are responsible for building history themselves.
- Ruff line length is 100; target Python 3.14+ (`target-version = "py314"`); `requires-python = ">=3.11"`.
- Test files for features are named `tests/test_feature_<feature_name>.py` (e.g. `line_length.py` → `test_feature_line_length.py`).
- `src/cobrabox/features/_dummy.py` is a negative reference (no useful docstring, no validation) — do not model new features after it.

## Build & CI

- Build backend: `uv_build` — auto-discovers `src/cobrabox` from the project name; use `uv publish` for PyPI.
- GitHub Actions: `.github/workflows/tests.yml` is a reusable workflow (inputs: `python-version`, `os`). `ci.yml` calls it on every push (defaults). `pr.yml` uses an explicit `include` matrix: Linux runs 3.11–3.14, Windows and macOS run 3.14 only, plus coverage and lint jobs.
- **Editing workflow files is blocked by a security hook** — `Edit`/`Write` tools are rejected on `.github/workflows/*.yml`. Use `Bash` with a heredoc: `cat > file.yml << 'WORKFLOW' ... WORKFLOW`.
- `EnricoMi/publish-unit-test-result-action` requires `contents: read` permission or it fails with 403.

## Agent skills

Four project-level Claude Code skills are in `.claude/skills/`:

- `/migrate-feature <path>` — migrates an old-style `@feature`-decorated function to the new `BaseFeature` dataclass pattern; also updates tests.
- `/review-feature <path>` — audits a feature file for code quality; writes report to `docs/agent-reviews/<feature>.md`.
- `/review-feature-tests <path>` — reviews or generates tests for a feature; writes plan to `docs/agent-reviews/<feature>-tests.md`.
- `/dnd-alignment [features...]` — rates features/pipelines on the D&D 9-alignment grid; no args prints full roster.
