# Contributing a Feature

This guide shows the recommended workflow for adding a new feature to CobraBox.

## Quick checklist

1. Make a new branch.
2. Make a new file in `src/cobrabox/features/` (for example `my_feature.py`)
3. Make a new file in `tests/` named `test_feature_my_feature.py`
4. Do work
5. Open a pull request. (When feature is finished)

## 1) Create a branch

Create a new branch from `main` before you start.

```bash
git checkout main
git pull
git checkout -b feature/add-mean-absolute-value
```

Pick a branch name that describes the feature.

## 2) Implement the feature

You can start from the template-like `dummy` feature, copy it and then adapt it. (Or just have some AI do it to save time)

Then edit `src/cobrabox/features/my_feature.py`:

- Rename `dummy` to your feature function name.
- Keep the `@dataclass` decorator.
- Choose appropriate base class (`BaseFeature[Data]`, `BaseFeature[SignalData]`, etc.).
- Return either `xr.DataArray` or `Data`.
- Add parameters needed by your feature (for example `dim`, `window_size`, etc.).
- Set `output_type = Data` if your feature removes the time dimension (e.g., correlation matrices).

Look at existing implementations for patterns:

- `src/cobrabox/features/line_length.py`
- `src/cobrabox/features/sliding_window.py`
- `src/cobrabox/features/mean.py`

Note: feature auto-discovery is enabled, so you do not need to manually register
the new feature in `features/__init__.py` or `feature.py`.

## 3) Add tests

You can start from the dummy feature test and adapt it for your new feature.

Then edit `tests/test_feature_my_feature.py`:

- Rename the test function(s).
- Verify expected numeric output for at least one simple known input.
- Verify metadata/history behavior where relevant.
- Add negative tests for invalid inputs/arguments.

Look at other feature tests for examples:

- `tests/test_feature_dummy.py`
- `tests/test_feature_line_length.py`
- `tests/test_feature_sliding_window.py`
- `tests/test_feature_mean.py`

Run tests locally:

```bash
uv run pytest -q
```

## 4) Commit and open a pull request

Small commits during development are good practice.

Example commit flow:

```bash
git add src/cobrabox/features/my_feature.py
git commit -m "add my_feature implementation"

git add tests/test_feature_my_feature.py
git commit -m "add tests for my_feature"
```

When complete:

```bash
git status
git push -u origin feature/add-mean-absolute-value
```

Create the pull request in the GitHub website.
For simple step-by-step instructions, see:
[`docs/how_to_make_a_pr.md`](how_to_make_a_pr.md)

In the pull request description, include:

- What the feature computes
