---
name: migrate-feature
description: This skill should be used when the user asks to "migrate this feature", "convert old-style feature", "update feature to new pattern", "migrate from @feature decorator", "port feature to class-based", "rebase onto feature-class", or when working on a feature branch that uses the old function_wrapper pattern (imports from cobrabox.function_wrapper, uses @feature decorator, or defines features as plain functions).
---

# Migrate Feature

Migrate an old-style `@feature`-decorated function to the new `BaseFeature` dataclass pattern.

## Branch Strategy

`main` currently uses the old `@feature` decorator pattern. The `feature-class` branch introduces
the new class-based system and **deletes `function_wrapper.py`**.

The intended workflow for contributors is:

1. **Merge your feature branch into `main` first** (old-style, as-is — no migration needed yet)
2. **Rebase your branch onto `feature-class`** — this is when `function_wrapper` disappears and
   import errors appear
3. **Run this migration** to convert the feature and its tests to the new pattern
4. Open a PR against `feature-class`

To rebase:
```bash
git fetch origin
git rebase origin/feature-class
# Resolve any conflicts, then run the migration steps below
```

After rebasing, broken files will have import errors like:
```
ModuleNotFoundError: No module named 'cobrabox.function_wrapper'
```
Those are the files to migrate.

## Quick Reference

| Old pattern                                            | New pattern                                                                  |
| ------------------------------------------------------ | ---------------------------------------------------------------------------- |
| `from cobrabox.function_wrapper import feature`        | `from ..base_feature import BaseFeature`                                     |
| `@feature` on a function                               | `@dataclass` on a class                                                      |
| `def my_feature(data: Data, ...)` (time-series)        | `class MyFeature(BaseFeature[SignalData])`                                   |
| `def my_feature(data: Data, ...)` (any dims)           | `class MyFeature(BaseFeature[Data])`                                         |
| `def sliding_window(data, ...) -> xr.DataArray` (windowing) | `class MySplit(SplitterFeature[SignalData]): ... -> Iterator[Data]`     |
| Manual aggregation loop                                | `class MyAgg(AggregatorFeature): ... -> Data`                               |
| `cb.feature.my_feature(data, param=val)`               | `cb.feature.MyFeature(param=val).apply(data)`                                |
| `cb.feature.my_feature \| cb.feature.other`            | `cb.feature.MyFeature() \| cb.feature.Other()`                              |
| `cb.from_numpy(arr, dims=[...])` in tests              | `cb.SignalData.from_numpy(arr, dims=[...])` for time-series tests            |
| Feature returns correlation matrix (no time dim)       | `output_type: ClassVar[type[Data]] = Data`                                   |
| Old output expands singleton `space`/`time` dims       | Drop the `expand_dims` — return the bare result DataArray                    |

## Procedure

### 1. Read the feature file

Read the file in full. Identify which pattern applies:

- **Standard transformation** (`Data → Data` or `Data → xr.DataArray`): use `BaseFeature`
- **Produces a stream of windows** (`Data → multiple Data`): use `SplitterFeature`
- **Folds a stream back into one result**: use `AggregatorFeature`

### 2. Rewrite as a dataclass

Apply the appropriate template from `references/patterns.md`.

Key rules:
- Class name is PascalCase of the old function name (`line_length` → `LineLength`)
- Filename stays snake_case and must match the class name (`line_length.py`)
- Parameters become dataclass fields; `data` is NOT a field — it is the argument to `__call__`
- Drop `**kwargs`; list every accepted parameter explicitly as a typed field
- The return type of `__call__` is `xr.DataArray | Data` for `BaseFeature`, `Iterator[Data]` for
  `SplitterFeature`, and `Data` for `AggregatorFeature`
- Do NOT implement `apply()` — it is inherited from `BaseFeature` and handles history automatically
- `AggregatorFeature` must build `history` manually (see patterns.md for the template)
- **Convert private/helper functions to class methods**: If the old file has helper functions (e.g.,
  functions starting with `_`), convert them to private methods on the new class. Move the logic
  inside the class as `def _helper(self, ...)` methods. Only use `@staticmethod` if the helper
  genuinely does not need access to `self` or any instance state. If the helper is shared across
  multiple feature functions in the same file, keep it as a module-level private function instead.
- **Strip legacy singleton dimensions**: Old `@feature` functions often wrapped scalar or matrix
  results in `.expand_dims("time").expand_dims("space")` to satisfy the old decorator's wrapping
  contract. In the new pattern this is unnecessary — `apply()` handles wrapping via
  `_copy_with_new_data`. Remove these `expand_dims` / `assign_coords` chains and return the bare
  result DataArray (scalar `xr.DataArray(value)` or `xr.DataArray(matrix, dims=[...])`).

### 3. Update imports

Replace:
```python
from cobrabox.function_wrapper import feature
```
With the appropriate import for the chosen base class:
```python
from ..base_feature import BaseFeature          # standard
from ..base_feature import SplitterFeature      # windowing / splitting
from ..base_feature import AggregatorFeature    # folding a stream
```

Also add `from __future__ import annotations` and `from dataclasses import dataclass` (plus
`field` if any field has a non-trivial default). Import `SignalData` from `..data` for
features that operate on time-series data, or `Data` for dimension-agnostic features.

### 4. Add or update the docstring

Write a Google-style docstring with:
- One-line summary
- `Args:` section listing each field (not `data`)
- `Example:` block showing `MyFeature(param=val).apply(data)`

### 5. Update tests

Old-style tests call the function directly:
```python
result = cb.feature.line_length(data)
```
New-style tests call `.apply()`:
```python
result = cb.feature.LineLength().apply(data)
```

Update assertions:
- History entry is the class name: `assert result.history == ["LineLength"]`  (not `"line_length"`)
- For `SplitterFeature`: the call returns a generator, not a `Data`; iterate or call inside a Chord
- For `AggregatorFeature`: test with a real stream via a Chord or by constructing `Data` objects

### 6. Lint and format

```bash
uvx ruff check --fix src/cobrabox/features/<file>.py tests/test_feature_<name>.py
uvx ruff format src/cobrabox/features/<file>.py tests/test_feature_<name>.py
```

### 7. Smoke-check

Run the tests for the migrated feature:
```bash
uv run pytest tests/test_feature_<name>.py -v
```

---

## Additional Resources

- **`references/patterns.md`** — full before/after code for all three feature types, including
  edge cases (default field values, `field()`, `AggregatorFeature` history propagation)
