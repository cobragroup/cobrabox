# Feature Review Criteria

Reference for evaluating a cobrabox feature file. Four categories, checked in order.

---

## 1. Signature & Structure

### `from __future__ import annotations`

Must be the first import line (after module docstring if any).

```python
# ✅
from __future__ import annotations

# ❌ missing entirely
import numpy as np
```

### `@feature` decorator

Must be present and must be the outermost decorator on the function.

```python
# ✅
@feature
def my_feature(data: Data, ...) -> xr.DataArray:

# ❌ missing decorator — function will not be auto-discovered
def my_feature(data: Data, ...) -> xr.DataArray:

# ❌ decorator not outermost
@some_other_decorator
@feature
def my_feature(...):
```

### First parameter

Must be `data: Data` with no default value.

```python
# ✅
def my_feature(data: Data, window_size: int = 10) -> xr.DataArray:

# ❌ wrong name
def my_feature(input: Data, ...) -> xr.DataArray:

# ❌ no type annotation on first param
def my_feature(data, ...) -> xr.DataArray:
```

### Return type annotation

Must be `xr.DataArray` or `Data` (or `xr.DataArray | Data`). Must be explicit.

```python
# ✅
def my_feature(data: Data) -> xr.DataArray:

# ✅
def my_feature(data: Data) -> Data:

# ❌ missing return type
def my_feature(data: Data):

# ❌ too broad
def my_feature(data: Data) -> Any:
```

### Imports

Only import what is used. No unused imports. Standard order:

1. `from __future__ import annotations`
2. stdlib
3. third-party (`numpy`, `xarray`)
4. internal (`from ..data import Data`, `from ..function_wrapper import feature`)

---

## 2. Docstring

All features require a Google-style docstring with all four sections. None are optional.

### One-line summary

First line: concise verb phrase, no period needed but consistent with existing style.

```python
# ✅
"""Compute line length of the signal along the time dimension."""

# ❌ too vague
"""Feature function."""

# ❌ missing entirely
```

### Extended description

Required if the algorithm or behaviour is non-trivial. Skip only for genuinely
self-evident functions. When present, explain *what* and *why*, not *how*.

### `Args:` section

Every parameter except `data` needs a type and description. `data` should describe
which dimensions and any preconditions.

```python
# ✅
Args:
    data: Input signal with 'time' and 'space' dimensions.
        If produced by sliding_window, 'window_index' is preserved.
    window_size: Number of timepoints per window. Must be > 0.

# ❌ missing Args section entirely

# ❌ Args present but parameters not documented
Args:
    data: Data object.
```

### `Returns:` section

Describe the shape, dimensions, and what the values represent.

```python
# ✅
Returns:
    xarray DataArray with 'time' dimension removed. Shape: (space,) or
    (window_index, space) if input had window_index.

# ❌ missing Returns section

# ❌ vague
Returns:
    DataArray.
```

### `Example:` section

At least one working snippet showing typical usage via `cb.feature.*`.

```python
# ✅
Example:
    >>> wdata = cb.feature.sliding_window(data, window_size=20, step_size=10)
    >>> feat = cb.feature.my_feature(wdata)

# ❌ missing Example section
```

---

## 3. Typing

### All parameters typed

Every parameter must have a type annotation. No bare untyped parameters.

```python
# ✅
def my_feature(data: Data, window_size: int = 10, threshold: float = 0.5) -> xr.DataArray:

# ❌ untyped parameter
def my_feature(data: Data, window_size=10) -> xr.DataArray:
```

### No bare `Any`

`Any` is only acceptable with an inline comment explaining why it cannot be narrowed.

```python
# ❌
def my_feature(data: Data, config: Any = None) -> xr.DataArray:

# ✅ (justified)
def my_feature(data: Data, config: Any = None) -> xr.DataArray:
    # config accepts heterogeneous user-supplied dicts; cannot narrow further
```

### Return type matches actual return

If the function returns `xr.DataArray`, the annotation must say `xr.DataArray`, not
`Data`. If it returns a `Data` instance, annotate as `Data`.

---

## 4. Safety & Style

### No `print()` statements

`print()` is not acceptable in feature functions. Use `logging` if output is needed.

```python
# ❌ — from dummy.py
print("whatever", whatever)

# ✅
import logging
logger = logging.getLogger(__name__)
logger.debug("whatever: %s", whatever)
```

### Input validation

Features should validate critical preconditions and raise `ValueError` with a clear
message. At minimum, validate that required dimensions are present.

```python
# ✅
if "time" not in data.data.dims:
    raise ValueError("data must have 'time' dimension")

# ❌ no validation — will produce a cryptic xarray error downstream
```

Parameters with numeric constraints should also be validated:

```python
# ✅
if window_size <= 0:
    raise ValueError(f"window_size must be positive, got {window_size}")
```

### No mutation of input `data`

`Data` is immutable. Features must not attempt to modify the input object. Always
work on `data.data` (the underlying `xr.DataArray`) and return a new array or object.

```python
# ❌
data._data = result  # will raise AttributeError but signals intent to mutate

# ✅
xr_data = data.data
result = xr_data.diff(dim="time")
return result
```

### Line length

Maximum 100 characters per line (enforced by ruff, but flag narratively if ruff is
not catching it due to configuration drift).

---

## Reference examples

### Positive reference: `src/cobrabox/features/line_length.py`

Compliant feature. Has all docstring sections, typed params, return type, input
validation, no print statements, `from __future__ import annotations`.

### Negative reference: `src/cobrabox/features/dummy.py`

Non-compliant. Issues:

- Incomplete docstring (one-liner only, no Args/Returns/Example sections)
- `print("whatever", whatever)` on line 16
- No input validation
- Returns `Data` but has no explicit return type annotation issue (actually annotated
  correctly), but the `print` and missing docstring sections are clear violations
