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

### `@dataclass` decorator + base class inheritance

The class must be decorated with `@dataclass` and inherit exactly one of the three base classes.
The correct base class depends on what the feature does:

| Base class                  | Use when                                      |
| --------------------------- | --------------------------------------------- |
| `BaseFeature[Data]`         | standard transformation, no time dimension required |
| `BaseFeature[SignalData]`   | transformation that requires a `time` dimension |
| `SplitterFeature[Data]`     | fan-out: yields multiple `Data` windows (generic) |
| `SplitterFeature[SignalData]` | fan-out: yields windows over the time dimension |
| `AggregatorFeature`         | fan-in: folds a stream back into one `Data`   |

The type parameter narrows what `__call__` accepts. Use `SignalData` for any feature that
operates on the time axis; use `Data` for dimension-agnostic features.

### `output_type` classvar (optional)

Features **only need to set `output_type` when changing from the default behavior**. The base class provides sensible defaults:

| `output_type` value | Result container | When to set it |
| ------------------- | ---------------- | -------------- |
| `None` (default)    | Same as input    | **Omit this** - preserves input container type automatically |
| `Data`              | Plain `Data`     | **Only set when** removing time dimension (e.g., correlation matrices) |

```python
# ✅ Feature that preserves container type - OMIT output_type entirely
@dataclass
class Spectrogram(BaseFeature[SignalData]):
    # No output_type needed - defaults to preserving SignalData
    ...

# ✅ Feature that returns plain Data - ONLY set when changing type
@dataclass
class Coherence(BaseFeature[SignalData]):
    output_type: ClassVar[type[Data]] = Data  # Removes time dimension
    ...
```

**Rule of thumb**: If your feature preserves the time dimension, don't declare `output_type`. Only declare it when you need to return a different container type than what was passed in.

```python
# ✅ time-series feature
@dataclass
class LineLength(BaseFeature[SignalData]):

# ✅ generic (any dims)
@dataclass
class Mean(BaseFeature[Data]):

# ✅
@dataclass
class SlidingWindow(SplitterFeature[SignalData]):

# ✅
@dataclass
class MeanAggregate(AggregatorFeature):

# ❌ missing @dataclass
class LineLength(BaseFeature[SignalData]):

# ❌ missing type parameter (still works at runtime, but incomplete typing)
class LineLength(BaseFeature):

# ❌ wrong base (function instead of class)
@feature
def line_length(data: Data, ...) -> xr.DataArray:
```

### Class name

Must be PascalCase matching the filename (`line_length.py` → `LineLength`).

#### Descriptive naming required

Class names must be descriptive and clearly indicate what the feature computes. Avoid abbreviations, acronyms, or shortened forms that are not immediately obvious.

```python
# ✅ Descriptive, clear names
class LineLength(BaseFeature[SignalData]):
class AmplitudeEntropy(BaseFeature[Data]):
class MutualInformation(BaseFeature[SignalData]):
class PhaseLockingValue(BaseFeature[SignalData]):

# ❌ Abbreviated, unclear names
class Ampent(BaseFeature[Data]):  # What is "Ampent"?
class Mi(BaseFeature[SignalData]):  # Abbreviation
class Pldv(BaseFeature[SignalData]):  # Acronym
class Fd(BaseFeature[Data]):  # Too short, unclear
```

**Guidelines:**

- Use full words: `AmplitudeEntropy` not `Ampent`
- Use widely recognized acronyms only if they are domain-standard: `PCA`, `FFT` are acceptable
- When in doubt, prefer the longer, clearer name
- The class name should describe *what* is being computed, not *how* it's implemented

### `_is_cobrabox_feature` marker (NOT NEEDED for class-based features)

**Class-based features automatically inherit `_is_cobrabox_feature = True` from `BaseFeature`.**

```python
# ✅ Class-based features - NO marker needed, inherited from BaseFeature
@dataclass
class LineLength(BaseFeature[SignalData]):
    # _is_cobrabox_feature is inherited as True
    ...

# ❌ Unnecessary - already inherited
_is_cobrabox_feature = True  # Don't add this for class-based features!

@dataclass
class LineLength(BaseFeature[SignalData]):
    ...
```

The marker is only needed at the **module level** for legacy function-based features (not using the dataclass pattern). Since all modern features use the class-based pattern, **never add `_is_cobrabox_feature = True` to new feature files**.

### `__call__` signature

`data` is the **argument** to `__call__`, not a class field. Parameters are dataclass fields.
The `data` type must match the type parameter of the base class.

```python
# ✅ BaseFeature[Data] (generic feature)
def __call__(self, data: Data) -> xr.DataArray | Data:

# ✅ BaseFeature[SignalData] (time-series feature)
def __call__(self, data: SignalData) -> xr.DataArray | Data:

# ✅ SplitterFeature[SignalData]
def __call__(self, data: SignalData) -> Iterator[Data]:

# ✅ AggregatorFeature
def __call__(self, data: Data, windows: Iterator[Data]) -> Data:

# ❌ data as a field
@dataclass
class LineLength(BaseFeature[SignalData]):
    data: SignalData  # wrong — data is never a field
```

### Do NOT implement `apply()`

`apply()` is inherited from `BaseFeature` and handles history and wrapping automatically.
Only `AggregatorFeature` subclasses must build `history` themselves.

### Imports

Only import what is used. No unused imports. Standard order:

1. `from __future__ import annotations`
2. stdlib (including `from dataclasses import dataclass, field` and `from collections.abc import Iterator`)
3. third-party (`numpy`, `xarray`)
4. internal (`from ..data import Data`, `from ..base_feature import BaseFeature`)

No import of `cobrabox.function_wrapper` — that module no longer exists.

---

## 2. Docstring

All features require a Google-style docstring with all required sections. None are optional.

### One-line summary

First line: concise verb phrase, no period needed but consistent with existing style.

```python
# ✅
"""Compute line length of the signal along the time dimension."""

# ❌ too vague
"""Feature class."""

# ❌ missing entirely
```

### Extended description

Required if the algorithm or behaviour is non-trivial. Skip only for genuinely
self-evident features. When present, explain *what* and *why*, not *how*.

### `Args:` section

Every **field** (dataclass parameter) needs a type and description. `data` is the
`__call__` argument and should not appear in `Args:` — it is described contextually
or omitted if obvious.

```python
# ✅
Args:
    window_size: Number of timepoints per window. Must be > 0.
    step_size: Step between consecutive windows. Must be > 0.

# ❌ missing Args section when fields exist

# ❌ documenting data as if it were a field
Args:
    data: Input Data object.
    window_size: ...
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

#### No fake singleton dimensions

Features that return scalar values (0-dimensional output) should return a proper 0-d xarray DataArray, NOT an array with fake singleton dimensions like `(1, 1)` with made-up dim names.

```python
# ✅ Return a proper 0-dimensional scalar
return xr.DataArray(scalar_value)

# ✅ Return reduced dimensions naturally
return data.sum(dim="time")  # Shape: (space,) not (1, space)

# ❌ Creating fake singleton dimensions
return xr.DataArray([[scalar]], dims=["time", "space"], coords={"time": [0], "space": [0]})

# ❌ Made-up dimension names that don't exist in input
return xr.DataArray(result, dims=["dim1", "dim2"])  # Where dim1/dim2 are not from input
```

Use `xr.apply_ufunc(..., output_core_dims=[[]])` for dimensionality reduction to scalars, or simply return the reduced array which naturally drops the reduced dimension.

### `Raises:` section

Recommended when the feature raises `ValueError` in `__call__` or `__post_init__`.
List each raised exception type and the condition that triggers it.

```python
# ✅
Raises:
    ValueError: If ``method`` is not a valid sift method.
    ValueError: If ``max_imfs`` is not positive.

# acceptable — omit if the feature raises nothing
```

### `References:` section

Required for features implementing a published algorithm. Include full citation(s)
so reviewers can verify correctness against the source. Omit for self-evident operations
(mean, max, etc.) with no specific algorithm reference.

```python
# ✅
References:
    Leuchter, A. F., et al. (1994). Cordance: a new method...
    *NeuroImage*, 1(3), 208-219.

# acceptable — omit for features with no literature basis
```

### `Example:` section

At least one working snippet showing typical usage via `.apply()`.

```python
# ✅
Example:
    >>> result = cb.feature.LineLength().apply(data)

    >>> chord = cb.feature.SlidingWindow(window_size=20, step_size=10) | \
    ...         cb.feature.LineLength() | \
    ...         cb.feature.MeanAggregate()
    >>> result = chord.apply(data)

# ❌ missing Example section

# ❌ old function-call style
Example:
    >>> result = cb.feature.line_length(data)
```

---

## 3. Typing

### All fields typed

Every dataclass field must have a type annotation. No bare untyped fields.

```python
# ✅
@dataclass
class SlidingWindow(SplitterFeature):
    window_size: int
    step_size: int = 1

# ❌ untyped field
@dataclass
class SlidingWindow(SplitterFeature):
    window_size = 10
```

### `__call__` return type annotation

Must be explicit and match the base class contract:

- `BaseFeature.__call__` → `xr.DataArray | Data`
- `SplitterFeature.__call__` → `Iterator[Data]`
- `AggregatorFeature.__call__` → `Data`

### No bare `Any`

`Any` is only acceptable with an inline comment explaining why it cannot be narrowed.

---

## 4. Safety & Style

### No `print()` statements

`print()` is not acceptable in feature classes. Use `logging` if output is needed.

```python
# ❌ — from dummy.py
print("whatever", whatever)

# ✅
import logging
logger = logging.getLogger(__name__)
logger.debug("whatever: %s", whatever)
```

### Input validation in `__call__`

Features should validate critical preconditions and raise `ValueError` with a clear message.

For **`BaseFeature[SignalData]`** features: `SignalData` already enforces the `time` dimension
at construction time, so a redundant check in `__call__` is not required. Focus validation on
feature-specific constraints (e.g. required non-time dims, parameter/data interactions).

For **`BaseFeature[Data]`** features that require `time`: validate explicitly:

```python
# ✅ required for Data-typed features that need time
if "time" not in data.data.dims:
    raise ValueError("data must have 'time' dimension")

# ❌ no validation — will produce a cryptic xarray error downstream
```

Dataclass fields with numeric constraints should be validated in `__post_init__`:

```python
# ✅
def __post_init__(self) -> None:
    if self.window_size <= 0:
        raise ValueError(f"window_size must be positive, got {self.window_size}")
```

### No mutation of input `data`

`Data` is immutable. Features must not attempt to modify the input object. Always
work on `data.data` (the underlying `xr.DataArray`) and return a new array or object.

```python
# ❌
data._data = result  # signals intent to mutate

# ✅
xr_data = data.data
result = xr_data.diff(dim="time")
return result
```

### Line length

Maximum 100 characters per line (enforced by ruff).

---

## Reference examples

### Positive reference: `src/cobrabox/features/line_length.py`

Compliant feature. Uses `BaseFeature[SignalData]` with `data: SignalData` in `__call__`.
Has all docstring sections, typed fields, `__call__` return type, no print statements,
`from __future__ import annotations`. No redundant time-dim check since `SignalData` enforces it.

### Negative reference: `src/cobrabox/features/dummy.py`

**SKIP this file during reviews.** It is intentionally non-compliant as a negative reference example.
The docstring explicitly warns: "This feature exists as a negative reference showing what a poorly
written feature looks like. Do not use this as a template for new features."

Known intentional issues:

- Missing type parameter on `BaseFeature` (should be `BaseFeature[SignalData]`)
- Wrong `__call__` return type (`Data` instead of `xr.DataArray | Data`)
- Missing `__post_init__` validation
- Drops metadata (subjectID, groupID, condition, extra)
