# Test Review Criteria

Criteria for evaluating an existing test file for a cobrabox feature.

---

## 0. Coverage requirement

The project targets **≥95% coverage**. Run `uv run pytest --cov-fail-under=95` to check.
When reviewing tests, consider whether the scenarios below are sufficient to meet this bar
for the feature under review.

**Per-file coverage requirement:** Each feature file in `src/cobrabox/features/` must have
≥95% coverage when running its dedicated test file. This ensures new features do not
depend on incidental coverage from other tests.

```bash
# Check per-file coverage for a specific feature
uv run pytest tests/test_feature_<feature_name>.py --cov=src/cobrabox/features/<feature_name>.py --cov-report=term-missing
```

If a feature has <95% coverage, the review must flag this as a **HIGH severity** issue
and add missing test scenarios to the Action List.

---

## 1. File & function conventions

### File naming

Must be `tests/test_feature_<feature_name>.py`, matching the feature file name exactly.

```text
# ✅
src/cobrabox/features/line_length.py  →  tests/test_feature_line_length.py

# ❌
tests/test_features.py  (too generic, mixes features)
tests/test_line_length.py  (missing feature_ prefix)
tests/line_length_test.py  (wrong convention)
```

### Function naming

Pattern: `test_<feature_name>_<scenario>`. The feature name prefix makes failures
immediately identifiable in pytest output.

```python
# ✅
def test_line_length_basic() -> None:
def test_line_length_history_updated() -> None:
def test_line_length_invalid_dims() -> None:

# ❌ missing feature name prefix
def test_basic() -> None:
def test_history() -> None:
```

### Return annotation

Every test function should have `-> None`. This is recommended for clarity but not strictly enforced.

```python
# ✅
def test_line_length_basic() -> None:

# ❌ (acceptable but not preferred)
def test_line_length_basic():
```

### Docstring

Every test function must have a one-line docstring describing what it verifies.

```python
# ✅
def test_line_length_basic() -> None:
    """LineLength returns correct shape with time dimension removed."""

# ❌ missing docstring
def test_line_length_basic() -> None:
    data = ...
```

---

## 2. Required scenarios

The following scenarios must be covered. A feature without params can skip
"invalid params" — all others are mandatory.

### Happy path

Basic call with valid input succeeds. Assert output shape, dims, and that values
are finite/non-NaN.

```python
# ✅ checks output shape and dims
assert result.data.sizes["space"] == 10
assert "time" not in result.data.dims
assert not np.any(np.isnan(result.to_numpy()))
```

### History updated

After calling the feature via `.apply()`, the **class name** (PascalCase) must appear
at the end of `history`.

```python
# ✅
assert result.history[-1] == "LineLength"

# ❌ old snake_case name — wrong in the new system
assert result.history[-1] == "line_length"

# ❌ only checks history is non-empty
assert len(result.history) > 0
```

### Metadata preserved

`subjectID`, `groupID`, `condition`, and `sampling_rate` from the input must all
survive the feature call unchanged.

```python
# ✅
assert result.subjectID == "s1"
assert result.groupID == "g1"
assert result.condition == "rest"
assert result.sampling_rate == pytest.approx(100.0)

# ❌ only checks one field
assert result.subjectID == "s1"
```

### Output type handling (`output_type = Data`)

When a feature sets `output_type: ClassVar[type[Data]] = Data` (removing the time
dimension), `sampling_rate` should become `None` since there is no time axis.

```python
# ✅ Feature that removes time dimension
assert result.sampling_rate is None

# ✅ Feature that preserves time dimension (default behavior)
assert result.sampling_rate == pytest.approx(100.0)
```

### Invalid dims — missing `time`

**Only required for `BaseFeature[Data]` features** that explicitly check `"time" in data.data.dims`
in their `__call__`. For `BaseFeature[SignalData]` features, `SignalData` enforces the time
dimension at construction — you cannot construct a `SignalData` without `time`, so there is
nothing to test here. Skip this scenario for `BaseFeature[SignalData]` features.

For `BaseFeature[Data]` features with an explicit time-dim guard, use `cb.Data.from_numpy`
directly — no internal bypasses needed:

```python
# ✅
def test_myfeature_missing_time_raises() -> None:
    """MyFeature raises ValueError when 'time' dimension is absent."""
    arr = np.random.default_rng(42).standard_normal((10,))
    data = cb.Data.from_numpy(arr, dims=["space"])
    with pytest.raises(ValueError, match="time"):
        cb.feature.MyFeature().apply(data)

# ❌ no error case tested (for BaseFeature[Data] features that validate time)
```

### Invalid params (skip if feature has no parameters)

Must raise `ValueError` for each field with a constraint (e.g. negative size, zero step).
Validation typically lives in `__post_init__` — test by constructing the class with bad args.

```python
# ✅ for SlidingWindow
with pytest.raises(ValueError):
    cb.feature.SlidingWindow(window_size=0, step_size=10)
with pytest.raises(ValueError):
    cb.feature.SlidingWindow(window_size=10, step_size=0)
# Also test runtime guard (window larger than signal):
with pytest.raises(ValueError):
    cb.feature.SlidingWindow(window_size=1000, step_size=10).apply(short_data)
```

### Output type is `Data`

`.apply()` always returns a `Data` instance (it calls `_copy_with_new_data` internally).

```python
# ✅
assert isinstance(result, cb.Data)

# ❌ checking the underlying array type instead
assert isinstance(result, xr.DataArray)
```

### No mutation of input

The input `Data` object must be unchanged after `.apply()`. Check `history` and shape.

```python
# ✅
original_history = list(data.history)
original_shape = data.data.shape
_ = cb.feature.LineLength().apply(data)
assert data.history == original_history
assert data.data.shape == original_shape
```

---

## 3. Assertion quality

### Prefer specific assertions over generic ones

```python
# ✅
assert result.data.dims == ("space",)
assert result.data.sizes["space"] == 10

# ❌ too loose
assert result is not None
assert result.data is not None
```

### Use `pytest.approx` for floats

```python
# ✅
assert result.sampling_rate == pytest.approx(100.0)

# ❌
assert result.sampling_rate == 100.0
```

### Include `match=` in `pytest.raises` where possible

```python
# ✅
with pytest.raises(ValueError, match="window_size"):

# acceptable (no match)
with pytest.raises(ValueError):
```

---

## 4. Test independence

Each test must be self-contained — create its own `Data` fixture inline or via a
module-level helper. Do not rely on test execution order.

```python
# ✅ each test builds its own data
def test_line_length_basic() -> None:
    """..."""
    arr = np.random.default_rng(42).standard_normal((100, 10))
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    result = cb.feature.LineLength().apply(data)
    ...

# ❌ test depends on module-level mutable state
data = cb.from_numpy(...)  # module-level — shared across tests
```

Small shared helpers (e.g. `_make_data()`) are acceptable and encouraged for
consistency. They must return a fresh object each call to maintain test isolation.
