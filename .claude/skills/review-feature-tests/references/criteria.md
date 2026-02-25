# Test Review Criteria

Criteria for evaluating an existing test file for a cobrabox feature.

The project targets **≥95% coverage**. Run `uv run pytest --cov-fail-under=95` to check.
When reviewing tests, consider whether the scenarios below are sufficient to meet this bar
for the feature under review.

---

## 1. File & function conventions

### File naming

Must be `tests/test_<feature_name>.py`, matching the feature file name exactly.

```text
# ✅
src/cobrabox/features/line_length.py  →  tests/test_line_length.py

# ❌
tests/test_features.py  (too generic, mixes features)
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

Every test function must have `-> None`.

```python
# ✅
def test_line_length_basic() -> None:

# ❌
def test_line_length_basic():
```

### Docstring

Every test function must have a one-line docstring describing what it verifies.

```python
# ✅
def test_line_length_basic() -> None:
    """line_length returns correct shape with time dimension removed."""

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
assert not np.any(np.isnan(result.asnumpy()))
```

### History updated

After calling the feature, the feature name must appear at the end of `history`.

```python
# ✅
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

### Invalid dims — missing `time`

Must raise `ValueError` when input lacks `time` dimension.

```python
# ✅
with pytest.raises(ValueError, match="time"):
    cb.feature.line_length(data_without_time)

# ❌ no error case tested
```

### Invalid params (skip if feature has no parameters)

Must raise `ValueError` for each parameter with a constraint (e.g. negative size,
zero step).

```python
# ✅ for sliding_window
with pytest.raises(ValueError):
    cb.feature.sliding_window(data, window_size=0)
with pytest.raises(ValueError):
    cb.feature.sliding_window(data, window_size=1000)  # larger than signal
```

### Output type is `Data`

The `@feature` decorator always repacks the result into a `Data` instance. Verify this.

```python
# ✅
assert isinstance(result, cb.Data)

# ❌ checking the underlying array type instead
assert isinstance(result, xr.DataArray)
```

### No mutation of input

The input `Data` object must be unchanged after the feature call. Check `history`
and shape.

```python
# ✅
original_history = list(data.history)
original_shape = data.data.shape
_ = cb.feature.line_length(data)
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
    arr = np.random.randn(100, 10)
    data = cb.from_numpy(arr, dims=["time", "space"], sampling_rate=100.0)
    result = cb.feature.line_length(data)
    ...

# ❌ test depends on module-level mutable state
data = cb.from_numpy(...)  # module-level — shared across tests
```

Small shared helpers (e.g. `_make_data()`) are fine if they return a fresh object
each call.
