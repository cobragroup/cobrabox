# Test Review: fractal_dimension

**Feature**: `src/cobrabox/features/fractal_dimension.py`
**Test file**: `tests/test_feature_fractal_dimension.py`
**Date**: 2026-03-05
**Verdict**: NEEDS WORK

## Coverage

```
fractal_dimension.py: 94% (50 statements, 3 missing — lines 88-89, 157)
```

Coverage is **94%**, just below the required 95% threshold. Three lines are uncovered:

- **Lines 88-89** (`_higuchi_1d`): the `n_steps == 0` fallback. Reachable when
  `N == k_max + 1` (e.g. N=11, k_max=10, k=10, m=10 → `n_steps = 1 // 10 = 0`).
- **Line 157** (`_katz_1d`): `return 1.0` degenerate guard. With `x = np.arange(N)`,
  `eu_length` and `max_dist` are always > 0 for N ≥ 2 (the x-component contributes
  at least `sqrt(1)` to every distance), so this line is dead code and should be
  removed from the feature rather than tested.

## Summary

The test file is well-structured with good docstrings, `-> None` annotations, correct
file naming, and comprehensive algorithmic coverage. The main gaps are: coverage just
below 95% (one missing test, one dead guard to remove), incomplete metadata assertions
(only `subjectID` and `sampling_rate` checked; `groupID` and `condition` omitted), and
no input-mutation guard test for either feature.

## Keep

- `test_higuchi_output_type_dims_history` — correct shape, type, key metadata, history ✅
- `test_higuchi_linear_signal_fd_equals_one` — analytical ground truth ✅
- `test_higuchi_known_value_matches_static_method` — round-trip via static method ✅
- `test_higuchi_random_more_complex_than_sine` — monotonicity property ✅
- `test_higuchi_fd_in_expected_range` — finite + > 1.5 for noise ✅
- `test_higuchi_multichannel_computed_independently` — per-channel isolation ✅
- `test_higuchi_custom_k_max` — parameter effect ✅
- `test_higuchi_raises_for_invalid_k_max` — construction-time guard with `match=` ✅
- `test_higuchi_raises_when_signal_too_short` — runtime guard with `match=` ✅
- `test_higuchi_via_chord` — pipeline integration ✅
- `test_katz_output_type_dims_history` — correct shape, type, key metadata, history ✅
- `test_katz_linear_signal_fd_equals_one` — analytical ground truth ✅
- `test_katz_known_value_matches_static_method` — round-trip via static method ✅
- `test_katz_random_more_complex_than_sine` — monotonicity property ✅
- `test_katz_via_chord` — pipeline integration ✅

## Fix

### `test_higuchi_output_type_dims_history` and `test_katz_output_type_dims_history`

Both only check `subjectID` and `sampling_rate`. Criteria require all four metadata
fields: `subjectID`, `groupID`, `condition`, `sampling_rate`.

```python
# extend both tests to include:
assert out.groupID == "g1"
assert out.condition == "rest"
```

## Add

### `test_higuchi_n_steps_zero_path` (fixes lines 88-89 coverage)

```python
def test_higuchi_n_steps_zero_path() -> None:
    """n_steps==0 fallback is exercised when N == k_max + 1."""
    # With N=11 and k_max=10: at k=10, m=10 → n_steps = (11-10)//10 = 0
    arr = np.arange(11, dtype=float).reshape(-1, 1)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    out = cb.feature.FractalDimHiguchi(k_max=10).apply(data)
    assert isinstance(out, cb.Data)
    assert np.isfinite(float(out.to_numpy()[0]))
```

### `test_higuchi_does_not_mutate_input`

```python
def test_higuchi_does_not_mutate_input() -> None:
    """FractalDimHiguchi.apply() leaves the input Data unchanged."""
    arr = np.random.default_rng(9).standard_normal((200, 2))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    _ = cb.feature.FractalDimHiguchi().apply(data)
    assert data.history == original_history
    assert data.data.shape == original_shape
```

### `test_katz_does_not_mutate_input`

```python
def test_katz_does_not_mutate_input() -> None:
    """FractalDimKatz.apply() leaves the input Data unchanged."""
    arr = np.random.default_rng(9).standard_normal((200, 2))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    original_history = list(data.history)
    original_shape = data.data.shape
    _ = cb.feature.FractalDimKatz().apply(data)
    assert data.history == original_history
    assert data.data.shape == original_shape
```

## Action List

1. [HIGH] Coverage is 94% — add `test_higuchi_n_steps_zero_path` to cover lines 88-89.
   Remove dead guard at line 157 of `fractal_dimension.py` (`_katz_1d` degenerate
   branch is unreachable with `x = np.arange(N)`).

2. [MEDIUM] Extend `test_higuchi_output_type_dims_history` and
   `test_katz_output_type_dims_history` to assert `groupID` and `condition` in
   addition to the currently-checked `subjectID` and `sampling_rate`.

3. [MEDIUM] Add `test_higuchi_does_not_mutate_input` and
   `test_katz_does_not_mutate_input`.
