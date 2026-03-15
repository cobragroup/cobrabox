# Test Review: fractal_dimension

**Feature**: `src/cobrabox/features/fractal_dimension.py`
**Test file**: `tests/test_feature_fractal_dimension.py`
**Date**: 2026-03-06
**Verdict**: PASS

## Coverage

```text
fractal_dimension.py: 100% (48 statements, 0 missing)
```

## Summary

Excellent test coverage for both FractalDimHiguchi and FractalDimKatz features. The test file includes 18 tests covering all required scenarios: happy path, history tracking, metadata preservation, output type handling, input immutability, parameter validation, runtime validation, Chord composition, and edge cases. All tests use proper naming conventions, return annotations, and docstrings. Both features have `output_type = Data` (time dimension removed) which is correctly tested with `sampling_rate is None` assertions.

## Keep

Tests that are correct and complete:

- `test_higuchi_output_type_dims_history` — checks output type, dims, metadata preservation, history, and sampling_rate=None
- `test_higuchi_linear_signal_fd_equals_one` — verifies linear signal yields FD=1 (analytic ground truth)
- `test_higuchi_known_value_matches_static_method` — validates feature output matches direct static method call
- `test_higuchi_random_more_complex_than_sine` — tests relative complexity ordering (noise > sine)
- `test_higuchi_fd_in_expected_range` — validates random data yields reasonable FD values
- `test_higuchi_multichannel_computed_independently` — confirms per-channel processing
- `test_higuchi_custom_k_max` — tests parameter effect on output
- `test_higuchi_n_steps_zero_path` — exercises edge case (n_steps==0 branch in `_higuchi_1d`)
- `test_higuchi_does_not_mutate_input` — validates input Data immutability
- `test_higuchi_raises_for_invalid_k_max` — tests `__post_init__` validation (k_max < 2)
- `test_higuchi_raises_when_signal_too_short` — tests runtime validation (N <= k_max)
- `test_higuchi_via_chord` — tests Chord composition with MeanAggregate
- `test_katz_output_type_dims_history` — checks output type, dims, metadata, history
- `test_katz_linear_signal_fd_equals_one` — verifies linear signal yields FD=1
- `test_katz_known_value_matches_static_method` — validates feature matches static method
- `test_katz_random_more_complex_than_sine` — tests relative complexity ordering
- `test_katz_via_chord` — tests Chord composition
- `test_katz_does_not_mutate_input` — validates input Data immutability

## Add

Missing scenarios for FractalDimKatz (symmetry with FractalDimHiguchi tests):

### `test_katz_multichannel_computed_independently`

```python
def test_katz_multichannel_computed_independently() -> None:
    """Each channel is processed independently; results differ across channels."""
    t = np.arange(256)
    ch0 = np.sin(2 * np.pi * t / 32)  # smooth sine
    ch1 = np.random.default_rng(7).standard_normal(256)  # noise
    arr = np.stack([ch0, ch1], axis=1)
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    out = cb.feature.FractalDimKatz().apply(data)

    assert out.data.shape == (2,)
    # Noise channel must be more complex than the sine channel
    assert out.to_numpy()[1] > out.to_numpy()[0]
```

### `test_katz_fd_in_expected_range`

```python
def test_katz_fd_in_expected_range(rng: np.random.Generator) -> None:
    """KFD for random EEG-like data should be finite and >= 1."""
    arr = rng.standard_normal((512, 4))
    data = cb.SignalData.from_numpy(arr, dims=["time", "space"], sampling_rate=256.0)
    values = cb.feature.FractalDimKatz().apply(data).to_numpy()

    # Random noise should be at least as complex as a line (FD >= 1)
    assert np.all(values >= 1.0)
    # Should be finite and reasonable
    assert np.all(np.isfinite(values))
```

## Action List

1. [Severity: LOW] Add `test_katz_multichannel_computed_independently` to test per-channel processing for FractalDimKatz
2. [Severity: LOW] Add `test_katz_fd_in_expected_range` to validate output bounds for FractalDimKatz
