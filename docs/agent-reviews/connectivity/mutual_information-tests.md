# Test Review: mutual_information

**Feature**: `src/cobrabox/features/mutual_information.py`
**Test file**: `tests/test_feature_mutual_information.py`
**Date**: 2026-03-06
**Verdict**: NEEDS WORK

## Coverage

```text
MutualInformation: 100% (70 statements, 0 missing)
```

Excellent per-file coverage at 100%.

## Summary

The test file covers the MutualInformation feature comprehensively including:

- Internal method testing (_vector_entropy, _get_binned)
- Parameter validation (negative bins, non-integer bins, missing dimensions)
- Low-dimensional data (2D: space x time) with both binning strategies
- High-dimensional data (4D: something x sample x space x time) with both binning strategies
- History, metadata preservation, sampling_rate handling, and immutability

However, 7 test functions lack docstrings, and some test naming doesn't follow the required convention.

## Keep

- `test_mutual_information_history_updated` — proper docstring, tests history correctly
- `test_mutual_information_metadata_preserved` — proper docstring, tests metadata correctly
- `test_mutual_information_sampling_rate_none` — proper docstring, tests output_type correctly
- `test_mutual_information_does_not_mutate_input` — proper docstring, tests immutability correctly
- `test_low_dim_equidistant_bins` — tests low-dim data with equidistant bins
- `test_low_dim_equiprobable_bins` — tests low-dim data with equiprobable bins
- `test_high_dim_equiprobable_bins` — tests high-dim data with equiprobable bins
- `test_high_dim_equidistant_bins` — tests high-dim data with equidistant bins

## Fix

### `test_entropy` → `test_mutual_information_vector_entropy`

Issue: Missing docstring, name lacks feature prefix.

```python
def test_mutual_information_vector_entropy() -> None:
    """_vector_entropy computes correct entropy for a binned distribution."""
```

### `test_binning` → `test_mutual_information_get_binned`

Issue: Missing docstring, name lacks feature prefix.

```python
def test_mutual_information_get_binned() -> None:
    """_get_binned correctly discretizes data into specified number of bins."""
```

### `test_bad_inits`

Issue: Missing docstring, should be split into multiple tests for clarity.

Replace with:

```python
def test_mutual_information_negative_bins_raises() -> None:
    """MutualInformation raises ValueError for negative bins."""
    with pytest.raises(ValueError, match="bins must be positive"):
        MutualInformation(bins=-1)


def test_mutual_information_non_integer_bins_raises() -> None:
    """MutualInformation raises ValueError for non-integer bins."""
    with pytest.raises(ValueError, match="bins must be an integer"):
        MutualInformation(bins=2.5)


def test_mutual_information_zero_bins_raises() -> None:
    """MutualInformation raises ValueError for zero bins."""
    with pytest.raises(ValueError, match="bins must be positive"):
        MutualInformation(bins=0)


def test_mutual_information_invalid_dim_type_raises() -> None:
    """MutualInformation raises ValueError when dim is not a string."""
    with pytest.raises(ValueError, match="dim must be a string"):
        MutualInformation(dim=123)


def test_mutual_information_invalid_other_dim_type_raises() -> None:
    """MutualInformation raises ValueError when other_dim is not a string or None."""
    with pytest.raises(ValueError, match="other_dim must be a string or None"):
        MutualInformation(other_dim=123)
```

### `test_low_dim_equidistant_bins`

Issue: Missing docstring.

```python
def test_low_dim_equidistant_bins(low_dim_data: cb.Data) -> None:
    """MutualInformation computes correct MI for 2D data with equidistant bins."""
```

### `test_low_dim_equiprobable_bins`

Issue: Missing docstring.

```python
def test_low_dim_equiprobable_bins(low_dim_data: cb.Data) -> None:
    """MutualInformation computes correct MI for 2D data with equiprobable bins."""
```

### `test_high_dim_equiprobable_bins`

Issue: Missing docstring.

```python
def test_high_dim_equiprobable_bins(high_dim_data: cb.Data) -> None:
    """MutualInformation handles 4D data correctly with equiprobable bins."""
```

### `test_high_dim_equidistant_bins`

Issue: Missing docstring.

```python
def test_high_dim_equidistant_bins(high_dim_data: cb.Data) -> None:
    """MutualInformation handles 4D data correctly with equidistant bins."""
```

## Add

None. All required scenarios are covered.

## Action List

1. [Severity: MEDIUM] Add docstrings to `test_entropy`, `test_binning`, `test_bad_inits`, `test_low_dim_equidistant_bins`, `test_low_dim_equiprobable_bins`, `test_high_dim_equiprobable_bins`, `test_high_dim_equidistant_bins`
2. [Severity: MEDIUM] Rename `test_entropy` to `test_mutual_information_vector_entropy` and `test_binning` to `test_mutual_information_get_binned`
3. [Severity: LOW] Split `test_bad_inits` into separate test functions for each validation case
