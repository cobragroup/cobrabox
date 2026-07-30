# Decompositions

*How do I decompose my signal into components?*

Features in the `cobrabox.decompositions` domain. Access them as `cb.<Feature>` (canonical), or as `cb.decompositions.<Feature>` / `cb.feature.<Feature>`.

Each feature has two forms: a **class** for building pipelines, and a one-shot **function** for a single call.

### EMD
Decompose a signal into Intrinsic Mode Functions (IMFs) using Empirical Mode Decomposition.

```python
cb.emd(data, ...)  # one-shot
cb.EMD(...).apply(data)  # composable, for pipelines
```

**Tags:** [`data-driven`](../tags.md#tag-data-driven), [`nonlinear`](../tags.md#tag-nonlinear), [`intrinsic-mode-functions`](../tags.md#tag-intrinsic-mode-functions), [`time-frequency`](../tags.md#tag-time-frequency), [`nonstationary`](../tags.md#tag-nonstationary), [`eeg`](../tags.md#tag-eeg), [`io:adds-dimension`](../tags.md#tag-io-adds-dimension)

### SVD
Compute truncated SVD over one dimension.

```python
cb.svd(data, ...)  # one-shot
cb.SVD(...).apply(data)  # composable, for pipelines
```

**Tags:** [`dimensionality-reduction`](../tags.md#tag-dimensionality-reduction), [`patterns`](../tags.md#tag-patterns), [`source-localization`](../tags.md#tag-source-localization), [`eeg`](../tags.md#tag-eeg), [`io:matrix`](../tags.md#tag-io-matrix)
