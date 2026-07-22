# Infometrics

*How complex/irregular is my signal?*

Features in the `cobrabox.infometrics` domain. Access them as `cb.<Feature>` (canonical), or as `cb.infometrics.<Feature>` / `cb.feature.<Feature>`.

Each feature has two forms: a **class** for building pipelines, and a one-shot **function** for a single call.

### AmplitudeEntropy
Compute amplitude entropy from time-series data using histogram-based probability estimation.

```python
cb.amplitude_entropy(data, ...)       # one-shot
cb.AmplitudeEntropy(...).apply(data)  # composable, for pipelines
```

**Tags:** [`shannon-entropy`](../tags.md#tag-shannon-entropy), [`histogram`](../tags.md#tag-histogram), [`probability-distribution`](../tags.md#tag-probability-distribution), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### FractalDimension
Compute fractal dimension over the time dimension.

```python
cb.fractal_dimension(data, ...)       # one-shot
cb.FractalDimension(...).apply(data)  # composable, for pipelines
```

**Tags:** [`fractal`](../tags.md#tag-fractal), [`self-similarity`](../tags.md#tag-self-similarity), [`higuchi`](../tags.md#tag-higuchi), [`katz`](../tags.md#tag-katz), [`box-counting`](../tags.md#tag-box-counting), [`aging`](../tags.md#tag-aging), [`anesthesia`](../tags.md#tag-anesthesia), [`dementia`](../tags.md#tag-dementia), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### LempelZiv
Compute Lempel-Ziv Complexity (LZC) over the time dimension.

```python
cb.lempel_ziv(data, ...)       # one-shot
cb.LempelZiv(...).apply(data)  # composable, for pipelines
```

**Tags:** [`compression`](../tags.md#tag-compression), [`algorithmic-complexity`](../tags.md#tag-algorithmic-complexity), [`binary`](../tags.md#tag-binary), [`epilepsy`](../tags.md#tag-epilepsy), [`anesthesia`](../tags.md#tag-anesthesia), [`consciousness`](../tags.md#tag-consciousness), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### Nonreversibility
Compute dc_norm: normalised deviation from causal normality (time-irreversibility).

```python
cb.nonreversibility(data, ...)        # one-shot
cb.Nonreversibility(...).apply(data)  # composable, for pipelines
```

**Tags:** [`time-irreversibility`](../tags.md#tag-time-irreversibility), [`causality`](../tags.md#tag-causality), [`kl-divergence`](../tags.md#tag-kl-divergence), [`entropy-production`](../tags.md#tag-entropy-production), [`mvar`](../tags.md#tag-mvar), [`epilepsy`](../tags.md#tag-epilepsy), [`nonlinear-dynamics`](../tags.md#tag-nonlinear-dynamics), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### RecurrenceMatrix
Compute a pairwise recurrence (self-similarity) matrix from a time-series.

```python
cb.recurrence_matrix(data, ...)       # one-shot
cb.RecurrenceMatrix(...).apply(data)  # composable, for pipelines
```

**Tags:** [`dynamical-systems`](../tags.md#tag-dynamical-systems), [`self-similarity`](../tags.md#tag-self-similarity), [`state-space`](../tags.md#tag-state-space), [`functional-connectivity`](../tags.md#tag-functional-connectivity), [`epilepsy`](../tags.md#tag-epilepsy), [`nonlinear-dynamics`](../tags.md#tag-nonlinear-dynamics), [`eeg`](../tags.md#tag-eeg), [`io:matrix`](../tags.md#tag-io-matrix)

### SampleEntropy
Sample Entropy feature.

```python
cb.sample_entropy(data, ...)       # one-shot
cb.SampleEntropy(...).apply(data)  # composable, for pipelines
```

**Tags:** [`entropy`](../tags.md#tag-entropy), [`regularity`](../tags.md#tag-regularity), [`predictability`](../tags.md#tag-predictability), [`dynamical-systems`](../tags.md#tag-dynamical-systems), [`heart-rate-variability`](../tags.md#tag-heart-rate-variability), [`epilepsy`](../tags.md#tag-epilepsy), [`anesthesia`](../tags.md#tag-anesthesia), [`eeg`](../tags.md#tag-eeg), [`ecg`](../tags.md#tag-ecg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)
