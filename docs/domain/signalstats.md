# Signal Statistics

*What are the basic properties of my signal?*

Features in the `cobrabox.signalstats` domain. Access them as `cb.<Feature>` (canonical), or as `cb.signalstats.<Feature>` / `cb.feature.<Feature>`.

Each feature has two forms: a **class** for building pipelines, and a one-shot **function** for a single call.

### AmplitudeVariation
Compute amplitude variation over the time dimension.

```python
cb.amplitude_variation(data, ...)       # one-shot
cb.AmplitudeVariation(...).apply(data)  # composable, for pipelines
```

**Tags:** [`variability`](../tags.md#tag-variability), [`standard-deviation`](../tags.md#tag-standard-deviation), [`anesthesia`](../tags.md#tag-anesthesia), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### Autocorrelation
Compute normalized autocorrelation at a given lag along a required dimension.

```python
cb.autocorrelation(data, ...)        # one-shot
cb.Autocorrelation(...).apply(data)  # composable, for pipelines
```

**Tags:** [`temporal-dependence`](../tags.md#tag-temporal-dependence), [`lag`](../tags.md#tag-lag), [`stationarity`](../tags.md#tag-stationarity), [`eeg`](../tags.md#tag-eeg), [`fmri`](../tags.md#tag-fmri), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### EpileptogenicityIndex
Compute the Epileptogenicity Index (EI) per channel (Bartolomei et al., 2008).

```python
cb.epileptogenicity_index(data, ...)       # one-shot
cb.EpileptogenicityIndex(...).apply(data)  # composable, for pipelines
```

**Tags:** [`epilepsy`](../tags.md#tag-epilepsy), [`seizure-onset-zone`](../tags.md#tag-seizure-onset-zone), [`energy-ratio`](../tags.md#tag-energy-ratio), [`beta-gamma`](../tags.md#tag-beta-gamma), [`page-hinkley`](../tags.md#tag-page-hinkley), [`intracranial-eeg`](../tags.md#tag-intracranial-eeg), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### LineLength
Compute line length over the time dimension.

```python
cb.line_length(data, ...)       # one-shot
cb.LineLength(...).apply(data)  # composable, for pipelines
```

**Tags:** [`variability`](../tags.md#tag-variability), [`signal-complexity`](../tags.md#tag-signal-complexity), [`epilepsy`](../tags.md#tag-epilepsy), [`seizure-detection`](../tags.md#tag-seizure-detection), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### Max
Compute the maximum value across a dimension.

```python
cb.max(data, ...)        # one-shot
cb.Max(...).apply(data)  # composable, for pipelines
```

**Tags:** [`reduction`](../tags.md#tag-reduction), [`io:scalar`](../tags.md#tag-io-scalar)

### Mean
Compute the mean value across a dimension.

```python
cb.mean(data, ...)        # one-shot
cb.Mean(...).apply(data)  # composable, for pipelines
```

**Tags:** [`reduction`](../tags.md#tag-reduction), [`io:scalar`](../tags.md#tag-io-scalar)

### Min
Compute the minimum value across a dimension.

```python
cb.min(data, ...)        # one-shot
cb.Min(...).apply(data)  # composable, for pipelines
```

**Tags:** [`reduction`](../tags.md#tag-reduction), [`io:scalar`](../tags.md#tag-io-scalar)

### SpikeCount
Calculate spikes in the input data using the IQR method.

```python
cb.spike_count(data, ...)       # one-shot
cb.SpikeCount(...).apply(data)  # composable, for pipelines
```

**Tags:** [`outlier-detection`](../tags.md#tag-outlier-detection), [`artifact`](../tags.md#tag-artifact), [`epilepsy`](../tags.md#tag-epilepsy), [`spike-detection`](../tags.md#tag-spike-detection), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)
