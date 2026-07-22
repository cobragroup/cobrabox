# Signal Statistics

*What are the basic properties of my signal?*

Features in the `cobrabox.signalstats` domain. Access them as `cb.<Feature>` (canonical), or as `cb.signalstats.<Feature>` / `cb.feature.<Feature>`.

### AmplitudeVariation
Compute amplitude variation over the time dimension.

**Tags:** [`variability`](../tags.md#tag-variability), [`standard-deviation`](../tags.md#tag-standard-deviation), [`anesthesia`](../tags.md#tag-anesthesia), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### Autocorrelation
Compute normalized autocorrelation at a given lag along a required dimension.

**Tags:** [`temporal-dependence`](../tags.md#tag-temporal-dependence), [`lag`](../tags.md#tag-lag), [`stationarity`](../tags.md#tag-stationarity), [`eeg`](../tags.md#tag-eeg), [`fmri`](../tags.md#tag-fmri), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### EpileptogenicityIndex
Compute the Epileptogenicity Index (EI) per channel (Bartolomei et al., 2008).

**Tags:** [`epilepsy`](../tags.md#tag-epilepsy), [`seizure-onset-zone`](../tags.md#tag-seizure-onset-zone), [`energy-ratio`](../tags.md#tag-energy-ratio), [`beta-gamma`](../tags.md#tag-beta-gamma), [`page-hinkley`](../tags.md#tag-page-hinkley), [`intracranial-eeg`](../tags.md#tag-intracranial-eeg), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### LineLength
Compute line length over the time dimension.

**Tags:** [`variability`](../tags.md#tag-variability), [`signal-complexity`](../tags.md#tag-signal-complexity), [`epilepsy`](../tags.md#tag-epilepsy), [`seizure-detection`](../tags.md#tag-seizure-detection), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)

### Max
Compute the maximum value across a dimension.

**Tags:** [`reduction`](../tags.md#tag-reduction), [`io:scalar`](../tags.md#tag-io-scalar)

### Mean
Compute the mean value across a dimension.

**Tags:** [`reduction`](../tags.md#tag-reduction), [`io:scalar`](../tags.md#tag-io-scalar)

### Min
Compute the minimum value across a dimension.

**Tags:** [`reduction`](../tags.md#tag-reduction), [`io:scalar`](../tags.md#tag-io-scalar)

### SpikeCount
Calculate spikes in the input data using the IQR method.

**Tags:** [`outlier-detection`](../tags.md#tag-outlier-detection), [`artifact`](../tags.md#tag-artifact), [`epilepsy`](../tags.md#tag-epilepsy), [`spike-detection`](../tags.md#tag-spike-detection), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel`](../tags.md#tag-io-scalar-per-channel)
