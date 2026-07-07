# Signal Statistics

*What are the basic properties of my signal?*

Features in the `cobrabox.signalstats` domain. Access them as `cb.signalstats.<Feature>` or `cb.feature.<Feature>`.

### AmplitudeVariation
Compute amplitude variation over the time dimension.

**Tags:** [`variability`](../tags/variability.md), [`standard-deviation`](../tags/standard-deviation.md), [`anesthesia`](../tags/anesthesia.md), [`eeg`](../tags/eeg.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)

### Autocorrelation
Compute normalized autocorrelation at a given lag along a required dimension.

**Tags:** [`temporal-dependence`](../tags/temporal-dependence.md), [`lag`](../tags/lag.md), [`stationarity`](../tags/stationarity.md), [`eeg`](../tags/eeg.md), [`fmri`](../tags/fmri.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)

### EpileptogenicityIndex
Compute the Epileptogenicity Index (EI) per channel (Bartolomei et al., 2008).

**Tags:** [`epilepsy`](../tags/epilepsy.md), [`seizure-onset-zone`](../tags/seizure-onset-zone.md), [`energy-ratio`](../tags/energy-ratio.md), [`beta-gamma`](../tags/beta-gamma.md), [`page-hinkley`](../tags/page-hinkley.md), [`intracranial-eeg`](../tags/intracranial-eeg.md), [`eeg`](../tags/eeg.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)

### LineLength
Compute line length over the time dimension.

**Tags:** [`variability`](../tags/variability.md), [`signal-complexity`](../tags/signal-complexity.md), [`epilepsy`](../tags/epilepsy.md), [`seizure-detection`](../tags/seizure-detection.md), [`eeg`](../tags/eeg.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)

### Max
Compute the maximum value across a dimension.

**Tags:** [`reduction`](../tags/reduction.md), [`io:scalar`](../tags/io-scalar.md)

### Mean
Compute the mean value across a dimension.

**Tags:** [`reduction`](../tags/reduction.md), [`io:scalar`](../tags/io-scalar.md)

### Min
Compute the minimum value across a dimension.

**Tags:** [`reduction`](../tags/reduction.md), [`io:scalar`](../tags/io-scalar.md)

### SpikeCount
Calculate spikes in the input data using the IQR method.

**Tags:** [`outlier-detection`](../tags/outlier-detection.md), [`artifact`](../tags/artifact.md), [`epilepsy`](../tags/epilepsy.md), [`spike-detection`](../tags/spike-detection.md), [`eeg`](../tags/eeg.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)
