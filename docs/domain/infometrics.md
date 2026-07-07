# Infometrics

*How complex/irregular is my signal?*

Features in the `cobrabox.infometrics` domain. Access them as `cb.infometrics.<Feature>` or `cb.feature.<Feature>`.

### AmplitudeEntropy
Compute amplitude entropy from time-series data using histogram-based probability estimation.

**Tags:** [`shannon-entropy`](../tags/shannon-entropy.md), [`histogram`](../tags/histogram.md), [`probability-distribution`](../tags/probability-distribution.md), [`eeg`](../tags/eeg.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)

### FractalDimension
Compute fractal dimension over the time dimension.

**Tags:** [`fractal`](../tags/fractal.md), [`self-similarity`](../tags/self-similarity.md), [`higuchi`](../tags/higuchi.md), [`katz`](../tags/katz.md), [`box-counting`](../tags/box-counting.md), [`aging`](../tags/aging.md), [`anesthesia`](../tags/anesthesia.md), [`dementia`](../tags/dementia.md), [`eeg`](../tags/eeg.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)

### LempelZiv
Compute Lempel-Ziv Complexity (LZC) over the time dimension.

**Tags:** [`compression`](../tags/compression.md), [`algorithmic-complexity`](../tags/algorithmic-complexity.md), [`binary`](../tags/binary.md), [`epilepsy`](../tags/epilepsy.md), [`anesthesia`](../tags/anesthesia.md), [`consciousness`](../tags/consciousness.md), [`eeg`](../tags/eeg.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)

### Nonreversibility
Compute dc_norm: normalised deviation from causal normality (time-irreversibility).

**Tags:** [`time-irreversibility`](../tags/time-irreversibility.md), [`causality`](../tags/causality.md), [`kl-divergence`](../tags/kl-divergence.md), [`entropy-production`](../tags/entropy-production.md), [`mvar`](../tags/mvar.md), [`epilepsy`](../tags/epilepsy.md), [`nonlinear-dynamics`](../tags/nonlinear-dynamics.md), [`eeg`](../tags/eeg.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)

### RecurrenceMatrix
Compute a pairwise recurrence (self-similarity) matrix from a time-series.

**Tags:** [`dynamical-systems`](../tags/dynamical-systems.md), [`self-similarity`](../tags/self-similarity.md), [`state-space`](../tags/state-space.md), [`functional-connectivity`](../tags/functional-connectivity.md), [`epilepsy`](../tags/epilepsy.md), [`nonlinear-dynamics`](../tags/nonlinear-dynamics.md), [`eeg`](../tags/eeg.md), [`io:matrix`](../tags/io-matrix.md)

### SampleEntropy
Sample Entropy feature.

**Tags:** [`entropy`](../tags/entropy.md), [`regularity`](../tags/regularity.md), [`predictability`](../tags/predictability.md), [`dynamical-systems`](../tags/dynamical-systems.md), [`heart-rate-variability`](../tags/heart-rate-variability.md), [`epilepsy`](../tags/epilepsy.md), [`anesthesia`](../tags/anesthesia.md), [`eeg`](../tags/eeg.md), [`ecg`](../tags/ecg.md), [`io:scalar-per-channel`](../tags/io-scalar-per-channel.md)
