# Connectivity

*Which regions are synchronized/interacting?*

Features in the `cobrabox.connectivity` domain. Access them as `cb.connectivity.<Feature>` or `cb.feature.<Feature>`.

### Coherence
Compute magnitude-squared coherence for all pairwise channel combinations.

**Tags:** [`cross-spectral`](../tags/cross-spectral.md), [`welch`](../tags/welch.md), [`frequency-domain`](../tags/frequency-domain.md), [`undirected`](../tags/undirected.md), [`resting-state`](../tags/resting-state.md), [`epilepsy`](../tags/epilepsy.md), [`eeg`](../tags/eeg.md), [`io:matrix`](../tags/io-matrix.md), [`req:multichannel`](../tags/req-multichannel.md), [`req:sampling-rate`](../tags/req-sampling-rate.md)

### Correlation
Compute pairwise Pearson or Spearman correlation between all channel pairs.

**Tags:** [`pearson`](../tags/pearson.md), [`spearman`](../tags/spearman.md), [`undirected`](../tags/undirected.md), [`linear`](../tags/linear.md), [`resting-state`](../tags/resting-state.md), [`default-mode`](../tags/default-mode.md), [`eeg`](../tags/eeg.md), [`fmri`](../tags/fmri.md), [`io:matrix`](../tags/io-matrix.md), [`req:multichannel`](../tags/req-multichannel.md)

### Covariance
Compute pairwise sample covariance between all channel pairs.

**Tags:** [`undirected`](../tags/undirected.md), [`linear`](../tags/linear.md), [`resting-state`](../tags/resting-state.md), [`eeg`](../tags/eeg.md), [`fmri`](../tags/fmri.md), [`io:matrix`](../tags/io-matrix.md), [`req:multichannel`](../tags/req-multichannel.md)

### DirectedTransferFunction
Estimate the Directed Transfer Function (DTF) between channels via a VAR model.

**Tags:** [`mvar`](../tags/mvar.md), [`var`](../tags/var.md), [`directed`](../tags/directed.md), [`frequency-domain`](../tags/frequency-domain.md), [`total-causality`](../tags/total-causality.md), [`epilepsy`](../tags/epilepsy.md), [`seizure-propagation`](../tags/seizure-propagation.md), [`eeg`](../tags/eeg.md), [`io:matrix-frequency`](../tags/io-matrix-frequency.md), [`req:multichannel`](../tags/req-multichannel.md), [`req:sampling-rate`](../tags/req-sampling-rate.md)

### EnvelopeCorrelation
Compute amplitude envelope correlation (AEC) between all channel pairs.

**Tags:** [`aec`](../tags/aec.md), [`orthogonalization`](../tags/orthogonalization.md), [`volume-conduction`](../tags/volume-conduction.md), [`undirected`](../tags/undirected.md), [`resting-state`](../tags/resting-state.md), [`meg`](../tags/meg.md), [`eeg`](../tags/eeg.md), [`io:matrix`](../tags/io-matrix.md), [`req:multichannel`](../tags/req-multichannel.md)

### GrangerCausality
Compute the Granger causality matrix across channels.

**Tags:** [`mvar`](../tags/mvar.md), [`var`](../tags/var.md), [`directed`](../tags/directed.md), [`prediction-error`](../tags/prediction-error.md), [`epilepsy`](../tags/epilepsy.md), [`seizure-propagation`](../tags/seizure-propagation.md), [`resting-state`](../tags/resting-state.md), [`eeg`](../tags/eeg.md), [`io:matrix`](../tags/io-matrix.md), [`req:multichannel`](../tags/req-multichannel.md), [`req:sampling-rate`](../tags/req-sampling-rate.md)

### MutualInformation
Compute mutual information (MI) between all pairs of series along a specified dimension (by

**Tags:** [`nonlinear`](../tags/nonlinear.md), [`entropy`](../tags/entropy.md), [`undirected`](../tags/undirected.md), [`resting-state`](../tags/resting-state.md), [`eeg`](../tags/eeg.md), [`io:matrix`](../tags/io-matrix.md), [`req:multichannel`](../tags/req-multichannel.md)

### PartialCorrelation
Compute the partial-correlation matrix across channels.

**Tags:** [`conditional-independence`](../tags/conditional-independence.md), [`precision-matrix`](../tags/precision-matrix.md), [`undirected`](../tags/undirected.md), [`resting-state`](../tags/resting-state.md), [`eeg`](../tags/eeg.md), [`fmri`](../tags/fmri.md), [`io:matrix`](../tags/io-matrix.md), [`req:multichannel`](../tags/req-multichannel.md)

### PartialDirectedCoherence
Estimate the Partial Directed Coherence (PDC) between channels via a VAR model.

**Tags:** [`mvar`](../tags/mvar.md), [`var`](../tags/var.md), [`directed`](../tags/directed.md), [`frequency-domain`](../tags/frequency-domain.md), [`direct-causality`](../tags/direct-causality.md), [`epilepsy`](../tags/epilepsy.md), [`seizure-propagation`](../tags/seizure-propagation.md), [`eeg`](../tags/eeg.md), [`io:matrix-frequency`](../tags/io-matrix-frequency.md), [`req:multichannel`](../tags/req-multichannel.md), [`req:sampling-rate`](../tags/req-sampling-rate.md)

### PhaseLockingValue
Compute the phase-locking-value (PLV) matrix across channels.

**Tags:** [`phase-synchrony`](../tags/phase-synchrony.md), [`undirected`](../tags/undirected.md), [`resting-state`](../tags/resting-state.md), [`epilepsy`](../tags/epilepsy.md), [`anesthesia`](../tags/anesthesia.md), [`eeg`](../tags/eeg.md), [`io:matrix`](../tags/io-matrix.md), [`req:multichannel`](../tags/req-multichannel.md), [`req:sampling-rate`](../tags/req-sampling-rate.md)

### ReciprocalConnectivity
Compute per-channel Reciprocal Connectivity (RC) from a directed connectivity matrix.

**Tags:** [`directed`](../tags/directed.md), [`sink-source`](../tags/sink-source.md), [`post-processing`](../tags/post-processing.md), [`epilepsy`](../tags/epilepsy.md), [`seizure-onset-zone`](../tags/seizure-onset-zone.md), [`eeg`](../tags/eeg.md), [`io:vector`](../tags/io-vector.md), [`req:asymmetric-matrix`](../tags/req-asymmetric-matrix.md)
