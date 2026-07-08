# Connectivity

*Which regions are synchronized/interacting?*

Features in the `cobrabox.connectivity` domain. Access them as `cb.connectivity.<Feature>` or `cb.feature.<Feature>`.

### Coherence
Compute magnitude-squared coherence for all pairwise channel combinations.

**Tags:** [`cross-spectral`](../tags.md#tag-cross-spectral), [`welch`](../tags.md#tag-welch), [`frequency-domain`](../tags.md#tag-frequency-domain), [`undirected`](../tags.md#tag-undirected), [`resting-state`](../tags.md#tag-resting-state), [`epilepsy`](../tags.md#tag-epilepsy), [`eeg`](../tags.md#tag-eeg), [`io:matrix`](../tags.md#tag-io-matrix), [`req:multichannel`](../tags.md#tag-req-multichannel), [`req:sampling-rate`](../tags.md#tag-req-sampling-rate)

### Correlation
Compute pairwise Pearson or Spearman correlation between all channel pairs.

**Tags:** [`pearson`](../tags.md#tag-pearson), [`spearman`](../tags.md#tag-spearman), [`undirected`](../tags.md#tag-undirected), [`linear`](../tags.md#tag-linear), [`resting-state`](../tags.md#tag-resting-state), [`default-mode`](../tags.md#tag-default-mode), [`eeg`](../tags.md#tag-eeg), [`fmri`](../tags.md#tag-fmri), [`io:matrix`](../tags.md#tag-io-matrix), [`req:multichannel`](../tags.md#tag-req-multichannel)

### Covariance
Compute pairwise sample covariance between all channel pairs.

**Tags:** [`undirected`](../tags.md#tag-undirected), [`linear`](../tags.md#tag-linear), [`resting-state`](../tags.md#tag-resting-state), [`eeg`](../tags.md#tag-eeg), [`fmri`](../tags.md#tag-fmri), [`io:matrix`](../tags.md#tag-io-matrix), [`req:multichannel`](../tags.md#tag-req-multichannel)

### DirectedTransferFunction
Estimate the Directed Transfer Function (DTF) between channels via a VAR model.

**Tags:** [`mvar`](../tags.md#tag-mvar), [`var`](../tags.md#tag-var), [`directed`](../tags.md#tag-directed), [`frequency-domain`](../tags.md#tag-frequency-domain), [`total-causality`](../tags.md#tag-total-causality), [`epilepsy`](../tags.md#tag-epilepsy), [`seizure-propagation`](../tags.md#tag-seizure-propagation), [`eeg`](../tags.md#tag-eeg), [`io:matrix-frequency`](../tags.md#tag-io-matrix-frequency), [`req:multichannel`](../tags.md#tag-req-multichannel), [`req:sampling-rate`](../tags.md#tag-req-sampling-rate)

### EnvelopeCorrelation
Compute amplitude envelope correlation (AEC) between all channel pairs.

**Tags:** [`aec`](../tags.md#tag-aec), [`orthogonalization`](../tags.md#tag-orthogonalization), [`volume-conduction`](../tags.md#tag-volume-conduction), [`undirected`](../tags.md#tag-undirected), [`resting-state`](../tags.md#tag-resting-state), [`meg`](../tags.md#tag-meg), [`eeg`](../tags.md#tag-eeg), [`io:matrix`](../tags.md#tag-io-matrix), [`req:multichannel`](../tags.md#tag-req-multichannel)

### GrangerCausality
Compute the Granger causality matrix across channels.

**Tags:** [`mvar`](../tags.md#tag-mvar), [`var`](../tags.md#tag-var), [`directed`](../tags.md#tag-directed), [`prediction-error`](../tags.md#tag-prediction-error), [`epilepsy`](../tags.md#tag-epilepsy), [`seizure-propagation`](../tags.md#tag-seizure-propagation), [`resting-state`](../tags.md#tag-resting-state), [`eeg`](../tags.md#tag-eeg), [`io:matrix`](../tags.md#tag-io-matrix), [`req:multichannel`](../tags.md#tag-req-multichannel), [`req:sampling-rate`](../tags.md#tag-req-sampling-rate)

### MutualInformation
Compute mutual information (MI) between all pairs of series along a specified dimension (by

**Tags:** [`nonlinear`](../tags.md#tag-nonlinear), [`entropy`](../tags.md#tag-entropy), [`undirected`](../tags.md#tag-undirected), [`resting-state`](../tags.md#tag-resting-state), [`eeg`](../tags.md#tag-eeg), [`io:matrix`](../tags.md#tag-io-matrix), [`req:multichannel`](../tags.md#tag-req-multichannel)

### PartialCorrelation
Compute the partial-correlation matrix across channels.

**Tags:** [`conditional-independence`](../tags.md#tag-conditional-independence), [`precision-matrix`](../tags.md#tag-precision-matrix), [`undirected`](../tags.md#tag-undirected), [`resting-state`](../tags.md#tag-resting-state), [`eeg`](../tags.md#tag-eeg), [`fmri`](../tags.md#tag-fmri), [`io:matrix`](../tags.md#tag-io-matrix), [`req:multichannel`](../tags.md#tag-req-multichannel)

### PartialDirectedCoherence
Estimate the Partial Directed Coherence (PDC) between channels via a VAR model.

**Tags:** [`mvar`](../tags.md#tag-mvar), [`var`](../tags.md#tag-var), [`directed`](../tags.md#tag-directed), [`frequency-domain`](../tags.md#tag-frequency-domain), [`direct-causality`](../tags.md#tag-direct-causality), [`epilepsy`](../tags.md#tag-epilepsy), [`seizure-propagation`](../tags.md#tag-seizure-propagation), [`eeg`](../tags.md#tag-eeg), [`io:matrix-frequency`](../tags.md#tag-io-matrix-frequency), [`req:multichannel`](../tags.md#tag-req-multichannel), [`req:sampling-rate`](../tags.md#tag-req-sampling-rate)

### PhaseLockingValue
Compute the phase-locking-value (PLV) matrix across channels.

**Tags:** [`phase-synchrony`](../tags.md#tag-phase-synchrony), [`undirected`](../tags.md#tag-undirected), [`resting-state`](../tags.md#tag-resting-state), [`epilepsy`](../tags.md#tag-epilepsy), [`anesthesia`](../tags.md#tag-anesthesia), [`eeg`](../tags.md#tag-eeg), [`io:matrix`](../tags.md#tag-io-matrix), [`req:multichannel`](../tags.md#tag-req-multichannel), [`req:sampling-rate`](../tags.md#tag-req-sampling-rate)

### ReciprocalConnectivity
Compute per-channel Reciprocal Connectivity (RC) from a directed connectivity matrix.

**Tags:** [`directed`](../tags.md#tag-directed), [`sink-source`](../tags.md#tag-sink-source), [`post-processing`](../tags.md#tag-post-processing), [`epilepsy`](../tags.md#tag-epilepsy), [`seizure-onset-zone`](../tags.md#tag-seizure-onset-zone), [`eeg`](../tags.md#tag-eeg), [`io:vector`](../tags.md#tag-io-vector), [`req:asymmetric-matrix`](../tags.md#tag-req-asymmetric-matrix)
