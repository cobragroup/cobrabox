# Spectral

*What's happening in frequency space?*

Features in the `cobrabox.spectral` domain. Access them as `cb.spectral.<Feature>` or `cb.feature.<Feature>`.

### BandPower
Compute band power for specified frequency bands using Welch's method.

**Tags:** [`power-spectrum`](../tags/power-spectrum.md), [`frequency-band`](../tags/frequency-band.md), [`welch`](../tags/welch.md), [`alpha`](../tags/alpha.md), [`theta`](../tags/theta.md), [`delta`](../tags/delta.md), [`beta`](../tags/beta.md), [`gamma`](../tags/gamma.md), [`epilepsy`](../tags/epilepsy.md), [`depression`](../tags/depression.md), [`anesthesia`](../tags/anesthesia.md), [`sleep`](../tags/sleep.md), [`eeg`](../tags/eeg.md), [`fmri`](../tags/fmri.md), [`io:scalar-per-channel-per-band`](../tags/io-scalar-per-channel-per-band.md)

### ContinuousWaveletTransform
Continuous wavelet transform (CWT) scalogram.

**Tags:** [`wavelet`](../tags/wavelet.md), [`scalogram`](../tags/scalogram.md), [`time-frequency`](../tags/time-frequency.md), [`scale-adaptive`](../tags/scale-adaptive.md), [`seizure-onset`](../tags/seizure-onset.md), [`erps`](../tags/erps.md), [`eeg`](../tags/eeg.md), [`io:frequency-time-output`](../tags/io-frequency-time-output.md)

### Cordance
Compute cordance, a qEEG measure combining absolute and relative bandpower.

**Tags:** [`qeeg`](../tags/qeeg.md), [`absolute-power`](../tags/absolute-power.md), [`relative-power`](../tags/relative-power.md), [`depression`](../tags/depression.md), [`treatment-response`](../tags/treatment-response.md), [`eeg`](../tags/eeg.md), [`io:scalar-per-channel-per-band`](../tags/io-scalar-per-channel-per-band.md)

### DiscreteWaveletTransform
Multi-level discrete wavelet decomposition (DWT).

**Tags:** [`wavelet`](../tags/wavelet.md), [`sub-band`](../tags/sub-band.md), [`dyadic`](../tags/dyadic.md), [`decomposition`](../tags/decomposition.md), [`denoising`](../tags/denoising.md), [`compression`](../tags/compression.md), [`eeg`](../tags/eeg.md), [`io:adds-dimension`](../tags/io-adds-dimension.md)

### Spectrogram
Compute the power spectrogram for each spatial channel.

**Tags:** [`stft`](../tags/stft.md), [`time-frequency`](../tags/time-frequency.md), [`power-spectrum`](../tags/power-spectrum.md), [`seizure-onset`](../tags/seizure-onset.md), [`event-related`](../tags/event-related.md), [`eeg`](../tags/eeg.md), [`fmri`](../tags/fmri.md), [`io:frequency-time-output`](../tags/io-frequency-time-output.md)
