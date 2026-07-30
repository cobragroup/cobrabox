# Spectral

*What's happening in frequency space?*

Features in the `cobrabox.spectral` domain. Access them as `cb.<Feature>` (canonical), or as `cb.spectral.<Feature>` / `cb.feature.<Feature>`.

Each feature has two forms: a **class** for building pipelines, and a one-shot **function** for a single call.

### BandPower
Compute band power for specified frequency bands using Welch's method.

```python
cb.band_power(data, ...)  # one-shot
cb.BandPower(...).apply(data)  # composable, for pipelines
```

**Tags:** [`power-spectrum`](../tags.md#tag-power-spectrum), [`frequency-band`](../tags.md#tag-frequency-band), [`welch`](../tags.md#tag-welch), [`alpha`](../tags.md#tag-alpha), [`theta`](../tags.md#tag-theta), [`delta`](../tags.md#tag-delta), [`beta`](../tags.md#tag-beta), [`gamma`](../tags.md#tag-gamma), [`epilepsy`](../tags.md#tag-epilepsy), [`depression`](../tags.md#tag-depression), [`anesthesia`](../tags.md#tag-anesthesia), [`sleep`](../tags.md#tag-sleep), [`eeg`](../tags.md#tag-eeg), [`fmri`](../tags.md#tag-fmri), [`io:scalar-per-channel-per-band`](../tags.md#tag-io-scalar-per-channel-per-band)

### ContinuousWaveletTransform
Continuous wavelet transform (CWT) scalogram.

```python
cb.continuous_wavelet_transform(data, ...)  # one-shot
cb.ContinuousWaveletTransform(...).apply(data)  # composable, for pipelines
```

**Tags:** [`wavelet`](../tags.md#tag-wavelet), [`scalogram`](../tags.md#tag-scalogram), [`time-frequency`](../tags.md#tag-time-frequency), [`scale-adaptive`](../tags.md#tag-scale-adaptive), [`seizure-onset`](../tags.md#tag-seizure-onset), [`erps`](../tags.md#tag-erps), [`eeg`](../tags.md#tag-eeg), [`io:frequency-time-output`](../tags.md#tag-io-frequency-time-output)

### Cordance
Compute cordance, a qEEG measure combining absolute and relative bandpower.

```python
cb.cordance(data, ...)  # one-shot
cb.Cordance(...).apply(data)  # composable, for pipelines
```

**Tags:** [`qeeg`](../tags.md#tag-qeeg), [`absolute-power`](../tags.md#tag-absolute-power), [`relative-power`](../tags.md#tag-relative-power), [`depression`](../tags.md#tag-depression), [`treatment-response`](../tags.md#tag-treatment-response), [`eeg`](../tags.md#tag-eeg), [`io:scalar-per-channel-per-band`](../tags.md#tag-io-scalar-per-channel-per-band)

### DiscreteWaveletTransform
Multi-level discrete wavelet decomposition (DWT).

```python
cb.discrete_wavelet_transform(data, ...)  # one-shot
cb.DiscreteWaveletTransform(...).apply(data)  # composable, for pipelines
```

**Tags:** [`wavelet`](../tags.md#tag-wavelet), [`sub-band`](../tags.md#tag-sub-band), [`dyadic`](../tags.md#tag-dyadic), [`decomposition`](../tags.md#tag-decomposition), [`denoising`](../tags.md#tag-denoising), [`compression`](../tags.md#tag-compression), [`eeg`](../tags.md#tag-eeg), [`io:adds-dimension`](../tags.md#tag-io-adds-dimension)

### Spectrogram
Compute the power spectrogram for each spatial channel.

```python
cb.spectrogram(data, ...)  # one-shot
cb.Spectrogram(...).apply(data)  # composable, for pipelines
```

**Tags:** [`stft`](../tags.md#tag-stft), [`time-frequency`](../tags.md#tag-time-frequency), [`power-spectrum`](../tags.md#tag-power-spectrum), [`seizure-onset`](../tags.md#tag-seizure-onset), [`event-related`](../tags.md#tag-event-related), [`eeg`](../tags.md#tag-eeg), [`fmri`](../tags.md#tag-fmri), [`io:frequency-time-output`](../tags.md#tag-io-frequency-time-output)
