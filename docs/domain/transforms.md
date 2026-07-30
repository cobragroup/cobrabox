# Transforms

*How do I transform my signal into another representation?*

Features in the `cobrabox.transforms` domain. Access them as `cb.<Feature>` (canonical), or as `cb.transforms.<Feature>` / `cb.feature.<Feature>`.

Each feature has two forms: a **class** for building pipelines, and a one-shot **function** for a single call.

### AnalyticSignal
Extract analytic-signal representations along the time axis.

```python
cb.analytic_signal(data, ...)  # one-shot
cb.AnalyticSignal(...).apply(data)  # composable, for pipelines
```

**Tags:** [`hilbert`](../tags.md#tag-hilbert), [`instantaneous-phase`](../tags.md#tag-instantaneous-phase), [`envelope`](../tags.md#tag-envelope), [`instantaneous-frequency`](../tags.md#tag-instantaneous-frequency), [`time-frequency`](../tags.md#tag-time-frequency), [`eeg`](../tags.md#tag-eeg), [`fmri`](../tags.md#tag-fmri), [`io:preserves-time`](../tags.md#tag-io-preserves-time)

### BandpassFilter
Filter a signal into frequency bands.

```python
cb.bandpass_filter(data, ...)  # one-shot
cb.BandpassFilter(...).apply(data)  # composable, for pipelines
```

**Tags:** [`filtering`](../tags.md#tag-filtering), [`butterworth`](../tags.md#tag-butterworth), [`frequency-band`](../tags.md#tag-frequency-band), [`preprocessing`](../tags.md#tag-preprocessing), [`eeg`](../tags.md#tag-eeg), [`io:preserves-time`](../tags.md#tag-io-preserves-time), [`io:adds-dimension`](../tags.md#tag-io-adds-dimension)

### FourierTransform
Real-valued FFT along the time axis.

```python
cb.fourier_transform(data, ...)  # one-shot
cb.FourierTransform(...).apply(data)  # composable, for pipelines
```

**Tags:** [`fft`](../tags.md#tag-fft), [`frequency-domain`](../tags.md#tag-frequency-domain), [`eeg`](../tags.md#tag-eeg), [`fmri`](../tags.md#tag-fmri), [`io:frequency-output`](../tags.md#tag-io-frequency-output)

### InverseFourierTransform
Inverse of :class:`~cobrabox.transforms.fourier_transform.FourierTransform`.

```python
cb.inverse_fourier_transform(data, ...)  # one-shot
cb.InverseFourierTransform(...).apply(data)  # composable, for pipelines
```

**Tags:** [`ifft`](../tags.md#tag-ifft), [`time-domain`](../tags.md#tag-time-domain), [`io:time-output`](../tags.md#tag-io-time-output), [`req:frequency-input`](../tags.md#tag-req-frequency-input)
