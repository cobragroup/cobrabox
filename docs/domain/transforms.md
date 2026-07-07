# Transforms

*How do I transform my signal into another representation?*

Features in the `cobrabox.transforms` domain. Access them as `cb.transforms.<Feature>` or `cb.feature.<Feature>`.

### AnalyticSignal
Extract analytic-signal representations along the time axis.

**Tags:** [`hilbert`](../tags/hilbert.md), [`instantaneous-phase`](../tags/instantaneous-phase.md), [`envelope`](../tags/envelope.md), [`instantaneous-frequency`](../tags/instantaneous-frequency.md), [`time-frequency`](../tags/time-frequency.md), [`eeg`](../tags/eeg.md), [`fmri`](../tags/fmri.md), [`io:preserves-time`](../tags/io-preserves-time.md)

### BandpassFilter
Filter a signal into frequency bands.

**Tags:** [`filtering`](../tags/filtering.md), [`butterworth`](../tags/butterworth.md), [`frequency-band`](../tags/frequency-band.md), [`preprocessing`](../tags/preprocessing.md), [`eeg`](../tags/eeg.md), [`io:preserves-time`](../tags/io-preserves-time.md), [`io:adds-dimension`](../tags/io-adds-dimension.md)

### FourierTransform
Real-valued FFT along the time axis.

**Tags:** [`fft`](../tags/fft.md), [`frequency-domain`](../tags/frequency-domain.md), [`eeg`](../tags/eeg.md), [`fmri`](../tags/fmri.md), [`io:frequency-output`](../tags/io-frequency-output.md)

### InverseFourierTransform
Inverse of :class:`~cobrabox.transforms.fourier_transform.FourierTransform`.

**Tags:** [`ifft`](../tags/ifft.md), [`time-domain`](../tags/time-domain.md), [`io:time-output`](../tags/io-time-output.md), [`req:frequency-input`](../tags/req-frequency-input.md)
