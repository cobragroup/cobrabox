# Time-Frequency Features

Time-frequency features provide joint time and frequency analysis of signals.

## Features

### Hilbert
Compute analytic signal, envelope, phase, or instantaneous frequency.

### WaveletTransform
Multi-level discrete wavelet decomposition (DWT) or continuous wavelet transform.

### AmplitudeEntropy
Amplitude entropy from histogram-based distribution.

### EMD
Empirical Mode Decomposition into Intrinsic Mode Functions (IMFs).

## Usage

```python
import cobrabox as cb
import numpy as np

data = cb.from_numpy(np.random.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

# Single feature
result = cb.feature.Hilbert(mode='envelope').apply(data)

# Pipeline with sliding window
pipeline = cb.Chord(
    split=cb.feature.SlidingWindow(window_size=20, step_size=10),
    pipeline=cb.feature.WaveletTransform(level=3),
    aggregate=cb.feature.MeanAggregate(),
)
result = pipeline.apply(data)
```

## See Also

- [Time Domain](time_domain.md) for temporal statistics
- [Frequency Domain](frequency_domain.md) for spectral analysis
