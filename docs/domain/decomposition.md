# Signal Decomposition Features

Decomposition features break signals into simpler components.

## Features

### FourierTransformSurrogates
Generate Fourier transform surrogates preserving power spectrum (splitter feature).

### EMD
Empirical Mode Decomposition into Intrinsic Mode Functions (IMFs).

## Usage

```python
import cobrabox as cb
import numpy as np

data = cb.from_numpy(np.random.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

# Single feature
result = cb.feature.EMD().apply(data)

# Pipeline with sliding window
pipeline = cb.Chord(
    split=cb.feature.SlidingWindow(window_size=20, step_size=10),
    pipeline=cb.feature.FourierTransformSurrogates(n_surrogates=10),
    aggregate=cb.feature.MeanAggregate(),
)
result = pipeline.apply(data)
```

## See Also

- [Time-Frequency](time_frequency.md) for joint time-frequency methods

