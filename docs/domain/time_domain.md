# Time Domain Features

Time-domain features compute morphological and temporal statistics directly on the signal in the time domain.

## Features

### LineLength
Compute line length over the time dimension — sum of absolute differences between consecutive timepoints.

### AmplitudeVariation
Measure the amplitude variation (standard deviation) over time.

### SampleEntropy
Compute sample entropy — a measure of signal regularity and complexity.

### LempelZiv
Compute Lempel-Ziv complexity — a measure of signal complexity based on pattern distinctness.

### FractalDimHiguchi
Compute Higuchi Fractal Dimension — a measure of signal roughness/complexity.

### FractalDimKatz
Compute Katz Fractal Dimension — a fast, parameter-free complexity measure.

### SpikeCount
Detect spikes using the IQR method and count outliers.

### Autocorr
Compute normalized autocorrelation at a single lag.

### EnvelopeCorrelation
Compute amplitude envelope correlation (AEC) between all channel pairs.

### RecurrenceMatrix
Compute pairwise recurrence (self-similarity) matrix across time-points or windows.

### Nonreversibility
Compute normalized deviation from causal normality — a measure of time-irreversibility.

## Usage

```python
import cobrabox as cb
import numpy as np

data = cb.from_numpy(np.random.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

# Single feature
result = cb.feature.LineLength().apply(data)

# Pipeline with sliding window
pipeline = cb.Chord(
    split=cb.feature.SlidingWindow(window_size=20, step_size=10),
    pipeline=cb.feature.LineLength(),
    aggregate=cb.feature.MeanAggregate(),
)
result = pipeline.apply(data)
```

## See Also

- [Windowing & Aggregation](windowing.md) for splitting signals into windows
- [Frequency Domain](frequency_domain.md) for spectral analysis
- [Time-Frequency](time_frequency.md) for joint time-frequency methods
