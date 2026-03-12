# Frequency Domain Features

Frequency-domain features compute spectral and frequency-based statistics from the signal.

## Features

### Bandpower
Compute power in specific frequency bands using Welch's method.

### BandFilter
Apply Butterworth bandpass filter to isolate frequency bands.

### Spectrogram
Compute time-frequency power spectrogram.

### EpileptogenicityIndex
Quantify epileptogenicity from SEEG (Bartolomei et al., 2008).

### Cordance
Quantitative EEG cordance combining absolute and relative bandpower.

## Usage

```python
import cobrabox as cb
import numpy as np

data = cb.from_numpy(np.random.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

# Single feature
result = cb.feature.Bandpower(bands={'theta': (4, 8), 'alpha': (8, 12)}).apply(data)

# Pipeline with sliding window
pipeline = cb.Chord(
    split=cb.feature.SlidingWindow(window_size=20, step_size=10),
    pipeline=cb.feature.Bandpower(bands={'theta': (4, 8)}),
    aggregate=cb.feature.MeanAggregate(),
)
result = pipeline.apply(data)
```

## See Also

- [Time Domain](time_domain.md) for temporal statistics
- [Time-Frequency](time_frequency.md) for joint time-frequency methods
