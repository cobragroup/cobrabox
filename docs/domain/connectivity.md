# Connectivity Features

Connectivity features compute pairwise interactions between channels.

## Features

### Correlation
Pairwise Pearson or Spearman correlation matrix between channels.

### Covariance
Pairwise sample covariance matrix between channels.

### PartialCorrelation / PartialCorrelationMatrix
Partial correlation controlling for other variables.

### PartialDirectedCoherence
Partial Directed Coherence via VAR model (directional frequency-domain connectivity).

### ReciprocalConnectivity
Net directional role per channel (source/sink detection from PDC).

### Coherence
Magnitude-squared coherence between channel pairs.

### PhaseLockingValue / PhaseLockingValueMatrix
Phase locking value between channels.

### GrangerCausality / GrangerCausalityMatrix
Granger causality testing.

### MutualInformation
Pairwise mutual information matrix between channels.

## Usage

```python
import cobrabox as cb
import numpy as np

data = cb.from_numpy(np.random.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

# Single feature
result = cb.feature.Correlation(method='pearson').apply(data)

# Pipeline with sliding window
pipeline = cb.Chord(
    split=cb.feature.SlidingWindow(window_size=20, step_size=10),
    pipeline=cb.feature.Coherence(),
    aggregate=cb.feature.MeanAggregate(),
)
result = pipeline.apply(data)
```

## See Also

- [Time Domain](time_domain.md) for temporal statistics
- [Frequency Domain](frequency_domain.md) for spectral analysis

