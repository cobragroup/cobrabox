# Reduction Features

Reduction features compute summary statistics by reducing over dimensions.

## Features

### Mean
Compute mean over any dimension.

### Min
Compute minimum over any dimension.

### Max
Compute maximum over any dimension.

## Usage

```python
import cobrabox as cb
import numpy as np

data = cb.from_numpy(np.random.normal(size=(100, 4)), dims=["time", "space"], sampling_rate=100.0)

# Reduce over time dimension (default)
result = cb.feature.Mean().apply(data)

# Reduce over space dimension
result = cb.feature.Mean(dim='space').apply(data)
```

## See Also

- [Time Domain](time_domain.md) for more complex temporal statistics
- [Windowing & Aggregation](windowing.md) for window-based reductions

