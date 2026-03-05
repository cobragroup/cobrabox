# Working with Datasets

CobraBox provides built-in dummy datasets for testing and development.

## Loading Datasets

```python
import cobrabox as cb

# Load a dataset (returns list[Data])
datasets = cb.dataset("dummy_chain")

# Access individual subjects
data = datasets[0]
print(f"Subject: {data.subjectID}")
print(f"Shape: {data.data.shape}")
```

## Available Datasets

### dummy_chain

Chain-like pattern data for testing pipelines.

### dummy_random

Random noise data for baseline testing.

### dummy_star

Star-shaped pattern for spatial analysis testing.

### dummy_noise

Pure noise data for null hypothesis testing.

## Dataset Structure

Each dataset returns a list of `Data` objects, one per subject:

```python
datasets = cb.dataset("dummy_chain")

for i, data in enumerate(datasets):
    print(f"Subject {i}:")
    print(f"  Shape: {data.data.shape}")
    print(f"  SubjectID: {data.subjectID}")
    print(f"  Sampling rate: {data.sampling_rate}")
```

## Data Location

Dummy datasets are stored in `data/dummy/` as compressed CSV files. The dataset loader reads and converts them to `Data` objects.

## Creating Custom Datasets

To load your own data:

```python
import cobrabox as cb
import numpy as np
import pandas as pd

# Load your data
df = pd.read_csv("my_data.csv")
arr = df.values

# Wrap in Data
data = cb.from_numpy(
    arr=arr,
    dims=["time", "space"],
    sampling_rate=100.0,
    subjectID="sub-01",
    condition="task"
)
```

## Batch Processing

Process multiple subjects:

```python
datasets = cb.dataset("dummy_chain")

results = []
for data in datasets:
    # Apply pipeline
    feat = cb.feature.LineLength().apply(data)
    results.append(feat)

# Combine results
all_results = cb.from_numpy(
    arr=np.stack([r.data.values for r in results]),
    dims=["subject", "space"]
)
```

## Inspecting Datasets

```python
data = cb.dataset("dummy_chain")[0]

# View summary
print(data)

# Access coordinates
print(data.data.coords)

# Check metadata
print(f"Subject: {data.subjectID}")
print(f"Condition: {data.condition}")

# View history (empty for raw data)
print(f"History: {data.history}")
```
