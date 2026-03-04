# Pipelines

Build reproducible analysis pipelines by chaining features.

## Basic Pipeline

```python
import cobrabox as cb

# Load data
data = cb.dataset("dummy_chain")[0]

# Apply pipeline
wdata = cb.feature.sliding_window(data, window_size=10, step_size=5)
win_min = cb.feature.min(wdata, dim="window_index")
win_max = cb.feature.max(wdata, dim="window_index")
line_len = cb.feature.line_length(data)

# Track history
print(win_min.history)  # ['sliding_window', 'min']
print(win_max.history)  # ['sliding_window', 'max']
print(line_len.history)  # ['line_length']
```

## Pipeline Function

Encapsulate pipelines in functions:

```python
def preprocess_and_extract(data: cb.Data) -> cb.Data:
    """Apply preprocessing and feature extraction."""
    # Sliding window
    windowed = cb.feature.sliding_window(data, window_size=10, step_size=5)
    
    # Extract features per window
    min_vals = cb.feature.min(windowed, dim="window_index")
    max_vals = cb.feature.max(windowed, dim="window_index")
    
    # Line length on original data
    line_len = cb.feature.line_length(data)
    
    return line_len  # or return multiple results
```

## Pipeline with Multiple Features

```python
def extract_all_features(data: cb.Data) -> dict:
    """Extract multiple features."""
    features = {}
    
    # Basic statistics
    features["mean"] = cb.feature.mean(data, dim="time")
    features["min"] = cb.feature.min(data, dim="time")
    features["max"] = cb.feature.max(data, dim="time")
    
    # Line length
    features["line_length"] = cb.feature.line_length(data)
    
    return features
```

## Batch Processing

```python
import numpy as np

def process_subjects(dataset_name: str) -> np.ndarray:
    """Process all subjects in a dataset."""
    datasets = cb.dataset(dataset_name)
    
    results = []
    for data in datasets:
        # Apply pipeline
        feat = cb.feature.line_length(data)
        results.append(feat.data.values)
    
    # Stack into array: [subjects, space]
    return np.stack(results)
```

## Pipeline with Conditional Logic

```python
def adaptive_pipeline(data: cb.Data, threshold: float) -> cb.Data:
    """Apply different features based on data properties."""
    # Compute basic stats
    mean_val = data.data.mean()
    
    if mean_val > threshold:
        # High activity: use sliding window
        windowed = cb.feature.sliding_window(data, window_size=10)
        return cb.feature.max(windowed, dim="window_index")
    else:
        # Low activity: simple line length
        return cb.feature.line_length(data)
```

## Tracking Pipeline History

The `history` attribute tracks all applied operations:

```python
data = cb.from_numpy(arr, dims=["time", "space"])

# Build pipeline
result = (
    data
    .pipe(cb.feature.sliding_window, window_size=10)
    .pipe(cb.feature.min, dim="window_index")
)

print(result.history)  # ['sliding_window', 'min']
```

## Saving Pipeline Results

```python
import pickle

# Process data
result = cb.feature.line_length(data)

# Save to file
with open("results.pkl", "wb") as f:
    pickle.dump(result, f)

# Load later
with open("results.pkl", "rb") as f:
    loaded = pickle.load(f)

print(loaded.history)  # Preserved!
```

## Pipeline Visualization

```python
def visualize_pipeline(data: cb.Data):
    """Print pipeline history."""
    print("Pipeline:")
    for i, step in enumerate(data.history, 1):
        print(f"  {i}. {step}")
    print(f"\nFinal shape: {data.data.shape}")
    print(f"Subject: {data.subjectID}")
```

## Best Practices

1. **Keep pipelines modular** - Break into small, testable functions
2. **Document each step** - Explain why each feature is applied
3. **Validate intermediate results** - Check shapes, ranges, NaN values
4. **Preserve history** - Don't manually modify the `history` attribute
5. **Test on dummy data** - Use `cb.dataset()` datasets for development
