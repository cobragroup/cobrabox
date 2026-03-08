# Working with Datasets

CobraBox provides built-in dummy datasets and a `Dataset[T]` collection class for working with groups of `Data` objects.

## Loading Datasets

```python
import cobrabox as cb

# Load a dataset — returns Dataset[SignalData]
ds = cb.dataset("dummy_chain")

# Inspect at a glance
ds.describe()
# Dataset  3 items  [SignalData]
#   subjectIDs : None, None, None
#   groupIDs   : None, None, None
#   conditions : None, None, None
#   shapes     : (4, 200) × 3
```

## The `Dataset[T]` Class

`Dataset[T]` is an immutable, typed collection of `Data` objects. It behaves like a read-only sequence.

### Indexing and iteration

```python
ds = cb.dataset("dummy_chain")

# Integer index → single item
item = ds[0]
print(item.data.shape)

# Slice → new Dataset
subset = ds[1:3]

# Iteration
for item in ds:
    print(item.sampling_rate)

# Length and membership
print(len(ds))
print(item in ds)
```

### Combining datasets

```python
ds1 = cb.dataset("dummy_chain")
ds2 = cb.dataset("dummy_random")

combined = ds1 + ds2   # → Dataset[SignalData]
print(len(combined))
```

### Representation

```python
repr(ds)    # 'Dataset(3 × SignalData)'
str(ds)     # multi-line summary with shapes and metadata
ds.describe()  # prints str(ds)
```

## Filtering

Filter by any combination of metadata fields (AND semantics):

```python
ds = cb.Dataset([
    cb.from_numpy(arr, dims=["time", "space"], subjectID="S1", groupID="control"),
    cb.from_numpy(arr, dims=["time", "space"], subjectID="S2", groupID="patient"),
    cb.from_numpy(arr, dims=["time", "space"], subjectID="S3", groupID="control"),
])

controls = ds.filter(groupID="control")   # Dataset with S1 and S3
s1_only  = ds.filter(subjectID="S1", groupID="control")  # Dataset with S1

# Returns empty Dataset (not an error) if nothing matches
empty = ds.filter(groupID="nonexistent")
print(len(empty))  # 0
```

## Grouping

Split a `Dataset` into sub-datasets keyed by a metadata attribute:

```python
groups = ds.groupby("groupID")
# {'control': Dataset(2 × Data), 'patient': Dataset(1 × Data)}

for name, group_ds in groups.items():
    print(f"{name}: {len(group_ds)} subjects")

# Items with no value for the attribute go to key "None"
groups_with_none = ds.groupby("condition")
print("None" in groups_with_none)
```

Valid attributes: `"subjectID"`, `"groupID"`, `"condition"`.

## Available Dummy Datasets

| Identifier | Description |
| ---------- | ----------- |
| `dummy_chain` | Chain-like pattern for pipeline testing |
| `dummy_random` | Random noise for baseline testing |
| `dummy_star` | Star-shaped pattern for spatial analysis |
| `dummy_noise` | Pure noise for null hypothesis testing |

## Building a Custom Dataset

Wrap any list of `Data` objects:

```python
import cobrabox as cb
import numpy as np

items = []
for i in range(5):
    arr = np.random.default_rng(i).normal(size=(100, 4))
    items.append(cb.from_numpy(
        arr=arr,
        dims=["time", "space"],
        sampling_rate=100.0,
        subjectID=f"S{i+1:02d}",
        groupID="control" if i < 3 else "patient",
        condition="rest",
    ))

ds = cb.Dataset(items)
ds.describe()
```

## Batch Processing

```python
ds = cb.dataset("dummy_chain")

pipeline = (
    cb.feature.SlidingWindow(window_size=20, step_size=10)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
)

results = cb.Dataset([pipeline.apply(item) for item in ds])
results.describe()
```

## Data Location

Dummy datasets are stored in `data/synthetic/dummy/` as compressed CSV files (`.csv.xz`) with optional JSON sidecar files for metadata.
