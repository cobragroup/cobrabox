# Working with Datasets

CobraBox provides built-in dummy datasets and a `Dataset[T]` collection class for working with groups of `Data` objects.

## Loading Datasets

```python
import cobrabox as cb

# Load a dataset — returns Dataset[SignalData]
ds = cb.load_dataset("dummy_chain")

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
ds = cb.load_dataset("dummy_chain")

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
ds1 = cb.load_dataset("dummy_chain")
ds2 = cb.load_dataset("dummy_random")

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

## Remote Datasets

CobraBox can download real EEG datasets from public repositories. Files are stored locally
under `data/remote/` and reused on subsequent calls — a dataset is only downloaded once.

### Listing available datasets

```python
cb.list_datasets()
# {
#   'local':  ['dummy_chain', 'dummy_noise', 'dummy_random', 'dummy_star', 'realistic_swiss'],
#   'remote': ['bonn_eeg', 'chb_mit', 'siena_eeg', 'sleep_ieeg', 'swiss_eeg_long', 'swiss_eeg_short', 'zurich_ieeg'],
# }
```

### Inspecting a dataset before downloading

`cb.dataset_info()` returns metadata without triggering any download:

```python
info = cb.dataset_info("chb_mit")
print(info)
# DatasetInfo: chb_mit
#   description : CHB-MIT Scalp EEG Database: pediatric patients with intractable seizures ...
#   size        : total ~30 GB, ~1.5 GB per subject (approximate)
#   subjects (24): chb01, chb02, chb03, ..., chb24
#   usage       : cb.load_dataset("chb_mit", subset=["chb01", "chb02"])
#   seizures/subject (200 total):
#     chb01  7   chb02  3   chb03  7   ...
#   seizure src : https://physionet.org/content/chbmit/1.0.0/
#   license     : Open Data Commons Attribution License v1.0 (ODC-By-1.0)
#   license url : https://physionet.org/content/chbmit/1.0.0/
```

### Downloading a dataset

By default, CobraBox shows a confirmation prompt before downloading anything. It displays
the dataset description, license, and estimated download size:

```
Dataset: chb_mit
  CHB-MIT Scalp EEG Database: ...

  License: Open Data Commons Attribution License v1.0 (ODC-By-1.0)
  More info: https://physionet.org/content/chbmit/1.0.0/

  Files to download: 664
  Estimated download size: ~30 GB

Proceed with download? [y/N]
```

Once you have reviewed and accepted the license, pass `accept=True` to skip the prompt
in scripts:

```python
ds = cb.load_dataset("bonn_eeg", accept=True)
```

### Available remote datasets

| Identifier | Description | Size |
| ---------- | ----------- | ---- |
| `bonn_eeg` | Bonn University EEG (Andrzejak et al. 2001) — 5 sets, single-channel, ictal/interictal | ~10 MB |
| `chb_mit` | CHB-MIT Scalp EEG — 24 pediatric subjects, 256 Hz, 23 channels | ~30 GB |
| `siena_eeg` | Siena Scalp EEG — 14 adult subjects, 512 Hz, 21+ channels | ~15 GB |
| `swiss_eeg_short` | BioCAS 2018 short-term scalp EEG — 18 subjects, ictal/interictal | ~11 GB |
| `swiss_eeg_long` | SWEZ long-term iEEG — 18 subjects, hourly files | >1 TB |
| `sleep_ieeg` | Sleep iEEG (OpenNeuro ds005398) — 185 subjects, interictal sleep ECoG/sEEG | ~13 GB |
| `zurich_ieeg` | Zurich iEEG HFO (OpenNeuro ds003498) — 20 epilepsy patients, interictal ECoG, 2000 Hz, with HFO markings | ~60 GB |

### Downloading a subset

Most datasets are large. Use `subset` to download only the subjects you need:

```python
# List form — all files for those subjects
ds = cb.load_dataset("chb_mit", subset=["chb01", "chb02"], accept=True)

# Dict form — fine-grained file-level control
ds = cb.load_dataset("swiss_eeg_long", subset={"ID01": 2}, accept=True)          # first 2 files
ds = cb.load_dataset("swiss_eeg_long", subset={"ID01": ["ID01_1h.mat"]}, accept=True)  # specific file
ds = cb.load_dataset("swiss_eeg_long", subset={"ID01": None, "ID02": 3}, accept=True)  # all of ID01, 3 of ID02
```

Call `cb.dataset_info()` to see the available subset keys for a dataset before downloading.

## Configuring the Data Directory

By default CobraBox stores downloaded files in a platform cache directory
(`~/.cache/cobrabox` on Linux, `~/Library/Caches/cobrabox` on macOS,
`%LOCALAPPDATA%\cobrabox` on Windows).

You can override this at any time:

```python
# Redirect to a project folder or shared storage (persists across restarts)
cb.set_dataset_dir("/mnt/data/cobrabox")

# In-process only (not written to disk)
cb.set_dataset_dir("/scratch/tmp", persist=False)

# See where data currently lives
print(cb.get_dataset_dir())
```

You can also set the `COBRABOX_DATA_DIR` environment variable before starting
Python — it takes priority over every other setting.

The directory is created automatically on the first download.

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
ds = cb.load_dataset("dummy_chain")

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
