# Working with Datasets

CobraBox provides built-in dummy datasets and a `Dataset[T]` collection class for working with groups of `Data` objects.

## Loading Datasets

```python
import cobrabox as cb

# Load a dataset — returns Dataset[SignalData]
ds = cb.load_dataset("dummy_chain")

# Inspect at a glance
ds.describe()
# Dataset  5 items  [SignalData]
#   subjectIDs : sub-01, sub-02, sub-03, sub-04, sub-05
#   groupIDs   : chain, chain, chain, chain, chain
#   conditions : None, None, None, None, None
#   shapes     : (10, 5000) × 5
#   labels     : sub-01, sub-02, sub-03, sub-04, sub-05
#   filter on  : subjectID, groupID, condition, runID, Description, Num_Samples
```

The last two rows tell you how to reach into the dataset: which labels are
available for lookup, and which fields `filter()` and `groupby()` will accept. The
`filter on` row is dataset-specific — it includes whatever the items carry in
`extra`, so `dummy_chain` lets you filter on `Description` and `Num_Samples` from
its sidecar metadata.

Each synthetic replicate stands in for one subject (`sub-01`, `sub-02`, …), and the
VAR topology becomes the `groupID`, which makes concatenated datasets group
cleanly:

```python
both = cb.load_dataset("dummy_chain") + cb.load_dataset("dummy_star")
{k: len(v) for k, v in both.groupby("groupID").items()}  # {'chain': 5, 'star': 4}
```

If the items carry no identifying metadata at all — as in a `Dataset` you build
yourself from bare arrays — the `labels` row says so, and names the fields that
will never match:

```text
  labels     : none — positional access only, not reachable by filter on subjectID, groupID, condition, runID
```

Only fields unset on *every* item are listed, so the row stays accurate as
metadata gets filled in.

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

### Labels

Items that carry identifying metadata can also be looked up by name. A label joins
`subjectID`, `condition` and `runID` with `/`, skipping whichever are unset — see
[Metadata fields](data-containers.md#metadata-fields) for what each means:

```python
ds = cb.load_dataset("dummy_chain")
ds.keys()  # ('sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05')
ds["sub-02"]  # the matching item
```

Labels are an index derived from the items, not a replacement for positional
storage — both reach the same object:

```python
ds[1] is ds["sub-02"]  # True
```

The two forms coexist without interfering, because positions stay integers and
labels stay strings. Even a subject literally named `"0"` is unambiguous: `ds[0]`
is the first item, `ds["0"]` is that subject.

Two consequences worth knowing:

- **Items without metadata contribute no label.** `keys()` may be shorter than
  `len(ds)`, and those items remain reachable by position.
- **Labels need not be unique.** Where several items share one, the lookup returns
  a `Dataset` of all of them rather than silently picking one.

### Reaching a whole subject

Labels get longer as metadata gets richer, but a lookup also accepts any *leading
part* of a label, so the coarser names keep working. `realistic_swiss` has one
subject `ID1`, two seizures, and three replicates:

```python
sw = cb.load_dataset("realistic_swiss")
sw.keys()  # ('ID1/sz13/1', 'ID1/sz13/2', 'ID1/sz7/3')

sw["ID1/sz13/2"]  # a SignalData — exact match
len(sw["ID1/sz13"])  # 2  → both replicates of that seizure
len(sw["ID1"])  # 3  → everything for that subject
```

This matters most on the long-monitoring datasets. Zurich subject `sub-01` has 39
recordings labelled `sub-01/01 … sub-01/39`; `ds["sub-01"]` still returns all 39 as
a `Dataset` rather than raising.

Matching is on `/` boundaries, so `ds["sub-0"]` matches nothing — it is a prefix of
the string but not of the label.

An unknown label raises `KeyError` listing what is available, so a typo is
reported rather than silently returning nothing:

```python
ds["sub-99"]
# KeyError: 'sub-99' is not a label in this Dataset.
#           Available: sub-01, sub-02, sub-03, sub-04, sub-05
```

### Discovering what is there

```python
ds.keys()  # labels available for lookup
ds.fields()  # every name filter() and groupby() accept
ds.unique("condition")  # the distinct values of one field
```

`fields()` includes the three standard metadata fields plus whatever the items
carry in `extra`, which varies by dataset — Zurich recordings add `ilae` and
`resected_zone`, for instance. `unique()` is the quickest way to see what a filter
could match before writing it.

### Combining datasets

```python
ds1 = cb.load_dataset("dummy_chain")
ds2 = cb.load_dataset("dummy_random")

combined = ds1 + ds2  # → Dataset[SignalData]
print(len(combined))
```

### Representation

```python
repr(ds)  # 'Dataset(3 × SignalData)'
str(ds)  # multi-line summary with shapes and metadata
ds.describe()  # prints str(ds)
```

## Filtering

Filter by any combination of metadata fields (AND semantics):

```python
ds = cb.Dataset(
    [
        cb.from_numpy(arr, dims=["time", "space"], subjectID="S1", groupID="control"),
        cb.from_numpy(arr, dims=["time", "space"], subjectID="S2", groupID="patient"),
        cb.from_numpy(arr, dims=["time", "space"], subjectID="S3", groupID="control"),
    ]
)

controls = ds.filter(groupID="control")  # Dataset with S1 and S3
s1_only = ds.filter(subjectID="S1", groupID="control")  # Dataset with S1

# Returns empty Dataset (not an error) if nothing matches
empty = ds.filter(groupID="nonexistent")
print(len(empty))  # 0
```

A criterion may also be a list, tuple, or set, matching any of the values:

```python
ds.filter(subjectID=["S1", "S2"])  # either subject
ds.filter(condition={"pre", "post"})  # either condition
```

Anything in an item's `extra` dict is filterable too — `fields()` lists what is
available for a given dataset:

```python
ds.filter(ilae=2)
```

A keyword that names no known field raises `ValueError` rather than quietly
matching nothing, so misspellings surface immediately.

### Expecting exactly one item

`filter()` always returns a `Dataset`, and an empty one when nothing matches. When
you expect a single item — and want a miss to be an error — use `one()`:

```python
sig = ds.one(subjectID="milan", condition="rest")  # the Data itself
```

It raises `ValueError` if nothing matches, and also if several do, so it can't
silently hand you the wrong recording when labels turn out not to be unique.

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

Group by several attributes at once to get one bucket per combination. With a
single attribute the keys are strings; with several they are tuples:

```python
by_both = ds.groupby("subjectID", "condition")
by_both[("milan", "rest")]
```

Valid attributes are anything `fields()` reports — the three standard metadata
fields, plus any key the items carry in `extra`.

## Available Dummy Datasets

<!-- local-dataset-table:start -->
| Identifier | Description |
| ---------- | ----------- |
| `dummy_chain` | Synthetic chain-topology VAR time-series (5 subjects). |
| `dummy_noise` | Synthetic uncorrelated noise time-series (5 subjects). |
| `dummy_random` | Synthetic random-topology VAR time-series (3 subjects). |
| `dummy_star` | Synthetic star-topology VAR time-series (4 subjects). |
| `realistic_swiss` | Simulated realistic Swiss VAR time-series (1 subject, 3 recordings). |
<!-- local-dataset-table:end -->

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

<!-- remote-dataset-table:start -->
| Identifier | Description | Subsets | Size | License |
| ---------- | ----------- | ------- | ---- | ------- |
| `bonn_eeg` | [Bonn University EEG dataset (Andrzejak et al. 2001): 5 sets of 100 single-channel recordings. Sets: Z = healthy eyes open, O = healthy eyes closed, N = interictal (seizure-free zone), F = interictal (epileptogenic zone), S = ictal (seizure). Hosted by Universitat Pompeu Fabra (DOI: 10.34810/data490).](https://repositori.upf.edu/handle/10230/42894) | 5 subsets | ~10 MB | Free for research and education only; commercial and military use prohibited. |
| `chb_mit` | [CHB-MIT Scalp EEG Database: pediatric patients with intractable seizures (24 subjects, 256 Hz, 23 channels, ictal/interictal). Children's Hospital Boston / MIT.](https://physionet.org/content/chbmit/1.0.0/) | 24 subjects | ~30 GB | Open Data Commons Attribution License v1.0 (ODC-By-1.0) |
| `siena_eeg` | [Siena Scalp EEG Database: adult epilepsy patients with annotated seizures (14 subjects, 512 Hz, 21+ channels, ictal/interictal). University of Siena.](https://physionet.org/content/siena-scalp-eeg/1.0.0/) | 14 subjects | ~15 GB | Creative Commons Attribution 4.0 International (CC-BY-4.0) |
| `sleep_ieeg` | [Sleep iEEG Dataset: interictal iEEG during slow-wave sleep from 185 epilepsy patients (135 Detroit at 1000 Hz, 50 UCLA at 2000 Hz). ECoG/sEEG recordings. DOI: 10.18112/openneuro.ds005398.v1.0.1.](https://openneuro.org/datasets/ds005398/versions/1.0.1) | 185 subjects | ~13 GB | CC0 1.0 Universal (public domain) |
| `swiss_eeg_long` | [Long-term intracranial EEG recordings from the SWEZ dataset (ETH Zurich, 18 subjects, ictal/interictal).](http://ieeg-swez.ethz.ch/) | 18 subjects | >1 TB (hundreds of hourly files per subject) | Free for research and education only; commercial and military use prohibited. |
| `swiss_eeg_short` | [Short-term scalp EEG recordings from the BioCAS 2018 challenge (18 subjects, ictal/interictal).](https://iis-people.ee.ethz.ch/~ieeg/BioCAS2018/) | 18 subjects | ~11 GB | Free for research and education only; commercial and military use prohibited. |
| `zurich_ieeg` | [Zurich iEEG HFO Dataset: interictal ECoG during slow-wave sleep from 20 epilepsy patients (TLE and extra-temporal), with HFO event markings. 2000 Hz, BrainVision format. DOI: 10.18112/openneuro.ds003498.v1.1.1.](https://openneuro.org/datasets/ds003498/versions/1.1.1) | 20 subjects | ~60 GB | CC0 1.0 Universal (public domain) |
| `zurich_ieeg_clean` | [Zurich iEEG HFO Dataset, cleaned: identical to 'zurich_ieeg' but with per-subject electrodes removed — channels lacking metadata and channels that produced motor/language responses under electrical stimulation (columns C+D of the Zurich electrode sheet). Shares the 'zurich_ieeg' download; no extra data is fetched.](https://openneuro.org/datasets/ds003498/versions/1.1.1) | 20 subjects | ~60 GB | CC0 1.0 Universal (public domain) |
<!-- remote-dataset-table:end -->

### Downloading a subset

Most datasets are large. Use `subset` to download only the subjects you need:

```python
# List form — all files for those subjects
ds = cb.load_dataset("chb_mit", subset=["chb01", "chb02"], accept=True)

# Dict form — fine-grained file-level control
ds = cb.load_dataset("swiss_eeg_long", subset={"ID01": 2}, accept=True)  # first 2 files
ds = cb.load_dataset(
    "swiss_eeg_long", subset={"ID01": ["ID01_1h.mat"]}, accept=True
)  # specific file
ds = cb.load_dataset(
    "swiss_eeg_long", subset={"ID01": None, "ID02": 3}, accept=True
)  # all of ID01, 3 of ID02
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
    items.append(
        cb.from_numpy(
            arr=arr,
            dims=["time", "space"],
            sampling_rate=100.0,
            subjectID=f"S{i + 1:02d}",
            groupID="control" if i < 3 else "patient",
            condition="rest",
        )
    )

ds = cb.Dataset(items)
ds.describe()
```

## Batch Processing

```python
ds = cb.load_dataset("dummy_chain")

pipeline = cb.SlidingWindow(window_size=20, step_size=10) | cb.LineLength() | cb.MeanAggregate()

results = cb.Dataset([pipeline.apply(item) for item in ds])
results.describe()
```

## Data Location

Dummy datasets are stored in `data/synthetic/dummy/` as compressed CSV files (`.csv.xz`) with optional JSON sidecar files for metadata.
