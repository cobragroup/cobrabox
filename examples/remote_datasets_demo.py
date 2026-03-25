"""Quick-start reference for working with remote datasets in cobrabox.

This demo walks through every dataset-related call in the public API, using
the Bonn EEG dataset as a concrete example (~2 MB per set, fast to download).
Everything here applies to any other remote dataset — just swap the identifier.

Available remote datasets
-------------------------
    bonn_eeg            ~10 MB total, ~2 MB per set (5 sets of 100 recordings)
    swiss_eeg_short     ~11 GB total, ~100 MB - 1 GB per subject (18 subjects)
    swiss_eeg_long      >1 TB total, ~100-200 GB per subject (18 subjects)
    chb_mit             ~30 GB total, ~1.5 GB per subject (24 subjects)
    siena_eeg           ~15 GB total, ~1 GB per subject (14 subjects)
    open_ieeg           ~13 GB total, ~70 MB per subject (185 subjects)

Local (synthetic) datasets — no download needed
------------------------------------------------
    dummy_chain     dummy_random     dummy_star     dummy_noise     realistic_swiss
"""

from __future__ import annotations

import cobrabox as cb

# =============================================================================
# 1. List all available dataset identifiers
# =============================================================================

# cb.list_datasets() returns a dict with "local" and "remote" keys.
datasets = cb.list_datasets()
print("Local datasets (no download):", datasets["local"])
print("Remote datasets:             ", datasets["remote"])
print()

# =============================================================================
# 2. Inspect a dataset before downloading anything
# =============================================================================

# cb.dataset_info(identifier) returns a DatasetInfo object.  No network call
# is made for datasets with a static file list (e.g. bonn_eeg, swiss_eeg_short,
# chb_mit, siena_eeg, helsinki_neonatal).
# Printing it gives a human-readable summary.
info = cb.dataset_info("bonn_eeg")
print(info)
print()

# The DatasetInfo fields are also accessible programmatically:
print(f"identifier      : {info.identifier}")
print(f"description     : {info.description}")
print(f"size_hint       : {info.size_hint}")
print(f"subset_key_name : {info.subset_key_name}")  # what the subsets represent
print(f"subsets         : {info.subsets}")  # all available subset keys
print(f"seizures/subset : {info.seizures_per_subject}")
print()

# =============================================================================
# 3. Load a subset (list form)
# =============================================================================

# Pass subset= as a list of subset keys to download only those entries.
# The first call downloads and caches the files; subsequent runs reuse them.
# For bonn_eeg the subset keys are set letters: Z, O, N, F, S.
#
# By default (verify=True) cobrabox shows a download summary and asks for
# confirmation before downloading any new files.  Pass verify=False to skip
# the prompt — useful in scripts and notebooks.
#
# Other examples:
#   cb.dataset("chb_mit",         subset=["chb01", "chb02"])
#   cb.dataset("siena_eeg",       subset=["PN00"])
#   cb.dataset("swiss_eeg_short", subset=["ID7", "ID8"])
#   cb.dataset("open_ieeg",       subset=["sub-Detroit001"])

ds = cb.dataset("bonn_eeg", subset=["S", "Z"], verify=False)
print(f"Loaded {len(ds)} recordings")
print()

# =============================================================================
# 4. Load a subset (dict form) — file-level control
# =============================================================================

# For datasets where each subject has many files (swiss_eeg_long), the dict
# form lets you control exactly how many files per subject to download:
#
#   cb.dataset("swiss_eeg_long", subset={"ID01": 2})
#       → first 2 hourly files for ID01 only
#
#   cb.dataset("swiss_eeg_long", subset={"ID01": ["ID01_1h.mat", "ID01_3h.mat"]})
#       → specific named files
#
#   cb.dataset("swiss_eeg_long", subset={"ID01": None, "ID02": 3})
#       → all files for ID01, first 3 for ID02

# =============================================================================
# 5. Inspect the Dataset object
# =============================================================================

# Dataset is an immutable sequence — supports len, indexing, iteration, and +.
print(f"Type        : {type(ds)}")
print(f"Length      : {len(ds)}")
print()

# ds.describe() prints a compact summary of subjects, groups, conditions, shapes.
ds.describe()
print()

# =============================================================================
# 6. Inspect a single SignalData recording
# =============================================================================

rec = ds[0]
print(f"Type           : {type(rec).__name__}")
print(f"Shape          : {dict(zip(rec.data.dims, rec.data.shape, strict=True))}")
print(f"Sampling rate  : {rec.sampling_rate} Hz")
print(f"subjectID      : {rec.subjectID}")
print(f"groupID        : {rec.groupID}")
print(f"condition      : {rec.condition}")
print(f"history        : {rec.history}")
print()

# =============================================================================
# 7. Filter and group a Dataset
# =============================================================================

# filter() returns a new Dataset with only matching recordings.
ictal = ds.filter(groupID="ictal")
healthy = ds.filter(groupID="healthy")
print(f"Ictal recordings  : {len(ictal)}")
print(f"Healthy recordings: {len(healthy)}")
print()

# groupby() splits a Dataset into a dict keyed by a metadata attribute.
by_condition = ds.groupby("condition")
print("Recordings per condition:")
for condition, group in by_condition.items():
    print(f"  {condition}: {len(group)}")
print()

# Datasets can be combined with +.
combined = ictal + healthy
print(f"Combined: {len(combined)} recordings")
print()

# =============================================================================
# 8. Run a feature pipeline across all recordings
# =============================================================================

# Features work the same regardless of which dataset you loaded.
# LineLength reduces over time → one value per channel (space dimension).
# Mean(dim="space") then averages across channels → scalar per recording.
pipeline = cb.LineLength() | cb.feature.Mean(dim="space")

print("Mean line length per recording:")
for rec in ds[:4]:
    result = pipeline.apply(rec)
    print(f"  {rec.subjectID} ({rec.condition}): {float(result.data):.1f}")
