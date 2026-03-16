"""Demo: loading the SWEZ long-term intracranial EEG dataset.

Uses the dict-form subset to download only the first 2 hours for subject ID01
(~1.2 GB) instead of all 294 hours (~180 GB).  Subsequent runs reuse the
cached files.

Dataset: 18 subjects with long-term intracranial EEG, ictal/interictal.
         ETH Zurich (SWEZ dataset).

WARNING: Each subject has hundreds of hourly .mat files (~619 MB each).
         cb.dataset("swiss_eeg_long", subset=["ID01"]) downloads ALL hours
         for that subject — potentially 100+ GB.  Use the dict form to limit
         the download to a manageable number of files.
"""

from __future__ import annotations

import cobrabox as cb

# Show dataset metadata (no download triggered).
ds_info = cb.dataset_info("swiss_eeg_long")
print(ds_info)
print()

# Dict-form subset: download only the first 2 hourly files for ID01 (~1.2 GB).
print("Loading first 2 hours for ID01 (dict-form subset)...")
ds = cb.dataset("swiss_eeg_long", subset={"ID01": 2})
print(f"Loaded recordings: {len(ds)}")

first = ds[0]
print("First recording type:  ", type(first).__name__)
print("Shape:                 ", dict(zip(first.data.dims, first.data.shape, strict=True)))
for dim in first.data.dims:
    coords = first.data.coords[dim].values if dim in first.data.coords else None
    print(f"  {dim} labels:        ", coords[:5] if coords is not None else None, "...")
print("Sampling rate:         ", first.sampling_rate, "Hz")
print("Subject ID:            ", first.subjectID)
print("History:               ", first.history)
print()

# You can also select specific files by name:
#   ds = cb.dataset("swiss_eeg_long", subset={"ID01": ["ID01_1h.mat", "ID01_5h.mat"]})
#
# Or load all files for multiple subjects (be aware of the size):
#   ds = cb.dataset("swiss_eeg_long", subset={"ID01": None, "ID02": 3})
