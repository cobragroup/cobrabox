"""Demo: loading the Swiss short-term scalp EEG dataset (BioCAS 2018).

Downloads two subjects from the BioCAS 2018 challenge dataset and inspects
the resulting Dataset.  Subsequent runs reuse the cached files.

Note: file sizes vary significantly across subjects (102 MB - 990 MB per zip).
      ID7 (~102 MB) and ID8 (~185 MB) are the two smallest subjects.

Dataset: 18 subjects with short-term scalp EEG, ictal/interictal.
         Hosted at iis-people.ee.ethz.ch/~ieeg/BioCAS2018.
"""

from __future__ import annotations

import cobrabox as cb

# Show dataset metadata (no download triggered).
ds_info = cb.dataset_info("swiss_eeg_short")
print(ds_info)
print()

# Download the two smallest subjects (ID7 ~102 MB, ID8 ~185 MB).
ds = cb.dataset("swiss_eeg_short", subset=["ID7", "ID8"])
print(f"Loaded recordings: {len(ds)}")
print()

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

# LineLength reduces over time → output is per-channel (space dimension only).
# Chain with Mean(dim="space") to get a single scalar per recording.
pipeline = cb.LineLength() | cb.feature.Mean(dim="space")
results = [pipeline.apply(rec) for rec in ds]
print("LineLength → Mean across channels:")
for rec, result in zip(ds, results, strict=True):
    shape = dict(zip(result.data.dims, result.data.shape, strict=True))
    print(f"  {rec.subjectID}: shape={shape}, value={float(result.data):.4f}")
