"""Demo: loading the CHB-MIT Scalp EEG database.

Downloads the EDF files for one subject (~1-2 GB) and inspects the resulting
Dataset.  Subsequent runs reuse the cached files.

Dataset: 22 pediatric patients with intractable seizures, 256 Hz, 23 channels.
         Children's Hospital Boston / MIT, hosted by PhysioNet.
         License: Open Data Commons Attribution License v1.0.

Note: The file list is fetched from the PhysioNet RECORDS index on first run.
      Each subject has multiple multi-hour EDF files.  Downloading one subject
      ("chb01") requires ~1-2 GB and may take several minutes.
"""

from __future__ import annotations

import cobrabox as cb

# Show dataset metadata.  The subset list is fetched dynamically from
# PhysioNet on first call, so this may take a moment.
ds_info = cb.dataset_info("chb_mit")
print(ds_info)
print()

# Download one subject.  "chb01" is the first available subject.
# Warning: this downloads ~1-2 GB worth of EDF files.
ds = cb.dataset("chb_mit", subset=["chb01"])
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
