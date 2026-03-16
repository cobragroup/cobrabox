"""Demo: loading the Siena Scalp EEG database.

Downloads the EDF files for one subject (~200-500 MB) and inspects the
resulting Dataset.  Subsequent runs reuse the cached files.

Dataset: 14 adult epilepsy patients, 512 Hz, 21+ channels, ictal/interictal.
         University of Siena, hosted by PhysioNet.
         License: CC BY 4.0.

Note: The file list is fetched from the PhysioNet RECORDS index on first run.
      Fetching one subject's files can take several minutes depending on
      connection speed.
"""

from __future__ import annotations

import cobrabox as cb

# Show dataset metadata.  For Siena the subset list is fetched dynamically
# from PhysioNet on first call, so this may take a moment.
ds_info = cb.dataset_info("siena_eeg")
print(ds_info)
print()

# Download one subject.  "PN00" is the first available subject.
ds = cb.dataset("siena_eeg", subset=["PN00"])
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
