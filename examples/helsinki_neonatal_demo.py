"""Demo: loading the Helsinki Neonatal EEG Seizure dataset.

Downloads one EDF file (~50-200 MB) for a single neonate and inspects the
resulting Dataset.  Subsequent runs reuse the cached file.

Dataset: 79 term neonates, 256 Hz, 19 EEG channels + ECG + respiratory,
         with expert seizure annotations.
         Helsinki University Hospital NICU.
         DOI: 10.5281/zenodo.2547147.
         License: CC BY 4.0.
"""

from __future__ import annotations

import cobrabox as cb

# Show dataset metadata (no download triggered).
ds_info = cb.dataset_info("helsinki_neonatal")
print(ds_info)
print()

# Download one neonate recording.  File size varies (~50-200 MB).
ds = cb.dataset("helsinki_neonatal", subset=["eeg1"])
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
