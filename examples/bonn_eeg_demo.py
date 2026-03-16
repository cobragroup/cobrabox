"""Demo: loading the Bonn University EEG dataset.

Downloads one set ("S" — ictal) from the Bonn EEG archive (~2 MB) and
inspects the resulting Dataset.  Subsequent runs reuse the cached files.

Dataset: Andrzejak et al. 2001, 5 sets of 100 single-channel recordings.
         Hosted by Universitat Pompeu Fabra (DOI: 10.34810/data490).

Sets:
    Z — healthy subjects, eyes open
    O — healthy subjects, eyes closed
    N — interictal EEG, contralateral hemisphere
    F — interictal EEG, epileptogenic zone
    S — ictal EEG (seizures)
"""

from __future__ import annotations

import cobrabox as cb

# Show dataset metadata (no download triggered).
ds_info = cb.dataset_info("bonn_eeg")
print(ds_info)
print()

# Download one set — the "S" (ictal) archive is ~2 MB.
ds = cb.dataset("bonn_eeg", subset=["S"])
print(f"Loaded recordings: {len(ds)}")

first = ds[0]
print("First recording type:  ", type(first).__name__)
print("Shape:                 ", dict(zip(first.data.dims, first.data.shape, strict=True)))
print("Sampling rate:         ", first.sampling_rate, "Hz")
print("Subject ID:            ", first.subjectID)
print("Group ID:              ", first.groupID)
print("Condition:             ", first.condition)
print("History:               ", first.history)
