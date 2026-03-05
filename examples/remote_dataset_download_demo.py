"""Demo: use cb.dataset for remote datasets.

Calling `cb.dataset("swiss_eeg_short")` should:
- download missing files if needed
- reuse local files when already present
"""

from __future__ import annotations

import cobrabox as cb

datasets = cb.dataset("swiss_eeg_short")
print(f"Loaded parts: {len(datasets)}")
if datasets:
    first = datasets[0]
    print("First part type:  ", type(first).__name__)
    print("Shape:            ", dict(zip(first.data.dims, first.data.shape, strict=True)))
    for dim in first.data.dims:
        coords = first.data.coords[dim].values if dim in first.data.coords else None
        print(f"  {dim} labels:   ", coords)
    print("Sampling rate:    ", first.sampling_rate, "Hz")
    print("Subject ID:       ", first.subjectID)
    print("Group ID:         ", first.groupID)
    print("Condition:        ", first.condition)
    print("History:          ", first.history)
