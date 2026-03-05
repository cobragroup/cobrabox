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
    print("First part type:", type(first).__name__)
