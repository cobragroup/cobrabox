"""Quick-start reference for working with remote datasets in cobrabox.

Uses the Bonn EEG dataset as a concrete example (~2 MB per set, fast to download).
Everything here applies to any other remote dataset — just swap the identifier.
"""

from __future__ import annotations

import cobrabox as cb

# 1. Browse all available datasets
cb.describe_datasets()
print()

# 2. List identifiers programmatically
datasets = cb.list_datasets()
print("Local :", datasets["local"])
print("Remote:", datasets["remote"])
print()

# 3. Inspect a dataset before downloading — shows subset keys, size, seizures, license.
info = cb.dataset_info("bonn_eeg")
print(info)
print()

# 4. Download without loading (useful for pre-fetching large datasets)
# accept=True skips the confirmation prompt; omit it (default False) in interactive use.
#   cb.download_dataset("chb_mit",         subset=["chb01", "chb02"], accept=True)
#   cb.download_dataset("swiss_eeg_long",  subset={"ID01": 2},        accept=True)
#   cb.download_dataset("sleep_ieeg",      subset=["sub-Detroit001"], accept=True)

# 5. Load a subset directly (downloads if needed, then loads into memory)
# List form — all files for the given subset keys:
ds = cb.load_dataset("bonn_eeg", subset=["S", "Z"], accept=True)
print(f"Loaded {len(ds)} recordings")
print()

# Dict form — file-level control for multi-file subjects:
#   cb.load_dataset("swiss_eeg_long", subset={"ID01": 2})                    # first 2 files
#   cb.load_dataset("swiss_eeg_long", subset={"ID01": ["ID01_1h.mat"]})      # specific files
#   cb.load_dataset("swiss_eeg_long", subset={"ID01": None, "ID02": 3})      # mix

# 6. Redirect downloads to a custom location (persisted across sessions)
#   cb.set_dataset_dir("/mnt/shared/cobrabox")

# 7. Inspect the loaded Dataset
ds.describe()
print()

rec = ds[0]
print(f"Shape         : {dict(zip(rec.data.dims, rec.data.shape, strict=True))}")
print(f"Sampling rate : {rec.sampling_rate} Hz")
print(f"subjectID     : {rec.subjectID}  groupID: {rec.groupID}  condition: {rec.condition}")
print()

# 8. Filter and group
ictal = ds.filter(groupID="ictal")
by_condition = ds.groupby("condition")
print(f"Ictal recordings: {len(ictal)}")
print("Recordings per condition:", {k: len(v) for k, v in by_condition.items()})
