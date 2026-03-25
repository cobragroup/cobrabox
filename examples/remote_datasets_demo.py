"""Quick-start reference for working with remote datasets in cobrabox.

Uses the Bonn EEG dataset as a concrete example (~2 MB per set, fast to download).
Everything here applies to any other remote dataset — just swap the identifier.

Remote datasets
---------------
    bonn_eeg            ~10 MB total, ~2 MB per set (5 sets of 100 recordings)
    swiss_eeg_short     ~11 GB total, ~100 MB - 1 GB per subject (18 subjects)
    swiss_eeg_long      >1 TB total, ~100-200 GB per subject (18 subjects)
    chb_mit             ~30 GB total, ~1.5 GB per subject (24 subjects)
    siena_eeg           ~15 GB total, ~1 GB per subject (14 subjects)
    sleep_ieeg           ~13 GB total, ~70 MB per subject (185 subjects)

Local (synthetic) datasets — no download needed
    dummy_chain     dummy_random     dummy_star     dummy_noise     realistic_swiss
"""

from __future__ import annotations

import cobrabox as cb

# 1. List all datasets
datasets = cb.list_datasets()
print("Local :", datasets["local"])
print("Remote:", datasets["remote"])
print()

# 2. Inspect before downloading — prints a human-readable summary including
#    subset keys, size hints, and seizure counts.
info = cb.dataset_info("bonn_eeg")
print(info)
print()

# 3. Load a subset (list form)
# accept=True skips the confirmation prompt; omit it (default True) in interactive use.
# Other examples:
#   cb.dataset("chb_mit",         subset=["chb01", "chb02"])
#   cb.dataset("swiss_eeg_short", subset=["ID7", "ID8"])
#   cb.dataset("sleep_ieeg",       subset=["sub-Detroit001"])
ds = cb.dataset("bonn_eeg", subset=["S", "Z"], accept=True)
print(f"Loaded {len(ds)} recordings")
print()

# 4. Load a subset (dict form) — file-level control for multi-file subjects
#   cb.dataset("swiss_eeg_long", subset={"ID01": 2})                          # first 2 files
#   cb.dataset("swiss_eeg_long", subset={"ID01": ["ID01_1h.mat"]})            # specific files
#   cb.dataset("swiss_eeg_long", subset={"ID01": None, "ID02": 3})            # mix

# 5. Inspect the loaded Dataset
ds.describe()
print()

rec = ds[0]
print(f"Shape         : {dict(zip(rec.data.dims, rec.data.shape, strict=True))}")
print(f"Sampling rate : {rec.sampling_rate} Hz")
print(f"subjectID     : {rec.subjectID}  groupID: {rec.groupID}  condition: {rec.condition}")
print()

# 6. Filter and group
ictal = ds.filter(groupID="ictal")
by_condition = ds.groupby("condition")
print(f"Ictal recordings: {len(ictal)}")
print("Recordings per condition:", {k: len(v) for k, v in by_condition.items()})
print()
