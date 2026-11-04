"""Load the swiss_demo dataset: whole, by subject, and by condition.

Run from the project root:
    uv run python examples/dataset_loader_demo.py

swiss_demo is simulated iEEG fitted to five subjects of the SWEC-ETHZ
short-term recordings. Each subject has one seizure, recorded as three
segments: preictal, ictal and postictal. Like every cobrabox dataset it is
downloaded on first use and cached under cb.get_dataset_dir().
"""

import cobrabox as cb

# What is in the dataset, and which subjects can be loaded on their own?
print(cb.dataset_info("swiss_demo"))

# Load only the subjects you need. Nothing else is read from disk.
ds = cb.load_dataset("swiss_demo", subset=["ID1", "ID4"])
ds.describe()

# Every recording carries its metadata.
for item in ds:
    print(
        f"{item.subjectID:>4}  {item.extra['seizureID']:>4}  {item.condition:<9}  "
        f"{item.data.sizes['space']} channels  {item.data.sizes['time'] / item.sampling_rate:.0f} s"
    )

# Slice by condition after loading.
ictal = ds.filter(condition="ictal")
print(f"\nIctal segments: {len(ictal)}")

# Group to compare states across subjects.
for condition, group in ds.groupby("condition").items():
    print(f"  {condition:<9}: {[item.subjectID for item in group]}")
