"""Demo: loading a subset of the Open iEEG Dataset via cobrabox.

The Open iEEG Dataset (ds005398) contains interictal iEEG recordings during
slow-wave sleep from 185 epilepsy patients (135 Detroit at 1000 Hz, 50 UCLA
at 2000 Hz). Files are downloaded from the OpenNeuro S3 mirror on first use
and cached locally. License: CC0.

DOI: 10.18112/openneuro.ds005398.v1.0.1
"""

import cobrabox as cb

# --- List available subjects ---
info = cb.dataset_info("open_ieeg")
print(info)

# --- Load two Detroit subjects ---
ds = cb.dataset("open_ieeg", subset=["sub-Detroit001", "sub-Detroit002"])
print(ds)

# --- Inspect the first recording ---
recording = ds[0]
print(f"Subject : {recording.subjectID}")
print(f"Shape   : {dict(recording.data.sizes)}")
print(f"Fs      : {recording.sampling_rate} Hz")
print(f"Channels: {list(recording.data.coords['space'].values[:5])} ...")
