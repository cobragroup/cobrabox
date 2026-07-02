# CLAUDE.md

This file provides guidance to Claude Code when working in this project.

## Project goal

Test whether **Reciprocal Connectivity (RC) computed from PDC** can reliably identify resected
intracranial electrode contacts in epilepsy patients with good surgical outcome (ILAE 1).

**Hypothesis:** Resected channels act as net sinks (epileptic drivers) and should show higher RC
than non-resected channels, detectable from interictal sleep iEEG.

## Data

- `data/Patient_Info_Zurich_Annicka.csv` — semicolon-delimited metadata for 20 patients:
  columns `Patient`, `Surgical Outcome`, `All Electrodes/Channels`,
  `Resected Electrodes/Channels`, `Excluded Electrodes/Channels`, `Pipeline Exclusions`.
- `data/zurich_ieeg/` — BrainVision iEEG recordings (`.eeg`/`.vhdr`/`.vmrk`) for sub-01…sub-20,
  multiple interictal sleep runs each; loaded via `cb.load_dataset("zurich_ieeg")`.

Only ILAE-1 patients (clean surgical outcome) are used for the main analysis.

## Cobrabox dependency

This project depends on CobraBox at `../cobrabox` (installed locally, importable as `cobrabox`).
See `../cobrabox/CLAUDE.md` for the full API reference.

Key API used here:

```python
import cobrabox as cb

# Load dataset (returns Dataset[SignalData], dims: space × time)
runs = cb.load_dataset("zurich_ieeg", subset=["sub-01"])

# PDC: fit VAR, returns (space_to × space_from × frequency) DataArray
# PDC[i,j,f] = influence from j→i; values in [0,1]
pdc = cb.feature.PartialDirectedCoherence(n_freqs=1000).apply(segment)

# RC: net directional role per channel from a pre-averaged (no frequency dim) PDC matrix
# RC[i] = in_strength[i] − out_strength[i]; positive = net sink, negative = net source
rc = cb.feature.ReciprocalConnectivity(connectivity='pdc', freq_band=None).apply(avg_pdc)
```

## Pipeline architecture (new)

The pipeline is now split into three notebooks plus a shared utilities module:

- **`utils.py`** — all shared functions and constants; imported by all notebooks.
- **`01_data_exploration.ipynb`** — tutorial walkthrough on sub-01: metadata, raw data stats,
  channel inspection, time-series and PSD plots, filtering and bipolar montage demos.
- **`02_preprocessing_and_computation.ipynb`** — batch pipeline: segments → connectivity → RC
  for all subjects, saved to disk. Modular: swap connectivity method via `CONNECTIVITY_METHOD`.
- **`03_analysis.ipynb`** — load saved RC from disk, Wilcoxon tests, paired-dot plots.

Disk layout:
- `data/segments/<subject_id>_segments.nc`
- `data/connectivity/<subject_id>_<method>_connectivity.nc`
- `data/rc/<subject_id>_<method>_rc.nc`

## Pipeline status

`pipeline_resection_zone.ipynb` (legacy monolith) has been superseded by the split notebooks
above. The old notebook remains untouched for reference.

The new notebooks cover:
1. Patient metadata loader (`load_patient_info()`)
2. Raw data loading, per-run statistics
3. Notch filter at 50 Hz, bipolar montage
4. Segment extraction (20 × 3 s per subject, random)
5. Connectivity computation — PDC with `var_order=VAR_ORDER` (default 10), band-averaged
6. RC computation per band, saved to `data/rc/`
7. Wilcoxon signed-rank tests (resected vs. non-resected per band), visualization

## Open questions

1. **Notch filter placement** — currently applied on unipolar signals before bipolar subtraction.
   Verify this is correct: filter should run on raw unipolar signals before computing bipolar
   differences (or vice versa — check whether bipolar subtraction should come first).
2. **Averaging order** — currently: average connectivity over frequency within band, then average
   over segments. The alternative (average over segments first, then over frequency) gives
   different results. Decide which is more appropriate for the hypothesis.

## Dataset directory

Cobrabox is told to look for data in `data/` via `cb.set_dataset_dir(PROJECT_DIR / "data", persist=False)`
in the second notebook cell. This is project-local and does not modify `~/.cobrabox/config.json`.
Run Jupyter from the project root (e.g. `uv run jupyter lab`) so relative paths resolve correctly.

## Environment

Dependencies in use:

```
cobrabox (local, ../cobrabox)
numpy
xarray
scipy
matplotlib
seaborn
tqdm
```

Run the notebook with whatever kernel has these packages. If setting up fresh:
```bash
cd ../cobrabox && uv sync   # installs cobrabox and its deps
# then launch jupyter from the same venv
```

## Frequency bands

```python
BANDS = {
    "delta":       (0.5,  4.0),
    "theta":       (4.0,  8.0),
    "alpha":       (8.0, 13.0),
    "beta":       (13.0, 30.0),
    "low_gamma":  (30.0, 80.0),
    "high_gamma": (80.0,150.0),
    "ripple":    (150.0,250.0),
    "fast_ripples": (250.0, 500.0),
}
```

## Pipeline constants

```python
SEGMENT_DURATION = 3.0   # seconds
N_SEGMENTS       = 20    # random segments per subject/run
VAR_ORDER        = 10    # VAR model order for PDC (None = auto via AIC)
```
