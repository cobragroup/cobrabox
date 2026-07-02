"""Shared utilities for the RC-from-connectivity pipeline."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import cobrabox as cb
import numpy as np
import xarray as xr
from scipy.signal import filtfilt, iirnotch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
SEGMENTS_DIR = DATA_DIR / "segments"
CONNECTIVITY_DIR = DATA_DIR / "connectivity"
RC_DIR = DATA_DIR / "rc"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEGMENT_DURATION = 3.0   # seconds
N_SEGMENTS = 20          # random segments per subject
VAR_ORDER = 10           # VAR model order for PDC (set to None for auto via AIC)

BANDS = {
    "delta":        (2.0,   4.0),
    "theta":        (5.0,   7.0),
    "alpha":        (8.0,  12.0),
    "beta":        (15.0,  29.0),
    "low_gamma":   (30.0,  59.0),
    "high_gamma":  (60.0,  79.0),
    "ripples":     (80.0, 249.0),
    "fast_ripples": (250.0, 500.0),
}

# ---------------------------------------------------------------------------
# Connectivity method registry
#
# Add new methods here. Each entry is a zero-argument callable that returns
# a cobrabox feature object. The feature must accept a SignalData segment and
# return a DataArray with dims (space_to, space_from, frequency).
# ---------------------------------------------------------------------------

def _make_pdc():
    return cb.feature.PartialDirectedCoherence(var_order=VAR_ORDER, n_freqs=1000)


CONNECTIVITY_METHODS: dict[str, callable] = {
    "pdc": _make_pdc,
}

# ---------------------------------------------------------------------------
# Patient metadata
# ---------------------------------------------------------------------------

def load_patient_info(path: Path | None = None) -> dict:
    """Parse Patient_Info CSV and return a dict keyed by subject ID.

    Returns:
        {
            "sub-01": {
                "outcome": "ILAE1",
                "resected": ["AHR1-AHR2", ...],
                "pipeline_exclusions": ["AHL4-AHL5", ...],
            },
            ...
        }
    """
    if path is None:
        path = DATA_DIR / "Patient_Info_Zurich_Annicka.csv"

    def _parse_channel_list(raw: str) -> list[str]:
        if not raw or raw.strip() in ("None", "No channels were removed", ""):
            return []
        return [
            line.strip().strip("'")
            for line in raw.split("\n")
            if line.strip().strip("'") and "-" in line
        ]

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        rows = list(reader)

    result = {}
    for row in rows[1:]:
        if len(row) < 6:
            continue
        m = re.match(r"Patient (\d+)", row[0].strip())
        if not m:
            continue
        sub_id = f"sub-{int(m.group(1)):02d}"
        result[sub_id] = {
            "outcome": row[1].strip(),
            "resected": _parse_channel_list(row[3]),
            "pipeline_exclusions": _parse_channel_list(row[5]),
        }
    return result

# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------

def notch_filter(da: xr.DataArray, freq: float = 50.0, q: float = 30.0, fs: float = None) -> xr.DataArray:
    """Apply a notch (IIR) filter along the time axis of a DataArray."""
    b, a = iirnotch(freq, q, fs)
    return xr.apply_ufunc(lambda x: filtfilt(b, a, x, axis=0), da)


def apply_bipolar_montage(da: xr.DataArray, valid_pairs: list[str]) -> xr.DataArray:
    """Subtract adjacent unipolar channels to form bipolar derivations."""
    return xr.concat(
        [
            (da.sel(space=a) - da.sel(space=b)).assign_coords(space=f"{a}-{b}")
            for a, b in (pair.split("-") for pair in valid_pairs)
        ],
        dim="space",
    )

# ---------------------------------------------------------------------------
# Channel validation
# ---------------------------------------------------------------------------

def get_valid_pairs(item, subject_id: str, patient_info: dict) -> list[str]:
    """Return bipolar pairs that are present, not excluded, not pipeline-excluded."""
    available = set(item.data.coords["space"].values)
    excluded = set(item.extra.get("excluded_channels", []))
    pipeline_exclusions = set(patient_info.get(subject_id, {}).get("pipeline_exclusions", []))

    monopolar = list(dict.fromkeys(
        ch for pair in item.extra["all_channels"] for ch in pair.split("-")
    ))
    missing = [ch for ch in monopolar if ch not in available]
    if missing:
        print(f"  [{subject_id}] Missing monopolar channels: {missing}")

    valid_pairs = [
        pair for pair in item.extra["all_channels"]
        if pair not in pipeline_exclusions
        and all(ch in available and ch not in excluded for ch in pair.split("-"))
    ]

    resected = set(patient_info.get(subject_id, {}).get("resected", []))
    missing_resected = resected - set(valid_pairs)
    if missing_resected:
        print(f"  [{subject_id}] WARNING: resected channels not in valid pairs: {missing_resected}")

    return valid_pairs

# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def build_segment_pool(ds, rng: np.random.Generator | None = None) -> list[tuple[int, int]]:
    """Return N_SEGMENTS randomly chosen (run_idx, start_sample) pairs from ds."""
    if rng is None:
        rng = np.random.default_rng(42)
    pool = []
    for run_idx, item in enumerate(ds):
        fs = item.sampling_rate
        n_samples = item.data.sizes["time"]
        seg_len = int(fs * SEGMENT_DURATION)
        n_possible = n_samples // seg_len
        for seg_idx in range(n_possible):
            pool.append((run_idx, seg_idx * seg_len))
    chosen = rng.choice(len(pool), size=N_SEGMENTS, replace=False)
    return [pool[i] for i in chosen]


def extract_segments(ds, subject_id: str, patient_info: dict,
                     rng: np.random.Generator | None = None) -> cb.Dataset:
    """Extract N_SEGMENTS × SEGMENT_DURATION-s bipolar segments from raw runs."""
    segments_list = build_segment_pool(ds, rng=rng)
    valid_pairs = get_valid_pairs(ds[0], subject_id, patient_info)

    segments_data = []
    for run_idx, start_sample in segments_list:
        item = ds[run_idx]
        seg_len = int(item.sampling_rate * SEGMENT_DURATION)
        seg_da = item.data.isel(time=slice(start_sample, start_sample + seg_len))
        seg_da = notch_filter(seg_da, fs=item.sampling_rate)
        seg_da = apply_bipolar_montage(seg_da, valid_pairs)
        segments_data.append(cb.SignalData.from_xarray(
            seg_da,
            sampling_rate=item.sampling_rate,
            extra={"run_idx": run_idx, "start_sample": start_sample},
        ))
    return cb.Dataset(segments_data)

# ---------------------------------------------------------------------------
# Segment I/O
# ---------------------------------------------------------------------------

def save_segments(segments_dataset: cb.Dataset, subject_id: str,
                  segments_dir: Path = SEGMENTS_DIR) -> None:
    segments_dir.mkdir(parents=True, exist_ok=True)
    n_time = segments_dataset[0].data.sizes["time"]
    common_time = np.arange(n_time)
    da = xr.concat(
        [seg.data.assign_coords(time=common_time) for seg in segments_dataset],
        dim="segment",
    )
    da = da.assign_coords(segment=np.arange(len(segments_dataset)))
    da.attrs["sampling_rate"] = segments_dataset[0].sampling_rate
    da.to_netcdf(segments_dir / f"{subject_id}_segments.nc")


def load_segments(subject_id: str, segments_dir: Path = SEGMENTS_DIR) -> cb.Dataset:
    da = xr.load_dataarray(segments_dir / f"{subject_id}_segments.nc")
    sr = da.attrs["sampling_rate"]
    return cb.Dataset([
        cb.SignalData.from_xarray(
            da.isel(segment=i).drop_vars("segment"),
            sampling_rate=sr,
            subjectID=subject_id,
        )
        for i in range(da.sizes["segment"])
    ])

# ---------------------------------------------------------------------------
# Connectivity computation (modular)
# ---------------------------------------------------------------------------

def compute_band_connectivity(segments_dataset: cb.Dataset,
                              method: str = "pdc") -> dict[str, xr.DataArray]:
    """Compute band-averaged connectivity matrices from a set of segments.

    Args:
        segments_dataset: Dataset of SignalData segments.
        method: Key into CONNECTIVITY_METHODS (e.g. "pdc").

    Returns:
        Dict mapping band name → (space_to × space_from) DataArray.
        Averaging order: per-segment connectivity averaged over segments, then
        averaged over the frequency axis within each band.
    """
    if method not in CONNECTIVITY_METHODS:
        raise ValueError(f"Unknown connectivity method {method!r}. "
                         f"Available: {list(CONNECTIVITY_METHODS)}")
    feat = CONNECTIVITY_METHODS[method]()
    results = [feat.apply(item) for item in
               tqdm(segments_dataset, desc=f"  Computing {method.upper()}", leave=False)]
    conn_mean = xr.concat([r.data for r in results], dim="segment").mean(dim="segment")
    return {
        band_name: conn_mean.sel(frequency=slice(low, high)).mean(dim="frequency")
        for band_name, (low, high) in BANDS.items()
    }

# ---------------------------------------------------------------------------
# Connectivity I/O
# ---------------------------------------------------------------------------

def save_connectivity(avg_conn: dict[str, xr.DataArray], subject_id: str,
                      method: str = "pdc",
                      connectivity_dir: Path = CONNECTIVITY_DIR) -> None:
    connectivity_dir.mkdir(parents=True, exist_ok=True)
    da = xr.concat(
        [mat.expand_dims("band").assign_coords(band=[band_name])
         for band_name, mat in avg_conn.items()],
        dim="band",
    )
    da.to_netcdf(connectivity_dir / f"{subject_id}_{method}_connectivity.nc")


def load_connectivity(subject_id: str, method: str = "pdc",
                      connectivity_dir: Path = CONNECTIVITY_DIR) -> dict[str, xr.DataArray]:
    da = xr.load_dataarray(connectivity_dir / f"{subject_id}_{method}_connectivity.nc")
    return {str(band): da.sel(band=band).drop_vars("band")
            for band in da.coords["band"].values}

# ---------------------------------------------------------------------------
# RC computation
# ---------------------------------------------------------------------------

def compute_rc(avg_conn: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Compute Reciprocal Connectivity per band from pre-averaged connectivity matrices."""
    rc_feat = cb.feature.ReciprocalConnectivity(connectivity="pdc", freq_band=None)
    return {
        band_name: rc_feat.apply(cb.Data.from_xarray(mat)).data
        for band_name, mat in avg_conn.items()
    }

# ---------------------------------------------------------------------------
# RC I/O
# ---------------------------------------------------------------------------

def save_rc(rc_values: dict[str, xr.DataArray], subject_id: str,
            method: str = "pdc", rc_dir: Path = RC_DIR) -> None:
    rc_dir.mkdir(parents=True, exist_ok=True)
    da = xr.concat(
        [v.expand_dims("band").assign_coords(band=[band_name])
         for band_name, v in rc_values.items()],
        dim="band",
    )
    da.to_netcdf(rc_dir / f"{subject_id}_{method}_rc.nc")


def load_rc(subject_id: str, method: str = "pdc",
            rc_dir: Path = RC_DIR) -> dict[str, xr.DataArray]:
    da = xr.load_dataarray(rc_dir / f"{subject_id}_{method}_rc.nc")
    return {str(band): da.sel(band=band).drop_vars("band")
            for band in da.coords["band"].values}
