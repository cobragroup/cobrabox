from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import xarray as xr

from .data import SignalData
from .dataset import Dataset


def _sidecar_json_for_csv(path: Path) -> Path:
    """Return expected JSON sidecar path for a given CSV file."""
    name = path.name
    if name.endswith(".csv.xz"):
        stem = name[: -len(".csv.xz")]
    elif name.endswith(".csv"):
        stem = name[: -len(".csv")]
    else:
        stem = path.stem
    return path.with_name(f"info_{stem}.json")


def _sampling_rate_from_info(info: dict) -> float | None:
    """Extract sampling rate from JSON info (Settings['fs'] or top-level 'fs')."""
    settings = info.get("Settings")
    fs = None
    if isinstance(settings, dict):
        fs = settings.get("fs")
    if fs is None:
        fs = info.get("fs")
    if fs is None:
        return None
    try:
        return float(fs)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid sampling rate 'fs' in metadata: {fs!r}") from e


def _load_one_csv_xz(path: Path, identifier: str) -> SignalData | None:
    """Load a single .csv.xz file into a SignalData, or return None if empty."""
    df = pd.read_csv(path, compression="xz")
    if df.empty:
        return None
    columns = list(df.columns)
    if not columns:  # pragma: no cover
        raise ValueError(f"{path.name}: expected at least one column")

    time = df.index.to_numpy(dtype=float)
    values = df.to_numpy()
    da = xr.DataArray(
        values,
        dims=["time", "space"],
        coords={"time": time, "space": columns},
        attrs={"source_file": path.name, "identifier": identifier},
    )
    extra: dict = {}
    info: dict = {}
    json_path = _sidecar_json_for_csv(path)
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                info = json.load(f)
            if isinstance(info, dict):
                extra.update(info)
        except Exception:
            pass
    sampling_rate = _sampling_rate_from_info(info) if info else None
    return SignalData.from_xarray(da, sampling_rate=sampling_rate, extra=extra or None)


def load_structured_dummy(identifier: str, repo_root: Path | None = None) -> Dataset[SignalData]:
    """Load dummy dataset parts from `data/dummy/struct`."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    struct_dir = repo_root / "data" / "synthetic" / "dummy" / "struct"
    variant = identifier.removeprefix("dummy_")
    candidates = sorted(struct_dir.glob(f"dummy_struct_VAR_{variant}_*.csv.xz"))
    if not candidates:
        raise FileNotFoundError(
            f"No files found for '{identifier}' in {struct_dir} "
            f"(expected: dummy_struct_VAR_{variant}_*.csv.xz)."
        )

    items = [item for path in candidates if (item := _load_one_csv_xz(path, identifier))]
    if not items:
        raise ValueError(f"All files for '{identifier}' are empty: {[p.name for p in candidates]}")
    return Dataset(items)


def load_noise_dummy(
    identifier: str = "dummy_noise", repo_root: Path | None = None
) -> Dataset[SignalData]:
    """Load dummy noise dataset parts from `data/dummy/noise`."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    noise_dir = repo_root / "data" / "synthetic" / "dummy" / "noise"
    candidates = sorted(noise_dir.glob("*.csv.xz"))
    if not candidates:
        raise FileNotFoundError(f"No .csv.xz files found for '{identifier}' in {noise_dir}.")

    items = [item for path in candidates if (item := _load_one_csv_xz(path, identifier))]
    if not items:
        raise ValueError(f"All files for '{identifier}' are empty: {[p.name for p in candidates]}")
    return Dataset(items)


def load_realistic_swiss(
    identifier: str = "realistic_swiss", repo_root: Path | None = None
) -> Dataset[SignalData]:
    """Load realistic Swiss VAR dataset parts from `data/synthetic/realistic`."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    realistic_dir = repo_root / "data" / "synthetic" / "realistic"
    candidates = sorted(realistic_dir.glob("fit_Swiss_VAR_ID1_*.csv.xz"))
    if not candidates:
        raise FileNotFoundError(
            f"No .csv.xz files found for '{identifier}' in {realistic_dir} "
            "matching pattern 'fit_Swiss_VAR_ID1_*.csv.xz'."
        )

    items = [item for path in candidates if (item := _load_one_csv_xz(path, identifier))]
    if not items:
        raise ValueError(f"All files for '{identifier}' are empty: {[p.name for p in candidates]}")
    return Dataset(items)


# ---------------------------------------------------------------------------
# Remote dataset loaders
# ---------------------------------------------------------------------------


def _extract_numeric_from_csv_bytes(raw: bytes) -> tuple[np.ndarray, list[str]]:
    """Parse a CSV-like payload and return numeric values plus channel names."""
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("CSV member has no numeric columns")
    values = numeric_df.to_numpy(dtype=float, copy=False)
    channels = [str(c) for c in numeric_df.columns]
    return values, channels


def _extract_numeric_from_npy_bytes(raw: bytes) -> tuple[np.ndarray, list[str]]:
    """Parse .npy/.npz payload and return a 2D array plus generated channel names."""
    obj = np.load(io.BytesIO(raw), allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if not obj.files:
            raise ValueError("NPZ archive contains no arrays")
        arr = np.asarray(obj[obj.files[0]], dtype=float)
    else:
        arr = np.asarray(obj, dtype=float)

    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")

    channels = [f"ch{i}" for i in range(arr.shape[1])]
    return arr, channels


def _extract_numeric_from_mat_bytes(raw: bytes) -> tuple[np.ndarray, list[str], float | None]:
    """Parse .mat payload and return signal data, channel names, and optional sampling rate."""
    data = scipy.io.loadmat(io.BytesIO(raw))
    sampling_rate: float | None = None
    for key in ("fs", "sampling_rate", "Fs", "srate"):
        if key in data:
            try:
                sampling_rate = float(np.asarray(data[key]).squeeze())
                break
            except TypeError:
                pass
            except ValueError:
                pass

    candidate: np.ndarray | None = None
    for key, value in data.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if not np.issubdtype(arr.dtype, np.number):
            continue
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        if arr.ndim == 2:
            candidate = np.asarray(arr, dtype=float)
            break

    if candidate is None:
        raise ValueError("MAT member does not contain a numeric 2D array")

    channels = [f"ch{i}" for i in range(candidate.shape[1])]
    return candidate, channels, sampling_rate


def _load_swiss_eeg_short(
    dataset_dir: Path, subset: Sequence[str] | None = None
) -> Dataset[SignalData]:
    """Load Swiss short EEG zip archives into SignalData objects.

    One SignalData object is produced per archive by reading the first supported
    numeric member found inside each zip file.

    Args:
        dataset_dir: Directory containing the ``.zip`` archives.
        subset: If given, only load archives whose stem (e.g. ``"ID1"``) is
            in this list. Useful when only a subset was downloaded or when all
            files are cached but only a subset is needed.
    """
    zip_paths = sorted(dataset_dir.glob("*.zip"))
    if subset is not None:
        subset_set = set(subset)
        zip_paths = [p for p in zip_paths if p.stem in subset_set]
    if not zip_paths:
        raise FileNotFoundError(f"No .zip files found for 'swiss_eeg_short' in {dataset_dir}.")

    items: list[SignalData] = []
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            members = [name for name in zf.namelist() if not name.endswith("/")]
            if not members:
                continue

            for member in members:
                suffix = Path(member).suffix.lower()
                if suffix not in {".csv", ".txt", ".tsv", ".npy", ".npz", ".mat"}:
                    continue

                raw = zf.read(member)
                sampling_rate: float | None = None
                if suffix in {".csv", ".txt", ".tsv"}:
                    values, channels = _extract_numeric_from_csv_bytes(raw)
                elif suffix in {".npy", ".npz"}:
                    values, channels = _extract_numeric_from_npy_bytes(raw)
                else:
                    values, channels, sampling_rate = _extract_numeric_from_mat_bytes(raw)

                if values.shape[0] == 0:
                    continue

                time = (
                    np.arange(values.shape[0], dtype=float) / sampling_rate
                    if sampling_rate
                    else np.arange(values.shape[0], dtype=float)
                )
                da = xr.DataArray(
                    values,
                    dims=["time", "space"],
                    coords={"time": time, "space": channels},
                    attrs={
                        "identifier": "swiss_eeg_short",
                        "source_archive": zip_path.name,
                        "source_member": member,
                    },
                )
                items.append(
                    SignalData.from_xarray(da, sampling_rate=sampling_rate, subjectID=zip_path.stem)
                )
                break
            else:
                raise ValueError(f"{zip_path.name}: no supported numeric member found in archive.")

    if not items:
        raise ValueError("All swiss_eeg_short archives were empty or unparsable.")
    return Dataset(items)
