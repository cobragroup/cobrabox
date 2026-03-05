from __future__ import annotations

import json
import lzma
from pathlib import Path

import pandas as pd
import xarray as xr

from .data import SignalData
from .dataset import Dataset


def _sidecar_json_for_csv(path: Path) -> Path:
    """Return expected JSON sidecar path for a given CSV.xz file."""
    name = path.name
    if name.endswith(".csv.xz"):
        stem = name[: -len(".csv.xz")]
    else:
        stem = path.stem
    return path.with_name(f"info_{stem}.json.xz")


def _sampling_rate_from_info(info: dict) -> float | None:
    """Extract sampling rate from JSON info (Settings['fs'] or top-level 'fs')."""
    settings = info.get("Settings")
    if isinstance(settings, dict):
        fs = settings.get("fs")
    else:
        fs = info.get("fs")
    if fs is None:
        return None
    try:
        return float(fs)
    except TypeError as e:
        raise ValueError(f"Invalid sampling rate 'fs' in metadata: {fs!r}") from e
    except ValueError as e:
        raise ValueError(f"Invalid sampling rate 'fs' in metadata: {fs!r}") from e


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

    items: list[SignalData] = []
    for path in candidates:
        df = pd.read_csv(path, compression="xz")
        if df.empty:
            continue
        columns = list(df.columns)
        if not columns:  # pragma: no cover
            raise ValueError(f"{path.name}: expected at least one column")

        # For these dummy datasets, time is implicit row index.
        time = df.index.to_numpy(dtype=float)
        space = columns
        values = df.to_numpy()
        da = xr.DataArray(
            values,
            dims=["time", "space"],
            coords={"time": time, "space": space},
            attrs={"source_file": path.name, "identifier": identifier},
        )
        # Attach optional metadata from matching JSON sidecar
        extra = {}
        info: dict = {}
        json_path = _sidecar_json_for_csv(path)
        if json_path.exists():
            try:
                with lzma.open(json_path, "rt", encoding="utf-8") as f:
                    info = json.load(f)
                if isinstance(info, dict):
                    extra.update(info)
            except Exception:
                # Ignore JSON parsing issues and continue without extra metadata
                pass
        sampling_rate = _sampling_rate_from_info(info) if info else None
        items.append(SignalData.from_xarray(da, sampling_rate=sampling_rate, extra=extra or None))

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

    items: list[SignalData] = []
    for path in candidates:
        df = pd.read_csv(path, compression="xz")
        if df.empty:
            continue
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
        extra = {}
        info: dict = {}
        json_path = _sidecar_json_for_csv(path)
        if json_path.exists():
            try:
                with lzma.open(json_path, "rt", encoding="utf-8") as f:
                    info = json.load(f)
                if isinstance(info, dict):
                    extra.update(info)
            except Exception:
                pass
        sampling_rate = _sampling_rate_from_info(info) if info else None
        items.append(SignalData.from_xarray(da, sampling_rate=sampling_rate, extra=extra or None))

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

    items: list[SignalData] = []
    for path in candidates:
        df = pd.read_csv(path, compression="xz")
        if df.empty:
            continue
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
        extra = {}
        info: dict = {}
        json_path = _sidecar_json_for_csv(path)
        if json_path.exists():
            try:
                with lzma.open(json_path, "rt", encoding="utf-8") as f:
                    info = json.load(f)
                if isinstance(info, dict):
                    extra.update(info)
            except Exception:
                pass
        sampling_rate = _sampling_rate_from_info(info) if info else None
        items.append(SignalData.from_xarray(da, sampling_rate=sampling_rate, extra=extra or None))

    if not items:
        raise ValueError(f"All files for '{identifier}' are empty: {[p.name for p in candidates]}")
    return Dataset(items)
