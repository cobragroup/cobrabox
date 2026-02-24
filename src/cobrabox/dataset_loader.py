from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from .data import Dataset


def load_structured_dummy(identifier: str, repo_root: Path | None = None) -> list[Dataset]:
    """Load dummy dataset parts from `data/dummy/struct`."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    struct_dir = repo_root / "data" / "dummy" / "struct"
    variant = identifier.removeprefix("dummy_")
    candidates = sorted(struct_dir.glob(f"dummy_struct_VAR_{variant}_*.csv.xz"))
    if not candidates:
        raise FileNotFoundError(
            f"No files found for '{identifier}' in {struct_dir} "
            f"(expected: dummy_struct_VAR_{variant}_*.csv.xz)."
        )

    datasets: list[Dataset] = []
    for path in candidates:
        df = pd.read_csv(path, compression="xz")
        if df.empty:
            continue
        columns = list(df.columns)
        if not columns:
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
        datasets.append(Dataset.from_xarray(da))

    if not datasets:
        raise ValueError(f"All files for '{identifier}' are empty: {[p.name for p in candidates]}")
    return datasets


def load_noise_dummy(
    identifier: str = "dummy_noise", repo_root: Path | None = None
) -> list[Dataset]:
    """Load dummy noise dataset parts from `data/dummy/noise`."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    noise_dir = repo_root / "data" / "dummy" / "noise"
    candidates = sorted(noise_dir.glob("*.csv.xz"))
    if not candidates:
        raise FileNotFoundError(f"No .csv.xz files found for '{identifier}' in {noise_dir}.")

    datasets: list[Dataset] = []
    for path in candidates:
        df = pd.read_csv(path, compression="xz")
        if df.empty:
            continue
        columns = list(df.columns)
        if not columns:
            raise ValueError(f"{path.name}: expected at least one column")

        time = df.index.to_numpy(dtype=float)
        values = df.to_numpy()
        da = xr.DataArray(
            values,
            dims=["time", "space"],
            coords={"time": time, "space": columns},
            attrs={"source_file": path.name, "identifier": identifier},
        )
        datasets.append(Dataset.from_xarray(da))

    if not datasets:
        raise ValueError(f"All files for '{identifier}' are empty: {[p.name for p in candidates]}")
    return datasets
