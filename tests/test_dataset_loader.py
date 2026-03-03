"""Tests for dataset loader helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobrabox.dataset_loader import load_noise_dummy, load_structured_dummy


def test_load_structured_dummy_reads_matching_files(tmp_path: Path) -> None:
    """Structured loader returns one Data object per matching non-empty file."""
    struct_dir = tmp_path / "data" / "synthetic" / "dummy" / "struct"
    struct_dir.mkdir(parents=True)

    pd.DataFrame({"ch0": [1.0, 2.0], "ch1": [3.0, 4.0]}).to_csv(
        struct_dir / "dummy_struct_VAR_chain_1.csv.xz", index=False, compression="xz"
    )
    pd.DataFrame({"ch0": [5.0], "ch1": [6.0]}).to_csv(
        struct_dir / "dummy_struct_VAR_chain_2.csv.xz", index=False, compression="xz"
    )

    out = load_structured_dummy("dummy_chain", repo_root=tmp_path)

    assert len(out) == 2
    assert out[0].data.dims == ("space", "time")
    assert list(out[0].data.coords["space"].values) == ["ch0", "ch1"]
    np.testing.assert_allclose(out[0].to_numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert out[0].data.attrs["identifier"] == "dummy_chain"


def test_load_structured_dummy_raises_when_no_files(tmp_path: Path) -> None:
    """Structured loader raises if no matching files exist."""
    (tmp_path / "data" / "synthetic" / "dummy" / "struct").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No files found"):
        load_structured_dummy("dummy_chain", repo_root=tmp_path)


def test_load_structured_dummy_raises_when_all_files_empty(tmp_path: Path) -> None:
    """Structured loader raises if all matching files are empty."""
    struct_dir = tmp_path / "data" / "synthetic" / "dummy" / "struct"
    struct_dir.mkdir(parents=True)
    pd.DataFrame(columns=["ch0", "ch1"]).to_csv(
        struct_dir / "dummy_struct_VAR_star_1.csv.xz", index=False, compression="xz"
    )

    with pytest.raises(ValueError, match="All files for 'dummy_star' are empty"):
        load_structured_dummy("dummy_star", repo_root=tmp_path)


def test_load_noise_dummy_reads_all_noise_files(tmp_path: Path) -> None:
    """Noise loader returns one Data object per non-empty noise file."""
    noise_dir = tmp_path / "data" / "synthetic" / "dummy" / "noise"
    noise_dir.mkdir(parents=True)
    pd.DataFrame({"n0": [0.1, 0.2]}).to_csv(noise_dir / "a.csv.xz", index=False, compression="xz")
    pd.DataFrame({"n0": [0.3, 0.4]}).to_csv(noise_dir / "b.csv.xz", index=False, compression="xz")

    out = load_noise_dummy(repo_root=tmp_path)

    assert len(out) == 2
    np.testing.assert_allclose(out[0].to_numpy(), np.array([[0.1, 0.2]]))
    assert out[0].data.attrs["identifier"] == "dummy_noise"


def test_load_noise_dummy_raises_when_all_files_empty(tmp_path: Path) -> None:
    """Noise loader raises if all discovered files are empty."""
    noise_dir = tmp_path / "data" / "synthetic" / "dummy" / "noise"
    noise_dir.mkdir(parents=True)
    pd.DataFrame(columns=["n0"]).to_csv(noise_dir / "a.csv.xz", index=False, compression="xz")

    with pytest.raises(ValueError, match="All files for 'dummy_noise' are empty"):
        load_noise_dummy(repo_root=tmp_path)
