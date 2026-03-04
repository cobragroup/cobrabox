"""Tests for dataset loader helpers."""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cobrabox.dataset_loader import (
    _sampling_rate_from_info,
    _sidecar_json_for_csv,
    load_noise_dummy,
    load_realistic_swiss,
    load_structured_dummy,
)


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
    a_csv = noise_dir / "a.csv.xz"
    b_csv = noise_dir / "b.csv.xz"
    pd.DataFrame({"n0": [0.1, 0.2]}).to_csv(a_csv, index=False, compression="xz")
    pd.DataFrame({"n0": [0.3, 0.4]}).to_csv(b_csv, index=False, compression="xz")

    # Sidecar JSON with settings metadata
    a_json = noise_dir / "info_a.json.xz"
    with lzma.open(a_json, "wt", encoding="utf-8") as f:
        json.dump({"Settings": {"Seizure start (sec)": 8, "SOZ": "[1 0 1 0]", "fs": 256}}, f)

    out = load_noise_dummy(repo_root=tmp_path)

    assert len(out) == 2
    np.testing.assert_allclose(out[0].to_numpy(), np.array([[0.1, 0.2]]))
    assert out[0].data.attrs["identifier"] == "dummy_noise"
    assert out[0].sampling_rate == 256.0
    # Extra metadata propagated from JSON, including SOZ array string
    assert "Settings" in out[0].extra
    settings = out[0].extra["Settings"]
    assert settings["Seizure start (sec)"] == 8
    assert settings["SOZ"] == "[1 0 1 0]"


def test_load_noise_dummy_raises_when_all_files_empty(tmp_path: Path) -> None:
    """Noise loader raises if all discovered files are empty."""
    noise_dir = tmp_path / "data" / "synthetic" / "dummy" / "noise"
    noise_dir.mkdir(parents=True)
    pd.DataFrame(columns=["n0"]).to_csv(noise_dir / "a.csv.xz", index=False, compression="xz")

    with pytest.raises(ValueError, match="All files for 'dummy_noise' are empty"):
        load_noise_dummy(repo_root=tmp_path)


def test_load_noise_dummy_raises_when_no_csv_xz_files(tmp_path: Path) -> None:
    """Noise loader raises FileNotFoundError when the noise dir has no .csv.xz files."""
    noise_dir = tmp_path / "data" / "synthetic" / "dummy" / "noise"
    noise_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"No \.csv\.xz files found"):
        load_noise_dummy(repo_root=tmp_path)


def test_load_realistic_swiss_reads_all_matching_files(tmp_path: Path) -> None:
    """Realistic loader returns one Data object per non-empty Swiss VAR file."""
    realistic_dir = tmp_path / "data" / "synthetic" / "realistic"
    realistic_dir.mkdir(parents=True)
    csv1 = realistic_dir / "fit_Swiss_VAR_ID1_sz13_simulated_data_1.csv.xz"
    csv2 = realistic_dir / "fit_Swiss_VAR_ID1_sz7_simulated_data_3.csv.xz"
    pd.DataFrame({"ch0": [1.0, 2.0]}).to_csv(csv1, index=False, compression="xz")
    pd.DataFrame({"ch0": [3.0, 4.0]}).to_csv(csv2, index=False, compression="xz")

    json1 = realistic_dir / "info_fit_Swiss_VAR_ID1_sz13_simulated_data_1.json.xz"
    with lzma.open(json1, "wt", encoding="utf-8") as f:
        json.dump({"Settings": {"Seizure start (sec)": 5, "SOZ": "[0 1 0]", "fs": 512}}, f)

    out = load_realistic_swiss(repo_root=tmp_path)

    assert len(out) == 2
    np.testing.assert_allclose(out[0].to_numpy(), np.array([[1.0, 2.0]]))
    assert out[0].data.attrs["identifier"] == "realistic_swiss"
    assert out[0].sampling_rate == 512.0
    assert "Settings" in out[0].extra
    settings = out[0].extra["Settings"]
    assert settings["Seizure start (sec)"] == 5
    assert settings["SOZ"] == "[0 1 0]"


def test_load_realistic_swiss_raises_when_no_matching_files(tmp_path: Path) -> None:
    """Realistic loader raises when there are no matching Swiss VAR files."""
    realistic_dir = tmp_path / "data" / "synthetic" / "realistic"
    realistic_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"No \.csv\.xz files found"):
        load_realistic_swiss(repo_root=tmp_path)


def test_load_realistic_swiss_raises_when_all_files_empty(tmp_path: Path) -> None:
    """Realistic loader raises when all matching files are empty."""
    realistic_dir = tmp_path / "data" / "synthetic" / "realistic"
    realistic_dir.mkdir(parents=True)
    pd.DataFrame(columns=["ch0"]).to_csv(
        realistic_dir / "fit_Swiss_VAR_ID1_sz13_simulated_data_1.csv.xz",
        index=False,
        compression="xz",
    )

    with pytest.raises(ValueError, match="All files for 'realistic_swiss' are empty"):
        load_realistic_swiss(repo_root=tmp_path)


def test_sidecar_json_for_csv_non_csv_xz_extension(tmp_path: Path) -> None:
    """_sidecar_json_for_csv falls back to path.stem for non-.csv.xz files."""
    p = tmp_path / "myfile.csv"
    result = _sidecar_json_for_csv(p)
    assert result.name == "info_myfile.json.xz"


def test_sampling_rate_from_info_raises_for_non_numeric_fs() -> None:
    """_sampling_rate_from_info raises ValueError when fs is not numeric."""
    with pytest.raises(ValueError, match="Invalid sampling rate"):
        _sampling_rate_from_info({"fs": "not-a-number"})


def test_sampling_rate_from_info_raises_for_unconvertible_type() -> None:
    """_sampling_rate_from_info raises ValueError when fs cannot be float-cast."""
    with pytest.raises(ValueError, match="Invalid sampling rate"):
        _sampling_rate_from_info({"fs": [1, 2]})


def test_load_structured_dummy_ignores_malformed_json_sidecar(tmp_path: Path) -> None:
    """Malformed JSON sidecar is silently ignored; data still loads."""
    struct_dir = tmp_path / "data" / "synthetic" / "dummy" / "struct"
    struct_dir.mkdir(parents=True)
    pd.DataFrame({"ch0": [1.0, 2.0]}).to_csv(
        struct_dir / "dummy_struct_VAR_chain_1.csv.xz", index=False, compression="xz"
    )
    json_path = struct_dir / "info_dummy_struct_VAR_chain_1.json.xz"
    with lzma.open(json_path, "wt", encoding="utf-8") as f:
        f.write("not valid json {{{")

    out = load_structured_dummy("dummy_chain", repo_root=tmp_path)
    assert len(out) == 1
    assert out[0].sampling_rate is None


def test_load_noise_dummy_ignores_malformed_json_sidecar(tmp_path: Path) -> None:
    """Malformed JSON sidecar in noise loader is silently ignored."""
    noise_dir = tmp_path / "data" / "synthetic" / "dummy" / "noise"
    noise_dir.mkdir(parents=True)
    pd.DataFrame({"n0": [1.0, 2.0]}).to_csv(noise_dir / "a.csv.xz", index=False, compression="xz")
    json_path = noise_dir / "info_a.json.xz"
    with lzma.open(json_path, "wt", encoding="utf-8") as f:
        f.write("{{invalid")

    out = load_noise_dummy(repo_root=tmp_path)
    assert len(out) == 1
    assert out[0].sampling_rate is None


def test_load_realistic_swiss_ignores_malformed_json_sidecar(tmp_path: Path) -> None:
    """Malformed JSON sidecar in realistic loader is silently ignored."""
    realistic_dir = tmp_path / "data" / "synthetic" / "realistic"
    realistic_dir.mkdir(parents=True)
    csv_path = realistic_dir / "fit_Swiss_VAR_ID1_sz1_simulated_data_1.csv.xz"
    pd.DataFrame({"ch0": [1.0, 2.0]}).to_csv(csv_path, index=False, compression="xz")
    json_path = realistic_dir / "info_fit_Swiss_VAR_ID1_sz1_simulated_data_1.json.xz"
    with lzma.open(json_path, "wt", encoding="utf-8") as f:
        f.write("{{invalid")

    out = load_realistic_swiss(repo_root=tmp_path)
    assert len(out) == 1
    assert out[0].sampling_rate is None


def test_load_structured_dummy_default_repo_root() -> None:
    """load_structured_dummy infers repo_root from package path and loads real data."""
    from cobrabox.data import Data

    out = load_structured_dummy("dummy_chain")
    assert len(out) > 0
    assert all(isinstance(d, Data) for d in out)


def test_load_noise_dummy_default_repo_root() -> None:
    """load_noise_dummy infers repo_root from package path and loads real data."""
    from cobrabox.data import Data

    try:
        out = load_noise_dummy()
    except FileNotFoundError:
        pytest.skip("Real dummy noise dataset files not available (likely LFS not fetched).")
    assert len(out) > 0
    assert all(isinstance(d, Data) for d in out)


def test_load_realistic_swiss_default_repo_root() -> None:
    """load_realistic_swiss infers repo_root and loads real data."""
    from cobrabox.data import Data

    try:
        out = load_realistic_swiss()
    except FileNotFoundError:
        pytest.skip("Real realistic_swiss dataset files not available (likely LFS not fetched).")
    assert len(out) > 0
    assert all(isinstance(d, Data) for d in out)
