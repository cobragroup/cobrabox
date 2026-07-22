"""Tests for the 'zurich_ieeg_clean' dataset: per-subject columns-C+D channel removal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import cobrabox as cb
from cobrabox.dataset_loader import _clean_zurich_dataset, _load_zurich_ieeg_clean
from cobrabox.downloader import _ZURICH_IEEG_CLEAN_REMOVE, REMOTE_DATASETS, get_remote_dataset_spec


def _make_item(
    subject: str, channels: list[str], n_time: int = 40, sr: float = 2000.0
) -> cb.SignalData:
    arr = np.random.default_rng(0).standard_normal((n_time, len(channels)))
    da = xr.DataArray(
        arr, dims=["time", "space"], coords={"time": np.arange(n_time) / sr, "space": channels}
    )
    return cb.SignalData.from_xarray(da, sampling_rate=sr, subjectID=subject)


# ---------------------------------------------------------------------------
# Core cleaning logic (pure, no download / MNE)
# ---------------------------------------------------------------------------


def test_clean_removes_flagged_channels() -> None:
    """Channels in columns C+D are dropped; others kept; time axis untouched."""
    keep = ["IAR1", "IAR2", "IAR3"]
    remove = _ZURICH_IEEG_CLEAN_REMOVE["sub-15"]  # this subject's full flagged set
    item = _make_item("sub-15", [*keep, *remove])

    out = _clean_zurich_dataset(cb.Dataset([item]))[0]

    assert list(out.data.coords["space"].values) == keep
    assert out.data.sizes["time"] == item.data.sizes["time"]
    assert out.subjectID == "sub-15"
    assert out.sampling_rate == pytest.approx(2000.0)
    assert out.history[-1] == "zurich_ieeg_clean"
    assert out.extra["removed_channels"] == remove


def test_clean_passthrough_for_subject_without_removals() -> None:
    """A subject absent from the removal dict is returned unchanged."""
    item = _make_item("sub-10", ["A1", "A2", "A3"])
    out = _clean_zurich_dataset(cb.Dataset([item]))[0]

    assert out is item  # untouched, same object
    assert "zurich_ieeg_clean" not in out.history


def test_clean_does_not_mutate_input() -> None:
    """Cleaning is immutable — the original item keeps all its channels."""
    channels = ["IAR1", *_ZURICH_IEEG_CLEAN_REMOVE["sub-15"]]
    item = _make_item("sub-15", channels)
    _clean_zurich_dataset(cb.Dataset([item]))

    assert list(item.data.coords["space"].values) == channels
    assert "zurich_ieeg_clean" not in item.history


def test_clean_warns_when_flagged_channel_absent() -> None:
    """A flagged channel not present in the data warns rather than fails."""
    # sub-18's only flagged channel is PML2; build data without it.
    item = _make_item("sub-18", ["X1", "X2"])
    with pytest.warns(UserWarning, match="PML2"):
        out = _clean_zurich_dataset(cb.Dataset([item]))[0]

    assert list(out.data.coords["space"].values) == ["X1", "X2"]
    assert out.extra["removed_channels"] == []


def test_clean_multiple_subjects_independent() -> None:
    """Each subject uses its own removal list; a mixed dataset is handled per-item."""
    a = _make_item("sub-18", ["PML2", "PML3"])  # removes PML2
    b = _make_item("sub-10", ["Z1", "Z2"])  # no removals
    out = _clean_zurich_dataset(cb.Dataset([a, b]))

    assert list(out[0].data.coords["space"].values) == ["PML3"]
    assert list(out[1].data.coords["space"].values) == ["Z1", "Z2"]


# ---------------------------------------------------------------------------
# Removal dict integrity
# ---------------------------------------------------------------------------


def test_removal_dict_wellformed() -> None:
    """Keys are sub-01..sub-20; every list is non-empty with no duplicates."""
    for sub, chans in _ZURICH_IEEG_CLEAN_REMOVE.items():
        assert sub.startswith("sub-")
        assert 1 <= int(sub[4:]) <= 20
        assert chans, f"{sub} has an empty removal list"
        assert len(chans) == len(set(chans)), f"{sub} has duplicate channels"


# ---------------------------------------------------------------------------
# Dataset registration: shares the raw download
# ---------------------------------------------------------------------------


def test_clean_spec_shares_raw_download() -> None:
    """zurich_ieeg_clean reuses zurich_ieeg's files and directory; only loader differs."""
    raw = REMOTE_DATASETS["zurich_ieeg"]
    clean = get_remote_dataset_spec("zurich_ieeg_clean")

    assert clean is not None
    assert clean.identifier == "zurich_ieeg_clean"
    assert clean.local_rel_dir == raw.local_rel_dir  # shared download dir
    assert clean.files == raw.files
    assert clean.known_subset_keys == raw.known_subset_keys
    assert clean.loader is _load_zurich_ieeg_clean
    assert clean.loader is not raw.loader


# ---------------------------------------------------------------------------
# End-to-end through the loader (MNE mocked)
# ---------------------------------------------------------------------------


class _MockRaw:
    def __init__(self, ch_names: list[str], n_samples: int = 100, sfreq: float = 2000.0) -> None:
        self._data = np.ones((len(ch_names), n_samples))
        self.ch_names = ch_names
        self.info = {"sfreq": sfreq}

    def get_data(self) -> np.ndarray:
        return self._data


def test_load_zurich_ieeg_clean_drops_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registered loader downloads-then-cleans: flagged channels are gone."""
    import mne

    dataset_dir = tmp_path / "zurich_ieeg"
    dataset_dir.mkdir()
    (dataset_dir / "sub-01_ses-interictalsleep_run-01_ieeg.vhdr").write_bytes(b"fake")

    keep = ["IAR1", "IAR2"]
    remove = _ZURICH_IEEG_CLEAN_REMOVE["sub-01"]  # this subject's full flagged set
    mock = _MockRaw([*keep, *remove])
    monkeypatch.setattr(mne.io, "read_raw_brainvision", lambda path, **kwargs: mock)

    out = _load_zurich_ieeg_clean(dataset_dir)

    assert len(out) == 1
    assert list(out[0].data.coords["space"].values) == keep
    assert set(out[0].extra["removed_channels"]) == set(remove)
    assert out[0].history[-1] == "zurich_ieeg_clean"
