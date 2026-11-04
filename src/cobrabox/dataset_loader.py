from __future__ import annotations

import io
import json
import warnings
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import xarray as xr

from .data import SignalData
from .dataset import Dataset

# ---------------------------------------------------------------------------
# swiss_demo — simulated iEEG fitted to the SWEC-ETHZ short-term recordings
# ---------------------------------------------------------------------------

# Sidecar fields copied into ``Data.extra``. The rest of the sidecar describes
# how the data was generated (Source, Generator, Quality_checks, ...) and is
# deliberately left out.
_SWISS_DEMO_EXTRA_FIELDS: tuple[str, ...] = ("seizureID", "Project", "Description")
_SWISS_DEMO_STATE_ORDER: tuple[str, ...] = ("preictal", "ictal", "postictal")


def _swiss_demo_sort_key(csv_path: Path) -> tuple[int, str, int, str]:
    """Order recordings by subject number, seizure, then preictal/ictal/postictal."""
    # Filename layout: fit_Swiss_VAR_<subject>_<seizure>_<state>.csv.xz
    _, _, _, subject, seizure, state = csv_path.name.removesuffix(".csv.xz").split("_", 5)
    subject_num = int(subject[2:]) if subject[2:].isdigit() else -1
    state_rank = _SWISS_DEMO_STATE_ORDER.index(state) if state in _SWISS_DEMO_STATE_ORDER else 99
    return subject_num, seizure, state_rank, csv_path.name


def _swiss_demo_sidecar(csv_path: Path) -> Path:
    """Return the ``info_*.json`` sidecar for a ``fit_*.csv.xz`` recording."""
    return csv_path.with_name(f"info_{csv_path.name.removesuffix('.csv.xz')}.json")


def _load_swiss_demo_recording(csv_path: Path) -> SignalData:
    """Load one swiss_demo recording and its sidecar metadata into a SignalData."""
    json_path = _swiss_demo_sidecar(csv_path)
    if not json_path.exists():
        raise FileNotFoundError(f"swiss_demo: missing metadata sidecar {json_path.name}.")
    info = json.loads(json_path.read_text(encoding="utf-8"))

    values = pd.read_csv(csv_path, compression="xz").to_numpy(dtype=float)
    fs = float(info["fs"])
    n_samples, n_channels = values.shape

    expected = {
        "Num_Samples": (info["Num_Samples"], n_samples),
        "Num_Electrodes": (info["Num_Electrodes"], n_channels),
        "Duration (sec)": (info["Duration (sec)"], n_samples / fs),
    }
    for field, (declared, actual) in expected.items():
        if not np.isclose(declared, actual):
            raise ValueError(
                f"swiss_demo: {csv_path.name} has {field}={actual:g} but its sidecar "
                f"declares {declared:g}."
            )

    da = xr.DataArray(
        values,
        dims=["time", "space"],
        coords={
            "time": np.arange(n_samples, dtype=float) / fs,
            "space": [f"ch{i}" for i in range(n_channels)],
        },
        attrs={"identifier": "swiss_demo", "source_file": csv_path.name},
    )
    return SignalData.from_xarray(
        da,
        sampling_rate=fs,
        subjectID=info["subjectID"],
        condition=info["condition"],
        extra={k: info[k] for k in _SWISS_DEMO_EXTRA_FIELDS if k in info},
    )


def _load_swiss_demo(dataset_dir: Path, subset: Sequence[str] | None = None) -> Dataset[SignalData]:
    """Load the swiss_demo dataset: simulated iEEG in three peri-ictal states.

    Each recording is a ``fit_Swiss_VAR_<subject>_<seizure>_<state>.csv.xz``
    file (samples x channels) with an ``info_*.json`` sidecar. One
    ``SignalData`` is produced per recording, with ``subjectID`` and
    ``condition`` (``"preictal"``, ``"ictal"`` or ``"postictal"``) set from
    the sidecar and ``seizureID``, ``Project`` and ``Description`` in
    ``extra``. The time axis is in seconds.

    Args:
        dataset_dir: Directory containing the recordings and their sidecars.
        subset: If given, only load recordings whose subject ID (e.g.
            ``"ID1"``) is in this list. Filtering happens on the filename, so
            unselected recordings are never decompressed.

    Raises:
        FileNotFoundError: If no matching recordings are in ``dataset_dir``,
            or a recording has no sidecar.
        ValueError: If a recording's shape or duration disagrees with its
            sidecar.
    """
    csv_paths = sorted(dataset_dir.glob("fit_Swiss_VAR_*.csv.xz"), key=_swiss_demo_sort_key)
    if subset is not None:
        subjects = set(subset)
        # Filename layout: fit_Swiss_VAR_<subject>_<seizure>_<state>.csv.xz
        csv_paths = [p for p in csv_paths if p.name.split("_")[3] in subjects]
    if not csv_paths:
        selected = f" for subjects {sorted(subset)}" if subset is not None else ""
        raise FileNotFoundError(
            f"No swiss_demo recordings found{selected} in {dataset_dir} "
            "(expected fit_Swiss_VAR_*.csv.xz with info_*.json sidecars). The dataset "
            "is not hosted yet: place the files there, or point cobrabox at the folder "
            "that contains swiss_demo/ with cb.set_dataset_dir() or COBRABOX_DATA_DIR."
        )
    return Dataset([_load_swiss_demo_recording(p) for p in csv_paths])


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


def _load_swez_sampling_rate(dataset_dir: Path, subject_id: str) -> float | None:
    """Try to read sampling rate from an IDxx_info.mat sidecar file."""
    info_path = dataset_dir / f"{subject_id}_info.mat"
    if not info_path.exists():
        return None
    try:
        info = scipy.io.loadmat(str(info_path))
        for key in ("fs", "Fs", "sampling_rate", "srate"):
            if key in info:
                return float(np.asarray(info[key]).squeeze())
    except Exception:
        pass
    return None


def _load_swiss_eeg_long(
    dataset_dir: Path, subset: Sequence[str] | None = None
) -> Dataset[SignalData]:
    """Load SWEZ long-term iEEG .mat files into SignalData objects.

    Each .mat file represents one hour of recording and produces one
    ``SignalData`` object. The ``EEG`` variable (channels x time) is read
    using ``scipy.io.loadmat`` with selective variable loading to avoid
    pulling the full file into memory unnecessarily. Sampling rate is
    looked up from an ``IDxx_info.mat`` sidecar when present.

    Args:
        dataset_dir: Directory containing the downloaded ``.mat`` files.
        subset: If given, restrict which files are loaded.  Two modes are
            detected automatically:

            - *Subject keys* (e.g. ``["ID01"]``, no ``"_"`` in any element):
              load all files whose subject ID matches.
            - *File stems* (e.g. ``["ID01_1h", "ID01_3h"]``, contain ``"_"``):
              load only the exact files named by those stems.  This is the
              form produced internally when a dict-form subset is used with
              :func:`~cobrabox.datasets.load_dataset`.
    """
    mat_paths = sorted(p for p in dataset_dir.glob("*.mat") if not p.stem.endswith("_info"))
    if subset is not None:
        subset_set = set(subset)
        if any("_" in s for s in subset_set):
            # Stem-based filtering: exact match against the file stem.
            mat_paths = [p for p in mat_paths if p.stem in subset_set]
        else:
            # Subject-key filtering: match subject ID derived from the stem.
            mat_paths = [p for p in mat_paths if p.stem.rsplit("_", 1)[0] in subset_set]
    if not mat_paths:
        raise FileNotFoundError(f"No .mat files found for 'swiss_eeg_long' in {dataset_dir}.")

    fs_cache: dict[str, float | None] = {}
    items: list[SignalData] = []
    for path in mat_paths:
        subject_id = path.stem.rsplit("_", 1)[0]  # "ID01_1h" -> "ID01"

        if subject_id not in fs_cache:
            fs_cache[subject_id] = _load_swez_sampling_rate(dataset_dir, subject_id)
        sampling_rate = fs_cache[subject_id]

        try:
            mat = scipy.io.loadmat(str(path), variable_names=["EEG"])
        except NotImplementedError as exc:
            raise RuntimeError(
                f"{path.name} appears to be a MATLAB v7.3 (HDF5) file, which is not "
                "supported by scipy. Install h5py and load the file manually, or "
                "convert it to an older MAT format."
            ) from exc

        eeg = mat.get("EEG")
        if eeg is None or np.asarray(eeg).size == 0:
            continue

        arr = np.asarray(eeg, dtype=float)
        if arr.ndim != 2:
            continue
        # SWEZ convention: channels x time — transpose to time x channels
        values = arr.T
        n_channels = values.shape[1]
        channels = [f"ch{i}" for i in range(n_channels)]
        time = (
            np.arange(values.shape[0], dtype=float) / sampling_rate
            if sampling_rate
            else np.arange(values.shape[0], dtype=float)
        )
        da = xr.DataArray(
            values,
            dims=["time", "space"],
            coords={"time": time, "space": channels},
            attrs={"identifier": "swiss_eeg_long", "source_file": path.name},
        )
        items.append(SignalData.from_xarray(da, sampling_rate=sampling_rate, subjectID=subject_id))

    if not items:
        raise ValueError("All swiss_eeg_long files were empty or unparsable.")
    return Dataset(items)


_BONN_SAMPLING_RATE: float = 173.61  # Hz, fixed for all Bonn EEG recordings

_BONN_SET_CONDITION: dict[str, str] = {
    "Z": "healthy_eyes_open",
    "O": "healthy_eyes_closed",
    "N": "interictal_contralateral",
    "F": "interictal_focal",
    "S": "ictal",
}

_BONN_SET_GROUP: dict[str, str] = {
    "Z": "healthy",
    "O": "healthy",
    "N": "interictal",
    "F": "interictal",
    "S": "ictal",
}


def _load_bonn_eeg(dataset_dir: Path, subset: Sequence[str] | None = None) -> Dataset[SignalData]:
    """Load Bonn University EEG zip archives into SignalData objects.

    Each of the five sets (Z, O, N, F, S) is a zip file containing 100
    single-channel .txt recordings at 173.61 Hz, 4096 samples each (~23.6 s).
    One ``SignalData`` is produced per .txt file.

    Set meanings (Andrzejak et al. 2001):
        Z — healthy subjects, eyes open
        O — healthy subjects, eyes closed
        N — interictal EEG, contralateral hemisphere
        F — interictal EEG, epileptogenic zone
        S — ictal EEG (seizures)

    Args:
        dataset_dir: Directory containing the downloaded ``.zip`` archives.
        subset: If given, only load sets whose letter (e.g. ``"S"``, ``"Z"``)
            is in this list.
    """
    zip_paths = sorted(dataset_dir.glob("*.zip"))
    if subset is not None:
        subset_set = set(subset)
        zip_paths = [p for p in zip_paths if p.stem in subset_set]
    if not zip_paths:
        raise FileNotFoundError(f"No .zip files found for 'bonn_eeg' in {dataset_dir}.")

    items: list[SignalData] = []
    for zip_path in zip_paths:
        set_letter = zip_path.stem  # "Z", "O", "N", "F", or "S"
        condition = _BONN_SET_CONDITION.get(set_letter, set_letter)
        group_id = _BONN_SET_GROUP.get(set_letter, set_letter)

        with zipfile.ZipFile(zip_path) as zf:
            txt_members = sorted(name for name in zf.namelist() if name.lower().endswith(".txt"))
            for member in txt_members:
                raw = zf.read(member)
                try:
                    signal = np.loadtxt(io.BytesIO(raw))
                except Exception:
                    continue
                if signal.ndim != 1 or signal.size == 0:
                    continue

                values = signal[:, np.newaxis]  # (T, 1) — single channel
                time = np.arange(len(values), dtype=float) / _BONN_SAMPLING_RATE
                subject_id = Path(member).stem  # e.g. "Z000"
                da = xr.DataArray(
                    values,
                    dims=["time", "space"],
                    coords={"time": time, "space": ["ch0"]},
                    attrs={
                        "identifier": "bonn_eeg",
                        "source_archive": zip_path.name,
                        "source_member": member,
                    },
                )
                items.append(
                    SignalData.from_xarray(
                        da,
                        sampling_rate=_BONN_SAMPLING_RATE,
                        subjectID=subject_id,
                        groupID=group_id,
                        condition=condition,
                    )
                )

    if not items:
        raise ValueError("All bonn_eeg archives were empty or unparsable.")
    return Dataset(items)


def _load_edf_dataset(
    dataset_dir: Path,
    identifier: str,
    subject_key_fn: Callable[[str], str],
    subset: Sequence[str] | None,
) -> Dataset[SignalData]:
    """Shared EDF loading logic for scalp EEG datasets.

    Reads all ``.edf`` files in ``dataset_dir``, optionally filtered by
    ``subset``. Uses MNE-Python for EDF parsing. One ``SignalData`` is
    produced per file.

    Args:
        dataset_dir: Directory containing the downloaded ``.edf`` files.
        identifier: Dataset identifier string (used in error messages and attrs).
        subject_key_fn: Maps a file stem to a subject ID string.
        subset: If given, only load files whose subject ID is in this list.
    """
    try:
        import mne
    except ImportError as e:
        raise RuntimeError(
            f"Loading {identifier!r} requires MNE-Python. "
            "It is pulled in transitively by mne-connectivity, but if missing: "
            "uv add mne"
        ) from e

    edf_paths = sorted(dataset_dir.glob("*.edf"))
    if subset is not None:
        subset_set = set(subset)
        edf_paths = [p for p in edf_paths if subject_key_fn(p.stem) in subset_set]
    if not edf_paths:
        raise FileNotFoundError(f"No .edf files found for {identifier!r} in {dataset_dir}.")

    items: list[SignalData] = []
    for path in edf_paths:
        subject_id = subject_key_fn(path.stem)
        try:
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
        except Exception:
            continue
        arr = raw.get_data().T  # (n_channels, n_samples) -> (n_samples, n_channels)
        if arr.shape[0] == 0:
            continue
        fs = float(raw.info["sfreq"])
        time = np.arange(arr.shape[0], dtype=float) / fs
        da = xr.DataArray(
            arr,
            dims=["time", "space"],
            coords={"time": time, "space": list(raw.ch_names)},
            attrs={"identifier": identifier, "source_file": path.name},
        )
        items.append(SignalData.from_xarray(da, sampling_rate=fs, subjectID=subject_id))

    if not items:
        raise ValueError(f"All {identifier!r} files were empty or unparsable.")
    return Dataset(items)


def _load_chb_mit(dataset_dir: Path, subset: Sequence[str] | None = None) -> Dataset[SignalData]:
    """Load CHB-MIT scalp EEG ``.edf`` files into SignalData objects.

    Each ``.edf`` file is one recording session (typically 1-2 hours) and
    produces one ``SignalData`` object. The subject ID is parsed from the
    filename stem (e.g. ``chb01_01.edf`` → subject ``chb01``).

    Args:
        dataset_dir: Directory containing the downloaded ``.edf`` files.
        subset: If given, only load files whose subject ID (e.g. ``"chb01"``)
            is in this list.
    """
    return _load_edf_dataset(
        dataset_dir,
        identifier="chb_mit",
        subject_key_fn=lambda stem: stem.split("_", 1)[0],
        subset=subset,
    )


def _load_siena_eeg(dataset_dir: Path, subset: Sequence[str] | None = None) -> Dataset[SignalData]:
    """Load Siena Scalp EEG ``.edf`` files into SignalData objects.

    Each ``.edf`` file is one long recording session (up to several hours) and
    produces one ``SignalData`` object. The subject ID is parsed from the
    filename stem (e.g. ``PN00-1.edf`` → subject ``PN00``).

    Args:
        dataset_dir: Directory containing the downloaded ``.edf`` files.
        subset: If given, only load files whose subject ID (e.g. ``"PN00"``)
            is in this list.
    """
    return _load_edf_dataset(
        dataset_dir,
        identifier="siena_eeg",
        subject_key_fn=lambda stem: stem.split("-", 1)[0],
        subset=subset,
    )


def _load_sleep_ieeg(dataset_dir: Path, subset: Sequence[str] | None = None) -> Dataset[SignalData]:
    """Load Sleep iEEG Dataset ``.edf`` files into SignalData objects.

    Each ``.edf`` file is one interictal sleep recording and produces one
    ``SignalData`` object. Subject IDs are parsed from the filename stem
    (e.g. ``sub-Detroit001_ses-01_task-sleep_ieeg.edf`` → ``sub-Detroit001``).

    Args:
        dataset_dir: Directory containing the downloaded ``.edf`` files.
        subset: If given, only load files whose subject ID (e.g.
            ``"sub-Detroit001"``) is in this list.
    """
    return _load_edf_dataset(
        dataset_dir,
        identifier="sleep_ieeg",
        subject_key_fn=lambda stem: stem.split("_", 1)[0],
        subset=subset,
    )


def _load_zurich_ieeg(
    dataset_dir: Path, subset: Sequence[str] | None = None
) -> Dataset[SignalData]:
    """Load Zurich iEEG HFO Dataset BrainVision files into SignalData objects.

    Each ``.vhdr`` file is one 5-minute interictal run and produces one
    ``SignalData`` object. Subject IDs are parsed from the filename stem
    (e.g. ``sub-01_ses-interictalsleep_run-01_ieeg.vhdr`` → ``sub-01``).
    The ``.eeg`` and ``.vmrk`` sidecar files must be present in the same
    directory.

    .. note:: Upstream data issue (OpenNeuro ds003498 v1.1.1)

       Several ``.vmrk`` marker files in the source dataset reference sample
       positions beyond the length of their corresponding ``.eeg`` data files.
       Sub-15 is the most affected: 8 of its 28 runs have truncated data files
       (65-183 s instead of the expected 300 s) while the marker files still
       reference the full 300 s range, resulting in MNE warnings about thousands
       of omitted annotations. A handful of other subjects have a single
       annotation that extends slightly past the recording end. These warnings
       are suppressed here because they are harmless — no sample data is lost,
       only out-of-range HFO event markers are discarded by MNE.

    Args:
        dataset_dir: Directory containing the downloaded BrainVision files.
        subset: If given, only load files whose subject ID (e.g. ``"sub-01"``)
            is in this list.
    """
    from .downloader import (
        _ZURICH_ALL_CHANNELS_PER_SUBJECT,
        _ZURICH_EXCLUDED_CHANNELS_PER_SUBJECT,
        _ZURICH_ILAE_PER_SUBJECT,
        _ZURICH_RESECTED_ZONE_PER_SUBJECT,
    )

    try:
        import mne
    except ImportError as e:
        raise RuntimeError(
            "Loading 'zurich_ieeg' requires MNE-Python. "
            "It is pulled in transitively by mne-connectivity, but if missing: "
            "uv add mne"
        ) from e

    vhdr_paths = sorted(dataset_dir.glob("*.vhdr"))
    if subset is not None:
        subset_set = set(subset)
        vhdr_paths = [p for p in vhdr_paths if p.stem.split("_", 1)[0] in subset_set]
    if not vhdr_paths:
        raise FileNotFoundError(f"No .vhdr files found for 'zurich_ieeg' in {dataset_dir}.")

    items: list[SignalData] = []
    for path in vhdr_paths:
        subject_id = path.stem.split("_", 1)[0]
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*annotation.*outside.*data range")
                raw = mne.io.read_raw_brainvision(str(path), preload=True, verbose=False)
        except Exception:
            continue
        arr = raw.get_data().T  # (n_channels, n_samples) → (n_samples, n_channels)
        if arr.shape[0] == 0:
            continue
        fs = float(raw.info["sfreq"])
        time = np.arange(arr.shape[0], dtype=float) / fs
        da = xr.DataArray(
            arr,
            dims=["time", "space"],
            coords={"time": time, "space": list(raw.ch_names)},
            attrs={"identifier": "zurich_ieeg", "source_file": path.name},
        )
        extra: dict[str, object] = {
            "ilae": _ZURICH_ILAE_PER_SUBJECT.get(subject_id),
            "resected_zone": _ZURICH_RESECTED_ZONE_PER_SUBJECT.get(subject_id, []),
            "excluded_channels": _ZURICH_EXCLUDED_CHANNELS_PER_SUBJECT.get(subject_id, []),
            "all_channels": _ZURICH_ALL_CHANNELS_PER_SUBJECT.get(subject_id, []),
        }
        items.append(
            SignalData.from_xarray(da, sampling_rate=fs, subjectID=subject_id, extra=extra)
        )

    if not items:
        raise ValueError("All 'zurich_ieeg' files were empty or unparsable.")
    return Dataset(items)


def _clean_zurich_dataset(ds: Dataset[SignalData]) -> Dataset[SignalData]:
    """Drop the per-subject columns-C+D channels from a loaded Zurich dataset.

    For each item, the electrodes flagged in ``_ZURICH_IEEG_CLEAN_REMOVE`` for its
    subject are removed from the ``space`` dimension. Subjects with no flagged
    channels pass through unchanged. Cleaned items gain ``"zurich_ieeg_clean"`` in
    ``history`` and the dropped channel names in ``extra["removed_channels"]``.
    Warns (rather than fails) if a flagged channel is not present in the data.

    Pure transform on an already-loaded dataset, so it is testable without the
    (large) download.
    """
    import warnings

    from .downloader import _ZURICH_IEEG_CLEAN_REMOVE

    cleaned: list[SignalData] = []
    for item in ds:
        remove = set(_ZURICH_IEEG_CLEAN_REMOVE.get(item.subjectID or "", []))
        if not remove:
            cleaned.append(item)
            continue

        ch_names = [str(c) for c in item.data.coords["space"].to_numpy()]
        missing = remove - set(ch_names)
        if missing:
            warnings.warn(
                f"zurich_ieeg_clean: {item.subjectID}: "
                f"{len(missing)} channel(s) flagged for removal not found in data: "
                f"{sorted(missing)}",
                stacklevel=2,
            )
        keep = [c for c in ch_names if c not in remove]
        removed = [c for c in ch_names if c in remove]
        cleaned.append(
            item._copy_with_new_data(
                item.data.sel(space=keep),
                operation_name="zurich_ieeg_clean",
                extra={"removed_channels": removed},
            )
        )
    return Dataset(cleaned)


def _load_zurich_ieeg_clean(
    dataset_dir: Path, subset: Sequence[str] | None = None
) -> Dataset[SignalData]:
    """Load the Zurich iEEG dataset, then apply the columns-C+D channel cleaning.

    Same files and download directory as :func:`_load_zurich_ieeg`; only the
    per-subject channel removal (see :func:`_clean_zurich_dataset`) differs.
    """
    return _clean_zurich_dataset(_load_zurich_ieeg(dataset_dir, subset=subset))


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
