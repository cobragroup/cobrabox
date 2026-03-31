from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable, Sequence
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
              :func:`~cobrabox.datasets.dataset`.
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

    Args:
        dataset_dir: Directory containing the downloaded BrainVision files.
        subset: If given, only load files whose subject ID (e.g. ``"sub-01"``)
            is in this list.
    """
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
        items.append(SignalData.from_xarray(da, sampling_rate=fs, subjectID=subject_id))

    if not items:
        raise ValueError("All 'zurich_ieeg' files were empty or unparsable.")
    return Dataset(items)


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
