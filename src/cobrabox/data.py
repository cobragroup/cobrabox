from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TimeSeriesDataset:
    """Container for multichannel time-series data.

    Data is expected in shape (n_samples, n_channels). For fMRI this typically
    means (timepoints, parcels/voxels). For EEG, (samples, channels).
    """

    data: np.ndarray
    sampling_rate_hz: float
    modality: str
    channel_names: Sequence[str] = field(default_factory=tuple)
    timestamps_s: Optional[np.ndarray] = None
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = np.asarray(self.data)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array shaped (n_samples, n_channels)")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.channel_names and len(self.channel_names) != data.shape[1]:
            raise ValueError("channel_names length must match n_channels")
        if self.timestamps_s is not None:
            timestamps = np.asarray(self.timestamps_s)
            if timestamps.shape != (data.shape[0],):
                raise ValueError("timestamps_s must be shape (n_samples,)")
            object.__setattr__(self, "timestamps_s", timestamps)

        object.__setattr__(self, "data", data)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sampling_rate_hz

