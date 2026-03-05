"""RawData component: always-first, non-removable pipeline head."""

from __future__ import annotations

from typing import ClassVar

import panel as pn

from cobrabox.data import Data

from ..base import PipelineComponent


class RawData(PipelineComponent):
    """Displays a summary of the initial raw data.

    This component is always the first element in a pipeline and cannot be
    removed or moved.  It passes data through unchanged.
    """

    display_name: ClassVar[str] = "Raw Data"
    component_kind: ClassVar[str] = "raw_data"

    def __init__(self, data: Data | None = None, **params: object) -> None:
        super().__init__(**params)
        self._raw_data = data

    def process(self, data: Data) -> Data:
        return data

    def sidebar_view(self) -> pn.viewable.Viewable:
        if self._raw_data is None:
            return pn.pane.Markdown("*No data loaded.*")

        d = self._raw_data
        lines = [
            f"**Dims:** {dict(d.data.sizes)}",
            f"**Shape:** {d.data.shape}",
            f"**Dtype:** {d.data.dtype}",
        ]
        if d.sampling_rate is not None:
            lines.append(f"**Sampling rate:** {d.sampling_rate} Hz")
        if d.subjectID is not None:
            lines.append(f"**Subject:** {d.subjectID}")
        if d.groupID is not None:
            lines.append(f"**Group:** {d.groupID}")
        if d.condition is not None:
            lines.append(f"**Condition:** {d.condition}")
        if d.extra:
            lines.append(f"**Extra keys:** {sorted(d.extra.keys())}")

        return pn.pane.Markdown("\n\n".join(lines))
