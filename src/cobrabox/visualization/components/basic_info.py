"""BasicInfo visualization: data summary table shown in a tab."""

from __future__ import annotations

from typing import ClassVar

import panel as pn

from cobrabox.data import Data

from ..base import VisualizationComponent


class BasicInfo(VisualizationComponent):
    """Renders a concise summary table of the data that enters this component."""

    display_name: ClassVar[str] = "Basic Info"

    def get_plot(self, data: Data) -> pn.viewable.Viewable:
        xd = data.data
        rows = [
            ("Dimensions", str(dict(xd.sizes))),
            ("Shape", str(xd.shape)),
            ("Dtype", str(xd.dtype)),
            ("Coordinates", ", ".join(str(c) for c in xd.coords)),
        ]
        if data.sampling_rate is not None:
            rows.append(("Sampling rate", f"{data.sampling_rate} Hz"))
        if data.subjectID is not None:
            rows.append(("Subject ID", data.subjectID))
        if data.groupID is not None:
            rows.append(("Group ID", data.groupID))
        if data.condition is not None:
            rows.append(("Condition", data.condition))
        if data.history:
            rows.append(("History", " -> ".join(data.history)))
        if data.extra:
            rows.append(("Extra keys", ", ".join(sorted(data.extra.keys()))))

        html_rows = "".join(
            f"<tr><td style='padding:4px 12px;font-weight:600'>{k}</td>"
            f"<td style='padding:4px 12px'>{v}</td></tr>"
            for k, v in rows
        )
        html = (
            f"<table style='border-collapse:collapse;width:100%'><tbody>{html_rows}</tbody></table>"
        )
        return pn.pane.HTML(html, sizing_mode="stretch_width")
