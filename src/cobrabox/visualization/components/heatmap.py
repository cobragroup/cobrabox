"""PlotHeatmap visualization: 2-D heatmap of channels over time."""

from __future__ import annotations

from typing import ClassVar

import holoviews as hv
import panel as pn
import param

from cobrabox.data import Data

from ..base import VisualizationComponent


class PlotHeatmap(VisualizationComponent):
    """Heatmap (image) plot with channels on the y-axis and time on the x-axis."""

    display_name: ClassVar[str] = "Heatmap"

    colormap = param.Selector(
        default="viridis",
        objects=["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "RdBu_r"],
        doc="Colormap for the heatmap",
    )

    def sidebar_view(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.widgets.Select.from_param(self.param.colormap, name="Colormap"),
            sizing_mode="stretch_width",
        )

    def get_plot(self, data: Data) -> pn.viewable.Viewable:
        xd = data.data
        space_labels = [str(s) for s in xd.coords["space"].values]
        n_channels = len(space_labels)
        time_vals = xd.coords["time"].values

        vals = xd.values  # (space, time)

        img = hv.Image(
            vals,
            bounds=(float(time_vals[0]), 0, float(time_vals[-1]), n_channels),
            kdims=["Time", "Channel"],
        ).opts(
            cmap=self.colormap,
            colorbar=True,
            xlabel="Time",
            ylabel="Channel",
            yticks=[(i + 0.5, label) for i, label in enumerate(space_labels)],
            responsive=True,
            height=400,
        )

        return pn.pane.HoloViews(img, sizing_mode="stretch_both")
