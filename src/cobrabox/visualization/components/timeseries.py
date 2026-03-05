"""PlotTimeseries visualization: interactive line plot of time-series channels."""

from __future__ import annotations

from typing import Any, ClassVar

import holoviews as hv
import hvplot.xarray  # noqa: F401 - registers .hvplot accessor
import numpy as np
import panel as pn
import param

from cobrabox.data import Data

from ..base import VisualizationComponent


class PlotTimeseries(VisualizationComponent):
    """Line plot of time-series data with optional channel overlay/offset."""

    display_name: ClassVar[str] = "Timeseries"

    overlay = param.Boolean(default=True, doc="Overlay all channels on the same y-axis")
    show_annotations = param.Boolean(default=True, doc="Show annotation regions from data.extra")

    def sidebar_view(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.widgets.Checkbox.from_param(self.param.overlay, name="Overlay channels"),
            pn.widgets.Checkbox.from_param(self.param.show_annotations, name="Show annotations"),
            sizing_mode="stretch_width",
        )

    def get_plot(self, data: Data) -> pn.viewable.Viewable:
        xd = data.data.rename("signal")

        if self.overlay:
            plot = xd.hvplot.line(x="time", by="space", responsive=True, height=400)
        else:
            plot = self._offset_plot(xd)

        if self.show_annotations:
            plot = self._add_annotations(plot, data)

        return pn.pane.HoloViews(plot, sizing_mode="stretch_both")

    def _offset_plot(self, xd: Any) -> Any:
        """Create a vertically-offset plot so channels don't overlap."""
        space_vals = xd.coords["space"].values
        n_channels = len(space_vals)
        if n_channels == 0:
            return xd.hvplot.line(x="time", responsive=True, height=400)

        # Compute per-channel offset based on global data range
        data_range = float(xd.max() - xd.min())
        offset_step = data_range * 1.1 if data_range > 0 else 1.0

        curves = []
        yticks = []
        for i, ch in enumerate(space_vals):
            channel_data = xd.sel(space=ch)
            offset = i * offset_step
            shifted = channel_data + offset
            curve = shifted.hvplot.line(x="time", label=str(ch), responsive=True, height=400)
            curves.append(curve)
            yticks.append((offset + float(channel_data.mean()), str(ch)))

        return hv.Overlay(curves).opts(yticks=yticks, ylabel="Channel")

    def _add_annotations(self, plot: Any, data: Data) -> Any:
        """Overlay shaded regions from data.extra annotation arrays."""
        extra = data.extra
        if not extra:
            return plot

        time_vals = np.asarray(data.data.coords["time"].values)
        colors = ["#ff000030", "#0000ff30", "#00ff0030", "#ff00ff30"]
        color_idx = 0

        for key, value in extra.items():
            if not key.endswith("_annot") and not key.endswith("_annotation"):
                continue

            arr = np.asarray(value)
            if arr.shape != time_vals.shape:
                continue

            # Find contiguous regions where annotation is nonzero/True
            mask = arr.astype(bool)
            spans = self._mask_to_spans(time_vals, mask)

            color = colors[color_idx % len(colors)]
            color_idx += 1

            for t_start, t_end in spans:
                plot = plot * hv.VSpan(t_start, t_end).opts(color=color, line_width=0)

        return plot

    @staticmethod
    def _mask_to_spans(time_vals: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
        """Convert a boolean mask to a list of (start, end) time spans."""
        spans: list[tuple[float, float]] = []
        in_span = False
        start = 0.0
        for i, val in enumerate(mask):
            if val and not in_span:
                start = float(time_vals[i])
                in_span = True
            elif not val and in_span:
                spans.append((start, float(time_vals[i])))
                in_span = False
        if in_span:
            spans.append((start, float(time_vals[-1])))
        return spans
