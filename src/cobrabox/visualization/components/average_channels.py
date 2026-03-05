"""AverageChannels feature: average across the space dimension."""

from __future__ import annotations

from typing import ClassVar

import panel as pn

from cobrabox.data import Data

from ..base import FeatureComponent


class AverageChannels(FeatureComponent):
    """Compute the mean across all spatial channels, producing a single-channel output."""

    display_name: ClassVar[str] = "Average Channels"

    def transform(self, data: Data) -> Data:
        xd = data.data
        averaged = xd.mean(dim="space").expand_dims(space=["mean"])
        return data._copy_with_new_data(averaged, operation_name="average_channels")

    def sidebar_view(self) -> pn.viewable.Viewable:
        return pn.pane.Markdown(
            "Averages all spatial channels into a single channel.", sizing_mode="stretch_width"
        )
