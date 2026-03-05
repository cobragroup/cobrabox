"""Pipeline engine: linear chain of components."""

from __future__ import annotations

from typing import Any

import param

from cobrabox.data import Data

from .base import PipelineComponent, VisualizationComponent


class Pipeline(param.Parameterized):
    """Ordered list of components through which data flows sequentially.

    Starts from ``raw_data``, feeds each component's output into the next.
    Visualization components also produce a plot alongside the pass-through data.
    """

    components = param.List(default=[], item_type=PipelineComponent)
    raw_data = param.Parameter(doc="Initial Data object (immutable)")

    def execute(self) -> dict[int, Any]:
        """Run the full pipeline, return a dict of {index: plot} for viz components."""
        plots: dict[int, Any] = {}
        if self.raw_data is None:
            return plots

        data: Data = self.raw_data
        for i, component in enumerate(self.components):
            data = component.process(data)
            if isinstance(component, VisualizationComponent):
                plots[i] = component.get_plot(data)

        return plots

    def add(self, component: PipelineComponent, index: int | None = None) -> None:
        """Append or insert *component*."""
        new = list(self.components)
        if index is None:
            index = len(new)
        new.insert(index, component)
        self.components = new

    def remove(self, index: int) -> PipelineComponent:
        """Remove and return the component at *index*."""
        new = list(self.components)
        removed = new.pop(index)
        self.components = new
        return removed

    def move(self, from_idx: int, to_idx: int) -> None:
        """Move component from *from_idx* to *to_idx*."""
        if from_idx == to_idx:
            return
        new = list(self.components)
        comp = new.pop(from_idx)
        new.insert(to_idx, comp)
        self.components = new
