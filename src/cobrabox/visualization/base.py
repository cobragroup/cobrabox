"""Base classes for pipeline components."""

from __future__ import annotations

from typing import Any, ClassVar

import panel as pn
import param

from cobrabox.data import Data


class PipelineComponent(pn.viewable.Viewer):
    """Abstract base for all pipeline components.

    Subclasses must set ``display_name`` and ``component_kind`` class variables
    and implement :meth:`process`.
    """

    display_name: ClassVar[str] = "Component"
    component_kind: ClassVar[str] = ""  # "feature" or "visualization"

    _instance_id = param.Integer(default=0, doc="Unique instance counter for tab naming")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

    def process(self, data: Data) -> Data:
        """Process data through this component. Override in subclasses."""
        raise NotImplementedError

    def sidebar_view(self) -> pn.viewable.Viewable:
        """Return the widget panel shown inside the sidebar card."""
        return pn.pane.Markdown("*No configurable parameters.*")

    def __panel__(self) -> pn.viewable.Viewable:
        return self.sidebar_view()


class FeatureComponent(PipelineComponent):
    """Component that transforms Data into new Data."""

    component_kind: ClassVar[str] = "feature"

    def transform(self, data: Data) -> Data:
        """Apply transformation. Override in subclasses."""
        raise NotImplementedError

    def process(self, data: Data) -> Data:
        return self.transform(data)


class VisualizationComponent(PipelineComponent):
    """Component that visualizes Data and passes it through unchanged."""

    component_kind: ClassVar[str] = "visualization"

    _last_data = param.Parameter(precedence=-1, doc="Data from last process() call")

    def get_plot(self, data: Data) -> pn.viewable.Viewable:
        """Create a visualization of *data*. Override in subclasses."""
        raise NotImplementedError

    def process(self, data: Data) -> Data:
        self._last_data = data
        return data

    def plot_view(self) -> pn.viewable.Viewable:
        """Return a reactive pane that re-renders when component params change."""
        # Collect user-facing params that should trigger a re-render
        reactive_params = {}
        for name in self.param:
            if name.startswith("_") or name == "name":
                continue
            reactive_params[name] = getattr(self.param, name)

        def _render(**kwargs: Any) -> pn.viewable.Viewable:
            if self._last_data is None:
                return pn.pane.Markdown("*No data*")
            return self.get_plot(self._last_data)

        if reactive_params:
            return pn.bind(_render, **reactive_params)
        return _render()
