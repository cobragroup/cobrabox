"""CobraboxApp: main Panel application wiring sidebar, tabs, and pipeline."""

from __future__ import annotations

from typing import Any

import panel as pn
import param

from cobrabox.data import Data

from .base import PipelineComponent, VisualizationComponent
from .components.raw_data import RawData
from .pipeline import Pipeline
from .registry import create_component, get_feature_components, get_visualization_components


class CobraboxApp(pn.viewable.Viewer):
    """Interactive pipeline-builder and visualization dashboard.

    Parameters
    ----------
    data : Data
        The initial (raw) dataset to visualise.
    """

    data = param.Parameter(doc="Initial Data object")

    def __init__(self, data: Data, **params: Any) -> None:
        params["data"] = data
        super().__init__(**params)

        self._pipeline = Pipeline(raw_data=data)
        self._raw_data_component = RawData(data=data)
        self._sidebar_column = pn.Column(sizing_mode="stretch_width")
        self._tabs = pn.Tabs(closable=True, sizing_mode="stretch_both")
        self._instance_counter: int = 0

        # Build add-component button menus
        viz_options = {
            cls.display_name: name for name, cls in get_visualization_components().items()
        }
        feat_options = {cls.display_name: name for name, cls in get_feature_components().items()}

        self._add_viz_button = pn.widgets.MenuButton(
            name="Add Visualization",
            items=list(viz_options.keys()) if viz_options else ["(none available)"],
            button_type="default",
            sizing_mode="stretch_width",
        )
        self._add_feat_button = pn.widgets.MenuButton(
            name="Add Feature",
            items=list(feat_options.keys()) if feat_options else ["(none available)"],
            button_type="default",
            sizing_mode="stretch_width",
        )
        self._viz_class_map = viz_options
        self._feat_class_map = feat_options

        self._add_viz_button.on_click(self._on_add_viz)
        self._add_feat_button.on_click(self._on_add_feat)

        self._refresh()

    # -- refresh: re-execute pipeline and rebuild all UI ---------------------

    def _refresh(self) -> None:
        """Run the pipeline and rebuild sidebar + tabs from scratch."""
        plots = self._pipeline.execute()
        self._rebuild_sidebar()
        self._sync_tabs(plots)

    # -- sidebar -------------------------------------------------------------

    def _rebuild_sidebar(self) -> None:
        cards: list[pn.viewable.Viewable] = [
            pn.Card(
                self._raw_data_component.sidebar_view(),
                title="Raw Data",
                collapsed=True,
                sizing_mode="stretch_width",
            )
        ]
        for i, comp in enumerate(self._pipeline.components):
            cards.append(self._make_component_card(i, comp))

        self._sidebar_column.objects = [
            *cards,
            pn.layout.Divider(),
            self._add_viz_button,
            self._add_feat_button,
        ]

    def _make_component_card(self, index: int, comp: PipelineComponent) -> pn.Card:
        kind_label = "V" if isinstance(comp, VisualizationComponent) else "F"
        title = f"[{kind_label}] {comp.display_name}"

        delete_btn = pn.widgets.Button(
            name="x", button_type="danger", width=32, height=32, margin=(0, 0)
        )
        delete_btn.on_click(lambda _event, idx=index: self._on_delete(idx))

        up_btn = pn.widgets.Button(name="^", width=32, height=32, margin=(0, 0))
        down_btn = pn.widgets.Button(name="v", width=32, height=32, margin=(0, 0))
        up_btn.on_click(lambda _event, idx=index: self._on_move(idx, idx - 1))
        down_btn.on_click(lambda _event, idx=index: self._on_move(idx, idx + 1))

        header_row = pn.Row(up_btn, down_btn, delete_btn, sizing_mode="stretch_width", align="end")

        return pn.Card(
            header_row,
            comp.sidebar_view(),
            title=title,
            collapsed=True,
            sizing_mode="stretch_width",
        )

    # -- tabs ----------------------------------------------------------------

    def _sync_tabs(self, plots: dict[int, Any]) -> None:
        self._tabs.clear()
        for i, comp in enumerate(self._pipeline.components):
            if isinstance(comp, VisualizationComponent):
                plot = plots.get(i, pn.pane.Markdown("*No data*"))
                self._tabs.append((f"{comp.display_name} #{comp._instance_id}", plot))

    # -- handlers ------------------------------------------------------------

    def _on_add_viz(self, event: Any) -> None:
        clicked = event.new if hasattr(event, "new") else event
        class_name = self._viz_class_map.get(clicked)
        if class_name is None:
            return
        self._instance_counter += 1
        comp = create_component(class_name)
        comp._instance_id = self._instance_counter
        self._pipeline.add(comp)
        self._refresh()

    def _on_add_feat(self, event: Any) -> None:
        clicked = event.new if hasattr(event, "new") else event
        class_name = self._feat_class_map.get(clicked)
        if class_name is None:
            return
        comp = create_component(class_name)
        self._pipeline.add(comp)
        self._refresh()

    def _on_delete(self, index: int) -> None:
        self._pipeline.remove(index)
        self._refresh()

    def _on_move(self, from_idx: int, to_idx: int) -> None:
        if to_idx < 0 or to_idx >= len(self._pipeline.components):
            return
        self._pipeline.move(from_idx, to_idx)
        self._refresh()

    # -- Panel interface -----------------------------------------------------

    def __panel__(self) -> pn.viewable.Viewable:
        return pn.Row(self._sidebar_column, self._tabs, sizing_mode="stretch_both")

    def as_template(self, title: str = "cobrabox") -> pn.template.FastListTemplate:
        """Return a FastListTemplate ready for ``panel serve``."""
        return pn.template.FastListTemplate(
            title=title,
            sidebar=[self._sidebar_column],
            main=[self._tabs],
            main_layout=None,
            sidebar_width=350,
        )
