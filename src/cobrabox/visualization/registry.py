"""Component registry with auto-discovery from the ``components`` subpackage."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

from . import components as _components_pkg
from .base import FeatureComponent, PipelineComponent, VisualizationComponent

_discovered: dict[str, type[PipelineComponent]] = {}

for _mod in pkgutil.iter_modules(_components_pkg.__path__):
    _module = importlib.import_module(f"{_components_pkg.__name__}.{_mod.name}")
    for _name, _obj in vars(_module).items():
        if (
            inspect.isclass(_obj)
            and issubclass(_obj, PipelineComponent)
            and _obj not in (PipelineComponent, FeatureComponent, VisualizationComponent)
            and getattr(_obj, "component_kind", "") != ""
        ):
            if _name in _discovered:
                raise ValueError(
                    f"Duplicate component class name '{_name}' found while importing "
                    f"module '{_module.__name__}'."
                )
            _discovered[_name] = _obj


def get_all_components() -> dict[str, type[PipelineComponent]]:
    """Return all discovered component classes keyed by class name."""
    return dict(_discovered)


def get_feature_components() -> dict[str, type[FeatureComponent]]:
    """Return only feature component classes."""
    return {k: v for k, v in _discovered.items() if issubclass(v, FeatureComponent)}


def get_visualization_components() -> dict[str, type[VisualizationComponent]]:
    """Return only visualization component classes."""
    return {k: v for k, v in _discovered.items() if issubclass(v, VisualizationComponent)}


def create_component(name: str, **kwargs: Any) -> PipelineComponent:
    """Instantiate a registered component by class name."""
    cls = _discovered.get(name)
    if cls is None:
        raise KeyError(f"Unknown component '{name}'. Available: {sorted(_discovered)}")
    return cls(**kwargs)
