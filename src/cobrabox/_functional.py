"""Helpers and the marker for the one-shot functional API (GH #116).

Every feature class has a companion function defined **in the same file**, just
below the class:

    cb.correlation(data, method="spearman")         # one-shot
    cb.Correlation(method="spearman").apply(data)   # class, for pipelines

The function is written as ordinary source and decorated with
``@functional(TheClass)``. Writing it by hand (rather than synthesising it at
import time) is deliberate: ``cb.correlation?`` then points at the feature file
where the real code lives, and type-checkers see its true signature. The marker
lets discovery (``feature.py``) and the doc/stub generators find the function
and tie it back to its class.

This module keeps only the shared helpers and the marker; the wrappers
themselves live with their classes. ``scripts/gen_functional_wrappers.py`` seeds
the wrapper for any feature that lacks one, deriving it from the class.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, TypeVar

from .base_feature import AggregatorFeature

_F = TypeVar("_F", bound=Callable[..., Any])


def functional(feature_cls: type) -> Callable[[_F], _F]:
    """Mark a module-level function as the functional form of ``feature_cls``.

    Returns the function unchanged — so ``inspect.getsourcefile`` still resolves
    to the feature file — while tagging it for auto-discovery.
    """

    def decorate(fn: _F) -> _F:
        # Attributes on a function object: set through Any, since a typed callable
        # does not admit arbitrary attributes.
        marked: Any = fn
        marked._wrapped_feature = feature_cls
        marked._is_cobrabox_feature_function = True
        return fn

    return decorate


def function_name(cls: type) -> str:
    """``cobrabox.signalstats._line_length`` -> ``line_length``."""
    return cls.__module__.rsplit(".", 1)[-1].lstrip("_")


def public_fields(cls: type) -> list[dataclasses.Field]:
    """Dataclass fields that are real constructor parameters."""
    return [f for f in dataclasses.fields(cls) if f.init and not f.name.startswith("_")]


def has_functional_form(cls: type) -> bool:
    """Aggregators fold a splitter's stream, so they get no standalone function."""
    return not issubclass(cls, AggregatorFeature)
