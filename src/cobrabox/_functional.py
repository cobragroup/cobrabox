"""Generate the one-shot functional API from the feature classes (GH #116).

    o = cb.autocorrelation(d, dim="time", fs=250.0)     # this module
    o = cb.Autocorrelation(dim="time", fs=250.0).apply(d)

The functional layer is **additive**. Pipelines compose feature *instances* —
``cb.SlidingWindow(...) | cb.LineLength() | cb.MeanAggregate()`` — so the classes
remain the primary API; these wrappers just remove the ``().apply()`` ceremony
from a single call.

Names come from the implementation module, not from the class: splitting
``CamelCase`` gives ``e_m_d`` and ``s_v_d``, while the filenames (``_emd.py``,
``_svd.py``) are already correct. The mapping is 1:1, so the module name minus
its leading underscore is the function name.

Wrappers are generated rather than hand-written so a new parameter cannot drift
out of sync with its dataclass. ``scripts/gen_stubs.py`` renders matching ``.pyi``
signatures for static analysis.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys
from collections.abc import Callable, Iterator
from typing import Any

from .base_feature import AggregatorFeature, SplitterFeature
from .data import Data

# Aggregators fold a stream produced by a splitter — `(Data, Iterator[Data]) -> Data`.
# Outside a Chord there is no stream to fold, so a standalone `mean_aggregate(d, windows)`
# would invite misuse. They stay class-only, deliberately; test_functional_api.py pins it.
_EXCLUDED_BASES = (AggregatorFeature,)


def function_name(cls: type) -> str:
    """`cobrabox.signalstats._line_length` -> `line_length`."""
    return cls.__module__.rsplit(".", 1)[-1].lstrip("_")


def public_fields(cls: type) -> list[dataclasses.Field]:
    """Dataclass fields that are real constructor parameters."""
    return [f for f in dataclasses.fields(cls) if f.init and not f.name.startswith("_")]


def has_functional_form(cls: type) -> bool:
    return not issubclass(cls, _EXCLUDED_BASES)


class _FactoryDefault:
    """Stand-in for a ``default_factory`` field in the displayed signature.

    The field is optional, but its value is built per call, so there is no single
    object to show. ``__signature__`` is metadata only — the wrapper really takes
    ``(data, *args, **kwargs)`` — so this is never passed to the dataclass.
    """

    def __repr__(self) -> str:
        return "<factory>"


FACTORY_DEFAULT = _FactoryDefault()


def _signature(cls: type) -> inspect.Signature:
    """`(data, <dataclass fields in declaration order>)`.

    Mirroring the dataclass keeps positional calls working, which is what the
    issue asked for — `cb.correlation(d, "time", "spearman")` rather than
    keyword-only.
    """
    params = [inspect.Parameter("data", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for f in public_fields(cls):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            # Optional in the dataclass, so it must be optional here too, or a
            # later defaulted field would make the signature invalid.
            default = FACTORY_DEFAULT
        else:
            default = inspect.Parameter.empty
        params.append(
            inspect.Parameter(
                f.name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default, annotation=f.type
            )
        )
    return inspect.Signature(params)


def _summary(cls: type) -> str:
    doc = inspect.getdoc(cls) or ""
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return f"Apply {cls.__name__}."


def _docstring(cls: type) -> str:
    """Compose a docstring rather than copying the class's.

    The class docstring's ``Example:`` shows the class form; reusing it verbatim
    would document the wrong call. Prose stays single-sourced on the class.
    """
    name = function_name(cls)
    fields = public_fields(cls)
    lines = [_summary(cls), ""]

    if fields:
        lines.append("Args:")
        lines.append("    data: Input data.")
        for f in fields:
            if f.default is not dataclasses.MISSING and f.default is not None:
                suffix = f" Defaults to ``{f.default!r}``."
            else:
                suffix = ""
            lines.append(f"    {f.name}: See :class:`~cobrabox.{cls.__name__}`.{suffix}")
    else:
        lines += ["Args:", "    data: Input data."]

    call = f"{name}(data)" if not fields else f"{name}(data, ...)"
    lines += [
        "",
        "Example:",
        f"    >>> result = cb.{call}",
        "",
        f"Equivalent to ``cb.{cls.__name__}(...).apply(data)``. Use the class form to",
        "compose a pipeline with ``|``, to serialize, or inside a ``Chord``.",
    ]
    return "\n".join(lines)


def _drop_factory_sentinels(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Let the dataclass apply its own factory if the displayed default is passed back.

    Guards the round-trip `f(d, **dict(inspect.signature(f).parameters ...))`, where
    the sentinel could be handed straight back to us.
    """
    return {k: v for k, v in kwargs.items() if v is not FACTORY_DEFAULT}


def make_functional(cls: type) -> Callable[..., Any]:
    """Build the one-shot wrapper for a feature class."""
    is_splitter = issubclass(cls, SplitterFeature)

    if is_splitter:

        def wrapper(data: Data, *args: Any, **kwargs: Any) -> Iterator[Data]:
            # Splitters have no .apply(); calling the instance yields the stream.
            return cls(*args, **_drop_factory_sentinels(kwargs))(data)
    else:

        def wrapper(data: Data, *args: Any, **kwargs: Any) -> Data:
            return cls(*args, **_drop_factory_sentinels(kwargs)).apply(data)

    name = function_name(cls)
    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__module__ = cls.__module__.rsplit(".", 1)[0]
    wrapper.__doc__ = _docstring(cls)
    wrapper.__signature__ = _signature(cls)  # type: ignore[attr-defined]
    wrapper.__wrapped_feature__ = cls  # type: ignore[attr-defined]
    return wrapper


def install(package_name: str) -> list[str]:
    """Install functional wrappers into a domain package and extend its ``__all__``.

    Called at the end of each domain's ``__init__.py``. Returns the names added.
    """
    package = sys.modules[package_name]
    added: list[str] = []
    for cls_name in list(getattr(package, "__all__", ())):
        cls = getattr(package, cls_name, None)
        if not isinstance(cls, type) or not getattr(cls, "_is_cobrabox_feature", False):
            continue
        if not has_functional_form(cls):
            continue
        fn = make_functional(cls)
        if hasattr(package, fn.__name__):
            raise RuntimeError(
                f"Cannot install functional wrapper {fn.__name__!r} in {package_name!r}: "
                f"the name is already taken by {getattr(package, fn.__name__)!r}."
            )
        setattr(package, fn.__name__, fn)
        added.append(fn.__name__)
    package.__all__ = sorted([*package.__all__, *added])  # type: ignore[attr-defined]
    return added
