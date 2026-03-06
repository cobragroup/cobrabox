"""Serialization and deserialization of cobrabox pipelines to YAML/JSON.

Public API:
    serialize(obj, fmt)   — pipeline → string
    deserialize(content)  — string → Pipeline
    save(obj, path)       — pipeline → file
    load(path)            — file → Pipeline

All objects are serialized as a ``pipeline:`` list. A single feature or Chord
is wrapped in a one-element list on the way out and unwrapped transparently
by the convenience methods on BaseFeature/Pipeline/Chord.

Security note: Use ``SafeLoader`` for YAML parsing — no arbitrary Python object
construction from YAML tags. Callable parameters are serialized with ``dill`` —
only load files from trusted sources.
"""

from __future__ import annotations

import dataclasses
import json
import typing
import warnings
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Literal

import dill
import yaml

# Lazily evaluated at first use to avoid import-time circular dependency.
# All public functions that need these perform the import inside the body.
_COBRABOX_VERSION: str | None = None


def _cb_version() -> str:
    global _COBRABOX_VERSION
    if _COBRABOX_VERSION is None:
        _COBRABOX_VERSION = _pkg_version("cobrabox")
    return _COBRABOX_VERSION


SCHEMA_VERSION = "1.0.0"


# ─── Error Classes ────────────────────────────────────────────────────────────


class SerializationError(Exception):
    """Base exception for serialization errors."""


class DeserializationError(SerializationError):
    """Error during deserialization."""


class SchemaVersionError(DeserializationError):
    """Schema version mismatch."""


class FeatureNotFoundError(DeserializationError):
    """Feature class not found during deserialization."""


class ValidationError(DeserializationError):
    """Parameter validation failed."""


# ─── Value Serialization ──────────────────────────────────────────────────────


def _serialize_value(value: Any) -> Any:
    """Serialize a parameter value to a YAML-compatible primitive."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, slice):
        return {"_type": "slice", "start": value.start, "stop": value.stop, "step": value.step}
    if callable(value):
        return {"_type": "callable", "format": "dill_hex", "data": dill.dumps(value).hex()}
    raise SerializationError(f"Cannot serialize value of type {type(value).__name__!r}: {value!r}")


def _annotation_is_tuple(annotation: Any) -> bool:
    """Return True if the annotation resolves to a tuple type."""
    if annotation is tuple:
        return True
    return typing.get_origin(annotation) is tuple


def _inner_annotation(annotation: Any, index: int) -> Any:
    """Return the element type annotation for a tuple at a given index."""
    args = typing.get_args(annotation)
    if not args:
        return Any
    # tuple[T, ...] — variable-length homogeneous
    if len(args) == 2 and args[1] is ...:
        return args[0]
    # tuple[T1, T2, ...] — fixed-length
    if index < len(args):
        return args[index]
    return Any


def _deserialize_value(value: Any, annotation: Any) -> Any:
    """Deserialize a parameter value, using the type annotation for reconstruction."""
    # Handle special dicts with _type markers
    if isinstance(value, dict) and "_type" in value:
        typ = value["_type"]
        if typ == "slice":
            return slice(value.get("start"), value.get("stop"), value.get("step"))
        if typ == "callable":
            return dill.loads(bytes.fromhex(value["data"]))
        raise DeserializationError(f"Unknown special type marker: {typ!r}")

    # Reconstruct tuple from list using type annotation
    if isinstance(value, list) and _annotation_is_tuple(annotation):
        return tuple(
            _deserialize_value(v, _inner_annotation(annotation, i)) for i, v in enumerate(value)
        )

    return value


# ─── Feature → Dict ───────────────────────────────────────────────────────────


def _dataclass_to_params(obj: Any) -> dict[str, Any]:
    """Serialize all dataclass instance fields to a params dict."""
    return {
        field.name: _serialize_value(getattr(obj, field.name)) for field in dataclasses.fields(obj)
    }


def _feature_to_step(feature: Any) -> dict[str, Any]:
    """Serialize a single pipeline step (plain feature or Chord)."""
    from cobrabox.base_feature import Chord

    if isinstance(feature, Chord):
        return _chord_to_step(feature)
    cls = type(feature)
    return {
        "class": cls.__name__,
        "module": cls.__module__,
        "params": _dataclass_to_params(feature),
    }


def _chord_to_step(chord: Any) -> dict[str, Any]:
    """Serialize a Chord to a ChordDefinition dict (used as a pipeline step)."""
    from cobrabox.base_feature import Pipeline

    # Inner pipeline: Pipeline or single BaseFeature
    if isinstance(chord.pipeline, Pipeline):
        inner = [_feature_to_step(f) for f in chord.pipeline.features]
    else:
        inner = [_feature_to_step(chord.pipeline)]

    split_cls = type(chord.split)
    agg_cls = type(chord.aggregate)

    return {
        "split": {
            "class": split_cls.__name__,
            "module": split_cls.__module__,
            "params": _dataclass_to_params(chord.split),
        },
        "pipeline": inner,
        "aggregate": {
            "class": agg_cls.__name__,
            "module": agg_cls.__module__,
            "params": _dataclass_to_params(chord.aggregate),
        },
    }


def _build_document(obj: Any) -> dict[str, Any]:
    """Build the full serialization document dict from a feature or pipeline."""
    from cobrabox.base_feature import Pipeline

    if isinstance(obj, Pipeline):
        steps = [_feature_to_step(f) for f in obj.features]
    else:
        steps = [_feature_to_step(obj)]

    return {"cobrabox_version": _cb_version(), "schema_version": SCHEMA_VERSION, "pipeline": steps}


# ─── Dict → Feature ───────────────────────────────────────────────────────────


def _resolve_class(class_name: str, module_path: str) -> type:
    """Import and return a feature class by name and module path."""
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise FeatureNotFoundError(
            f"Cannot import module {module_path!r} for feature {class_name!r}: {exc}"
        ) from exc

    cls = getattr(module, class_name, None)
    if cls is None:
        available = [
            name
            for name, obj in vars(module).items()
            if isinstance(obj, type) and getattr(obj, "_is_cobrabox_feature", False)
        ]
        raise FeatureNotFoundError(
            f"Feature {class_name!r} not found in module {module_path!r}. "
            f"Available features: {available}"
        )
    if not getattr(cls, "_is_cobrabox_feature", False):
        raise FeatureNotFoundError(
            f"Class {class_name!r} in {module_path!r} is not a cobrabox feature "
            f"(missing _is_cobrabox_feature = True)."
        )
    return cls


def _instantiate(cls: type, params: dict[str, Any]) -> Any:
    """Reconstruct a dataclass instance from its class and params dict."""
    if not dataclasses.is_dataclass(cls):
        raise ValidationError(f"Class {cls.__name__!r} is not a dataclass.")

    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {}

    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.name not in params:
            # Missing — rely on dataclass default; raise if no default exists
            no_default = (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            )
            if no_default:
                raise ValidationError(
                    f"Feature {cls.__name__!r} missing required parameter {field.name!r}. "
                    f"Expected type: {hints.get(field.name, 'unknown')}"
                )
            continue
        annotation = hints.get(field.name, Any)
        kwargs[field.name] = _deserialize_value(params[field.name], annotation)

    return cls(**kwargs)


def _deserialize_feature_dict(feat_dict: dict[str, Any]) -> Any:
    """Deserialize a FeatureDefinition dict into a feature instance."""
    cls = _resolve_class(feat_dict["class"], feat_dict["module"])
    return _instantiate(cls, feat_dict.get("params", {}))


def _deserialize_chord_dict(chord_dict: dict[str, Any]) -> Any:
    """Deserialize a ChordDefinition dict into a Chord instance."""
    from cobrabox.base_feature import Chord, Pipeline

    split = _deserialize_feature_dict(chord_dict["split"])
    aggregate = _deserialize_feature_dict(chord_dict["aggregate"])
    inner_steps = [_deserialize_step(s) for s in chord_dict["pipeline"]]

    inner: Any = inner_steps[0] if len(inner_steps) == 1 else Pipeline(*inner_steps)
    return Chord(split=split, pipeline=inner, aggregate=aggregate)


def _deserialize_step(step_dict: dict[str, Any]) -> Any:
    """Deserialize a pipeline step — either a ChordDefinition or FeatureDefinition."""
    if {"split", "pipeline", "aggregate"} <= step_dict.keys():
        return _deserialize_chord_dict(step_dict)
    return _deserialize_feature_dict(step_dict)


# ─── Version Checking ─────────────────────────────────────────────────────────


def _check_versions(doc: dict[str, Any]) -> None:
    """Validate version fields in a parsed document."""
    cb_ver = doc.get("cobrabox_version")
    schema_ver = doc.get("schema_version")

    if cb_ver is None or schema_ver is None:
        raise DeserializationError(
            "Missing version fields. Both 'cobrabox_version' and 'schema_version' are required."
        )

    # Schema version: major must not exceed current
    try:
        schema_major = int(str(schema_ver).split(".")[0])
        current_major = int(SCHEMA_VERSION.split(".")[0])
    except ValueError, IndexError:
        raise SchemaVersionError(f"Unreadable schema_version: {schema_ver!r}") from None

    if schema_major > current_major:
        raise SchemaVersionError(
            f"Schema version {schema_ver!r} is not compatible with this installation "
            f"(max supported: {SCHEMA_VERSION!r}). "
            "Please upgrade cobrabox or regenerate the file."
        )

    # cobrabox_version: warn if major differs
    try:
        file_major = int(str(cb_ver).split(".")[0])
        installed_major = int(_cb_version().split(".")[0])
    except ValueError, IndexError:
        return

    if file_major != installed_major:
        warnings.warn(
            f"File was created with cobrabox {cb_ver!r} but installed version is "
            f"{_cb_version()!r}. Some parameters may behave differently.",
            UserWarning,
            stacklevel=3,
        )


# ─── Document Parsing ─────────────────────────────────────────────────────────


def _parse_document(doc: dict[str, Any]) -> Any:
    """Parse a document dict and return a Pipeline."""
    from cobrabox.base_feature import Pipeline

    if "pipeline" not in doc:
        raise DeserializationError(
            "Invalid cobrabox document: missing required 'pipeline' key. "
            "Only 'pipeline' is supported as the top-level content key."
        )
    pipeline_list = doc["pipeline"]
    if not isinstance(pipeline_list, list) or len(pipeline_list) == 0:
        raise DeserializationError("'pipeline' must be a non-empty list.")

    steps = [_deserialize_step(s) for s in pipeline_list]
    return Pipeline(*steps)


# ─── Public API ───────────────────────────────────────────────────────────────


def serialize(obj: Any, fmt: Literal["yaml", "json"] = "yaml") -> str:
    """Serialize a feature or pipeline to a string.

    Args:
        obj: Feature, Pipeline, or Chord to serialize.
        fmt: Output format — ``"yaml"`` (default) or ``"json"``.

    Returns:
        String representation of the pipeline.
    """
    doc = _build_document(obj)
    if fmt == "json":
        return json.dumps(doc, indent=2)
    return yaml.dump(doc, default_flow_style=False, sort_keys=False, allow_unicode=True)


def deserialize(content: str, fmt: Literal["yaml", "json"] = "yaml") -> Any:
    """Deserialize a pipeline from a content string.

    Args:
        content: Raw YAML or JSON string. Pass a file path to :func:`load` instead.
        fmt: Input format — ``"yaml"`` (default) or ``"json"``.

    Returns:
        :class:`~cobrabox.Pipeline` reconstructed from the content string.
    """
    if fmt == "json":
        doc = json.loads(content)
    else:
        doc = yaml.safe_load(content)
    _check_versions(doc)
    return _parse_document(doc)


def save(obj: Any, path: str | Path, fmt: Literal["yaml", "json"] | None = None) -> None:
    """Save a feature or pipeline to a file.

    Args:
        obj: Feature, Pipeline, or Chord to save.
        path: Destination file path.
        fmt: File format. If ``None``, inferred from the extension
            (``.yaml``/``.yml`` → yaml, ``.json`` → json).
    """
    path = Path(path)
    if fmt is None:
        ext = path.suffix.lower()
        if ext in (".yaml", ".yml"):
            fmt = "yaml"
        elif ext == ".json":
            fmt = "json"
        else:
            raise SerializationError(
                f"Cannot infer format from extension {ext!r}. "
                "Pass fmt='yaml' or fmt='json' explicitly."
            )
    path.write_text(serialize(obj, fmt=fmt), encoding="utf-8")


def load(path: str | Path) -> Any:
    """Load a pipeline from a file.

    Format is auto-detected from the file extension (``.yaml``/``.yml`` or ``.json``).

    Args:
        path: Source file path.

    Returns:
        :class:`~cobrabox.Pipeline` loaded from the file.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        fmt: Literal["yaml", "json"] = "yaml"
    elif ext == ".json":
        fmt = "json"
    else:
        raise SerializationError(
            f"Cannot infer format from extension {ext!r}. Supported extensions: .yaml, .yml, .json"
        )
    return deserialize(path.read_text(encoding="utf-8"), fmt=fmt)
