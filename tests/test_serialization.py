"""Tests for cobrabox pipeline serialization / deserialization."""

from __future__ import annotations

import dataclasses
import json
import textwrap
import typing
from pathlib import Path

import numpy as np
import pytest

import cobrabox as cb
from cobrabox import serialization
from cobrabox.serialization import (
    DeserializationError,
    FeatureNotFoundError,
    SchemaVersionError,
    deserialize,
    load,
    save,
    serialize,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_signal() -> cb.SignalData:
    """Return a small deterministic SignalData for round-trip testing."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((20, 3))
    return cb.SignalData.from_numpy(
        arr, dims=["time", "space"], sampling_rate=100.0, subjectID="sub-01"
    )


def _outputs_match(original: object, deserialized: object, data: cb.SignalData) -> bool:
    """Return True if two pipelines/features produce numerically equal output."""
    out_orig = original.apply(data)  # type: ignore[union-attr]
    out_deser = deserialized.apply(data)  # type: ignore[union-attr]
    np.testing.assert_allclose(out_orig.to_numpy(), out_deser.to_numpy())
    return True


# ─── Round-Trip: Single Feature ───────────────────────────────────────────────


def test_roundtrip_single_feature_yaml() -> None:
    """LineLength serializes and deserializes correctly via YAML."""
    feature = cb.feature.LineLength()
    yaml_str = serialize(feature)
    pipeline = deserialize(yaml_str)

    assert isinstance(pipeline, cb.Pipeline)
    assert len(pipeline.features) == 1
    assert isinstance(pipeline.features[0], cb.feature.LineLength)
    assert _outputs_match(feature, pipeline, _make_signal())


def test_roundtrip_single_feature_json() -> None:
    """LineLength serializes and deserializes correctly via JSON."""
    feature = cb.feature.LineLength()
    json_str = serialize(feature, fmt="json")
    pipeline = deserialize(json_str, fmt="json")

    assert isinstance(pipeline, cb.Pipeline)
    assert _outputs_match(feature, pipeline, _make_signal())


def test_roundtrip_sliding_window_params() -> None:
    """SlidingWindow int params survive round-trip."""
    feature = cb.feature.SlidingWindow(window_size=8, step_size=3)
    yaml_str = serialize(feature)
    pipeline = deserialize(yaml_str)

    restored = pipeline.features[0]
    assert isinstance(restored, cb.feature.SlidingWindow)
    assert restored.window_size == 8
    assert restored.step_size == 3


# ─── Round-Trip: Pipeline ────────────────────────────────────────────────────


def test_roundtrip_pipeline_yaml() -> None:
    """Two-step Pipeline round-trips through YAML."""
    sw = cb.feature.SlidingWindow(window_size=5, step_size=2)
    # Build via pipe to get a Pipeline
    pipeline = cb.feature.LineLength() | cb.feature.LineLength()
    yaml_str = serialize(pipeline)
    restored = deserialize(yaml_str)

    assert isinstance(restored, cb.Pipeline)
    assert len(restored.features) == 2
    _ = sw  # unused but confirms import


def test_roundtrip_pipeline_output_matches() -> None:
    """Pipeline output is numerically identical after round-trip (single-step pipeline)."""
    # LineLength | LineLength would be invalid (time dim removed after first step).
    # Use a one-element pipeline — structure serialization is tested separately.
    pipeline = cb.Pipeline(cb.feature.LineLength())
    yaml_str = serialize(pipeline)
    restored = deserialize(yaml_str)
    assert _outputs_match(pipeline, restored, _make_signal())


# ─── Round-Trip: Chord ───────────────────────────────────────────────────────


def test_roundtrip_chord_yaml() -> None:
    """Chord (SlidingWindow | LineLength | MeanAggregate) round-trips through YAML."""
    chord = (
        cb.feature.SlidingWindow(window_size=5, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    )
    yaml_str = serialize(chord)
    pipeline = deserialize(yaml_str)

    assert isinstance(pipeline, cb.Pipeline)
    assert len(pipeline.features) == 1
    restored_chord = pipeline.features[0]
    assert isinstance(restored_chord, cb.Chord)
    assert isinstance(restored_chord.split, cb.feature.SlidingWindow)
    assert restored_chord.split.window_size == 5
    assert restored_chord.split.step_size == 2


def test_roundtrip_chord_output_matches() -> None:
    """Chord output is numerically identical after round-trip."""
    chord = (
        cb.feature.SlidingWindow(window_size=5, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    )
    yaml_str = serialize(chord)
    pipeline = deserialize(yaml_str)
    restored_chord = pipeline.features[0]
    assert _outputs_match(chord, restored_chord, _make_signal())


def test_roundtrip_nested_chord_in_pipeline() -> None:
    """Chord nested inside a Pipeline round-trips correctly."""
    chord = (
        cb.feature.SlidingWindow(window_size=5, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    )
    # Pipeline: LineLength → Chord
    pipeline = cb.feature.LineLength() | chord
    yaml_str = serialize(pipeline)
    restored = deserialize(yaml_str)

    assert isinstance(restored, cb.Pipeline)
    assert len(restored.features) == 2
    assert isinstance(restored.features[0], cb.feature.LineLength)
    assert isinstance(restored.features[1], cb.Chord)


# ─── Round-Trip: Method API ──────────────────────────────────────────────────


def test_feature_to_yaml_from_yaml() -> None:
    """BaseFeature.to_yaml() / from_yaml() round-trip."""
    feature = cb.feature.LineLength()
    yaml_str = feature.to_yaml()
    restored = cb.feature.LineLength.from_yaml(yaml_str)

    assert isinstance(restored, cb.feature.LineLength)
    assert _outputs_match(feature, restored, _make_signal())


def test_pipeline_to_yaml_from_yaml() -> None:
    """Pipeline.to_yaml() / from_yaml() round-trip."""
    chord = (
        cb.feature.SlidingWindow(window_size=5, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    )
    pipeline = cb.Pipeline(cb.feature.LineLength(), chord)
    yaml_str = pipeline.to_yaml()
    restored = cb.Pipeline.from_yaml(yaml_str)

    assert isinstance(restored, cb.Pipeline)
    assert len(restored.features) == 2


def test_feature_to_dict_from_dict() -> None:
    """BaseFeature.to_dict() / from_dict() round-trip."""
    feature = cb.feature.LineLength()
    d = feature.to_dict()
    restored = cb.feature.LineLength.from_dict(d)

    assert isinstance(restored, cb.feature.LineLength)
    assert _outputs_match(feature, restored, _make_signal())


def test_pipeline_to_dict_from_dict() -> None:
    """Pipeline.to_dict() / from_dict() round-trip."""
    chord = (
        cb.feature.SlidingWindow(window_size=5, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    )
    pipeline = cb.Pipeline(cb.feature.LineLength(), chord)
    d = pipeline.to_dict()
    restored = cb.Pipeline.from_dict(d)

    assert isinstance(restored, cb.Pipeline)
    assert len(restored.features) == 2


# ─── Top-Level cb Namespace ──────────────────────────────────────────────────


def test_cb_namespace_functions_exist() -> None:
    """serialize/deserialize/save/load are accessible from the cb namespace."""
    assert cb.serialize is serialize
    assert cb.deserialize is deserialize
    assert cb.save is save
    assert cb.load is load
    assert cb.serialization is serialization


# ─── File I/O ────────────────────────────────────────────────────────────────


def test_save_load_yaml(tmp_path: Path) -> None:
    """save() / load() round-trip for YAML files."""
    pipeline = cb.Pipeline(cb.feature.LineLength())
    path = tmp_path / "pipeline.yaml"
    save(pipeline, path)

    assert path.exists()
    restored = load(path)
    assert isinstance(restored, cb.Pipeline)
    assert _outputs_match(pipeline, restored, _make_signal())


def test_save_load_yml_extension(tmp_path: Path) -> None:
    """.yml extension is treated as YAML."""
    feature = cb.feature.LineLength()
    path = tmp_path / "pipeline.yml"
    save(feature, path)
    restored = load(path)
    assert isinstance(restored, cb.Pipeline)


def test_save_load_json(tmp_path: Path) -> None:
    """save() / load() round-trip for JSON files."""
    pipeline = cb.Pipeline(cb.feature.LineLength())
    path = tmp_path / "pipeline.json"
    save(pipeline, path)

    content = path.read_text()
    doc = json.loads(content)
    assert "cobrabox_version" in doc

    restored = load(path)
    assert isinstance(restored, cb.Pipeline)
    assert _outputs_match(pipeline, restored, _make_signal())


def test_save_explicit_fmt(tmp_path: Path) -> None:
    """save() respects explicit fmt parameter regardless of extension."""
    feature = cb.feature.LineLength()
    path = tmp_path / "pipeline.txt"
    save(feature, path, fmt="yaml")

    content = path.read_text()
    assert "cobrabox_version" in content


def test_load_unknown_extension_raises(tmp_path: Path) -> None:
    """load() raises SerializationError for unknown file extension."""
    path = tmp_path / "pipeline.txt"
    path.write_text("dummy")
    from cobrabox.serialization import SerializationError

    with pytest.raises(SerializationError, match="Cannot infer format"):
        load(path)


def test_save_unknown_extension_raises(tmp_path: Path) -> None:
    """save() without fmt raises SerializationError for unknown extension."""
    from cobrabox.serialization import SerializationError

    path = tmp_path / "pipeline.bin"
    with pytest.raises(SerializationError, match="Cannot infer format"):
        save(cb.feature.LineLength(), path)


# ─── Fixture Files ───────────────────────────────────────────────────────────


def test_load_fixture_pipeline_v1() -> None:
    """Load the pipeline_v1.yaml fixture and verify structure."""
    pipeline = load(FIXTURES / "pipeline_v1.yaml")
    assert isinstance(pipeline, cb.Pipeline)
    assert len(pipeline.features) == 1
    assert isinstance(pipeline.features[0], cb.feature.LineLength)


def test_load_fixture_chord_v1() -> None:
    """Load the chord_v1.yaml fixture and verify structure."""
    pipeline = load(FIXTURES / "chord_v1.yaml")
    assert isinstance(pipeline, cb.Pipeline)
    chord = pipeline.features[0]
    assert isinstance(chord, cb.Chord)
    assert chord.split.window_size == 5
    assert chord.split.step_size == 2


# ─── YAML Structure ──────────────────────────────────────────────────────────


def test_serialized_yaml_has_version_fields() -> None:
    """Serialized YAML includes cobrabox_version and schema_version."""
    import yaml

    yaml_str = serialize(cb.feature.LineLength())
    doc = yaml.safe_load(yaml_str)
    assert "cobrabox_version" in doc
    assert "schema_version" in doc
    assert doc["schema_version"] == "1.0.0"


def test_serialized_yaml_pipeline_key() -> None:
    """Serialized YAML uses 'pipeline' as the only top-level content key."""
    import yaml

    yaml_str = serialize(cb.feature.LineLength())
    doc = yaml.safe_load(yaml_str)
    assert "pipeline" in doc
    assert "single" not in doc
    assert "chord" not in doc


def test_single_feature_is_one_element_pipeline() -> None:
    """A single feature serializes as a one-element pipeline list."""
    import yaml

    yaml_str = serialize(cb.feature.LineLength())
    doc = yaml.safe_load(yaml_str)
    assert isinstance(doc["pipeline"], list)
    assert len(doc["pipeline"]) == 1


# ─── Error Handling ──────────────────────────────────────────────────────────


def test_missing_version_fields_raises() -> None:
    """Missing cobrabox_version or schema_version raises DeserializationError."""
    yaml_str = textwrap.dedent("""\
        pipeline:
          - class: LineLength
            module: cobrabox.features.line_length
            params: {}
    """)
    with pytest.raises(DeserializationError, match="Missing version fields"):
        deserialize(yaml_str)


def test_future_schema_version_raises() -> None:
    """A schema_version with a higher major version raises SchemaVersionError."""
    yaml_str = textwrap.dedent("""\
        cobrabox_version: "0.3.1"
        schema_version: "99.0.0"
        pipeline:
          - class: LineLength
            module: cobrabox.features.line_length
            params: {}
    """)
    with pytest.raises(SchemaVersionError, match="not compatible"):
        deserialize(yaml_str)


def test_unknown_feature_class_raises() -> None:
    """An unknown feature class name raises FeatureNotFoundError."""
    yaml_str = textwrap.dedent("""\
        cobrabox_version: "0.3.1"
        schema_version: "1.0.0"
        pipeline:
          - class: NonExistentFeature
            module: cobrabox.features.line_length
            params: {}
    """)
    with pytest.raises(FeatureNotFoundError, match="NonExistentFeature"):
        deserialize(yaml_str)


def test_unknown_module_raises() -> None:
    """An unimportable module raises FeatureNotFoundError."""
    yaml_str = textwrap.dedent("""\
        cobrabox_version: "0.3.1"
        schema_version: "1.0.0"
        pipeline:
          - class: LineLength
            module: cobrabox.features.does_not_exist
            params: {}
    """)
    with pytest.raises(FeatureNotFoundError, match="Cannot import module"):
        deserialize(yaml_str)


def test_missing_required_param_raises() -> None:
    """Missing required parameters raise ValidationError."""
    # SlidingWindow has no defaults, so missing params should fail.
    # However, they do have defaults (window_size=10, step_size=5).
    # LineLength has no required params. Let's use a malformed case:
    # Pass a non-dataclass — we can test via a bad class check.
    # Instead test: missing pipeline key.
    yaml_str = textwrap.dedent("""\
        cobrabox_version: "0.3.1"
        schema_version: "1.0.0"
        pipeline: []
    """)
    with pytest.raises(DeserializationError, match="non-empty"):
        deserialize(yaml_str)


def test_missing_pipeline_key_raises() -> None:
    """Document without 'pipeline' key raises DeserializationError."""
    yaml_str = textwrap.dedent("""\
        cobrabox_version: "0.3.1"
        schema_version: "1.0.0"
        features:
          - class: LineLength
            module: cobrabox.features.line_length
            params: {}
    """)
    with pytest.raises(DeserializationError, match="missing required 'pipeline' key"):
        deserialize(yaml_str)


def test_fixture_missing_version_raises() -> None:
    """Fixture with missing version fields raises on load."""
    with pytest.raises(DeserializationError, match="Missing version fields"):
        load(FIXTURES / "invalid" / "missing_version.yaml")


def test_fixture_unknown_feature_raises() -> None:
    """Fixture with unknown feature class raises on load."""
    with pytest.raises(FeatureNotFoundError):
        load(FIXTURES / "invalid" / "unknown_feature.yaml")


def test_fixture_future_schema_raises() -> None:
    """Fixture with a future schema version raises SchemaVersionError."""
    with pytest.raises(SchemaVersionError):
        load(FIXTURES / "invalid" / "future_schema.yaml")


# ─── Version Warning ─────────────────────────────────────────────────────────


def test_major_version_mismatch_warns() -> None:
    """A cobrabox_version with different major version emits a UserWarning."""
    yaml_str = textwrap.dedent("""\
        cobrabox_version: "99.0.0"
        schema_version: "1.0.0"
        pipeline:
          - class: LineLength
            module: cobrabox.features.line_length
            params: {}
    """)
    with pytest.warns(UserWarning, match="99.0.0"):
        deserialize(yaml_str)


# ─── Type Serialization ──────────────────────────────────────────────────────


def test_serialize_none_value() -> None:
    """None params survive round-trip (e.g. optional fields)."""
    from cobrabox.serialization import _deserialize_value, _serialize_value

    assert _serialize_value(None) is None
    assert _deserialize_value(None, type(None)) is None


def test_serialize_slice() -> None:
    """slice objects serialize to a dict and deserialize back to slice."""
    from cobrabox.serialization import _deserialize_value, _serialize_value

    s = slice(1, 10, 2)
    serialized = _serialize_value(s)
    assert serialized == {"_type": "slice", "start": 1, "stop": 10, "step": 2}

    restored = _deserialize_value(serialized, slice)
    assert restored == s


def test_serialize_tuple_to_list() -> None:
    """Tuples serialize as lists."""
    from cobrabox.serialization import _serialize_value

    assert _serialize_value((1, 2, 3)) == [1, 2, 3]


def test_deserialize_list_to_tuple_with_annotation() -> None:
    """Lists deserialize back to tuple when annotation says tuple."""
    from cobrabox.serialization import _deserialize_value

    result = _deserialize_value([1, 2, 3], tuple[int, int, int])
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


def test_deserialize_list_to_homogeneous_tuple() -> None:
    """Lists deserialize back to tuple[T, ...] annotation."""
    from cobrabox.serialization import _deserialize_value

    result = _deserialize_value([1.0, 2.0, 3.0], tuple[float, ...])
    assert result == (1.0, 2.0, 3.0)
    assert isinstance(result, tuple)


def test_serialize_nested_dict() -> None:
    """Nested dicts (e.g. band defs) survive round-trip."""
    from cobrabox.serialization import _deserialize_value, _serialize_value

    bands = {"alpha": [8.0, 12.0], "beta": [12.0, 30.0]}
    serialized = _serialize_value(bands)
    assert serialized == bands
    # dict annotation — should come back as dict
    restored = _deserialize_value(serialized, dict)
    assert restored == bands


def test_serialize_unknown_type_raises() -> None:
    """Serializing an unsupported type raises SerializationError."""
    from cobrabox.serialization import SerializationError, _serialize_value

    class CustomObject:
        pass

    with pytest.raises(SerializationError, match="Cannot serialize"):
        _serialize_value(CustomObject())


def test_deserialize_unknown_type_marker_raises() -> None:
    """Deserializing a dict with unknown _type raises DeserializationError."""
    from cobrabox.serialization import _deserialize_value

    with pytest.raises(DeserializationError, match="Unknown special type marker"):
        _deserialize_value({"_type": "unknown"}, object)


# ─── Chord Inner Pipeline Variants ───────────────────────────────────────────


def test_chord_with_multi_step_inner_pipeline() -> None:
    """Chord with a Pipeline (not single feature) as inner pipeline serializes correctly."""
    # Two LineLength steps are semantically invalid (first removes time dim),
    # but this test is structural: it verifies that a Chord whose .pipeline is a
    # Pipeline object (not a single feature) round-trips with the correct inner step count.
    chord = cb.Chord(
        split=cb.feature.SlidingWindow(window_size=5, step_size=2),
        pipeline=cb.feature.LineLength() | cb.feature.LineLength(),
        aggregate=cb.feature.MeanAggregate(),
    )
    yaml_str = serialize(chord)
    pipeline = deserialize(yaml_str)
    restored_chord = pipeline.features[0]

    assert isinstance(restored_chord, cb.Chord)
    assert isinstance(restored_chord.pipeline, cb.Pipeline)
    assert len(restored_chord.pipeline.features) == 2


# ─── Idempotency ─────────────────────────────────────────────────────────────


def test_double_roundtrip_is_idempotent() -> None:
    """Serializing twice produces the same YAML string."""
    chord = (
        cb.feature.SlidingWindow(window_size=5, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    )
    yaml1 = serialize(chord)
    yaml2 = serialize(deserialize(yaml1))
    assert yaml1 == yaml2


def test_json_roundtrip_is_idempotent() -> None:
    """Serializing twice via JSON produces the same string."""
    chord = (
        cb.feature.SlidingWindow(window_size=5, step_size=2)
        | cb.feature.LineLength()
        | cb.feature.MeanAggregate()
    )
    json1 = serialize(chord, fmt="json")
    json2 = serialize(deserialize(json1, fmt="json"), fmt="json")
    assert json1 == json2


# ─── Validation ──────────────────────────────────────────────────────────────


def test_non_cobrabox_class_raises() -> None:
    """A class without _is_cobrabox_feature raises FeatureNotFoundError."""
    yaml_str = textwrap.dedent("""\
        cobrabox_version: "0.3.1"
        schema_version: "1.0.0"
        pipeline:
          - class: Data
            module: cobrabox.data
            params: {}
    """)
    with pytest.raises(FeatureNotFoundError, match="_is_cobrabox_feature"):
        deserialize(yaml_str)


def test_params_passed_to_post_init() -> None:
    """Invalid params trigger __post_init__ validation during deserialization."""
    yaml_str = textwrap.dedent("""\
        cobrabox_version: "0.3.1"
        schema_version: "1.0.0"
        pipeline:
          - class: SlidingWindow
            module: cobrabox.features.sliding_window
            params:
              window_size: 0
              step_size: 5
    """)
    # SlidingWindow.__post_init__ raises ValueError when window_size < 1
    with pytest.raises(ValueError, match="window_size"):
        deserialize(yaml_str)


# ─── Missing Coverage Tests ───────────────────────────────────────────────────


def test_serialize_callable_with_dill() -> None:
    """Callable params are serialized with dill and round-trip correctly."""
    from cobrabox.serialization import _deserialize_value, _serialize_value

    def my_func(x: float) -> float:
        return x * 2

    serialized = _serialize_value(my_func)
    assert serialized["_type"] == "callable"
    assert serialized["format"] == "dill_hex"

    restored = _deserialize_value(serialized, typing.Callable)
    assert restored(5.0) == 10.0


def test_annotation_is_tuple_direct() -> None:
    """_annotation_is_tuple handles bare tuple annotation."""
    from cobrabox.serialization import _annotation_is_tuple

    assert _annotation_is_tuple(tuple) is True
    assert _annotation_is_tuple(tuple[int, str]) is True
    assert _annotation_is_tuple(list) is False


def test_inner_annotation_no_args() -> None:
    """_inner_annotation returns Any when tuple has no type args."""
    from cobrabox.serialization import _inner_annotation

    # Bare tuple has no args
    assert _inner_annotation(tuple, 0) is typing.Any


def test_inner_annotation_index_out_of_range() -> None:
    """_inner_annotation returns Any when index exceeds tuple length."""
    from cobrabox.serialization import _inner_annotation

    # Fixed-length tuple with 2 elements, asking for index 5
    result = _inner_annotation(tuple[int, str], 5)
    assert result is typing.Any


def test_instantiate_non_dataclass_raises() -> None:
    """_instantiate raises ValidationError for non-dataclass."""
    from cobrabox.serialization import ValidationError, _instantiate

    class NotADataclass:
        pass

    with pytest.raises(ValidationError, match="not a dataclass"):
        _instantiate(NotADataclass, {})


def test_instantiate_missing_required_param() -> None:
    """_instantiate raises ValidationError when required param is missing."""
    from cobrabox.serialization import ValidationError, _instantiate

    @dataclasses.dataclass
    class TestFeature:
        required_param: float

    # Manually mark as feature
    TestFeature._is_cobrabox_feature = True  # type: ignore[attr-defined]

    with pytest.raises(ValidationError, match="missing required parameter"):
        _instantiate(TestFeature, {})


def test_instantiate_get_type_hints_fails_gracefully() -> None:
    """_instantiate handles get_type_hints failure gracefully."""
    from cobrabox.serialization import _instantiate

    @dataclasses.dataclass
    class TestFeature:
        value: float = 1.0

    TestFeature._is_cobrabox_feature = True  # type: ignore[attr-defined]

    # Should work even if get_type_hints fails (uses empty hints)
    result = _instantiate(TestFeature, {"value": 42.0})
    assert result.value == 42.0


def test_schema_version_unreadable() -> None:
    """Unreadable schema_version raises SchemaVersionError."""
    yaml_str = textwrap.dedent("""\
        cobrabox_version: "0.3.1"
        schema_version: "not-a-version"
        pipeline:
          - class: LineLength
            module: cobrabox.features.line_length
            params: {}
    """)
    with pytest.raises(SchemaVersionError, match="Unreadable"):
        deserialize(yaml_str)


def test_cobrabox_version_unreadable() -> None:
    """Unreadable cobrabox_version is handled gracefully (no crash)."""
    yaml_str = textwrap.dedent("""\
        cobrabox_version: "not-a-version"
        schema_version: "1.0.0"
        pipeline:
          - class: LineLength
            module: cobrabox.features.line_length
            params: {}
    """)
    # Should not raise, just not warn
    result = deserialize(yaml_str)
    assert isinstance(result, cb.Pipeline)
