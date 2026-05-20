"""Pipeline serialization demo. From project root: python examples/serialization_demo.py"""

import tempfile
from pathlib import Path

import cobrabox as cb

data = cb.load_dataset("dummy_chain")[0]

# ─── Single feature ───────────────────────────────────────────────────────────

feature = cb.feature.LineLength()

yaml_str = cb.serialize(feature)
print("=== Single feature YAML ===")
print(yaml_str)

restored = cb.deserialize(yaml_str)
print("Restored pipeline:", restored)
print("Output shape:", restored.apply(data).data.shape)

# ─── Pipeline ─────────────────────────────────────────────────────────────────

pipeline = (
    cb.feature.SlidingWindow(window_size=10, step_size=5)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
)

yaml_str = cb.serialize(pipeline)
print("\n=== Chord YAML ===")
print(yaml_str)

restored_pipeline = cb.deserialize(yaml_str)
out_original = pipeline.apply(data)
out_restored = restored_pipeline.apply(data)
print("Original output shape:", out_original.data.shape)
print("Restored output shape:", out_restored.data.shape)
print("Outputs match:", (out_original.to_numpy() == out_restored.to_numpy()).all())

# ─── JSON format ──────────────────────────────────────────────────────────────

json_str = cb.serialize(pipeline, fmt="json")
print("\n=== JSON format (first 200 chars) ===")
print(json_str[:200], "...")

restored_from_json = cb.deserialize(json_str, fmt="json")
print("Restored from JSON — history:", restored_from_json.apply(data).history)

# ─── File I/O ─────────────────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    yaml_path = Path(tmp) / "pipeline.yaml"
    json_path = Path(tmp) / "pipeline.json"

    cb.save(pipeline, yaml_path)
    cb.save(pipeline, json_path)

    loaded_yaml = cb.load(yaml_path)
    loaded_json = cb.load(json_path)

    print("\n=== File I/O ===")
    print("Loaded from .yaml — type:", type(loaded_yaml).__name__)
    print("Loaded from .json — type:", type(loaded_json).__name__)
    print("YAML file contents:\n", yaml_path.read_text())

# ─── Method API ───────────────────────────────────────────────────────────────

chord = (
    cb.feature.SlidingWindow(window_size=10, step_size=5)
    | cb.feature.LineLength()
    | cb.feature.MeanAggregate()
)

yaml_via_method = chord.to_yaml()
restored_via_method = cb.Pipeline.from_yaml(yaml_via_method)

print("=== Method API ===")
print("chord.to_yaml() round-trip — history:", restored_via_method.apply(data).history)

d = chord.to_dict()
restored_from_dict = cb.Pipeline.from_dict(d)
print("chord.to_dict() round-trip — history:", restored_from_dict.apply(data).history)

# ─── D&D alignment of a saved pipeline ───────────────────────────────────────

from cobrabox.egg.dnd_alignment import main as dnd_main  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    yaml_path = Path(tmp) / "chord.yaml"
    cb.save(chord, yaml_path)

    print("\n=== D&D alignment of saved pipeline ===")
    dnd_main([str(yaml_path)])
