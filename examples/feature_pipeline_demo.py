"""Minimal example. From project root: python examples/feature_pipeline_demo.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cobrabox as cb

data = cb.dataset("dummy_chain")[0]
wdata = cb.feature.sliding_window(data)
feat = cb.feature.line_length(wdata)
dummy = cb.feature.dummy(feat)

print("feature.data.shape:", feat.data.shape)
print("feature.history:", feat.history)

print("dummy", dummy.data)