"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed random generator for deterministic tests."""
    return np.random.default_rng(42)
