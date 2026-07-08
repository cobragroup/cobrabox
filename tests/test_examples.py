"""Smoke tests: every script in examples/ must run to completion.

These guard against examples silently rotting when the public API changes
(e.g. a feature rename or signature change). Each example runs in its own
subprocess; a non-zero exit code fails the test with the captured output.

Examples that download or read real datasets are marked ``slow`` so the
default suite stays hermetic; run them with ``pytest -m slow``.

Subprocesses run with UTF-8 I/O forced (``PYTHONUTF8``/``PYTHONIOENCODING``).
The examples print Unicode — plain ``->`` arrows plus the D&D-alignment "egg",
which renders a rich grid with ``★`` and box-drawing. On a non-UTF-8 console
(e.g. Windows cp1252) that raises ``UnicodeEncodeError``, which is a terminal-
encoding concern, not an example bug — the examples and docs assume a UTF-8
terminal. Forcing UTF-8 keeps this check about whether each example *runs*
(catching API drift) rather than about the host locale.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

# Examples that need network access or real dataset files on disk.
_NETWORK_EXAMPLES = {
    "dataset_loader_demo.py",
    "realistic_dataset_loader_demo.py",
    "remote_datasets_demo.py",
}


def _example_params() -> list:
    params = []
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        marks = [pytest.mark.slow] if path.name in _NETWORK_EXAMPLES else []
        params.append(pytest.param(path, id=path.name, marks=marks))
    return params


@pytest.mark.parametrize("example_path", _example_params())
def test_example_runs(example_path: Path) -> None:
    """Running the example script exits cleanly (rc == 0)."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, str(example_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{example_path.name} exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
