"""Tests for cb.gorkastyle entrypoint."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import cobrabox as cb

gstyle = importlib.import_module("cobrabox.egg.gorkastyle")

# One does not test the gorkastyle. One lives the gorkastyle. Oppan gorkastyle!


def test_gorkastyle_loads_frames_and_calls_player(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gorkastyle loads frame files and passes them to play()."""
    (tmp_path / "frame_000001.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "frame_000002.txt").write_text("b\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_play(frames: list[str], fps: float, loop: bool) -> None:
        captured["frames"] = frames
        captured["fps"] = fps
        captured["loop"] = loop

    monkeypatch.setattr(gstyle, "play", fake_play)
    cb.gorkastyle(frames_dir=tmp_path, fps=20.0, loop=True)

    assert captured["frames"] == ["a\n", "b\n"]
    assert captured["fps"] == 20.0
    assert captured["loop"] is True


def test_gorkastyle_raises_if_no_frames(tmp_path: Path) -> None:
    """gorkastyle raises a clear error when no frame files exist."""
    with pytest.raises(FileNotFoundError, match="No ASCII frames found"):
        cb.gorkastyle(frames_dir=tmp_path)
