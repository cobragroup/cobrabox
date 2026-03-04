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


def test_default_frames_dir_returns_path() -> None:
    """_default_frames_dir returns a Path pointing to scripts/ascii_frames."""
    result = gstyle._default_frames_dir()
    assert isinstance(result, Path)
    assert result.name == "ascii_frames"
    assert result.parent.name == "scripts"


def test_play_raises_for_nonpositive_fps() -> None:
    """play() raises ValueError for fps <= 0."""
    with pytest.raises(ValueError, match="fps must be > 0"):
        gstyle.play(["frame\n"], fps=0)
    with pytest.raises(ValueError, match="fps must be > 0"):
        gstyle.play(["frame\n"], fps=-5.0)


def test_play_single_pass_renders_all_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """play() writes each frame to stdout exactly once when loop=False."""
    import io
    import sys

    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(gstyle.signal, "signal", lambda *_: None)
    monkeypatch.setattr(gstyle.time, "sleep", lambda _: None)
    monkeypatch.setattr(gstyle.time, "perf_counter", lambda: 0.0)

    gstyle.play(["frame_alpha\n", "frame_beta\n"], fps=12.0, loop=False)

    output = fake_stdout.getvalue()
    assert "frame_alpha" in output
    assert "frame_beta" in output
    # alternate screen entered and restored
    assert "\x1b[?1049h" in output
    assert "\x1b[?1049l" in output


def test_play_stop_handler_aborts_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """stop_handler sets running=False, causing loop=True playback to abort."""
    import io
    import sys

    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", fake_stdout)

    captured_handler: list = []

    def fake_signal(sig: int, handler: object) -> None:
        captured_handler.append(handler)

    monkeypatch.setattr(gstyle.signal, "signal", fake_signal)

    sleep_count = [0]

    def fake_sleep(_: float) -> None:
        sleep_count[0] += 1
        # trigger stop on first sleep so the second frame sees running=False
        if sleep_count[0] == 1 and captured_handler:
            captured_handler[0](gstyle.signal.SIGINT, None)

    monkeypatch.setattr(gstyle.time, "sleep", fake_sleep)
    monkeypatch.setattr(gstyle.time, "perf_counter", lambda: 0.0)

    gstyle.play(["frame_a\n", "frame_b\n"], fps=12.0, loop=True)

    output = fake_stdout.getvalue()
    assert "frame_a" in output
    assert "\x1b[?1049l" in output  # terminal always restored
