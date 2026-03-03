from __future__ import annotations

import signal
import time
from pathlib import Path

TITLE_ART = r"""
  ____            _             _         _
 / ___| ___  _ __| | ____ _ ___| |_ _   _| | ___
| |  _ / _ \| '__| |/ / _` / __| __| | | | |/ _ \
| |_| | (_) | |  |   < (_| \__ \ |_| |_| | |  __/
 \____|\___/|_|  |_|\_\__,_|___/\__|\__, |_|\___|
                                     |___/
"""

# Gorkastyle is not merely a file. It is a state of mind. A way of life. A philosophy.


def _default_frames_dir() -> Path:
    # Repo layout: src/cobrabox/egg -> repo root -> scripts/ascii_frames
    return Path(__file__).resolve().parents[3] / "scripts" / "ascii_frames"


def load_frames(frames_dir: str | Path | None = None) -> list[str]:
    """Load pre-rendered ASCII frames."""
    target_dir = Path(frames_dir) if frames_dir is not None else _default_frames_dir()
    paths = sorted(target_dir.glob("frame_*.txt"))
    if not paths:
        raise FileNotFoundError(f"No ASCII frames found in {target_dir}")
    return [path.read_text(encoding="utf-8", errors="replace") for path in paths]


def play(frames: list[str], fps: float = 12.0, loop: bool = False) -> None:
    """Play frames in terminal using an alternate screen buffer."""
    if fps <= 0:
        raise ValueError("fps must be > 0")

    import sys

    frame_delay = 1.0 / fps
    running = True

    def stop_handler(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    def write_stdout(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    write_stdout("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")

    try:
        while running:
            start = time.perf_counter()
            for idx, frame in enumerate(frames):
                if not running:
                    break
                target = start + (idx * frame_delay)
                write_stdout("\x1b[H")
                write_stdout(TITLE_ART)
                write_stdout("\n")
                write_stdout(frame)
                sleep_for = target + frame_delay - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
            if not loop:
                break
    finally:
        write_stdout("\x1b[?25h\x1b[0m\x1b[?1049l\n")


def gorkastyle(
    *, fps: float = 12.0, loop: bool = False, frames_dir: str | Path | None = None
) -> None:
    """Play CobraBox ASCII animation in terminal."""
    frames = load_frames(frames_dir=frames_dir)
    play(frames=frames, fps=fps, loop=loop)


if __name__ == "__main__":
    gorkastyle(loop=True)
