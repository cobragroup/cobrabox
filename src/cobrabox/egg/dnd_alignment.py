"""Compute the D&D alignment of a cobrabox feature pipeline.

Usage
-----
    uv run python -m cobrabox.egg.dnd_alignment SlidingWindow LineLength MeanAggregate
    uv run python -m cobrabox.egg.dnd_alignment my_pipeline.yaml
    uv run python -m cobrabox.egg.dnd_alignment --roster

Options
-------
--roster        Print the full alignment table and ASCII grid, then exit.
--chord         Treat the pipeline as a Chord (see Chord Modifier below).
feature [...]   Feature class names to include in the pipeline.
file [...]      One or more .yaml / .yml / .json pipeline files to load.
                File paths and feature names may be mixed freely.

File Loading
------------
Any argument ending in .yaml, .yml, or .json is treated as a pipeline file
and loaded via cobrabox.serialization.load().  Feature names are extracted
by walking the pipeline in order.  If the file contains a single top-level
Chord, chord-weighting is applied automatically (overridden by --chord /
--no-chord).

Chord Modifier
--------------
When --chord is passed the script applies the Chord structural modifier:

    The first feature must be a SplitterFeature and the last must be an
    AggregatorFeature.  These two framing features are given double weight
    in the aggregate calculation -- they define the structure of the pipeline
    and therefore have outsized influence on its alignment.  The inner
    (map) features are weighted normally.

    Example:  SlidingWindow | LineLength | MeanAggregate
              weights:  2 (splitter) + 1 (map) + 2 (aggregator)
              vs sequential: 1 + 1 + 1

    This means a Chord dominated by Lawful framing features will be pulled
    more strongly Lawful than an equivalent sequential pipeline.
"""

from __future__ import annotations

import sys

from cobrabox.egg.alignments import ALIGNMENTS, label_for, snap

# ── file loading helpers ───────────────────────────────────────────────────────


def _is_file_arg(arg: str) -> bool:
    """Return True if the argument looks like a pipeline file path."""
    return arg.lower().endswith((".yaml", ".yml", ".json"))


def _names_from_step(step: object) -> list[str]:
    """Recursively extract feature class names from a single pipeline step."""
    from cobrabox.base_feature import Chord, Pipeline

    if isinstance(step, Chord):
        names = [type(step.split).__name__]
        if isinstance(step.pipeline, Pipeline):
            for f in step.pipeline.features:
                names.extend(_names_from_step(f))
        else:
            names.extend(_names_from_step(step.pipeline))
        names.append(type(step.aggregate).__name__)
        return names
    return [type(step).__name__]


def _load_file(path: str) -> tuple[list[str], bool]:
    """Load a pipeline file and return (feature_names, is_top_level_chord).

    is_top_level_chord is True when the file contains a single Chord step,
    which triggers automatic chord-weighting unless overridden by the caller.
    """
    from cobrabox.base_feature import Chord
    from cobrabox.serialization import load

    try:
        pipeline = load(path)
    except Exception as exc:
        print(f"✖ Could not load {path!r}: {exc}", file=sys.stderr)
        sys.exit(1)

    is_chord = len(pipeline.features) == 1 and isinstance(pipeline.features[0], Chord)
    names: list[str] = []
    for step in pipeline.features:
        names.extend(_names_from_step(step))
    return names, is_chord


# ── grid rendering ────────────────────────────────────────────────────────────

_GRID_LAW = [1, 0, -1]
_GRID_GOOD = [1, 0, -1]
_ROW_LABEL = {1: "GOOD", 0: "NEUT", -1: "EVIL"}
_COL_LABEL = {1: "LAW", 0: "NEUTRAL", -1: "CHAOS"}


def _render_grid(cells: dict[tuple[int, int], str]) -> str:
    """Render the 3x3 alignment grid.

    Args:
        cells: mapping of (law, good) -> cell content string (<=6 chars).
    """
    header = "         LAW        NEUTRAL      CHAOS"
    divider = "         " + "\u2500" * 33
    rows = [header, divider]
    for good in _GRID_GOOD:
        row_parts = []
        for law in _GRID_LAW:
            content = cells.get((law, good), "")
            row_parts.append(f"[{content:^6}]")
        rows.append(f"{_ROW_LABEL[good]:<4}  \u2502  {'   '.join(row_parts)}  \u2502")
    return "\n".join(rows)


# ── roster mode ───────────────────────────────────────────────────────────────


def print_roster() -> None:
    print("## Cobrabox Feature Alignment Roster\n")
    cells: dict[tuple[int, int], str] = {}
    for feat_name, entry in ALIGNMENTS.items():
        print(f"{feat_name}  \u2014  {entry['label']}")
        print(f'  "{entry["lore"]}"\n')
        key = (entry["law"], entry["good"])
        existing = cells.get(key, "")
        abbrev = entry["abbrev"]
        cells[key] = f"{existing},{abbrev}" if existing else abbrev
    print(_render_grid(cells))


# ── pipeline mode ─────────────────────────────────────────────────────────────


def print_pipeline(features: list[str], chord: bool = False) -> None:
    valid: list[tuple[str, dict]] = []
    for name in features:
        if name in ALIGNMENTS:
            valid.append((name, ALIGNMENTS[name]))
        else:
            print(f"\u26a0 Unknown feature: {name} \u2014 skipped", file=sys.stderr)

    if not valid:
        print("No valid features provided.", file=sys.stderr)
        sys.exit(1)

    # ── weights ──────────────────────────────────────────────────────────────
    if chord and len(valid) >= 2:
        weights = [1] * len(valid)
        weights[0] = 2  # splitter gets double weight
        weights[-1] = 2  # aggregator gets double weight
        mode_note = " (chord-weighted: splitter x2, aggregator x2)"
    else:
        weights = [1] * len(valid)
        mode_note = ""

    total_weight = sum(weights)
    law_avg = sum(w * e["law"] for (_, e), w in zip(valid, weights, strict=True)) / total_weight
    good_avg = sum(w * e["good"] for (_, e), w in zip(valid, weights, strict=True)) / total_weight

    agg_law = snap(law_avg)
    agg_good = snap(good_avg)
    agg_label = label_for(agg_law, agg_good)

    # ── print pipeline header ─────────────────────────────────────────────────
    pipeline_str = " \u2192 ".join(name for name, _ in valid)
    print(f"Pipeline: {pipeline_str}{mode_note}\n")

    law_detail = ", ".join(f"{'+' if e['law'] >= 0 else ''}{e['law']}" for _, e in valid)
    good_detail = ", ".join(f"{'+' if e['good'] >= 0 else ''}{e['good']}" for _, e in valid)
    print(f"Law scores:  {law_detail}  \u2192  avg {law_avg:+.2f}  \u2192  {_COL_LABEL[agg_law]}")
    print(
        f"Good scores: {good_detail}  \u2192  avg {good_avg:+.2f}  \u2192  {_ROW_LABEL[agg_good]}\n"
    )

    # ── build grid cells ─────────────────────────────────────────────────────
    cells: dict[tuple[int, int], str] = {}
    for _name, entry in valid:
        key = (entry["law"], entry["good"])
        abbrev = entry["abbrev"]
        existing = cells.get(key, "")
        cells[key] = f"{existing},{abbrev}" if existing else abbrev

    agg_key = (agg_law, agg_good)
    existing = cells.get(agg_key, "")
    cells[agg_key] = f"{existing} \u2605" if existing else "\u2605"

    print(_render_grid(cells))
    print(f"\nAggregate Alignment: {agg_label}  \u2605")


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv

    if not args or "--roster" in args:
        print_roster()
        return

    explicit_chord = "--chord" in args
    explicit_no_chord = "--no-chord" in args
    positional = [a for a in args if not a.startswith("--")]

    features: list[str] = []
    auto_chord = False  # set True if any loaded file is a top-level Chord

    for arg in positional:
        if _is_file_arg(arg):
            file_names, file_is_chord = _load_file(arg)
            features.extend(file_names)
            if file_is_chord:
                auto_chord = True
        else:
            features.append(arg)

    # --chord / --no-chord override auto-detection from file content
    if explicit_chord:
        chord = True
    elif explicit_no_chord:
        chord = False
    else:
        chord = auto_chord

    print_pipeline(features, chord=chord)


if __name__ == "__main__":
    main()
