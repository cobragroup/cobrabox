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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.table import Table

from cobrabox.egg.alignments import ALIGNMENTS, label_for, snap

# ── alignment axis labels ─────────────────────────────────────────────────────

_ROW_LABEL = {1: "GOOD", 0: "NEUT", -1: "EVIL"}
_COL_LABEL = {1: "LAW", 0: "NEUTRAL", -1: "CHAOS"}
_GRID_LAW = [1, 0, -1]
_GRID_GOOD = [1, 0, -1]

# ── cell styling ──────────────────────────────────────────────────────────────

_CELL_STYLE: dict[tuple[int, int], str] = {
    (1, 1): "bold green",
    (0, 1): "cyan",
    (-1, 1): "blue",
    (1, 0): "bold yellow",
    (0, 0): "dim",
    (-1, 0): "bold",
    (1, -1): "bold red",
    (0, -1): "red",
    (-1, -1): "bold dark_red",
}

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
    from rich.console import Console

    from cobrabox.base_feature import Chord
    from cobrabox.serialization import load

    try:
        pipeline = load(path)
    except Exception as exc:
        Console(stderr=True).print(f"[bold red]\u2716 Could not load {path!r}:[/bold red] {exc}")
        sys.exit(1)

    is_chord = len(pipeline.features) == 1 and isinstance(pipeline.features[0], Chord)
    names: list[str] = []
    for step in pipeline.features:
        names.extend(_names_from_step(step))
    return names, is_chord


# ── grid rendering ────────────────────────────────────────────────────────────


def _rich_grid(cells: dict[tuple[int, int], str]) -> Table:
    """Build a Rich Table representing the 3x3 alignment grid."""
    import rich.box
    from rich.table import Table

    table = Table(box=rich.box.SIMPLE_HEAD, show_header=True, padding=(0, 2))
    table.add_column("", style="dim", no_wrap=True)
    table.add_column("[bold]LAW[/bold]", justify="center", min_width=10)
    table.add_column("[bold]NEUTRAL[/bold]", justify="center", min_width=10)
    table.add_column("[bold]CHAOS[/bold]", justify="center", min_width=10)

    for good in _GRID_GOOD:
        row_label = _ROW_LABEL[good]
        cells_row: list[str] = []
        for law in _GRID_LAW:
            content = cells.get((law, good), "")
            style = _CELL_STYLE.get((law, good), "")
            cells_row.append(f"[{style}]{content}[/{style}]" if content else "")
        table.add_row(row_label, *cells_row)

    return table


# ── roster mode ───────────────────────────────────────────────────────────────


def print_roster() -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    feat_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    feat_table.add_column("Feature", style="bold", no_wrap=True)
    feat_table.add_column("Alignment", style="dim", no_wrap=True)
    feat_table.add_column("Lore", style="italic")

    cells: dict[tuple[int, int], str] = {}
    for feat_name, entry in ALIGNMENTS.items():
        feat_table.add_row(feat_name, entry["label"], f'"{entry["lore"]}"')
        key = (entry["law"], entry["good"])
        existing = cells.get(key, "")
        abbrev = entry["abbrev"]
        cells[key] = f"{existing},{abbrev}" if existing else abbrev

    console.print(Panel(feat_table, title="[bold]Cobrabox Feature Alignment Roster[/bold]"))
    console.print()
    console.print(Panel(_rich_grid(cells), title="[bold]Alignment Grid[/bold]"))


# ── pipeline mode ─────────────────────────────────────────────────────────────


def print_pipeline(features: list[str], chord: bool = False) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    stderr = Console(stderr=True)

    valid: list[tuple[str, dict]] = []
    for name in features:
        if name in ALIGNMENTS:
            valid.append((name, ALIGNMENTS[name]))
        else:
            stderr.print(f"[yellow]\u26a0 Unknown feature: {name} \u2014 skipped[/yellow]")

    if not valid:
        stderr.print("[red]No valid features provided.[/red]")
        sys.exit(1)

    # ── weights ──────────────────────────────────────────────────────────────
    if chord and len(valid) >= 2:
        weights = [1] * len(valid)
        weights[0] = 2  # splitter gets double weight
        weights[-1] = 2  # aggregator gets double weight
        mode_note = "  [dim](chord-weighted: splitter \u00d72, aggregator \u00d72)[/dim]"
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
    console.print(f"[bold]Pipeline:[/bold] {pipeline_str}{mode_note}\n")

    law_detail = ", ".join(f"{'+' if e['law'] >= 0 else ''}{e['law']}" for _, e in valid)
    good_detail = ", ".join(f"{'+' if e['good'] >= 0 else ''}{e['good']}" for _, e in valid)
    console.print(
        f"[dim]Law scores :[/dim]  {law_detail}  \u2192  "
        f"avg {law_avg:+.2f}  \u2192  {_COL_LABEL[agg_law]}"
    )
    console.print(
        f"[dim]Good scores:[/dim]  {good_detail}  \u2192  "
        f"avg {good_avg:+.2f}  \u2192  {_ROW_LABEL[agg_good]}\n"
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

    console.print(Panel(_rich_grid(cells), title="[bold]Alignment Grid[/bold]"))
    agg_style = _CELL_STYLE.get((agg_law, agg_good), "bold")
    console.print(
        f"\n[bold]Aggregate Alignment:[/bold] [{agg_style}]{agg_label}[/{agg_style}]  \u2605"
    )


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
