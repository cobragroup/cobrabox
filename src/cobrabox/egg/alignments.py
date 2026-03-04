"""D&D alignment registry for cobrabox features.

Each entry maps a feature class name to its alignment scores and lore.

Axes
----
law  : +1 = Lawful, 0 = Neutral, -1 = Chaotic
good : +1 = Good,   0 = Neutral, -1 = Evil

New entries are added by the /dnd-alignment Claude skill when a feature
is ranked for the first time.  The script src/cobrabox/egg/dnd_alignment.py
reads this table to compute pipeline aggregate alignments.
"""

from __future__ import annotations

# fmt: off
ALIGNMENTS: dict[str, dict] = {
    "SlidingWindow": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "sw",
        "lore":  "Rigidly structured, principled expansion of data — serves understanding",
    },
    "SpikesCalc": {
        "law":   1,
        "good":  -1,
        "label": "Lawful Evil",
        "abbrev": "sc",
        "lore":  "Judges by the book of IQR, condemning outliers to mere enumeration",
    },
    "MeanAggregate": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "ma",
        "lore":  "Collapses by strict rule; neither creates nor destroys meaning",
    },
    "Max": {
        "law":   1,
        "good": -1,
        "label": "Lawful Evil",
        "abbrev": "mx",
        "lore":  "Obeys the law of the maximum, ruthlessly discards everything else",
    },
    "Min": {
        "law":   0,
        "good": -1,
        "label": "Neutral Evil",
        "abbrev": "mi",
        "lore":  "Seeks only the lowest — no principle beyond pessimism",
    },
    "LineLength": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "ll",
        "lore":  "Measures without judgement, in service of signal",
    },
    "Dummy": {
        "law":  -1,
        "good":  0,
        "label": "Chaotic Neutral",
        "abbrev": "du",
        "lore":  "Print statements, no validation — chaos without malice, just bad practice",
    },
    "Mean": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "mn",
        "lore":  "Averages faithfully and without prejudice — the purest bureaucrat",
    },
    "Bandpower": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "bp",
        "lore":  "Integrates the spectrum with precision and purpose — a scholar of oscillations",
    },
    "Coherence": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "co",
        "lore":  "Seeks channel connection without imposing structure — empathic, unbiased",
    },
    "Autocorr": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "ac",
        "lore":  "Interrogates its own past with clockwork precision, faithful to every lag",
    },
    "Spectrogram": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "sg",
        "lore":  "Unfolds time into frequency — the scholarly cartographer of oscillations",
    },
    "EnvelopeCorrelation": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "ec",
        "lore":  "Exorcises zero-lag phantoms, revealing genuine kinship between channels",
    },
    "EpileptogenicityIndex": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "ei",
        "lore":  "Follows Bartolomei's law to the letter; renders its verdict in [0, 1]",
    },
    "PartialCorrelation": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "pc",
        "lore":  "Controls for the guilty bystanders, exonerating the true connection",
    },
    "PartialCorrelationMatrix": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "pm",
        "lore":  "Maps the full tribunal of partial truths — no pair left unjudged",
    },
}
# fmt: on

# ── helpers ──────────────────────────────────────────────────────────────────

_LABEL: dict[tuple[int, int], str] = {
    (1, 1): "Lawful Good",
    (1, 0): "Lawful Neutral",
    (1, -1): "Lawful Evil",
    (0, 1): "Neutral Good",
    (0, 0): "True Neutral",
    (0, -1): "Neutral Evil",
    (-1, 1): "Chaotic Good",
    (-1, 0): "Chaotic Neutral",
    (-1, -1): "Chaotic Evil",
}


def snap(value: float) -> int:
    """Snap a float average to the nearest alignment axis value {+1, 0, -1}."""
    if value >= 0.34:
        return 1
    if value <= -0.34:
        return -1
    return 0


def label_for(law: int, good: int) -> str:
    """Return the alignment label string for a (law, good) pair."""
    return _LABEL[(law, good)]
