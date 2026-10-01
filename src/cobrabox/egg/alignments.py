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
    "AmplitudeEntropy": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "ae",
        "lore":  "Reads the disorder of amplitudes through histograms — faithful entropy",
    },
    "AmplitudeVariation": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "av",
        "lore":  "Measures the breath of the signal — faithfully, without judgment",
    },
    "SlidingWindow": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "sw",
        "lore":  "Rigidly structured, principled expansion of data — serves understanding",
    },
    "SlidingWindowReduce": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "sr",
        "lore":  "Methodically carves time into windows, then summarizes without prejudice",
    },
    "SpikeCount": {
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
        "law":   1,
        "good": -1,
        "label": "Lawful Evil",
        "abbrev": "mi",
        "lore":  "The mirror of Max — same cold devotion to order, different floor",
    },
    "LempelZiv": {
        "law":   0,
        "good":  0,
        "label": "True Neutral",
        "abbrev": "lz",
        "lore":  "Counts pattern distinctness — impartial accountant of signal complexity",
    },
    "LineLength": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "ll",
        "lore":  "Measures without judgement, in service of signal",
    },
    "DiscreteWaveletTransform": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "dw",
        "lore":  "Decomposes time into multi-resolution levels — reveals hidden scales",
    },
    "EMD": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "em",
        "lore":  "Decomposes adaptively without imposing basis — patient midwife of IMFs",
    },
    "Mean": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "mn",
        "lore":  "Averages faithfully and without prejudice — the purest bureaucrat",
    },
    "MutualInformation": {
        "law":   0,
        "good":  0,
        "label": "True Neutral",
        "abbrev": "mu",
        "lore":  "Measures statistical dependence — impartial accountant, no semantic intent",
    },
    "Nonreversibility": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "nr",
        "lore":  "Reads the arrow of time through VAR(1) asymmetry — faithful time's arrow",
    },
    "RecurrenceMatrix": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "rm",
        "lore":  "Maps self-similarity across time — cartographer of recurrence plots",
    },
    "ConcatAggregate": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "ca",
        "lore":  "Collapses by strict rule; neither creates nor destroys meaning",
    },
    "Correlation": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "cr",
        "lore":  "Seeks linear kinship between channels — faithful, unbiased judge of association",
    },
    "Covariance": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "cv",
        "lore":  "Measures joint variability honestly — no structure imposed, only truth revealed",
    },
    "ContinuousWaveletTransform": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "cw",
        "lore":  "Unfolds time into frequency with continuous scales — reveals hidden rhythms",
    },
    "Cordance": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "cd",
        "lore":  "Classifies channels by Leuchter's law — absolute and relative power united",
    },
    "BandpassFilter": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "bf",
        "lore":  "Imposes the classical order of brain rhythms upon chaotic oscillations",
    },
    "BandPower": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "bp",
        "lore":  "Integrates the spectrum with precision and purpose — a scholar of oscillations",
    },
    "AnalyticSignal": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "hi",
        "lore":  "Reveals the hidden complex soul of oscillations without imposing form",
    },
    "Coherence": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "co",
        "lore":  "Seeks channel connection without imposing structure — empathic, unbiased",
    },
    "Autocorrelation": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "ac",
        "lore":  "Holds a mirror to time and reads the echo — without judgment, only fidelity",
    },
    "Spectrogram": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "sg",
        "lore":  "Unfolds time into frequency — cartographer of oscillations, imposing nothing",
    },
    "EnvelopeCorrelation": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "ec",
        "lore":  "Exorcises zero-lag phantoms, revealing genuine kinship between channels",
    },
    "FractalDimension": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "fd",
        "lore":  "Reads the roughness of the signal through the lens of fractal geometry",
    },
    "FourierTransformSurrogates": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "fs",
        "lore":  "Imposes null hypotheses via strict phase-shuffling protocol",
    },
    "GrangerCausality": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "gc",
        "lore":  "Tests if the past of one channel predicts another — honest temporal judge",
    },
    "EpileptogenicityIndex": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "ei",
        "lore":  "Follows Bartolomei's law to the letter; renders its verdict in [0, 1]",
    },
    "PartialCorrelation": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "pc",
        "lore":  "Controls for the guilty bystanders, exonerating the true connection",
    },
    "PhaseLockingValue": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "pl",
        "lore":  "Listens for rhythmic sympathy between channels — neither judge nor jailer",
    },
    "PartialDirectedCoherence": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "pd",
        "lore":  "Reveals directional influence through spectral VAR analysis",
    },
    "ReciprocalConnectivity": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "rc",
        "lore":  "Summarizes net flow to name sources and sinks",
    },
    "SampleEntropy": {
        "law":   0,
        "good":  0,
        "label": "True Neutral",
        "abbrev": "se",
        "lore":  "Measures regularity of complexity — impartial accountant of signal disorder",
    },
    "SVD": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "sv",
        "lore": (
            "Imposes the rigid geometry of linear subspaces — "
            "order from chaos, without malice"
        ),
    },
    "DirectDirectedTransferFunction": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "dd",
        "lore":  "Strips away the middlemen, naming only who speaks directly to whom",
    },
    "DirectedTransferFunction": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "dt",
        "lore":  "Traces every whisper of influence, direct or relayed, without prejudice",
    },
    "FourierTransform": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "ft",
        "lore":  "Translates time into frequency, losing not a single coefficient",
    },
    "InverseFourierTransform": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "if",
        "lore":  "Returns the frequencies home to time, faithful to every phase",
    },
    "InwardStrength": {
        "law":   0,
        "good":  0,
        "label": "True Neutral",
        "abbrev": "is",
        "lore":  "Tallies what each channel receives, then forgets from whom",
    },
    "OutwardStrength": {
        "law":   0,
        "good":  0,
        "label": "True Neutral",
        "abbrev": "os",
        "lore":  "Tallies what each channel sends, then forgets to whom",
    },
    "Normalize": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "no",
        "lore":  "Forces every slice onto the same scale, indifferent to its magnitude",
    },
    "NotchFilter": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "nf",
        "lore":  "Silences the mains hum with surgical precision, sparing all else",
    },
    "PowerSpectralDensity": {
        "law":   0,
        "good":  1,
        "label": "Neutral Good",
        "abbrev": "ps",
        "lore":  "Weighs the power of each frequency honestly, phase set aside",
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
