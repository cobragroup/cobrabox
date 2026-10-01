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
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "ae",
        "lore":  "Herds amplitudes into fixed bins before reading their disorder",
    },
    "AmplitudeVariation": {
        "law":   0,
        "good":  0,
        "label": "True Neutral",
        "abbrev": "av",
        "lore":  "Reports how much the signal wobbles, caring not why",
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
        "law":   1,
        "good": -1,
        "label": "Lawful Evil",
        "abbrev": "lz",
        "lore":  "Crushes every amplitude to one bit at the mean, then counts",
    },
    "LineLength": {
        "law":   0,
        "good":  0,
        "label": "True Neutral",
        "abbrev": "ll",
        "lore":  "Walks the signal's path and reports the distance, nothing more",
    },
    "DiscreteWaveletTransform": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "dw",
        "lore":  "Divides the signal into dyadic estates — perfectly reconstructible order",
    },
    "EMD": {
        "law":  -1,
        "good":  1,
        "label": "Chaotic Good",
        "abbrev": "em",
        "lore":  "Sifts by instinct with no basis decreed — yet every IMF sums home",
    },
    "Mean": {
        "law":   0,
        "good":  0,
        "label": "True Neutral",
        "abbrev": "mn",
        "lore":  "Averages the crowd into one voice — no malice, no memory",
    },
    "MutualInformation": {
        "law":   0,
        "good":  0,
        "label": "True Neutral",
        "abbrev": "mu",
        "lore":  "Measures statistical dependence — impartial accountant, no semantic intent",
    },
    "Nonreversibility": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "nr",
        "lore":  "Forces dynamics into VAR(1) shackles, then asks if time runs backward",
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
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "ca",
        "lore":  "Stitches every window back in order — nothing lost, nothing invented",
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
        "good":  0,
        "label": "True Neutral",
        "abbrev": "cv",
        "lore":  "Counts co-movement in raw units, blind to scale and meaning",
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
    "BandDecomposition": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "bd",
        "lore":  "Imposes the classical order of brain rhythms, granting each band its own estate",
    },
    "BandpassFilter": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "bf",
        "lore":  "Declares one band lawful and the rest outlaw, returning a single purified signal",
    },
    "BandPower": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "bp",
        "lore":  "Weighs each named rhythm by decree; phase and waveform discarded",
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
        "good":  0,
        "label": "True Neutral",
        "abbrev": "sg",
        "lore":  "Trades phase and resolution for a picture of time and frequency",
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
        "law":  -1,
        "good": -1,
        "label": "Chaotic Evil",
        "abbrev": "fs",
        "lore":  "Scrambles every phase on purpose — forges impostors to frame the truth",
    },
    "GrangerCausality": {
        "law":   1,
        "good":  0,
        "label": "Lawful Neutral",
        "abbrev": "gc",
        "lore":  "Fits the linear law, then hands down a verdict on predictive guilt",
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
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "pd",
        "lore":  "Binds channels to a VAR covenant, then reveals who drives whom",
    },
    "ReciprocalConnectivity": {
        "law":   0,
        "good": -1,
        "label": "Neutral Evil",
        "abbrev": "rc",
        "lore":  "Nets out every dialogue, leaving only who shouted louder",
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
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "dd",
        "lore":  "Enforces the VAR code and prosecutes only direct influence",
    },
    "DirectedTransferFunction": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "dt",
        "lore":  "Swears by the VAR model to trace every path of influence",
    },
    "FourierTransform": {
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "ft",
        "lore":  "Imposes the sinusoid's law on time, yet keeps every coefficient",
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
        "law":   1,
        "good":  1,
        "label": "Lawful Good",
        "abbrev": "nf",
        "lore":  "Declares the mains frequency outlaw and excises it, sparing the rest",
    },
    "PowerSpectralDensity": {
        "law":   0,
        "good":  0,
        "label": "True Neutral",
        "abbrev": "ps",
        "lore":  "Keeps the power, throws the phase overboard without a second thought",
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
