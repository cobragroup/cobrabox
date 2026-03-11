---
name: dnd-alignment
description: Rank cobrabox features and pipelines on the D&D 9-alignment grid (Law/Chaos × Good/Evil). Use when the user invokes "/dnd-alignment", asks "what is the alignment of X", asks to "align my pipeline", or wants to know the moral character of a feature or analysis chain.
---

# DnD Alignment

Determine the D&D 9-alignment of a cobrabox feature and append it to the canonical table.

**Pipeline and roster rendering are handled by the Python script — do not compute them here.**

## Invocation

```text
/dnd-alignment <FeatureName>
```

---

## Responsibility split

| Task                                                       | Owner                                                          |
| ---------------------------------------------------------- | -------------------------------------------------------------- |
| Rank an individual feature (assign Law/Good scores + lore) | This skill                                                     |
| Append ranking to the canonical table                      | This skill                                                     |
| Print the full roster                                      | `uv run python -m cobrabox.egg.dnd_alignment --roster`         |
| Compute pipeline aggregate                                 | `uv run python -m cobrabox.egg.dnd_alignment F1 F2 F3`         |
| Compute Chord pipeline aggregate                           | `uv run python -m cobrabox.egg.dnd_alignment --chord F1 F2 F3` |

---

## Canonical alignment table

The source of truth is **`src/cobrabox/egg/alignments.py`**.

Read that file before ranking a new feature — the feature may already be present.

### Axis encoding

- **Law axis:** Lawful = +1, Neutral = 0, Chaotic = −1
- **Good axis:** Good = +1, Neutral = 0, Evil = −1

### Label grid

```text
             LAW (+1)          NEUTRAL (0)        CHAOS (-1)
GOOD  (+1)   Lawful Good       Neutral Good       Chaotic Good
NEUT  ( 0)   Lawful Neutral    True Neutral       Chaotic Neutral
EVIL  (-1)   Lawful Evil       Neutral Evil       Chaotic Evil
```

---

## Procedure

### 1. Read the alignment table

Read `src/cobrabox/egg/alignments.py`. Check whether the feature is already present
in `ALIGNMENTS`. If it is, print its existing entry and stop — do not re-rank.

### 2. Read the feature file

Read `src/cobrabox/features/<feature_snake_case>.py` to understand what the feature
actually does before assigning an alignment.

### 3. Assign scores

Choose Law and Good scores using the rubrics below. Write 1–2 sentences of reasoning
for each axis before committing to a score.

#### Law axis rubric

| Score      | Meaning                              | Indicators                                                                                          |
| ---------- | ------------------------------------ | --------------------------------------------------------------------------------------------------- |
| +1 Lawful  | **Actively imposes** structure       | Creates segments (windowing), hard threshold classification (IQR), named category ontology (frequency bands), strict published-protocol adherence |
| 0 Neutral  | **Passively describes** existing patterns | Correlation/synchrony measures, spectral descriptions, statistical summaries — even with fixed formulas |
| -1 Chaotic | Disrupts or ignores conventions      | `print` statements, missing validation, unpredictable output shape                                  |

> **Common trap:** A fixed, deterministic formula does **not** make a feature Lawful — almost all
> signal processing is deterministic. Ask instead: does this feature *impose* structure onto the
> data, or *describe* structure already present in it?
>
> Lawful examples: `SlidingWindow` (creates window segments), `Bandpower` (names frequency
> categories), `SpikesCalc` (classifies by IQR rule).
>
> Neutral examples: `Coherence`, `PLV`, `Autocorr`, `Spectrogram`, `EnvelopeCorrelation`,
> `PartialCorrelation` — all use precise formulas but measure existing patterns without imposing.

#### Good axis rubric

| Score     | Meaning                               | Indicators                                                                 |
| --------- | ------------------------------------- | -------------------------------------------------------------------------- |
| +1 Good   | Preserves or enhances signal meaning  | Increases interpretability, faithful to data, good metadata practice       |
| 0 Neutral | Indifferent to meaning                | Mechanical reduction with no semantic intent (pure aggregation)            |
| -1 Evil   | Discards or distorts signal meaning   | Selects extremes ruthlessly, drops metadata, lossy without documentation   |

### 4. Write one lore sentence

One punchy sentence (≤ 15 words) in the style of the existing entries. It should
capture the *moral character* of what the feature does to data, not just describe it
technically.

### 5. Choose a 2-char abbreviation

Use the first two lowercase letters of the class name, unless that conflicts with an
existing abbreviation in the table — in that case pick the most recognisable 2-char
substring.

### 6. Append to the table

Edit `src/cobrabox/egg/alignments.py` — add a new entry to the `ALIGNMENTS` dict
following the existing format exactly:

```python
"FeatureName": {
    "law":    <+1|0|-1>,
    "good":   <+1|0|-1>,
    "label":  "<Alignment Label>",
    "abbrev": "<2-char>",
    "lore":   "<lore sentence>",
},
```

Place it alphabetically by key, or at the end if alphabetical order is not obvious.

### 7. Run the roster to confirm

```bash
uv run python -m cobrabox.egg.dnd_alignment --roster
```

Confirm the new feature appears correctly in the output.

### 8. Report to conversation

Print:

```text
<FeatureName>  —  <Alignment Label>
  Law:  <score>  (<reasoning>)
  Good: <score>  (<reasoning>)
  "<lore sentence>"

Table updated: src/cobrabox/egg/alignments.py
Run `uv run python -m cobrabox.egg.dnd_alignment --roster` to see the full grid.
```

---

## Pipeline alignment — delegate to script

When the user asks about a pipeline or sequence of features, do **not** compute it
yourself. Instead, tell them to run:

```bash
# Sequential pipeline
uv run python -m cobrabox.egg.dnd_alignment SlidingWindow LineLength MeanAggregate

# Chord pipeline (splitter + map + aggregator, framing features weighted ×2)
uv run python -m cobrabox.egg.dnd_alignment --chord SlidingWindow LineLength MeanAggregate
```

---

## Roster — delegate to script

When the user asks for the full roster or grid:

```bash
uv run python -m cobrabox.egg.dnd_alignment --roster
```
