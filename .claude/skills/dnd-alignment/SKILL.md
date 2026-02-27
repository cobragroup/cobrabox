---
name: dnd-alignment
description: Rank cobrabox features and pipelines on the D&D 9-alignment grid (Law/Chaos × Good/Evil). Use when the user invokes "/dnd-alignment", asks "what is the alignment of X", asks to "align my pipeline", or wants to know the moral character of a feature or analysis chain.
---

# DnD Alignment

Determine the D&D alignment of cobrabox features — individually or as a pipeline.

## Invocation

```text
/dnd-alignment [feature1 feature2 ...]
```

- **No arguments** → print the full feature roster with all alignments and a 3×3 chart
- **With feature names** → compute pipeline aggregate, print chart and label

---

## Canonical Alignment Table

These alignments are fixed lore. Do not recompute them — look them up.

| Feature | Law | Good | Alignment | Lore |
|---|---|---|---|---|
| `sliding_window` | +1 | +1 | Lawful Good | Rigidly structured, principled expansion of data — serves understanding |
| `mean` | +1 | 0 | Lawful Neutral | Collapses by strict rule; neither creates nor destroys meaning |
| `max` | +1 | -1 | Lawful Evil | Obeys the law of the maximum, ruthlessly discards everything else |
| `min` | 0 | -1 | Neutral Evil | Seeks only the lowest — no principle beyond pessimism |
| `line_length` | 0 | +1 | Neutral Good | Measures without judgement, in service of signal |
| `dummy` | -1 | 0 | Chaotic Neutral | Print statements, no validation — chaos without malice, just bad practice |

### Axis encoding

- **Law axis:** Lawful = +1, Neutral = 0, Chaotic = −1
- **Good axis:** Good = +1, Neutral = 0, Evil = −1

### Cell labels

```
        LAW (+1)         NEUTRAL (0)       CHAOS (-1)
GOOD(+1)  Lawful Good      Neutral Good      Chaotic Good
NEUT (0)  Lawful Neutral   True Neutral      Chaotic Neutral
EVIL(-1)  Lawful Evil      Neutral Evil      Chaotic Evil
```

---

## Procedure

### Mode A — No arguments (full roster)

1. Print the header line: `## Cobrabox Feature Alignment Roster`
2. For each feature in the table (in the order listed), print:
   ```
   feature_name  —  Alignment Label
     "lore sentence"
   ```
3. Print the ASCII grid (see Grid Format below) with every feature's abbreviation placed in its cell.

### Mode B — Pipeline mode (feature names given as arguments)

1. Parse `$ARGUMENTS` as a space-separated list of feature names.
2. For each name, look up its Law and Good scores from the table. If a name is not in the table, print a warning: `⚠ Unknown feature: <name> — skipped` and continue.
3. Compute averages:
   - `law_avg = mean of all Law scores for valid features`
   - `good_avg = mean of all Good scores for valid features`
4. Snap each average to {+1, 0, −1}:
   - ≥ 0.34 → +1
   - ≤ −0.34 → −1
   - else → 0
5. Look up the aggregate alignment label from the cell labels table.
6. Print the pipeline sequence line:
   ```
   Pipeline: feat1 → feat2 → feat3
   ```
7. Print the ASCII grid (see Grid Format below) with:
   - Each feature's abbreviation in its own cell
   - `[★]` in the aggregate cell (if no individual feature already occupies it; if overlap, add ★ after the abbrev: `[sw ★]`)
8. Print the aggregate label:
   ```
   Aggregate Alignment: Lawful Good  ★
   ```

---

## Grid Format

Use this exact template. Replace cell contents appropriately.

```
         LAW        NEUTRAL      CHAOS
         ─────────────────────────────────
GOOD  │  [      ]   [      ]   [      ]  │
NEUT  │  [      ]   [      ]   [      ]  │
EVIL  │  [      ]   [      ]   [      ]  │
```

Cell contents (6 chars wide):
- Empty cell: `      ` (6 spaces)
- Single feature: 2-char abbreviation centered: `  sw  `
- Multiple features in same cell: comma-separated: `sw,mn `
- Aggregate marker only: `  ★   `
- Feature + aggregate: `sw ★  `

### Feature abbreviations

| Feature | Abbrev |
|---|---|
| `sliding_window` | `sw` |
| `mean` | `mn` |
| `max` | `mx` |
| `min` | `mi` |
| `line_length` | `ll` |
| `dummy` | `du` |

For unknown/user features not in the table, use first 2 chars of the name.

---

## Example outputs

### Roster mode (`/dnd-alignment`)

```
## Cobrabox Feature Alignment Roster

sliding_window  —  Lawful Good
  "Rigidly structured, principled expansion of data — serves understanding"

mean  —  Lawful Neutral
  "Collapses by strict rule; neither creates nor destroys meaning"

max  —  Lawful Evil
  "Obeys the law of the maximum, ruthlessly discards everything else"

min  —  Neutral Evil
  "Seeks only the lowest — no principle beyond pessimism"

line_length  —  Neutral Good
  "Measures without judgement, in service of signal"

dummy  —  Chaotic Neutral
  "Print statements, no validation — chaos without malice, just bad practice"

         LAW        NEUTRAL      CHAOS
         ─────────────────────────────────
GOOD  │  [  sw  ]   [  ll  ]   [      ]  │
NEUT  │  [  mn  ]   [      ]   [  du  ]  │
EVIL  │  [  mx  ]   [  mi  ]   [      ]  │
```

### Pipeline mode (`/dnd-alignment sliding_window line_length mean`)

```
Pipeline: sliding_window → line_length → mean

Law scores:  +1, 0, +1  →  avg +0.67  →  Lawful
Good scores: +1, +1, 0  →  avg +0.67  →  Good

         LAW        NEUTRAL      CHAOS
         ─────────────────────────────────
GOOD  │  [sw ★ ]   [  ll  ]   [      ]  │
NEUT  │  [  mn  ]   [      ]   [      ]  │
EVIL  │  [      ]   [      ]   [      ]  │

Aggregate Alignment: Lawful Good  ★
```
