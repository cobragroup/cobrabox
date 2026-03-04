# DnD Alignment Skill Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a `/dnd-alignment` skill that ranks every cobrabox feature on the D&D 9-alignment grid and computes a pipeline's aggregate alignment.

**Architecture:** A single `SKILL.md` file under `.claude/skills/dnd-alignment/`. The skill hardcodes the canonical alignment for each feature, defines the numeric axis encoding, and gives the LLM exact instructions for both roster mode (no args) and pipeline mode (feature names as args). No Python code — pure LLM-executed skill.

**Tech Stack:** Markdown skill file only. No dependencies. Invoked via `/dnd-alignment` in Claude Code.

---

### Task 1: Create the skill directory and SKILL.md

**Files:**
- Create: `.claude/skills/dnd-alignment/SKILL.md`

**Step 1: Create the directory**

```bash
mkdir -p .claude/skills/dnd-alignment
```

**Step 2: Write the skill file**

Create `.claude/skills/dnd-alignment/SKILL.md` with the following content:

```markdown
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
        LAW (+1)       NEUTRAL (0)     CHAOS (-1)
GOOD(+1)  Lawful Good    Neutral Good    Chaotic Good
NEUT (0)  Lawful Neutral True Neutral    Chaotic Neutral
EVIL(-1)  Lawful Evil    Neutral Evil    Chaotic Evil
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
   - `[★]` in the aggregate cell (if no individual feature already occupies it; if overlap, add ★ after the abbrev: `[sw★]`)
8. Print the aggregate label:
   ```
   Aggregate Alignment: Lawful Good  ★
   ```

---

## Grid Format

Use this exact template. Replace `[ ]` with the appropriate content.

```
         LAW        NEUTRAL      CHAOS
         ─────────────────────────────────
GOOD  │  [      ]   [      ]   [      ]  │
NEUT  │  [      ]   [      ]   [      ]  │
EVIL  │  [      ]   [      ]   [      ]  │
```

Cell contents (6 chars wide, left-padded):
- Empty cell: `      ` (6 spaces)
- Single feature: 2-char abbreviation centered: `  sw  `
- Multiple features in same cell: comma-separated abbrevs: `sw,mn `
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
```

**Step 3: Verify the file was created**

```bash
cat .claude/skills/dnd-alignment/SKILL.md | head -5
```

Expected: first 5 lines including `---` and `name: dnd-alignment`.

**Step 4: Test roster mode**

Invoke the skill with no arguments in Claude Code:

```
/dnd-alignment
```

Expected: roster of 6 features with alignment labels and a 3×3 chart showing `sw`, `mn`, `mx`, `mi`, `ll`, `du` in their correct cells.

**Step 5: Test pipeline mode**

Invoke the skill with a pipeline:

```
/dnd-alignment sliding_window line_length mean
```

Expected: pipeline line, arithmetic shown, chart with `sw★` in Lawful Good, `ll` in Neutral Good, `mn` in Lawful Neutral, label `Aggregate Alignment: Lawful Good ★`.

**Step 6: Test unknown feature warning**

```
/dnd-alignment sliding_window banana mean
```

Expected: `⚠ Unknown feature: banana — skipped`, pipeline computed from `sliding_window` and `mean` only.

**Step 7: Commit**

```bash
git add .claude/skills/dnd-alignment/SKILL.md docs/plans/2026-02-27-dnd-alignment-skill.md docs/plans/2026-02-27-dnd-alignment-skill-design.md
git commit -m "feat: add dnd-alignment skill — easter egg feature alignment roster and pipeline aggregator"
```

---

## Notes for implementer

- The skill file uses a nested code block (triple backtick inside triple backtick). Write it carefully — the inner examples use backtick fences that must not accidentally close the outer one. In the actual SKILL.md file there is no outer wrapper — the examples use ` ``` ` normally.
- The `description:` frontmatter field is what Claude Code uses to decide when to invoke the skill — keep it specific enough to not fire on unrelated queries.
- No Python, no tests beyond manual invocation. This is lore, not logic.
