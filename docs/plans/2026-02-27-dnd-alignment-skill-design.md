# Design: DnD Alignment Skill

**Date:** 2026-02-27
**Status:** Approved

## Summary

Add a `/dnd-alignment` skill that ranks every cobrabox feature on the D&D 9-alignment grid (Law/Chaos × Good/Evil). When given a list of feature names it computes the aggregate pipeline alignment and renders an ASCII chart. This is a project easter egg — a repeatable, consistent canonical alignment for every feature.

## Skill location

`.claude/skills/dnd-alignment/SKILL.md`

## Feature alignment table

| Feature | Law score | Good score | Alignment | Rationale |
|---|---|---|---|---|
| `sliding_window` | +1 (Lawful) | +1 (Good) | Lawful Good | Rigidly structured, principled expansion of data — serves understanding |
| `mean` | +1 (Lawful) | 0 (Neutral) | Lawful Neutral | Collapses information by strict rule, neither creates nor destroys meaning |
| `max` | +1 (Lawful) | -1 (Evil) | Lawful Evil | Obeys the law of the maximum, ruthlessly discards everything else |
| `min` | 0 (Neutral) | -1 (Evil) | Neutral Evil | Seeks only the lowest — no principle beyond pessimism |
| `line_length` | 0 (Neutral) | +1 (Good) | Neutral Good | Measures without judgement, in service of signal |
| `dummy` | -1 (Chaotic) | 0 (Neutral) | Chaotic Neutral | Print statements, no validation — chaos without malice, just bad practice |

## Axes encoding

- Law axis: Lawful = +1, Neutral = 0, Chaotic = −1
- Good axis: Good = +1, Neutral = 0, Evil = −1
- Snapping rule: average ≥ 0.33 → +1, ≤ −0.33 → −1, else 0

## Modes

**No arguments** (`/dnd-alignment`):
- Print full roster: one line per feature with alignment label and justification
- Print full 3×3 ASCII chart with all features placed

**With feature list** (`/dnd-alignment feat1 feat2 ...`):
- Look up each feature (warn if unknown)
- Average Law scores and Good scores independently
- Snap to nearest cell
- Print pipeline sequence, ASCII chart with `[X]` at aggregate and initials at individual cells, and alignment label

## Output format (pipeline mode)

```
Pipeline: sliding_window → line_length → mean

         LAW      NEUTRAL    CHAOS
         ───────────────────────────
GOOD  │  [sw]      [ll]       [ ]  │
NEUT  │  [mn]      [ ]        [ ]  │
EVIL  │  [ ]       [ ]        [ ]  │

Aggregate: Lawful Good  ★
```

## File structure

```
.claude/skills/dnd-alignment/
└── SKILL.md
```

No helper scripts — pure skill, LLM does the arithmetic.
