"""Dataset card for an exported SFT set."""
from __future__ import annotations

from typing import Any

from .schema import Config


def render(a: dict[str, Any], cfg: Config, source_name: str) -> str:
    g = {x["gate"]: x for x in a["gates"]}
    cats = "\n".join(f"| {c} | {v['in']} | {v['final']} |" for c, v in a["categories"].items())
    return f"""---
task_categories:
- text-generation
tags:
- instruction-tuning
- sft
- curated
---

# Cleaned SFT set from {source_name}

Produced by SFT Forge {a['version']} (seed {a['seed']}). Source rows: {a['rows_in']}. Kept: {a['rows_kept']}
({a['keep_rate']:.1%}); train {a['rows_train']}, validation {a['rows_val']}.

Check the licence of the source data before reuse: this card does not change it.

## What was done

1. Normalisation: Unicode NFKC, whitespace, control characters, citation markers.
2. Deduplication: exact (SHA-1), near (MinHash/LSH, Jaccard >= {cfg.dedupe.near.threshold}), semantic
   (embedding cosine >= {cfg.dedupe.semantic.threshold} within category). Removed {g['dedupe']['rejected']}.
3. Quality rules (length, character mix, repetition, truncation, boilerplate/refusal, language, relevance).
   Removed {g['quality']['rejected']}.
4. Safety: secrets rejected, PII redacted to typed placeholders, toxicity and unsafe-request screening.
   Removed {g['safety']['rejected']}.
5. Balance: no category above {cfg.balance.max_share:.0%} of the final set (cap {a['balance_cap'] or 'not needed'}).
   Removed {g['balance']['rejected']}.

## Categories

| Category | Input rows | Final rows |
|---|---:|---:|
{cats}

## Known limitations

- PII detection is regex based: names and street addresses are not found.
- The toxicity classifier (unitary/toxic-bert) has known biases and was applied to the first 128 tokens.
- Thresholds are in the audit report; every rejected row and its reasons are in `rejects.jsonl`.

## Files

`train.jsonl`, `val.jsonl` (Alpaca schema), `chat_*.jsonl` (messages), `text_*.jsonl` ({cfg.export.chat_template} formatted),
`hf_dataset/`, `rejects.jsonl`, `audit.json`, `audit.md`.
"""
