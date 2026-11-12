"""Shared helpers for the scripts: disjoint Dolly slices and JSONL IO."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOLLY = ROOT / "data" / "databricks-dolly-15k" / "databricks-dolly-15k.jsonl"
SPLIT_SEED = 7
# Three disjoint slices of the shuffled Dolly rows: threshold tuning, benchmark report, headline run.
SLICES = {"dev": (0, 3000), "bench": (3000, 6000), "headline": (6000, 11000)}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def dolly_slice(name: str) -> list[dict]:
    rows = read_jsonl(DOLLY)
    for i, r in enumerate(rows):
        r["id"] = f"d{i:05d}"
    random.Random(SPLIT_SEED).shuffle(rows)
    a, b = SLICES[name]
    return rows[a:b]
