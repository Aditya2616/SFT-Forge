"""Audit report: what each gate removed, why, and what the balanced set looks like."""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import pandas as pd

from . import __version__
from .schema import GATES, Config


def input_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for a, b, c in zip(df["instruction"], df["context"], df["response"], strict=True):
        h.update(f"{a}\x1f{b}\x1f{c}\x1e".encode())
    return h.hexdigest()


def build(df: pd.DataFrame, cfg: Config, raw_hash: str, cap: int | None, timings: dict[str, float],
          load_errors: int = 0) -> dict[str, Any]:
    n = len(df)
    gates: list[dict[str, Any]] = []
    remaining = n
    for g in GATES:
        rej = df[df["gate"] == g]
        reasons = Counter()
        for rs in rej["reasons"]:
            if rs and rs[0]["g"] == g:
                reasons[rs[0]["c"]] += 1
        gates.append({"gate": g, "in": remaining, "rejected": len(rej), "out": remaining - len(rej),
                      "reasons": dict(reasons.most_common())})
        remaining -= len(rej)
    bal = df[df["gate"] == "balance"]
    gates.append({"gate": "balance", "in": remaining, "rejected": len(bal), "out": remaining - len(bal),
                  "reasons": {"category_cap": len(bal)} if len(bal) else {}})

    survivors_before = df[~df["gate"].isin(["dedupe", "quality", "safety"])]
    kept = df[df["status"] == "kept"]
    cats_before = survivors_before["category"].value_counts().to_dict()
    cats_after = kept["category"].value_counts().to_dict()
    cats_in = df["category"].value_counts().to_dict()
    redactions: Counter[str] = Counter()
    for rr in kept["redactions"]:
        for r in rr:
            redactions[r["k"]] += 1
    edits: Counter[str] = Counter()
    for ee in df["edits"]:
        for e in ee:
            edits[e] += 1
    return {
        "tool": "sftforge",
        "version": __version__,
        "seed": cfg.seed,
        "input_sha256": raw_hash,
        "rows_in": n,
        "load_errors": load_errors,
        "rows_kept": len(kept),
        "rows_train": int((df["split"] == "train").sum()),
        "rows_val": int((df["split"] == "val").sum()),
        "keep_rate": round(len(kept) / n, 4) if n else 0.0,
        "gates": gates,
        "balance_cap": cap,
        "categories": {
            c: {"in": int(cats_in.get(c, 0)), "after_gates": int(cats_before.get(c, 0)), "final": int(cats_after.get(c, 0))}
            for c in sorted(cats_in)
        },
        "redactions_in_kept": dict(redactions),
        "edits": dict(edits),
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "config": cfg.model_dump(),
    }


def to_markdown(a: dict[str, Any]) -> str:
    lines = [
        "# Audit report",
        "",
        f"- Rows in: **{a['rows_in']}**, kept: **{a['rows_kept']}** ({a['keep_rate']:.1%}); "
        f"train {a['rows_train']}, validation {a['rows_val']}",
        f"- Input SHA-256: `{a['input_sha256'][:16]}...`, seed {a['seed']}, sftforge {a['version']}",
        "",
        "## Gates",
        "",
        "| Gate | In | Rejected | Out | Reasons |",
        "|---|---:|---:|---:|---|",
    ]
    for g in a["gates"]:
        why = ", ".join(f"{k} {v}" for k, v in g["reasons"].items()) or "-"
        lines.append(f"| {g['gate']} | {g['in']} | {g['rejected']} | {g['out']} | {why} |")
    lines += ["", "## Categories", "", "| Category | Input | After gates | Final |", "|---|---:|---:|---:|"]
    for c, v in a["categories"].items():
        lines.append(f"| {c} | {v['in']} | {v['after_gates']} | {v['final']} |")
    if a["balance_cap"]:
        lines.append(f"\nPer-category cap applied: {a['balance_cap']} rows.")
    if a["redactions_in_kept"]:
        lines += ["", "## Redactions in kept rows", ""] + [f"- {k}: {v}" for k, v in a["redactions_in_kept"].items()]
    if a["edits"]:
        lines += ["", "## Normalisation edits (not rejections)", ""] + [f"- {k}: {v}" for k, v in a["edits"].items()]
    lines += ["", "## Timings (seconds)", ""] + [f"- {k}: {v}" for k, v in a["timings_s"].items()]
    return "\n".join(lines) + "\n"
