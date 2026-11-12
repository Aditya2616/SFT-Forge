"""Serialise a pipeline result to the compact JSON the web UI replays (live runs and recorded demos)."""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from .detectors import secrets
from .pipeline import Result

STATUS = {"kept": "k", "rejected": "x", "balanced_out": "b"}
LIMITS = {"instruction": 400, "context": 300, "response": 700}


def _show(text: str, limit: int) -> str:
    for h in sorted(secrets.find_secrets(text), key=lambda h: -h.start):
        text = text[: h.start] + secrets.mask(h.text) + text[h.end :]
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def build(result: Result, meta: dict[str, Any], truth: dict[str, str] | None = None,
          withhold_toxic: bool = False) -> dict[str, Any]:
    df: pd.DataFrame = result.df
    recs = []
    for i in range(len(df)):
        row = df.iloc[i]
        reasons = row["reasons"]
        toxic = withhold_toxic and any(r["c"] == "toxicity" for r in reasons)
        rec: dict[str, Any] = {
            "i": i,
            "id": row["id"],
            "c": row["category"],
            "s": STATUS.get(row["status"], "x"),
            "a": _show(row["instruction"], LIMITS["instruction"]),
            "r": _show(row["response"], LIMITS["response"]),
            "n": [len(row["instruction"]), len(row["response"])],
        }
        if toxic:
            rec["a"] = "[text withheld in the public recording: toxic content]"
            rec["r"] = "[text withheld in the public recording: toxic content]"
            rec["w"] = 1
        if row["context"]:
            rec["x"] = "[text withheld]" if toxic else _show(row["context"], LIMITS["context"])
        if row["gate"]:
            rec["g"] = row["gate"]
        if row["split"]:
            rec["sp"] = row["split"]
        if reasons:
            rec["rs"] = reasons
        if row["checks"]:
            rec["ck"] = row["checks"]
        if row["twin"]:
            rec["tw"] = row["twin"]
        if row["redactions"]:
            rec["rd"] = row["redactions"]
        if row["edits"]:
            rec["ed"] = row["edits"]
        if row["scores"]:
            rec["sc"] = row["scores"]
        if truth is not None:
            rec["t"] = truth.get(row["id"], "clean")
        recs.append(rec)
    meta = {"created": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"), **meta}
    return {"meta": meta, "audit": result.audit, "records": recs}
