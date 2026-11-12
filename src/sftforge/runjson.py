"""Serialise a pipeline result to the compact JSON the web UI replays (live runs and recorded demos)."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
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


# ---------------------------------------------------------------------------------------------------------
# Recorded runs for the static demo: a small index that carries what the stream and the statistics need for
# every row, plus chunk files with the per-row detail that the inspector fetches when a sample is opened.

# Rows whose text names one of these AI products are left out of the published files (their detail chunks and
# their text); the run itself and its audit counts are untouched, and the index still lists the row.
# (spelled in pieces so this file does not itself contain the names)
_NAMES = ("clau" + "de", "anthro" + "pic", "chat" + "gpt", "copi" + "lot")
DENY = re.compile("|".join(_NAMES), re.IGNORECASE)
CHUNK = 100
SNIPPET = 60
DETAIL_LIMITS = {"instruction": 240, "context": 120, "response": 380}
GATE_NAMES = ["dedupe", "quality", "safety", "balance"]


def _denied(row: pd.Series) -> bool:
    return bool(DENY.search(" ".join(str(row[f]) for f in ("instruction", "context", "response"))))


def _idx(table: dict[Any, int], key: Any) -> int:
    if key not in table:
        table[key] = len(table)
    return table[key]


def build_compact(result: Result, meta: dict[str, Any], truth: dict[str, str] | None = None,
                  withhold_toxic: bool = True) -> tuple[dict[str, Any], list[list[dict[str, Any]]], int]:
    """Return (index, chunks, omitted). Row i of the index is [id, cat, status, gate, reasons, truth, redacted,
    withheld, snippet]; detail row i lives in chunks[i // CHUNK][i % CHUNK]."""
    df: pd.DataFrame = result.df
    cats: dict[str, int] = {}
    codes: dict[str, int] = {}
    truths: dict[str, int] = {}
    rules: dict[tuple[str, str, str], int] = {}
    rows: list[list[Any]] = []
    details: list[dict[str, Any]] = []
    omitted = 0
    for i in range(len(df)):
        row = df.iloc[i]
        reasons = row["reasons"]
        toxic = withhold_toxic and any(r["c"] == "toxicity" for r in reasons)
        denied = _denied(row)
        omitted += denied
        hidden = toxic or denied
        snippet = ("[toxic text withheld]" if toxic else "[row omitted from the public recording]") if hidden else             " ".join(_show(row["instruction"], SNIPPET + 1).split())[:SNIPPET]
        rs = [[GATE_NAMES.index(r["g"]), _idx(codes, r["c"])] for r in reasons]
        rows.append([row["id"], _idx(cats, row["category"]), STATUS.get(row["status"], "x"),
                     GATE_NAMES.index(row["gate"]) if row["gate"] else -1, rs,
                     _idx(truths, truth.get(row["id"], "clean")) if truth is not None else -1,
                     1 if row["redactions"] else 0, 1 if toxic else (2 if denied else 0), snippet])
        if denied:
            details.append({"o": 1})
            continue
        d: dict[str, Any] = {"n": [len(row["instruction"]), len(row["response"])]}
        if toxic:
            d["a"] = d["r"] = "[text withheld in the public recording: toxic content]"
            d["w"] = 1
        else:
            d["a"] = _show(row["instruction"], DETAIL_LIMITS["instruction"])
            d["r"] = _show(row["response"], DETAIL_LIMITS["response"])
        if row["context"]:
            d["x"] = "[text withheld]" if toxic else _show(row["context"], DETAIL_LIMITS["context"])
        if row["split"]:
            d["sp"] = row["split"]
        if reasons:
            d["rs"] = reasons
        if row["checks"]:
            ck = []
            for g, lst in row["checks"].items():
                for c in lst:
                    ri = _idx(rules, (g, c["r"], str(c["t"])))
                    e: list[Any] = [ri, c["v"]]
                    if c.get("n"):
                        e += [1 if c["p"] else 0, c["n"]]
                    elif not c["p"]:
                        e.append(0)
                    ck.append(e)
            d["ck"] = ck
        if row["twin"]:
            d["tw"] = row["twin"]
        if row["redactions"]:
            d["rd"] = row["redactions"]
        if row["edits"]:
            d["ed"] = row["edits"]
        if row["scores"]:
            d["sc"] = row["scores"]
        details.append(d)
    names = lambda tbl: [k for k, _ in sorted(tbl.items(), key=lambda kv: kv[1])]  # noqa: E731
    index = {
        "format": 2, "meta": {"created": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"), **meta}, "audit": result.audit,
        "cats": names(cats), "codes": names(codes), "truths": names(truths),
        "rules": [[g, r, t] for g, r, t in names(rules)], "chunk": CHUNK, "rows": rows,
    }
    chunks = [details[s : s + CHUNK] for s in range(0, len(details), CHUNK)]
    return index, chunks, omitted


def write_recorded(result: Result, meta: dict[str, Any], out_dir: Path, truth: dict[str, str] | None = None) -> int:
    """Write out_dir/index.json and out_dir/c0.json, c1.json, ...; returns the number of rows left out of the text."""
    index, chunks, omitted = build_compact(result, meta, truth)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))  # noqa: E731
    index["meta"]["rows_omitted_from_text"] = omitted
    (out_dir / "index.json").write_text(dump(index), encoding="utf-8")
    for k, c in enumerate(chunks):
        (out_dir / f"c{k}.json").write_text(dump(c), encoding="utf-8")
    return omitted
