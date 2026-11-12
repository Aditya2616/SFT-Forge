"""Read Alpaca, Dolly, chat-messages and ShareGPT JSON/JSONL into one DataFrame."""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import CATEGORIES_NONE

COLUMNS = ["id", "instruction", "context", "response", "category", "category_inferred", "source_row", "turn"]

_ROLE_USER = {"user", "human", "prompter"}
_ROLE_ASSISTANT = {"assistant", "gpt", "bot", "model"}

_CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("summarization", re.compile(r"\b(summari[sz]e|summary|tl;?dr|shorten|condense)\b", re.IGNORECASE)),
    ("classification", re.compile(r"\b(classify|categori[sz]e|which category|sentiment)\b", re.IGNORECASE)),
    ("information_extraction", re.compile(r"\b(extract|pull out)\b", re.IGNORECASE)),
    ("creative_writing", re.compile(r"\b(write (a|an|me) (poem|story|song|letter|essay|haiku)|compose|imagine)\b", re.IGNORECASE)),
    ("brainstorming", re.compile(r"\b(brainstorm|ideas?|suggest)\b", re.IGNORECASE)),
    ("closed_qa", re.compile(r"\b(based on|according to|given (the|this)|from the (text|passage|paragraph))\b", re.IGNORECASE)),
]


@dataclass
class LoadResult:
    df: pd.DataFrame
    errors: list[str] = field(default_factory=list)
    fmt: dict[str, int] = field(default_factory=dict)


def infer_category(instruction: str, context: str = "") -> str:
    """Cheap rule-based task type, used only when the input has no category column."""
    for name, pat in _CATEGORY_RULES:
        if pat.search(instruction):
            return name
    if context:
        return "closed_qa"
    return "general_qa" if instruction.rstrip().endswith("?") else "generation"


def _text(v: Any) -> str:
    if v is None:
        return ""
    return v if isinstance(v, str) else str(v)


def _pairs_from_turns(turns: list[tuple[str, str]]) -> list[tuple[int, str, str]]:
    """Pair each user turn with the assistant turn that follows it."""
    out: list[tuple[int, str, str]] = []
    pending: str | None = None
    n = 0
    for role, content in turns:
        r = role.lower()
        if r in _ROLE_USER:
            pending = content
        elif r in _ROLE_ASSISTANT and pending is not None:
            out.append((n, pending, content))
            n += 1
            pending = None
    return out


def _records_from_obj(obj: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Return (format name, list of partial records) for one JSON object."""
    turns: list[tuple[str, str]] = []
    fmt = ""
    if isinstance(obj.get("messages"), list):
        turns = [(_text(m.get("role")), _text(m.get("content"))) for m in obj["messages"] if isinstance(m, dict)]
        fmt = "chat"
    elif isinstance(obj.get("conversations"), list):
        turns = [(_text(m.get("from")), _text(m.get("value"))) for m in obj["conversations"] if isinstance(m, dict)]
        fmt = "sharegpt"
    if fmt:
        pairs = _pairs_from_turns(turns)
        if not pairs:
            raise ValueError("no user/assistant pair found")
        return fmt, [
            {"instruction": ins, "context": "", "response": res, "category": obj.get("category"), "turn": t}
            for t, ins, res in pairs
        ]
    if "instruction" in obj and ("response" in obj or "output" in obj):
        ctx = obj.get("context", obj.get("input", ""))
        fmt = "dolly" if "response" in obj else "alpaca"
        return fmt, [
            {
                "instruction": _text(obj["instruction"]),
                "context": _text(ctx),
                "response": _text(obj.get("response", obj.get("output"))),
                "category": obj.get("category"),
                "turn": 0,
                "id": obj.get("id"),
            }
        ]
    raise ValueError("unrecognised schema (need instruction+output/response, messages, or conversations)")


def load_lines(lines: Iterable[str]) -> LoadResult:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    fmts: dict[str, int] = {}
    for n, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                raise ValueError("line is not a JSON object")
            fmt, recs = _records_from_obj(obj)
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(f"line {n}: {exc}")
            continue
        fmts[fmt] = fmts.get(fmt, 0) + len(recs)
        for r in recs:
            r["source_row"] = n
            rows.append(r)
    return LoadResult(_frame(rows), errors, fmts)


def load_text(text: str) -> LoadResult:
    """Accept JSONL, or a single JSON array of objects."""
    if text.lstrip().startswith("["):
        try:
            arr = json.loads(text)
        except json.JSONDecodeError as exc:
            return LoadResult(_frame([]), [f"invalid JSON array: {exc}"])
        return load_lines(json.dumps(o) for o in arr if isinstance(o, dict))
    return load_lines(text.splitlines())


def load_path(path: str | Path) -> LoadResult:
    return load_text(Path(path).read_text(encoding="utf-8"))


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    recs = []
    seen: set[str] = set()
    for i, r in enumerate(rows):
        rid = r.get("id")
        rid = str(rid) if rid not in (None, "") and str(rid) not in seen else f"r{i:05d}"
        seen.add(rid)
        ins, ctx, res = r["instruction"], r.get("context", ""), r["response"]
        cat = r.get("category")
        inferred = False
        if not cat:
            cat, inferred = infer_category(ins, ctx), True
        recs.append(
            {
                "id": rid,
                "instruction": ins,
                "context": ctx,
                "response": res,
                "category": cat or CATEGORIES_NONE,
                "category_inferred": inferred,
                "source_row": r["source_row"],
                "turn": r.get("turn", 0),
            }
        )
    return pd.DataFrame(recs, columns=COLUMNS)
