"""Orchestration: normalise, dedupe, quality, safety, balance, audit."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import audit, balance
from .gates import dedupe, quality, safety
from .gates.base import Context, init_state
from .models import Models
from .normalize import normalize_text
from .schema import Config

Progress = Callable[[str, float], None]


@dataclass
class Result:
    df: pd.DataFrame
    audit: dict[str, Any]


def normalise_frame(df: pd.DataFrame, cfg: Config) -> None:
    for i in range(len(df)):
        for f in ("instruction", "context", "response"):
            new, edits = normalize_text(df.at[i, f], cfg.normalize.strip_citations)
            if new != df.at[i, f]:
                df.at[i, f] = new
            for e in edits:
                if e not in df.at[i, "edits"]:
                    df.at[i, "edits"].append(e)


def run(raw: pd.DataFrame, cfg: Config | None = None, models: Models | None = None,
        progress: Progress | None = None, load_errors: int = 0) -> Result:
    cfg = cfg or Config()
    models = models or Models(cfg.models)
    raw_hash = audit.input_hash(raw)
    df = init_state(raw)
    ctx = Context(models, progress)
    timings: dict[str, float] = {}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        out = fn()
        timings[name] = time.perf_counter() - t0
        return out

    timed("normalise", lambda: normalise_frame(df, cfg))
    timed("dedupe", lambda: dedupe.run(df, cfg, ctx))
    timed("quality", lambda: quality.run(df, cfg, ctx))
    timed("safety", lambda: safety.run(df, cfg, ctx))
    info = timed("balance", lambda: balance.run(df, cfg.balance, cfg.seed))
    timings["total"] = sum(timings.values())
    rep = audit.build(df, cfg, raw_hash, info["cap"], timings, load_errors, models.device_label)  # type: ignore[arg-type]
    return Result(df, rep)
