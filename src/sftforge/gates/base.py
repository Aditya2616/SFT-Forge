"""Shared state helpers. The audit trail lives in object columns of the working DataFrame."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..models import Models

ALIVE = "alive"
REJECTED = "rejected"


class Context:
    """Per-run shared objects: models, embedding cache, progress callback."""

    def __init__(self, models: Models, progress=None):
        self.models = models
        self.progress = progress or (lambda stage, frac: None)
        self.e_ins: np.ndarray | None = None
        self.e_res: np.ndarray | None = None
        self.have_emb: np.ndarray | None = None

    def ensure_embeddings(self, df: pd.DataFrame, idx: list[int]) -> None:
        todo = [i for i in idx if self.have_emb is None or not self.have_emb[i]]
        if not todo:
            return
        enc = self.models.embedder.encode
        a = enc([df.at[i, "instruction"][:500] for i in todo])
        b = enc([df.at[i, "response"][:800] for i in todo])
        if self.e_ins is None:
            n = len(df)
            self.e_ins = np.zeros((n, a.shape[1]), dtype=np.float32)
            self.e_res = np.zeros((n, a.shape[1]), dtype=np.float32)
            self.have_emb = np.zeros(n, dtype=bool)
        self.e_ins[todo], self.e_res[todo] = a, b
        self.have_emb[todo] = True


def init_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    n = len(df)
    df["status"] = ALIVE
    df["gate"] = None
    df["twin"] = None
    df["reasons"] = [[] for _ in range(n)]
    df["checks"] = [{} for _ in range(n)]
    df["scores"] = [{} for _ in range(n)]
    df["redactions"] = [[] for _ in range(n)]
    df["edits"] = [[] for _ in range(n)]
    df["split"] = None
    return df


def alive_idx(df: pd.DataFrame) -> list[int]:
    return [int(i) for i in np.flatnonzero((df["status"] == ALIVE).to_numpy())]


def check(df: pd.DataFrame, i: int, gate: str, rule: str, value: Any, threshold: Any, passed: bool, note: str = "") -> None:
    entry: dict[str, Any] = {"r": rule, "v": value, "t": threshold, "p": passed}
    if note:
        entry["n"] = note
    df.at[i, "checks"].setdefault(gate, []).append(entry)


def fail(df: pd.DataFrame, i: int, gate: str, code: str, msg: str, value: Any = None, threshold: Any = None) -> None:
    """Record a failed rule. The row is marked rejected at the first gate that fails it."""
    df.at[i, "reasons"].append({"g": gate, "c": code, "m": msg, "v": value, "t": threshold})
    if df.at[i, "status"] == ALIVE:
        df.at[i, "status"] = REJECTED
        df.at[i, "gate"] = gate


def rnd(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(float(x), nd)
