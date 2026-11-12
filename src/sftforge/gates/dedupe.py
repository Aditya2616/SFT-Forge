"""Gate 1: exact, near (MinHash + LSH) and semantic (embedding) deduplication.

The first occurrence in input order survives; later copies are rejected with a pointer to their twin.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

from ..normalize import key_text
from ..schema import Config
from .base import Context, alive_idx, check, fail, rnd

GATE = "dedupe"


def _shingles(text: str, k: int) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) <= k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def run(df: pd.DataFrame, cfg: Config, ctx: Context) -> None:
    d = cfg.dedupe
    ids = df["id"].tolist()

    if d.exact:
        seen: dict[str, int] = {}
        for i in alive_idx(df):
            row = df.loc[i]
            h = hashlib.sha1(key_text(row.instruction, row.context, row.response).encode()).hexdigest()
            if h in seen:
                df.at[i, "twin"] = ids[seen[h]]
                fail(df, i, GATE, "exact_duplicate", f"identical to {ids[seen[h]]} after normalisation", 1.0, 1.0)
            else:
                seen[h] = i
                check(df, i, GATE, "exact_duplicate", False, "sha1 match", True)
    ctx.progress("dedupe", 0.15)

    if d.near.enabled:
        # LSH only proposes candidates, so it runs a little below the target; the exact Jaccard decides
        lsh = MinHashLSH(threshold=max(0.1, d.near.threshold - 0.1), num_perm=d.near.num_perm)
        sets: dict[int, set[str]] = {}
        for i in alive_idx(df):
            sh = _shingles(df.at[i, "instruction"] + " " + df.at[i, "response"], d.near.shingle_words)
            mh = MinHash(num_perm=d.near.num_perm)
            for s in sh:
                mh.update(s.encode())
            best, best_j = None, 0.0
            for key in lsh.query(mh):
                j = _jaccard(sh, sets[int(key)])
                if j > best_j:
                    best, best_j = int(key), j
            if best is not None and best_j >= d.near.threshold:
                df.at[i, "twin"] = ids[best]
                fail(df, i, GATE, "near_duplicate", f"Jaccard {best_j:.2f} with {ids[best]}", rnd(best_j, 2),
                     d.near.threshold)
            else:
                sets[i] = sh
                lsh.insert(str(i), mh)
                check(df, i, GATE, "near_duplicate_jaccard", rnd(best_j, 2) if best is not None else None,
                      d.near.threshold, True, "" if best is not None else "no LSH candidate")
    ctx.progress("dedupe", 0.35)

    if d.semantic.enabled:
        idx = alive_idx(df)
        ctx.ensure_embeddings(df, idx)
        ctx.progress("dedupe", 0.8)
        vec = ctx.e_ins[idx] + ctx.e_res[idx]
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / np.clip(norm, 1e-9, None)
        by_cat: dict[str, list[int]] = {}
        for pos, i in enumerate(idx):
            by_cat.setdefault(df.at[i, "category"], []).append(pos)
        for positions in by_cat.values():
            m = vec[positions]
            sims = m @ m.T
            kept = np.ones(len(positions), dtype=bool)
            for k in range(len(positions)):
                i = idx[positions[k]]
                if k == 0:
                    check(df, i, GATE, "semantic_max_cosine", None, d.semantic.threshold, True, "first in category")
                    continue
                row = np.where(kept[:k], sims[k, :k], -1.0)
                j = int(np.argmax(row))
                best = float(row[j])
                if best >= d.semantic.threshold:
                    kept[k] = False
                    twin = idx[positions[j]]
                    df.at[i, "twin"] = ids[twin]
                    fail(df, i, GATE, "semantic_duplicate", f"cosine {best:.2f} with {ids[twin]}", rnd(best, 2),
                         d.semantic.threshold)
                else:
                    check(df, i, GATE, "semantic_max_cosine", rnd(best, 2), d.semantic.threshold, True)
    ctx.progress("dedupe", 1.0)
