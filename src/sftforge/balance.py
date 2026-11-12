"""Final stage: cap dominant categories, then a seeded stratified train/val split."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gates.base import alive_idx, check
from .schema import BalanceCfg

BALANCED_OUT = "balanced_out"
KEPT = "kept"


def category_cap(counts: dict[str, int], max_share: float) -> int | None:
    """Largest per-category cap c such that no category exceeds max_share of the capped total.

    Returns None when no cap is needed. A category can only be trimmed, never padded, so with very
    few categories the target share may be unreachable; the cap then stops at the smallest feasible value.
    """
    if not counts:
        return None
    biggest = max(counts.values())
    for c in range(biggest, 0, -1):
        total = sum(min(n, c) for n in counts.values())
        if c <= max_share * total + 1e-9:
            return None if c >= biggest else c
    return 1


def run(df: pd.DataFrame, cfg: BalanceCfg, seed: int) -> dict[str, object]:
    idx = alive_idx(df)
    info: dict[str, object] = {"cap": None}
    by_cat: dict[str, list[int]] = {}
    for i in idx:
        by_cat.setdefault(df.at[i, "category"], []).append(i)
    # with k categories no set can have every share below 1/k, so the target never goes under 1/k + 5 points
    share = max(cfg.max_share, 1 / max(len(by_cat), 1) + 0.05)
    cap = category_cap({k: len(v) for k, v in by_cat.items()}, share) if cfg.enabled and len(by_cat) > 1 and len(idx) >= cfg.min_rows else None
    info["cap"], info["max_share"] = cap, round(share, 3)
    for cat, members in by_cat.items():
        members_sorted = sorted(members, key=lambda i: (-(df.at[i, "scores"].get("q") or 0.0), i))
        keep = set(members_sorted if cap is None else members_sorted[:cap])
        for i in members:
            if i in keep:
                df.at[i, "status"] = KEPT
                check(df, i, "balance", "category_cap", cap, cap, True)
            else:
                df.at[i, "status"] = BALANCED_OUT
                df.at[i, "gate"] = "balance"
                df.at[i, "reasons"].append({
                    "g": "balance", "c": "category_cap",
                    "m": f"category '{cat}' capped at {cap} (max share {share:.0%}); lower quality score dropped",
                    "v": df.at[i, "scores"].get("q"), "t": cap,
                })
    _split(df, cfg.val_fraction, seed)
    return info


def _split(df: pd.DataFrame, val_fraction: float, seed: int) -> None:
    """Seeded split stratified by category, using Hugging Face Datasets when there is enough data."""
    kept = [int(i) for i in np.flatnonzero((df["status"] == KEPT).to_numpy())]
    for i in kept:
        df.at[i, "split"] = "train"
    if val_fraction <= 0 or len(kept) < 20:
        return
    from datasets import Dataset

    cats = df.loc[kept, "category"]
    counts = cats.value_counts()
    # singletons cannot be stratified; they stay in train
    splittable = [i for i in kept if counts[df.at[i, "category"]] >= 2]
    if len(splittable) < 20:
        return
    ds = Dataset.from_dict({"pos": splittable, "category": [df.at[i, "category"] for i in splittable]})
    ds = ds.class_encode_column("category")
    n_val = max(len(counts), round(len(splittable) * val_fraction))
    parts = ds.train_test_split(test_size=n_val, seed=seed, stratify_by_column="category")
    for pos in parts["test"]["pos"]:
        df.at[pos, "split"] = "val"
