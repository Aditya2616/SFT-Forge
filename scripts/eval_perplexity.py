"""Does the optional SmolLM2-135M-Instruct perplexity check help? Tuned on the dev slice, reported on the bench slice.

Pre-specified rule: the check becomes a default only if, on the dev slice, some max_ppl raises the recall of the
truncated and off-topic rows by at least 5 percentage points while the false-reject rate on untouched rows rises
by at most 0.1 percentage points. The chosen threshold is then applied unchanged to the bench slice.

Writes results/perplexity_check.json. Usage: python scripts/eval_perplexity.py [--device cuda]
"""
from __future__ import annotations

import argparse
import json

from common import ROOT
from eval_benchmark import evaluate
from sftforge import pipeline
from sftforge.loaders import load_path
from sftforge.models import Models
from sftforge.schema import Config

GRID = [150, 250, 400, 600, 1000, 1500, 2500]
TARGETS = ("truncated", "off_topic")
MIN_GAIN_PP = 5.0
MAX_FALSE_PP = 0.1


def roc_auc(labels: list[int], scores: list[float]) -> float:
    """Probability that a random positive scores higher than a random negative (ties count half)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = [r for r, y in zip(ranks, labels, strict=True) if y == 1]
    n_pos, n_neg = len(pos), len(labels) - len(pos)
    return (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def run_slice(name: str, device: str | None, thresholds: list[int | None], models_cache) -> dict:
    data = ROOT / "data" / f"contaminated_{name}"
    loaded = load_path(data / "benchmark.jsonl")
    truth = json.loads((data / "truth.json").read_text(encoding="utf-8"))
    out: dict = {"n_rows": len(loaded.df), "settings": {}}
    for thr in thresholds:
        cfg = Config.load(ROOT / "configs" / "default.yaml")
        if thr is not None:
            cfg = cfg.model_copy(deep=True)
            cfg.quality.perplexity.enabled = True
            cfg.quality.perplexity.max_ppl = thr
        else:
            cfg.quality.perplexity.enabled = False
        models = Models(cfg.models, cache_dir=models_cache, device=device)
        res = pipeline.run(loaded.df, cfg, models)
        ev = evaluate(res.df, truth)
        tgt_n = sum(ev["per_defect"][d]["n"] for d in TARGETS)
        tgt_caught = sum(ev["per_defect"][d]["caught"] for d in TARGETS)
        added = res.df[res.df["reasons"].apply(lambda r: bool(r) and r[0]["c"] == "high_perplexity")]
        out["settings"]["off" if thr is None else str(thr)] = {
            "max_ppl": thr, "target_recall": round(tgt_caught / tgt_n, 4), "target_caught": tgt_caught, "target_n": tgt_n,
            "false_reject_rate": ev["untouched"]["false_reject_rate"], "false_rejects": ev["untouched"]["rejected"],
            "untouched_n": ev["untouched"]["n"], "rows_kept": res.audit["rows_kept"],
            "high_perplexity_rejects": len(added),
            "high_perplexity_on_injected": int(sum(1 for i in added["id"] if i in truth)),
            "overall_injected_recall": round(sum(v["caught"] for v in ev["per_defect"].values()) /
                                             sum(v["n"] for v in ev["per_defect"].values()), 4),
            "device": res.audit["device"],
        }
    # separation of the raw score, independent of any threshold
    cfg = Config.load(ROOT / "configs" / "default.yaml")
    models = Models(cfg.models, cache_dir=models_cache, device=device)
    scores = models.perplexity.score(list(loaded.df["response"]))
    ids = list(loaded.df["id"])
    auc = {}
    for d in TARGETS:
        y, s = [], []
        for i, sc in zip(ids, scores, strict=True):
            dfn = truth.get(i, {}).get("defect")
            if dfn == d:
                y.append(1)
                s.append(sc)
            elif dfn is None:
                y.append(0)
                s.append(sc)
        auc[d] = round(roc_auc(y, s), 3)
    out["auc_vs_untouched"] = auc
    # which untouched rows does the loosest threshold flag? (the score is the response alone, so short answers suffer)
    import statistics

    flagged = [len(r.split()) for r, i, s in zip(loaded.df["response"], ids, scores, strict=True) if i not in truth and s > GRID[-1]]
    others = [len(r.split()) for r, i, s in zip(loaded.df["response"], ids, scores, strict=True) if i not in truth and s <= GRID[-1]]
    out["untouched_flagged_at_loosest"] = {"max_ppl": GRID[-1], "n": len(flagged), "median_words": statistics.median(flagged),
                                           "median_words_not_flagged": statistics.median(others)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cache = ROOT / ".cache"
    dev = run_slice("dev", args.device, [None, *GRID], cache)
    base = dev["settings"]["off"]
    chosen, best_gain = None, 0.0
    for thr in GRID:
        s = dev["settings"][str(thr)]
        gain = 100 * (s["target_recall"] - base["target_recall"])
        extra_false = 100 * (s["false_reject_rate"] - base["false_reject_rate"])
        s["gain_pp"], s["extra_false_pp"] = round(gain, 2), round(extra_false, 2)
        if gain >= MIN_GAIN_PP and extra_false <= MAX_FALSE_PP and gain > best_gain:
            chosen, best_gain = thr, gain
    bench = run_slice("bench", args.device, [None, *([chosen] if chosen else GRID[-1:])], cache)
    for k, s in bench["settings"].items():
        if k != "off":
            b = bench["settings"]["off"]
            s["gain_pp"] = round(100 * (s["target_recall"] - b["target_recall"]), 2)
            s["extra_false_pp"] = round(100 * (s["false_reject_rate"] - b["false_reject_rate"]), 2)
    result = {"rule": {"min_gain_pp": MIN_GAIN_PP, "max_extra_false_pp": MAX_FALSE_PP, "targets": list(TARGETS),
                       "grid": GRID}, "chosen_max_ppl": chosen, "enable_by_default": chosen is not None,
              "dev": dev, "bench": bench, "model": "HuggingFaceTB/SmolLM2-135M-Instruct"}
    (ROOT / "results" / "perplexity_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chosen": chosen, "dev_auc": dev["auc_vs_untouched"], "bench_auc": bench["auc_vs_untouched"],
                      "dev": {k: (v["target_recall"], v["false_reject_rate"]) for k, v in dev["settings"].items()}}, indent=1))


if __name__ == "__main__":
    main()
