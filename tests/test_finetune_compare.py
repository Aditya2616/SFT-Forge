from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import finetune_compare as fc  # noqa: E402


def test_rouge_l_and_verbatim_helpers():
    assert fc.rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert fc.rouge_l("alpha beta", "gamma delta") == 0.0
    assert 0 < fc.rouge_l("the cat sat", "the cat sat on the mat") < 1
    grams = fc.ngram_set(["one two three four five six seven eight nine ten eleven twelve thirteen"], 8)
    assert fc.is_verbatim("one two three four five six seven eight nine ten eleven twelve", grams)
    assert not fc.is_verbatim("a completely different sentence that shares nothing with the training row at all", grams)
    assert not fc.is_verbatim("one two three", grams)  # too short to judge


def test_learning_rate_schedule_warms_up_then_decays():
    cfg = fc.CFG
    assert fc.lr_at(0) < fc.lr_at(cfg["warmup_steps"] - 1) <= cfg["lr"]
    assert math.isclose(fc.lr_at(cfg["warmup_steps"]), cfg["lr"], rel_tol=1e-6)
    assert math.isclose(fc.lr_at(cfg["steps"] - 1), cfg["lr"] * cfg["min_lr_fraction"], rel_tol=0.02)


def test_trim_keeps_short_prompts_and_clips_long_ones():
    assert fc.trim("a b c", 5) == "a b c"
    assert fc.trim("a b c d e f", 3) == "a b c"


def _run(cond: str, seed: int, loss: float, rouge: float) -> dict:
    m = {k: 0.1 for k, _, _ in fc.METRICS}
    m.update(loss=loss, rouge_l=rouge, perplexity=math.exp(loss))
    return {"condition": cond, "seed": seed, "device": "cpu, float32", "train_rows": 10, "injected_rows": 1, "metrics": m,
            "curve_every_10_steps": [3.0, 2.0], "train_seconds": 1.0, "eval_seconds": 1.0, "peak_vram_gib": 0.0}


def test_summary_pairs_runs_by_seed_and_reports_intervals(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    losses = {"raw": [2.30, 2.32, 2.31], "cleaned": [2.20, 2.22, 2.21], "random": [2.25, 2.27, 2.26]}
    for cond, ls in losses.items():
        for seed, loss in enumerate(ls):
            (runs / f"{cond}_s{seed}.json").write_text(json.dumps(_run(cond, seed, loss, 0.25)), encoding="utf-8")
    monkeypatch.setattr(fc, "RUNS", runs)
    s = fc.summarize("cpu, float32", {})
    d = s["paired"]["cleaned_minus_raw"]["loss"]
    assert math.isclose(d["mean_diff"], -0.10, abs_tol=1e-9)
    assert d["excludes_zero"] and d["favours_cleaned"] is True
    flat = s["paired"]["cleaned_minus_raw"]["rouge_l"]
    assert flat["mean_diff"] == 0 and flat["favours_cleaned"] is None
    assert s["conditions"]["raw"]["metrics"]["loss"]["sd"] > 0


def test_roc_auc_helper_in_perplexity_script():
    import eval_perplexity as ep

    assert ep.roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert ep.roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert ep.roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5
