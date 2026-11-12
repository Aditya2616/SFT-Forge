from __future__ import annotations

from collections import Counter
from pathlib import Path

from sftforge import pipeline
from sftforge.loaders import load_path
from sftforge.schema import Config

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "messy_sample.jsonl"


def _reasons(res) -> Counter:
    return Counter(r[0]["c"] for r in res.df["reasons"] if r and r[0]["g"] != "balance")


def test_messy_sample_trips_every_gate(models):
    loaded = load_path(SAMPLE)
    assert len(loaded.df) == 60 and not loaded.errors
    # The stub embedder only compares shared words, so the relevance rule is exercised with real models (slow tests).
    cfg = Config.model_validate({"quality": {"relevance": {"enabled": False}}})
    res = pipeline.run(loaded.df, cfg, models)
    reasons = _reasons(res)
    assert reasons["exact_duplicate"] == 2
    assert reasons["near_duplicate"] >= 2
    assert reasons["refusal_boilerplate"] == 5
    assert reasons["truncated"] == 3
    assert reasons["wrong_language"] == 2
    assert reasons["secret"] == 2
    assert reasons["toxicity"] == 1
    gates = {g["gate"]: g["rejected"] for g in res.audit["gates"]}
    assert gates["dedupe"] >= 4 and gates["quality"] == 10 and gates["safety"] == 3
    text = " ".join(res.df[res.df["status"] == "kept"]["response"])
    assert "[EMAIL]" in text and "[PHONE]" in text and "[CARD]" in text
    assert "example.com" not in text and "4111" not in text

