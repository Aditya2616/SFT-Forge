"""Run with: pytest -m slow. Loads the real models (needs the one-time download)."""
from __future__ import annotations

import pytest

from sftforge.models import Models
from sftforge.schema import ModelsCfg

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real() -> Models:
    return Models(ModelsCfg())


def test_embedder_ranks_related_text_higher(real):
    e = real.embedder.encode(["What is the capital of France?", "Paris is the capital of France.", "Bananas ripen near apples."])
    assert float(e[0] @ e[1]) > float(e[0] @ e[2]) + 0.2


def test_toxicity_model_separates_insults_from_benign_text(real):
    s = real.toxicity.score(["Thank you for the helpful explanation.", "You are a worthless idiot and everyone hates you."])
    assert s[0]["toxic"] < 0.1 and s[1]["toxic"] > 0.9


def test_end_to_end_with_real_models():
    from sftforge import pipeline
    from sftforge.loaders import load_text

    text = "\n".join([
        '{"instruction": "Why is the sky blue?", "output": "Air scatters short blue wavelengths of sunlight more than red ones."}',
        '{"instruction": "Why is the sky blue?", "output": "Air scatters short blue wavelengths of sunlight more than red ones."}',
        '{"instruction": "Tell me about my neighbour", "output": "He is a worthless idiot and everybody hates him."}',
    ])
    res = pipeline.run(load_text(text).df)
    reasons = [r[0]["c"] if r else None for r in res.df["reasons"]]
    assert reasons == [None, "exact_duplicate", "toxicity"]


def test_messy_sample_with_real_models():
    from collections import Counter
    from pathlib import Path

    from sftforge import pipeline
    from sftforge.loaders import load_path

    sample = Path(__file__).resolve().parent.parent / "examples" / "messy_sample.jsonl"
    res = pipeline.run(load_path(sample).df)
    c = Counter(r[0]["c"] for r in res.df["reasons"] if r and r[0]["g"] != "balance")
    assert c["off_topic"] >= 2 and c["semantic_duplicate"] >= 1 and c["toxicity"] == 1
    assert res.audit["rows_kept"] < 45
