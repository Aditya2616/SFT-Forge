from __future__ import annotations

import json
import re
import zlib

import numpy as np
import pytest

from sftforge.loaders import load_lines
from sftforge.models import Models
from sftforge.schema import BalanceCfg, Config


class HashEmbedder:
    """Deterministic bag-of-words embedding: similar word sets give similar vectors. No model needed."""

    dim = 1024

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for k, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[k, zlib.crc32(w.encode()) % self.dim] += 1.0
            n = np.linalg.norm(out[k])
            if n:
                out[k] /= n
        return out


class StubToxicity:
    WORDS = ("idiot", "moron", "i will hurt you")

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        res = []
        for t in texts:
            hit = any(w in t.lower() for w in self.WORDS)
            res.append({"toxic": 0.98 if hit else 0.01, "severe_toxic": 0.05, "obscene": 0.02,
                        "threat": 0.9 if "hurt you" in t.lower() else 0.01, "insult": 0.5 if hit else 0.01,
                        "identity_hate": 0.01})
        return res


@pytest.fixture()
def cfg() -> Config:
    # balancing is off by default in tests so small fixtures are not capped; balance tests turn it on
    return Config(balance=BalanceCfg(enabled=False, val_fraction=0))


@pytest.fixture()
def models(cfg: Config) -> Models:
    return Models(cfg.models, embedder=HashEmbedder(), toxicity=StubToxicity())


def make_df(rows: list[dict]):
    return load_lines(json.dumps(r) for r in rows).df


def row(ins: str, res: str, cat: str = "open_qa", ctx: str = "", rid: str | None = None) -> dict:
    r = {"instruction": ins, "context": ctx, "response": res, "category": cat}
    if rid:
        r["id"] = rid
    return r


@pytest.fixture()
def good_rows() -> list[dict]:
    topics = [
        ("What is the capital of France?", "The capital of France is Paris, a city on the river Seine known for museums."),
        ("Explain how photosynthesis works.", "Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar and oxygen."),
        ("Name three primary colours.", "The three primary colours are red, yellow and blue, from which other colours are mixed."),
        ("How do I boil an egg?", "Place the egg in boiling water for about nine minutes, then cool it under cold water."),
        ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare around the year sixteen hundred, as a tragedy."),
        ("Why is the sky blue?", "The sky looks blue because air scatters short blue wavelengths of sunlight more than red."),
        ("What does a compiler do?", "A compiler translates source code written in a programming language into machine code."),
        ("Describe the water cycle.", "Water evaporates, condenses into clouds, falls as rain and flows back to the sea again."),
    ]
    return [row(i, r, cat=("open_qa" if k % 2 == 0 else "general_qa"), rid=f"g{k}") for k, (i, r) in enumerate(topics)]
