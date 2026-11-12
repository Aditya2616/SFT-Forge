"""Lazy wrappers around the small local models. Each sits behind a tiny protocol so tests can stub it."""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .schema import ModelsCfg


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class ToxicityScorer(Protocol):
    def score(self, texts: list[str]) -> list[dict[str, float]]: ...


class PerplexityScorer(Protocol):
    def score(self, texts: list[str]) -> list[float]: ...


class SbertEmbedder:
    def __init__(self, name: str, revision: str | None = None, max_seq_length: int = 128):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, revision=revision, device="cpu")
        self.model.max_seq_length = max_seq_length

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self.model.encode(
            texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)


class ToxicBert:
    """unitary/toxic-bert: multi-label (toxic, severe_toxic, obscene, threat, insult, identity_hate)."""

    def __init__(self, name: str, revision: str | None = None, max_tokens: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision).eval()
        n = len(self.model.config.id2label)
        self.labels = [self.model.config.id2label[i] for i in range(n)]
        self.max_tokens = max_tokens

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        res: dict[int, dict[str, float]] = {}
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), 32):
            idx = order[s : s + 32]
            enc = self.tok([texts[i] for i in idx], truncation=True, max_length=self.max_tokens,
                           padding=True, return_tensors="pt")
            with self.torch.no_grad():
                p = self.torch.sigmoid(self.model(**enc).logits).numpy()
            for k, i in enumerate(idx):
                res[i] = {lab: float(p[k, j]) for j, lab in enumerate(self.labels)}
        return [res[i] for i in range(len(texts))]


class SmolPerplexity:
    """Perplexity of a text under SmolLM2-135M, on CPU."""

    def __init__(self, name: str, revision: str | None = None, max_tokens: int = 128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision).eval()
        self.max_tokens = max_tokens

    def score(self, texts: list[str]) -> list[float]:
        torch = self.torch
        res = [0.0] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), 16):
            idx = order[s : s + 16]
            enc = [self.tok(texts[i], truncation=True, max_length=self.max_tokens)["input_ids"] or [0] for i in idx]
            width = max(len(e) for e in enc)
            pad = self.tok.eos_token_id or 0
            ids = torch.tensor([e + [pad] * (width - len(e)) for e in enc])
            mask = torch.tensor([[1] * len(e) + [0] * (width - len(e)) for e in enc])
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask).logits
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            nll = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:].float()
            mean = (nll * m).sum(1) / m.sum(1).clamp(min=1)
            for k, i in enumerate(idx):
                res[i] = float(torch.exp(mean[k]))
        return res


class DiskCache:
    """Pickle-backed dict keyed by text hash. Used by the scripts so threshold tuning does not re-run models."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {}
        if path.exists():
            self.data = pickle.loads(path.read_bytes())

    @staticmethod
    def key(tag: str, text: str) -> str:
        return hashlib.sha1((tag + "|" + text).encode()).hexdigest()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(pickle.dumps(self.data))


class _Cached:
    def __init__(self, inner: Any, cache: DiskCache, tag: str):
        self.inner, self.cache, self.tag = inner, cache, tag

    def _run(self, texts: list[str], fn) -> list[Any]:
        keys = [self.cache.key(self.tag, t) for t in texts]
        missing = sorted({k: t for k, t in zip(keys, texts, strict=True) if k not in self.cache.data}.items())
        if missing:
            for k, v in zip((k for k, _ in missing), fn([t for _, t in missing]), strict=True):
                self.cache.data[k] = v
            self.cache.save()
        return [self.cache.data[k] for k in keys]


class CachedEmbedder(_Cached):
    def encode(self, texts: list[str]) -> np.ndarray:
        out = self._run(texts, lambda t: list(self.inner.encode(t)))
        return np.stack(out) if out else np.zeros((0, 384), dtype=np.float32)


class CachedToxicity(_Cached):
    def score(self, texts: list[str]) -> list[dict[str, float]]:
        return self._run(texts, self.inner.score)


class Models:
    """Builds real models on first use; tests pass stubs instead. cache_dir enables on-disk result caching."""

    def __init__(self, cfg: ModelsCfg, embedder: Embedder | None = None,
                 toxicity: ToxicityScorer | None = None, perplexity: PerplexityScorer | None = None,
                 cache_dir: str | Path | None = None):
        self.cfg = cfg
        self._embedder, self._tox, self._ppl = embedder, toxicity, perplexity
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            emb: Embedder = SbertEmbedder(self.cfg.embedder, self.cfg.embedder_revision)
            if self.cache_dir:
                emb = CachedEmbedder(emb, DiskCache(self.cache_dir / "embed.pkl"), self.cfg.embedder)
            self._embedder = emb
        return self._embedder

    @property
    def toxicity(self) -> ToxicityScorer:
        if self._tox is None:
            tox: ToxicityScorer = ToxicBert(self.cfg.toxicity, self.cfg.toxicity_revision)
            if self.cache_dir:
                tox = CachedToxicity(tox, DiskCache(self.cache_dir / "toxicity.pkl"), self.cfg.toxicity)
            self._tox = tox
        return self._tox

    @property
    def perplexity(self) -> PerplexityScorer:
        if self._ppl is None:
            self._ppl = SmolPerplexity(self.cfg.perplexity, self.cfg.perplexity_revision)
        return self._ppl
