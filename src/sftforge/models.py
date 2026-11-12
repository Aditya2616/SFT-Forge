"""Lazy wrappers around the small local models. Each sits behind a tiny protocol so tests can stub it."""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .schema import ModelsCfg


def resolve_device(requested: str | None = None) -> str:
    """'cuda' when available, else 'cpu'. Override with the argument or the SFTFORGE_DEVICE variable."""
    want = (requested or os.environ.get("SFTFORGE_DEVICE") or "auto").lower()
    if want != "auto":
        return want
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class ToxicityScorer(Protocol):
    def score(self, texts: list[str]) -> list[dict[str, float]]: ...


class PerplexityScorer(Protocol):
    def score(self, texts: list[str]) -> list[float]: ...


class SbertEmbedder:
    def __init__(self, name: str, revision: str | None = None, max_seq_length: int = 128, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.device = device
        self.model = SentenceTransformer(name, revision=revision, device=device)
        self.model.max_seq_length = max_seq_length

    def _encode(self, texts: list[str], batch: int) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        try:
            return self._encode(texts, 64)
        except Exception as exc:  # a CUDA out-of-memory error retries smaller, then on the CPU
            if not _is_oom(exc):
                raise
            import torch

            torch.cuda.empty_cache()
            try:
                return self._encode(texts, 8)
            except Exception as exc2:
                if not _is_oom(exc2):
                    raise
                self.model.to("cpu")
                self.device = "cpu"
                return self._encode(texts, 32)


class ToxicBert:
    """unitary/toxic-bert: multi-label (toxic, severe_toxic, obscene, threat, insult, identity_hate)."""

    def __init__(self, name: str, revision: str | None = None, max_tokens: int = 128, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision).eval().to(device)
        n = len(self.model.config.id2label)
        self.labels = [self.model.config.id2label[i] for i in range(n)]
        self.max_tokens = max_tokens

    def score(self, texts: list[str]) -> list[dict[str, float]]:
        res: dict[int, dict[str, float]] = {}
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        batch, s = 32, 0
        while s < len(order):
            idx = order[s : s + batch]
            enc = self.tok([texts[i] for i in idx], truncation=True, max_length=self.max_tokens,
                           padding=True, return_tensors="pt")
            try:
                with self.torch.no_grad():
                    p = self.torch.sigmoid(self.model(**enc.to(self.device)).logits).float().cpu().numpy()
            except Exception as exc:  # CUDA out of memory: retry smaller, then fall back to the CPU
                if not _is_oom(exc):
                    raise
                self.torch.cuda.empty_cache()
                if batch > 4:
                    batch //= 4
                else:
                    self.model.to("cpu")
                    self.device = "cpu"
                continue
            s += len(idx)
            for k, i in enumerate(idx):
                res[i] = {lab: float(p[k, j]) for j, lab in enumerate(self.labels)}
        return [res[i] for i in range(len(texts))]


class SmolPerplexity:
    """Perplexity of a text under SmolLM2-135M-Instruct (float32 on the chosen device)."""

    def __init__(self, name: str, revision: str | None = None, max_tokens: int = 128, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision).eval().to(device)
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
            ids = torch.tensor([e + [pad] * (width - len(e)) for e in enc], device=self.device)
            mask = torch.tensor([[1] * len(e) + [0] * (width - len(e)) for e in enc], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask).logits
            lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            nll = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            m = mask[:, 1:].float()
            mean = (nll * m).sum(1) / m.sum(1).clamp(min=1)
            for k, i in enumerate(idx):
                res[i] = float(torch.exp(mean[k]).cpu())
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


class CachedPerplexity(_Cached):
    def score(self, texts: list[str]) -> list[float]:
        return self._run(texts, self.inner.score)


class Models:
    """Builds real models on first use; tests pass stubs instead. cache_dir enables on-disk result caching."""

    def __init__(self, cfg: ModelsCfg, embedder: Embedder | None = None,
                 toxicity: ToxicityScorer | None = None, perplexity: PerplexityScorer | None = None,
                 cache_dir: str | Path | None = None, device: str | None = None):
        self.cfg = cfg
        self._device_req = device
        self._device: str | None = None
        self._embedder, self._tox, self._ppl = embedder, toxicity, perplexity
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @property
    def device(self) -> str:
        """Device the real models run on, chosen on first use. Injected stubs ignore it."""
        if self._device is None:
            self._device = resolve_device(self._device_req)
            if self._device.startswith("cuda"):
                import torch

                torch.set_num_threads(2)  # the GPU does the work; keep the CPU free
        return self._device

    @property
    def device_label(self) -> str:
        """Device description recorded in the audit report."""
        if self._embedder is not None and not hasattr(self._embedder, "model") and not hasattr(self._embedder, "inner"):
            return "stub models"
        dev = self.device
        if dev.startswith("cuda"):
            import torch

            return f"{dev} ({torch.cuda.get_device_name(0)}), float32"
        return f"{dev}, float32"

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            emb: Embedder = SbertEmbedder(self.cfg.embedder, self.cfg.embedder_revision, device=self.device)
            if self.cache_dir:
                emb = CachedEmbedder(emb, DiskCache(self.cache_dir / "embed.pkl"), f"{self.cfg.embedder}@{self.device}")
            self._embedder = emb
        return self._embedder

    @property
    def toxicity(self) -> ToxicityScorer:
        if self._tox is None:
            tox: ToxicityScorer = ToxicBert(self.cfg.toxicity, self.cfg.toxicity_revision, device=self.device)
            if self.cache_dir:
                tox = CachedToxicity(tox, DiskCache(self.cache_dir / "toxicity.pkl"), f"{self.cfg.toxicity}@{self.device}")
            self._tox = tox
        return self._tox

    @property
    def perplexity(self) -> PerplexityScorer:
        if self._ppl is None:
            ppl: PerplexityScorer = SmolPerplexity(self.cfg.perplexity, self.cfg.perplexity_revision, device=self.device)
            if self.cache_dir:
                ppl = CachedPerplexity(ppl, DiskCache(self.cache_dir / "perplexity.pkl"), f"{self.cfg.perplexity}@{self.device}")
            self._ppl = ppl
        return self._ppl
