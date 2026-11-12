"""Config models and shared constants."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelsCfg(_Cfg):
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedder_revision: str | None = None
    toxicity: str = "unitary/toxic-bert"
    toxicity_revision: str | None = None
    perplexity: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    perplexity_revision: str | None = None


class NormalizeCfg(_Cfg):
    strip_citations: bool = True


class NearCfg(_Cfg):
    enabled: bool = True
    threshold: float = 0.80
    shingle_words: int = 5
    num_perm: int = 128


class SemanticCfg(_Cfg):
    enabled: bool = True
    threshold: float = 0.95


class DedupeCfg(_Cfg):
    exact: bool = True
    near: NearCfg = NearCfg()
    semantic: SemanticCfg = SemanticCfg()


class RepetitionCfg(_Cfg):
    min_words: int = 30
    max_repeated_trigram_share: float = 0.50


class TruncationCfg(_Cfg):
    enabled: bool = True
    unpunctuated_min_words: int = 0


class LanguageCfg(_Cfg):
    enabled: bool = True
    allowed: list[str] = ["ENGLISH"]
    min_chars: int = 40
    max_other_confidence: float = 0.80
    max_allowed_confidence: float = 0.20


class RelevanceCfg(_Cfg):
    enabled: bool = True
    min_cosine: float = 0.10
    min_response_words: int = 8


class PerplexityCfg(_Cfg):
    enabled: bool = False
    max_ppl: float = 2500


class QualityCfg(_Cfg):
    min_instruction_chars: int = 5
    min_response_chars: int = 2
    max_response_chars: int = 20000
    max_length_ratio: float = 400
    min_alnum_ratio: float = 0.40
    repetition: RepetitionCfg = RepetitionCfg()
    truncation: TruncationCfg = TruncationCfg()
    refusal_boilerplate: bool = True
    language: LanguageCfg = LanguageCfg()
    relevance: RelevanceCfg = RelevanceCfg()
    perplexity: PerplexityCfg = PerplexityCfg()


class PiiCfg(_Cfg):
    action: str = "redact"
    types: list[str] = ["email", "phone", "card", "ssn", "ipv4", "iban"]


class ToxicityCfg(_Cfg):
    enabled: bool = True
    labels: list[str] = ["toxic", "severe_toxic", "threat", "identity_hate"]
    threshold: float = 0.50
    max_tokens: int = 128
    chunk_chars: int = 500
    max_chunks: int = 5
    allow_categories: list[str] = ["classification"]


class SafetyCfg(_Cfg):
    pii: PiiCfg = PiiCfg()
    secrets: bool = True
    toxicity: ToxicityCfg = ToxicityCfg()
    unsafe_rules: bool = True


class BalanceCfg(_Cfg):
    enabled: bool = True
    max_share: float = 0.20
    val_fraction: float = 0.05
    min_rows: int = 100


class ExportCfg(_Cfg):
    chat_template: str = "chatml"
    hf_dataset: bool = True


class Config(_Cfg):
    seed: int = 42
    models: ModelsCfg = ModelsCfg()
    normalize: NormalizeCfg = NormalizeCfg()
    dedupe: DedupeCfg = DedupeCfg()
    quality: QualityCfg = QualityCfg()
    safety: SafetyCfg = SafetyCfg()
    balance: BalanceCfg = BalanceCfg()
    export: ExportCfg = ExportCfg()

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        if path is None:
            return cls()
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)


GATES = ("dedupe", "quality", "safety")
CATEGORIES_NONE = "uncategorised"
