"""Language identification with lingua (bundled models, no download), restricted to a language subset."""
from __future__ import annotations

from functools import lru_cache

_SUBSET = [
    "ENGLISH", "SPANISH", "FRENCH", "GERMAN", "ITALIAN", "PORTUGUESE", "DUTCH", "RUSSIAN", "CHINESE",
    "JAPANESE", "KOREAN", "ARABIC", "HINDI", "TURKISH", "POLISH", "UKRAINIAN", "VIETNAMESE", "INDONESIAN",
    "SWEDISH",
]


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    langs = [getattr(Language, n) for n in _SUBSET]
    return LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()


def language_confidences(text: str) -> dict[str, float]:
    """Map language name -> confidence in [0, 1] (sums to 1 over the subset)."""
    vals = _detector().compute_language_confidence_values(text)
    return {v.language.name: float(v.value) for v in vals}
