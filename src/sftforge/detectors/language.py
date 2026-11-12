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


def strip_proper_nouns(text: str) -> str:
    """Drop capitalised words that do not start a sentence, so lists of names do not decide the language."""
    out: list[str] = []
    start = True
    for tok in text.split():
        first = next((c for c in tok if c.isalpha()), "")
        if not (first.isupper() and not start):
            out.append(tok)
        start = tok.endswith((".", "!", "?", ":"))
    return " ".join(out)


def language_confidences(text: str) -> dict[str, float]:
    """Map language name -> confidence in [0, 1] (sums to 1 over the subset)."""
    vals = _detector().compute_language_confidence_values(text)
    return {v.language.name: float(v.value) for v in vals}
