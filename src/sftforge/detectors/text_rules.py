"""Small explainable text rules: refusal/boilerplate, truncation, repetition, unsafe-request patterns."""
from __future__ import annotations

import re

# AI-identity and template phrases count anywhere in the response.
_BOILERPLATE = re.compile("|".join([
    r"\bas an ai(?: language model| assistant)?\b",
    r"\bas a (?:large )?language model\b",
    r"\bi(?:'m| am) (?:just )?an ai\b",
    r"\bmy (?:knowledge|training) (?:cut-?off|data)\b",
    r"\bi do(?:n't| not) have (?:personal )?(?:opinions|feelings|access to real-time)\b",
    r"\blorem ipsum\b",
    r"\[(?:insert|your|placeholder)[^\]]{0,40}\]",
]), re.IGNORECASE)
# Refusal verbs only count at the very start of a short response, so quoted speech is not caught.
_REFUSAL_START = re.compile("|".join([
    r"^\W*(?:i(?:'m| am) (?:so |very )?sorry,? but,? )?i (?:cannot|can't|can not|am unable to|won't be able to|will not) "
    r"(?:assist|help|provide|fulfil+|comply|answer|do that|do this|complete|generate|create|write)",
    r"^\W*i apologi[sz]e,? but",
    r"^\W*i(?:'m| am) (?:so |very )?sorry,? but",
    r"^\W*(?:sorry|unfortunately),? (?:but )?i (?:cannot|can't|am unable|won't)",
]), re.IGNORECASE)
_PLACEHOLDER_ONLY = re.compile(r"^\W*(?:n/?a|none|null|todo|tbd|\.{2,}|-+|placeholder)\W*$", re.IGNORECASE)

_DANGLING = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "with", "by", "from", "and", "or", "but", "as",
    "that", "which", "who", "is", "are", "was", "were", "be", "been", "has", "have", "had", "will", "would",
    "can", "could", "should", "may", "might", "its", "their", "his", "her", "our", "your", "this", "these",
    "those", "than", "then", "if", "because", "while", "when", "into", "about", "between", "such",
}

_UNSAFE = {
    "self_harm_instructions": re.compile(
        r"\b(?:how (?:do i|to|can i)|ways to|best way to) (?:kill myself|commit suicide|end my life|hurt myself)\b", re.IGNORECASE),
    "weapon_construction": re.compile(
        r"\b(?:how (?:do i|to|can i)|steps to|instructions (?:for|to)) (?:make|build|construct|assemble) (?:a |an )?"
        r"(?:pipe bomb|bomb|explosive|nerve agent|chemical weapon|ghost gun)\b", re.IGNORECASE),
    "drug_synthesis": re.compile(
        r"\b(?:how (?:do i|to|can i)|steps to|synthesi[sz]e|cook) (?:make |synthesi[sz]e |cook |produce )?"
        r"(?:meth(?:amphetamine)?|fentanyl|heroin|crack cocaine)\b", re.IGNORECASE),
}


def find_refusal(text: str) -> str | None:
    """Return the matched phrase, or None. Placeholder-only responses also count."""
    t = text.strip()
    if _PLACEHOLDER_ONLY.match(t):
        return t[:40]
    m = _BOILERPLATE.search(t)
    if m:
        return m.group(0)
    if len(t.split()) <= 60:
        m = _REFUSAL_START.search(t[:160])
        if m:
            return t[: m.end()].strip()
    return None


def truncation_signal(text: str, unpunctuated_min_words: int = 0) -> str | None:
    """Name of the truncation signal found at the end of the text, or None."""
    t = text.rstrip()
    if not t:
        return None
    if t.count("```") % 2 == 1:
        return "unclosed_code_fence"
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    for a, b in pairs:
        if t.count(a) > t.count(b) and len(t) > 40:
            return f"unbalanced_{a}{b}"
    if t.count('"') % 2 == 1 and len(t) > 60 and t[-1] not in '."!?':
        return "unbalanced_quote"
    last = t[-1]
    if last in ",;:-–—/(":
        return "ends_with_connector_punctuation"
    if last in ".!?\"')]}”’*`>%" or not (last.isalnum()):
        return None
    words = re.findall(r"[\w']+", t)
    if len(words) >= 8 and words[-1].lower() in _DANGLING:
        return "ends_on_function_word"
    if unpunctuated_min_words and len(t.split()) >= unpunctuated_min_words and "\n" not in t[-80:]:
        return "long_text_without_terminal_punctuation"
    return None


def repeated_trigram_share(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 6:
        return 0.0
    grams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    return 1.0 - len(set(grams)) / len(grams)


def unsafe_rule_hits(text: str) -> list[str]:
    return [name for name, pat in _UNSAFE.items() if pat.search(text)]
