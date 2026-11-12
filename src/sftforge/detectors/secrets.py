"""Secret and credential detection: known token formats plus an entropy-checked generic assignment."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SecretHit:
    kind: str
    start: int
    end: int
    text: str


_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "aws_secret_key": re.compile(r"(?i)aws_?secret_?(?:access_)?key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "stripe_key": re.compile(r"\b[sr]k_live_[A-Za-z0-9]{16,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY(?: BLOCK)?-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
}
_GENERIC = re.compile(
    r"(?i)\b(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|passwd|password)\b"
    r"\s*[=:]\s*['\"]([^'\"\s]{12,})['\"]"
)


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {c: s.count(c) for c in set(s)}
    return -sum(n / len(s) * math.log2(n / len(s)) for n in counts.values())


def find_secrets(text: str, min_entropy: float = 3.0) -> list[SecretHit]:
    hits = [SecretHit(k, m.start(), m.end(), m.group()) for k, p in _PATTERNS.items() for m in p.finditer(text)]
    for m in _GENERIC.finditer(text):
        if shannon_entropy(m.group(1)) >= min_entropy:
            hits.append(SecretHit("generic_credential", m.start(1), m.end(1), m.group(1)))
    return sorted(hits, key=lambda h: h.start)


def mask(value: str) -> str:
    """Display form of a secret: first 4 and last 4 characters only."""
    if len(value) <= 10:
        return "█" * len(value)
    return value[:4] + "█" * (len(value) - 8) + value[-4:]
