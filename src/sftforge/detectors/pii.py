"""Regex PII detection with validators (Luhn, IBAN mod-97, SSN ranges, IPv4 octets).

Names and street addresses are out of scope for regexes; see the README limitations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PLACEHOLDER = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "card": "[CARD]",
    "ssn": "[SSN]",
    "ipv4": "[IP]",
    "iban": "[IBAN]",
}


@dataclass(frozen=True)
class Span:
    kind: str
    start: int
    end: int
    text: str


_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w-])"
    r"|(?<![\w.])\+\d{1,3}[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?![\w-])"
)
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_SSN = re.compile(r"(?<![\w-])(\d{3})-(\d{2})-(\d{4})(?![\w-])")
_IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.]*\w)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def iban_ok(raw: str) -> bool:
    s = raw.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in moved)
    return int(num) % 97 == 1


def _ssn_ok(m: re.Match[str]) -> bool:
    area, group, serial = m.group(1), m.group(2), m.group(3)
    return area not in ("000", "666") and not area.startswith("9") and group != "00" and serial != "0000"


def _ip_ok(text: str) -> bool:
    parts = text.split(".")
    if not all(p.isdigit() and 0 <= int(p) <= 255 and (p == "0" or not p.startswith("0")) for p in parts):
        return False
    return not (text.startswith("127.") or text == "0.0.0.0")


def find_pii(text: str, types: list[str] | None = None) -> list[Span]:
    want = set(types or PLACEHOLDER)
    spans: list[Span] = []
    if "email" in want:
        spans += [Span("email", m.start(), m.end(), m.group()) for m in _EMAIL.finditer(text)]
    if "iban" in want:
        spans += [Span("iban", m.start(), m.end(), m.group()) for m in _IBAN.finditer(text) if iban_ok(m.group())]
    if "card" in want:
        for m in _CARD.finditer(text):
            digits = re.sub(r"[ -]", "", m.group())
            if 13 <= len(digits) <= 19 and luhn_ok(digits) and len(set(digits)) > 1:
                spans.append(Span("card", m.start(), m.end(), m.group()))
    if "ssn" in want:
        spans += [Span("ssn", m.start(), m.end(), m.group()) for m in _SSN.finditer(text) if _ssn_ok(m)]
    if "ipv4" in want:
        spans += [Span("ipv4", m.start(), m.end(), m.group()) for m in _IPV4.finditer(text) if _ip_ok(m.group())]
    if "phone" in want:
        spans += [Span("phone", m.start(), m.end(), m.group()) for m in _PHONE.finditer(text)]
    return _drop_overlaps(spans)


def _drop_overlaps(spans: list[Span]) -> list[Span]:
    """Keep the earliest, longest span when two detectors hit the same characters."""
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    for s in spans:
        if out and s.start < out[-1].end:
            continue
        out.append(s)
    return out


def redact(text: str, spans: list[Span]) -> str:
    out, pos = [], 0
    for s in spans:
        out.append(text[pos : s.start])
        out.append(PLACEHOLDER[s.kind])
        pos = s.end
    out.append(text[pos:])
    return "".join(out)
