"""Gate 0: text normalisation. Edits are logged, never counted as rejections."""
from __future__ import annotations

import re
import unicodedata

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CITE = re.compile(r"\[\d{1,3}\]")
_SPACES = re.compile(r"[ \t ]+")
_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str, strip_citations: bool = True) -> tuple[str, list[str]]:
    """Return (clean text, list of edit names applied)."""
    edits: list[str] = []
    out = unicodedata.normalize("NFKC", text or "")
    if out != (text or ""):
        edits.append("unicode_nfkc")
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    no_ctrl = _CTRL.sub("", out)
    if no_ctrl != out:
        edits.append("control_chars")
    out = no_ctrl
    if strip_citations:
        no_cite = _CITE.sub("", out)
        if no_cite != out:
            edits.append("citation_markers")
        out = no_cite
    collapsed = _SPACES.sub(" ", out)
    collapsed = "\n".join(line.strip() for line in collapsed.split("\n"))
    collapsed = _NEWLINES.sub("\n\n", collapsed).strip()
    if collapsed != out.strip():
        edits.append("whitespace")
    return collapsed, edits


def key_text(*parts: str) -> str:
    """Case- and whitespace-insensitive key used for exact duplicate hashing."""
    joined = " ␟ ".join(p for p in parts)
    return re.sub(r"\s+", " ", joined.lower()).strip()
