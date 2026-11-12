"""Gate 3: secrets (reject), PII (redact or reject), toxicity and unsafe-request rules (reject)."""
from __future__ import annotations

import pandas as pd

from ..detectors import pii, secrets, text_rules
from ..schema import Config
from .base import Context, alive_idx, check, fail, rnd

GATE = "safety"
FIELDS = ("instruction", "context", "response")


def run(df: pd.DataFrame, cfg: Config, ctx: Context) -> None:
    s = cfg.safety

    if s.secrets:
        for i in alive_idx(df):
            hits = [(f, h) for f in FIELDS for h in secrets.find_secrets(df.at[i, f])]
            check(df, i, GATE, "secrets", len(hits), 0, not hits)
            for f, h in hits:
                fail(df, i, GATE, "secret", f"{h.kind.replace('_', ' ')} in {f}", h.kind, None)
                df.at[i, "redactions"].append({"f": f, "k": "secret:" + h.kind, "was": secrets.mask(h.text)})

    for i in alive_idx(df):
        found = []
        for f in FIELDS:
            text = df.at[i, f]
            spans = pii.find_pii(text, s.pii.types)
            if not spans:
                continue
            found.append((f, spans))
        n = sum(len(sp) for _, sp in found)
        check(df, i, GATE, "pii", n, 0, n == 0, f"action: {s.pii.action}" if n else "")
        if not n:
            continue
        for f, spans in found:
            for sp in spans:
                df.at[i, "redactions"].append({"f": f, "k": sp.kind, "was": secrets.mask(sp.text)})
            if s.pii.action == "redact":
                df.at[i, f] = pii.redact(df.at[i, f], spans)
        kinds = sorted({sp.kind for _, spans in found for sp in spans})
        if s.pii.action == "redact":
            df.at[i, "edits"].append("pii_redacted")
        else:
            fail(df, i, GATE, "pii", f"contains {', '.join(kinds)}", n, 0)
    ctx.progress("safety", 0.4)

    if s.unsafe_rules:
        for i in alive_idx(df):
            hits = text_rules.unsafe_rule_hits(df.at[i, "instruction"])
            check(df, i, GATE, "unsafe_request_rules", ",".join(hits) or None, "none", not hits)
            if hits:
                fail(df, i, GATE, "unsafe_rule", f"instruction matches rule {hits[0]}", hits[0], None)
    ctx.progress("safety", 0.5)

    t = s.toxicity
    if t.enabled:
        idx = alive_idx(df)
        if idx:
            scorer = ctx.models.toxicity
            ins_scores = scorer.score([df.at[i, "instruction"] for i in idx])
            ctx.progress("safety", 0.7)
            res_scores = scorer.score([df.at[i, "response"] for i in idx])
            for i, a, b in zip(idx, ins_scores, res_scores, strict=True):
                worst_label, worst = max(
                    ((f"{where}:{lab}", v[lab]) for where, v in (("instruction", a), ("response", b))
                     for lab in t.labels if lab in v),
                    key=lambda kv: kv[1],
                )
                df.at[i, "scores"]["tox"] = rnd(worst)
                allowed = df.at[i, "category"] in t.allow_categories
                ok = worst < t.threshold or allowed
                note = f"{worst_label}" + (" (category allowlisted)" if allowed and worst >= t.threshold else "")
                check(df, i, GATE, "toxicity", rnd(worst), f"< {t.threshold}", ok, note)
                if not ok:
                    fail(df, i, GATE, "toxicity", f"{worst_label.split(':')[1]} {worst:.2f} in "
                         f"{worst_label.split(':')[0]}", rnd(worst), t.threshold)
    ctx.progress("safety", 1.0)
