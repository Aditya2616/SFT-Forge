"""Gate 2: transparent quality rules. Every rule records its measured value and threshold."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..detectors import language, text_rules
from ..schema import Config
from .base import Context, alive_idx, check, fail, rnd

GATE = "quality"
_WORD = re.compile(r"\w+")


def _words(s: str) -> int:
    return len(_WORD.findall(s))


def run(df: pd.DataFrame, cfg: Config, ctx: Context) -> None:
    q = cfg.quality
    idx = alive_idx(df)

    for i in idx:
        ins, res = df.at[i, "instruction"], df.at[i, "response"]

        # length bounds and ratio
        ok = len(ins.strip()) >= q.min_instruction_chars
        check(df, i, GATE, "instruction_chars", len(ins.strip()), f">= {q.min_instruction_chars}", ok)
        if not ok:
            fail(df, i, GATE, "instruction_too_short", f"instruction has {len(ins.strip())} characters",
                 len(ins.strip()), q.min_instruction_chars)
        ok = len(res.strip()) >= q.min_response_chars
        check(df, i, GATE, "response_chars", len(res.strip()), f">= {q.min_response_chars}", ok)
        if not ok:
            fail(df, i, GATE, "response_too_short", f"response has {len(res.strip())} characters",
                 len(res.strip()), q.min_response_chars)
        ok = len(res) <= q.max_response_chars
        if not ok:
            check(df, i, GATE, "response_chars_max", len(res), f"<= {q.max_response_chars}", False)
            fail(df, i, GATE, "response_too_long", f"response has {len(res)} characters", len(res), q.max_response_chars)
        iw, rw = max(_words(ins), 1), _words(res)
        ratio = rw / iw
        ok = ratio <= q.max_length_ratio
        check(df, i, GATE, "length_ratio", rnd(ratio, 1), f"<= {q.max_length_ratio}", ok)
        if not ok:
            fail(df, i, GATE, "length_ratio", f"response is {ratio:.0f}x longer than the instruction",
                 rnd(ratio, 1), q.max_length_ratio)

        # character mix
        nonspace = re.sub(r"\s", "", res)
        alnum = (sum(c.isalnum() for c in nonspace) / len(nonspace)) if nonspace else 0.0
        ok = alnum >= q.min_alnum_ratio
        check(df, i, GATE, "alnum_ratio", rnd(alnum), f">= {q.min_alnum_ratio}", ok)
        if not ok:
            fail(df, i, GATE, "low_alnum_ratio", f"only {alnum:.0%} of the response is letters or digits",
                 rnd(alnum), q.min_alnum_ratio)

        # repetition
        rep = text_rules.repeated_trigram_share(res) if rw >= q.repetition.min_words else 0.0
        ok = rep <= q.repetition.max_repeated_trigram_share
        check(df, i, GATE, "repeated_trigrams", rnd(rep), f"<= {q.repetition.max_repeated_trigram_share}", ok)
        if not ok:
            fail(df, i, GATE, "repetitive", f"{rep:.0%} of word trigrams repeat", rnd(rep),
                 q.repetition.max_repeated_trigram_share)
        df.at[i, "scores"]["rep"] = rnd(rep)
        df.at[i, "scores"]["alnum"] = rnd(alnum)

        # refusal / boilerplate
        if q.refusal_boilerplate:
            hit = text_rules.find_refusal(res)
            check(df, i, GATE, "refusal_boilerplate", hit, "none", hit is None)
            if hit:
                fail(df, i, GATE, "refusal_boilerplate", f"matches boilerplate phrase \"{hit}\"", hit, None)

        # truncation
        if q.truncation.enabled:
            sig = text_rules.truncation_signal(res, q.truncation.unpunctuated_min_words)
            check(df, i, GATE, "truncation", sig, "none", sig is None)
            if sig:
                fail(df, i, GATE, "truncated", f"response looks cut off ({sig.replace('_', ' ')})", sig, None)
    ctx.progress("quality", 0.3)

    # language
    if q.language.enabled:
        allowed = {a.upper() for a in q.language.allowed}
        for i in alive_idx(df):
            text = (df.at[i, "instruction"] + " " + df.at[i, "response"])[:700]
            letters = sum(c.isalpha() for c in text)
            if letters < q.language.min_chars:
                check(df, i, GATE, "language", None, f"other <= {q.language.max_other_confidence}", True,
                      "too short to judge")
                continue
            conf = language.language_confidences(text)
            other_lang, other = max(((k, v) for k, v in conf.items() if k not in allowed), key=lambda kv: kv[1])
            mine = sum(v for k, v in conf.items() if k in allowed)
            ok = not (other >= q.language.max_other_confidence and mine < q.language.max_allowed_confidence)
            check(df, i, GATE, "language", f"{other_lang.title()} {other:.2f}",
                  f"other < {q.language.max_other_confidence} or allowed >= {q.language.max_allowed_confidence}", ok)
            if not ok:
                fail(df, i, GATE, "wrong_language", f"text reads as {other_lang.title()} ({other:.2f})",
                     rnd(other, 2), q.language.max_other_confidence)
    ctx.progress("quality", 0.5)

    # relevance (embeddings)
    rel_all: dict[int, float] = {}
    todo = alive_idx(df)
    if q.relevance.enabled and todo:
        ctx.ensure_embeddings(df, todo)
        ctx_idx = [i for i in todo if df.at[i, "context"].strip()]
        e_ctx = {}
        if ctx_idx:
            vecs = ctx.models.embedder.encode([df.at[i, "context"][:800] for i in ctx_idx])
            e_ctx = {i: vecs[k] for k, i in enumerate(ctx_idx)}
        for i in todo:
            cos = float(ctx.e_ins[i] @ ctx.e_res[i])
            if i in e_ctx:
                cos = max(cos, float(e_ctx[i] @ ctx.e_res[i]))
            rel_all[i] = cos
            if _words(df.at[i, "response"]) < q.relevance.min_response_words:
                check(df, i, GATE, "relevance_cosine", rnd(cos), f">= {q.relevance.min_cosine}", True,
                      "short answer, not judged")
                continue
            ok = cos >= q.relevance.min_cosine
            check(df, i, GATE, "relevance_cosine", rnd(cos), f">= {q.relevance.min_cosine}", ok)
            if not ok:
                fail(df, i, GATE, "off_topic", f"instruction and response are unrelated (cosine {cos:.2f})",
                     rnd(cos), q.relevance.min_cosine)
    ctx.progress("quality", 0.8)

    # optional perplexity
    if q.perplexity.enabled:
        todo = alive_idx(df)
        ppl = ctx.models.perplexity.score([df.at[i, "response"] for i in todo])
        for i, p in zip(todo, ppl, strict=True):
            ok = p <= q.perplexity.max_ppl
            check(df, i, GATE, "perplexity", rnd(p, 1), f"<= {q.perplexity.max_ppl}", ok)
            df.at[i, "scores"]["ppl"] = rnd(p, 1)
            if not ok:
                fail(df, i, GATE, "high_perplexity", f"response perplexity {p:.0f} under SmolLM2-135M", rnd(p, 1),
                     q.perplexity.max_ppl)

    # combined 0-1 score (display and balancing tie-break only; verdicts come from the rules above)
    for i in range(len(df)):
        sc = df.at[i, "scores"]
        if "alnum" not in sc:
            continue
        rel = rel_all.get(i)
        rel_n = 0.7 if rel is None else float(np.clip(rel / 0.5, 0, 1))
        length_n = min(1.0, _words(df.at[i, "response"]) / 20)
        score = 0.45 * rel_n + 0.20 * (1 - sc["rep"]) + 0.20 * length_n + 0.15 * min(1.0, sc["alnum"] / 0.7)
        sc["q"] = rnd(score)
        if rel is not None:
            sc["rel"] = rnd(rel)
    ctx.progress("quality", 1.0)
