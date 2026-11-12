"""Build a synthetic-contamination benchmark from a clean Dolly slice.

Known defects are injected at known rates with a seeded RNG. The ground truth is written beside the data,
so the pipeline can be scored per gate and per defect type. Everything here is synthetic contamination:
the PII and secrets are documentation or test values that match real formats but are not valid credentials.

Sources: Dolly-15k rows (CC BY-SA 3.0), non-English chat turns from OpenAssistant oasst1 (Apache-2.0),
and toxic snippets from AllenAI real-toxicity-prompts (Apache-2.0).
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from pathlib import Path

from common import ROOT, dolly_slice, read_jsonl, write_jsonl

N_PER = {
    "exact_dup": 60, "near_dup": 60, "paraphrase_dup": 60, "truncated": 60, "refusal": 60,
    "wrong_language": 60, "off_topic": 60, "toxic": 45,
}
N_PII = 15      # per PII type
N_SECRET = 15   # per secret type

REFUSALS = [
    "I'm sorry, but I can't help with that request.",
    "I cannot assist with this. Please contact a professional for help.",
    "I apologize, but I am unable to provide that information.",
    "Sorry, but I can't do that.",
    "I'm sorry, but I cannot complete this task.",
    "I am unable to answer questions like this one.",
]
AI_BOILERPLATE = [
    "As an AI language model, I don't have personal opinions, but here is what I can say.",
    "As an AI, I do not have access to real-time information.",
    "As a language model, I cannot browse the internet, but I can try to help.",
    "My knowledge cutoff means this may be out of date.",
]
SYNONYMS = {
    "big": "large", "large": "big", "small": "little", "begin": "start", "start": "begin", "use": "employ",
    "help": "assist", "show": "display", "buy": "purchase", "many": "numerous", "also": "additionally",
    "however": "but", "but": "however", "important": "significant", "main": "primary", "get": "obtain",
    "make": "create", "need": "require", "often": "frequently", "usually": "typically", "people": "individuals",
    "very": "really", "good": "great", "bad": "poor", "fast": "quick", "quick": "fast", "find": "locate",
    "give": "provide", "provide": "give", "about": "regarding", "because": "since", "since": "because",
    "different": "distinct", "common": "frequent", "famous": "well-known", "city": "town", "country": "nation",
    "world": "globe", "company": "firm", "build": "construct", "choose": "select", "end": "finish",
    "keep": "retain", "place": "location", "said": "stated", "know": "understand", "try": "attempt",
}
TEMPLATES = ["Could you tell me: {x}", "Question: {x}", "{x} Please answer.", "I would like to know the following. {x}"]

EMAILS = ["jane.doe@example.com", "support@example.org", "john_smith92@example.net", "test.user@mail.example.com"]
PHONES = ["(415) 555-0142", "+1 212-555-0188", "650.555.0123", "415-555-0199", "+44 20 7946 0958"]
CARDS = ["4111 1111 1111 1111", "5555 5555 5555 4444", "3782 822463 10005", "4242-4242-4242-4242"]
SSNS = ["078-05-1120", "123-45-6789", "219-09-9999"]
IPS = ["192.0.2.44", "198.51.100.23", "203.0.113.9", "192.0.2.201"]
IBANS = ["GB82 WEST 1234 5698 7654 32", "DE89 3704 0044 0532 0130 00", "FR14 2004 1010 0505 0001 3M02 606"]
PII_SENTENCES = {
    "pii_email": ("email", EMAILS, "For follow-up questions, write to {v}."),
    "pii_phone": ("phone", PHONES, "You can also call me on {v}."),
    "pii_card": ("card", CARDS, "Charge it to the card {v} if needed."),
    "pii_ssn": ("ssn", SSNS, "My social security number is {v}."),
    "pii_ipv4": ("ipv4", IPS, "The server is reachable at {v}."),
    "pii_iban": ("iban", IBANS, "Please send the payment to IBAN {v}."),
}


def secret_snippets(rng: random.Random) -> dict[str, str]:
    """Fake credentials built at runtime from fragments (never stored as literals in the repo)."""
    aws = "AKIA" + "IOSFODNN7" + "EXAMPLE"
    gh = "gh" + "p_" + "0123456789abcdefghijklmnopqrstuvwxyz"
    pem = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIEFAKEFAKEFAKEFAKEFAKE\n-----END RSA " + "PRIVATE KEY-----"
    generic = "f3A9xQ2LmP7vZk0TbW8y" + str(rng.randint(10, 99))
    return {
        "secret_aws": f"Here is my config:\nAWS_ACCESS_KEY_ID={aws}",
        "secret_github": f"I used this token in CI: {gh}",
        "secret_private_key": f"Key file contents:\n{pem}",
        "secret_generic": f'In settings.py I set:\nAPI_KEY = "{generic}"',
    }


def words(s: str) -> int:
    return len(re.findall(r"\w+", s))


def paraphrase(text: str, rng: random.Random) -> str:
    def swap(m: re.Match[str]) -> str:
        w = m.group(0)
        rep = SYNONYMS.get(w.lower())
        if rep is None or rng.random() > 0.8:
            return w
        return rep.capitalize() if w[0].isupper() else rep
    out = re.sub(r"[A-Za-z]+", swap, text)
    sents = re.split(r"(?<=[.!?])\s+", out)
    if len(sents) >= 3:
        k = rng.randrange(len(sents) - 1)
        sents[k], sents[k + 1] = sents[k + 1], sents[k]
        if rng.random() < 0.5:
            sents.pop(rng.randrange(len(sents)))
    return " ".join(sents)


def near_edit(text: str, rng: random.Random) -> str:
    toks = text.split(" ")
    for _ in range(rng.choice([1, 2])):
        k = rng.randrange(len(toks))
        w = toks[k]
        if len(w) > 3:
            j = rng.randrange(1, len(w) - 1)
            toks[k] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]  # swap two letters: a typo
        else:
            toks[k] = w + "s"
    return " ".join(toks)


def load_foreign_pairs(rng: random.Random, need: int) -> list[tuple[str, str, str]]:
    """Non-English (prompter, assistant) pairs from oasst1, 'ready' split."""
    path = ROOT / "data" / "oasst1" / "2023-04-12_oasst_ready.messages.jsonl.gz"
    msgs: dict[str, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            msgs[m["message_id"]] = m
    pairs = []
    for m in msgs.values():
        if m["role"] == "assistant" and m.get("lang") not in (None, "en") and m.get("parent_id") in msgs:
            p = msgs[m["parent_id"]]
            if len(m["text"]) >= 120 and len(p["text"]) >= 30 and len(m["text"]) <= 900:
                pairs.append((p["text"], m["text"], m["lang"]))
    rng.shuffle(pairs)
    return pairs[:need]


def load_toxic(rng: random.Random, need: int) -> list[str]:
    out = []
    for r in read_jsonl(ROOT / "data" / "real-toxicity-prompts" / "prompts.jsonl"):
        p = r.get("prompt") or {}
        t, tox = p.get("text") or "", p.get("toxicity")
        if tox is not None and tox >= 0.9 and 40 <= len(t) <= 220:
            out.append(t.strip())
    rng.shuffle(out)
    return out[:need]


def build(slice_name: str, seed: int, n: int) -> tuple[list[dict], dict[str, dict]]:
    rng = random.Random(seed)
    base = dolly_slice(slice_name)[:n]
    base = [dict(r, category=r["category"]) for r in base]
    by_id = {r["id"]: r for r in base}
    truth: dict[str, dict] = {}
    pool = [r["id"] for r in base]
    rng.shuffle(pool)
    cursor = 0

    def take(k: int, pred=lambda r: True) -> list[dict]:
        nonlocal cursor
        out = []
        while len(out) < k and cursor < len(pool):
            r = by_id[pool[cursor]]
            cursor += 1
            if pred(r):
                out.append(r)
        if len(out) < k:
            raise RuntimeError("not enough rows for the requested defects; use a larger slice")
        return out

    def long_enough(r: dict) -> bool:
        return words(r["response"]) >= 40 and len(r["response"]) >= 200

    copies: list[tuple[dict, str]] = []

    for r in take(N_PER["exact_dup"]):
        c = dict(r, id=f"x_exact_{r['id']}")
        copies.append((c, r["id"]))
        truth[c["id"]] = {"defect": "exact_dup", "source": r["id"]}
    for r in take(N_PER["near_dup"], long_enough):
        c = dict(r, id=f"x_near_{r['id']}", response=near_edit(r["response"], rng))
        copies.append((c, r["id"]))
        truth[c["id"]] = {"defect": "near_dup", "source": r["id"]}
    for r in take(N_PER["paraphrase_dup"], lambda r: words(r["response"]) >= 25):
        ins = rng.choice(TEMPLATES).format(x=paraphrase(r["instruction"], rng))
        c = dict(r, id=f"x_para_{r['id']}", instruction=ins, response=paraphrase(r["response"], rng))
        copies.append((c, r["id"]))
        truth[c["id"]] = {"defect": "paraphrase_dup", "source": r["id"]}

    for r in take(N_PER["truncated"], long_enough):
        toks = r["response"].split(" ")
        cut = max(8, int(len(toks) * rng.uniform(0.4, 0.8)))
        r["response"] = " ".join(toks[:cut]).rstrip(".,;:!?")
        truth[r["id"]] = {"defect": "truncated"}
    for k, r in enumerate(take(N_PER["refusal"])):
        if k % 2 == 0:
            r["response"] = rng.choice(REFUSALS)
        else:
            first = re.split(r"(?<=[.!?])\s+", r["response"])[0]
            r["response"] = rng.choice(AI_BOILERPLATE) + " " + first
        truth[r["id"]] = {"defect": "refusal"}
    foreign = load_foreign_pairs(rng, N_PER["wrong_language"])
    for r, (ins, res, lang) in zip(take(N_PER["wrong_language"]), foreign, strict=True):
        r["instruction"], r["context"], r["response"] = ins, "", res
        truth[r["id"]] = {"defect": "wrong_language", "lang": lang}
    off = take(N_PER["off_topic"], lambda r: words(r["response"]) >= 15)
    for r in off:
        other = rng.choice([o for o in base if o["category"] != r["category"] and words(o["response"]) >= 15])
        r["response"] = other["response"]
        truth[r["id"]] = {"defect": "off_topic"}
    for name, (_, vals, tpl) in PII_SENTENCES.items():
        for r in take(N_PII):
            r["response"] = r["response"].rstrip() + " " + tpl.format(v=rng.choice(vals))
            truth[r["id"]] = {"defect": name}
    snippets = secret_snippets(rng)
    for name, snip in snippets.items():
        for r in take(N_SECRET):
            r["response"] = r["response"].rstrip() + "\n\n" + snip
            truth[r["id"]] = {"defect": name}
    toxic = load_toxic(rng, N_PER["toxic"])
    for r, t in zip(take(N_PER["toxic"], lambda r: r["category"] != "classification"), toxic, strict=True):
        r["response"] = r["response"].rstrip() + "\n\n" + t
        truth[r["id"]] = {"defect": "toxic"}

    # shuffle; every copy lands after its original so the original is the one that survives deduplication
    key = {r["id"]: rng.random() for r in base}
    placed = [(key[r["id"]], r) for r in base]
    for c, src in copies:
        k = key[src]
        placed.append((k + (1 - k) * rng.random() * 0.999 + 1e-6, c))
    placed.sort(key=lambda kv: kv[0])
    rows = [r for _, r in placed]
    return rows, truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="bench", choices=["dev", "bench"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ROOT / "data" / f"contaminated_{args.slice}")
    rows, truth = build(args.slice, args.seed, args.n)
    write_jsonl(out / "benchmark.jsonl", rows)
    (out / "truth.json").write_text(json.dumps(truth, indent=0), encoding="utf-8")
    from collections import Counter

    print(f"{len(rows)} rows, {len(truth)} injected ({len(truth) / len(rows):.1%}) -> {out}")
    print(dict(Counter(v["defect"] for v in truth.values())))


if __name__ == "__main__":
    main()
