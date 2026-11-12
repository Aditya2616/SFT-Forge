from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_df, row
from sftforge import audit, balance, export, pipeline
from sftforge.schema import BalanceCfg, Config


def run(rows, cfg, models):
    return pipeline.run(make_df(rows), cfg, models)


def by_id(res, rid):
    return res.df[res.df["id"] == rid].iloc[0]


def test_clean_rows_all_survive(good_rows, cfg, models):
    res = run(good_rows, cfg, models)
    assert (res.df["status"] == "kept").all()
    assert res.audit["rows_kept"] == len(good_rows)


def test_exact_duplicate_rejected_with_twin(good_rows, cfg, models):
    rows = good_rows + [dict(good_rows[0], id="copy", response=good_rows[0]["response"].upper())]
    res = run(rows, cfg, models)
    r = by_id(res, "copy")
    assert r["status"] == "rejected" and r["gate"] == "dedupe"
    assert r["reasons"][0]["c"] == "exact_duplicate" and r["twin"] == "g0"
    assert by_id(res, "g0")["status"] != "rejected"  # the original survives


def test_near_duplicate(cfg, models):
    base = ("Photosynthesis is the process by which plants use sunlight water and carbon dioxide to produce "
            "oxygen and energy in the form of sugar inside their leaves during the day. It takes place in "
            "chloroplasts which contain chlorophyll, the green pigment that absorbs light, mostly red and blue "
            "wavelengths, and it is the basis of almost every food chain on the planet earth today")
    rows = [row("Explain photosynthesis please", base, rid="a"),
            row("Explain photosynthesis please", base.replace("leaves", "leafs"), rid="b")]
    res = run(rows, cfg, models)
    assert by_id(res, "b")["reasons"][0]["c"] == "near_duplicate"


def test_semantic_duplicate_same_category_only(cfg, models):
    a = row("How tall is Mount Everest in metres", "Mount Everest is eight thousand eight hundred forty eight metres tall", rid="a")
    b = row("How tall is Mount Everest in metres exactly", "Mount Everest is eight thousand eight hundred forty eight metres tall indeed", rid="b")
    # same words in reverse order: identical for the bag-of-words stub, different shingles, other category
    c = dict(b, id="c", category="brainstorming", instruction=" ".join(reversed(b["instruction"].split())),
             response=" ".join(reversed(b["response"].split())))
    d = dict(a, id="d", instruction=" ".join(sorted(a["instruction"].split())),
             response=" ".join(sorted(a["response"].split())))
    res = run([a, b, c, d], cfg.model_copy(update={"balance": BalanceCfg(enabled=False, val_fraction=0)}), models)
    assert by_id(res, "b")["reasons"][0]["c"] in ("semantic_duplicate", "near_duplicate")
    assert by_id(res, "c")["status"] == "kept"  # same text, different category: not compared
    assert by_id(res, "d")["reasons"][0]["c"] == "semantic_duplicate"


@pytest.mark.parametrize("resp,code", [
    ("As an AI language model, I do not have a favourite colour at all.", "refusal_boilerplate"),
    ("The treaty was signed by many nations after a long war and it was meant to end the", "truncated"),
    ("la la " * 40, "repetitive"),
    ("", "response_too_short"),
    ("!!! ??? ### $$$ %%% ^^^ &&& *** ((( )))", "low_alnum_ratio"),
])
def test_quality_rules_reject_with_reason(good_rows, cfg, models, resp, code):
    res = run(good_rows + [row("Tell me something about colours and their names", resp, rid="bad")], cfg, models)
    r = by_id(res, "bad")
    assert r["status"] == "rejected" and r["gate"] == "quality"
    assert code in [x["c"] for x in r["reasons"]]
    assert all({"g", "c", "m"} <= x.keys() for x in r["reasons"])  # every reject carries reasons


def test_wrong_language_and_off_topic(good_rows, cfg, models):
    es = row("Cuéntame sobre la historia de España durante la edad media",
             "España durante la edad media estuvo dividida en varios reinos cristianos y musulmanes que lucharon durante siglos.",
             rid="es")
    off = row("What is the capital of Germany and why is it famous?",
              "Bananas ripen faster when kept next to apples inside a paper bag on a warm shelf.", rid="off")
    res = run(good_rows + [es, off], cfg, models)
    assert "wrong_language" in [x["c"] for x in by_id(res, "es")["reasons"]]
    assert "off_topic" in [x["c"] for x in by_id(res, "off")["reasons"]]


def test_short_answers_not_judged_on_relevance(good_rows, cfg, models):
    res = run(good_rows + [row("What is two plus two?", "Four", rid="short")], cfg, models)
    assert by_id(res, "short")["status"] != "rejected"


def test_pii_redacted_and_secret_rejected(good_rows, cfg, models):
    aws = "AKIA" + "IOSFODNN7" + "EXAMPLE"
    pii_row = row("How do I contact support about my order?", "You can email a.b@example.com or call (415) 555-0142 any time to contact support about your order.", rid="pii")
    sec_row = row("How do I configure the cloud client?", f"Set AWS_ACCESS_KEY_ID={aws} in your shell profile.", rid="sec")
    res = run(good_rows + [pii_row, sec_row], cfg, models)
    p = by_id(res, "pii")
    assert p["status"] != "rejected" and "[EMAIL]" in p["response"] and "[PHONE]" in p["response"]
    assert {r["k"] for r in p["redactions"]} == {"email", "phone"}
    s = by_id(res, "sec")
    assert s["status"] == "rejected" and s["gate"] == "safety" and s["reasons"][0]["c"] == "secret"
    assert aws not in json.dumps(s["redactions"])  # only a masked form is kept in the audit trail


def test_pii_reject_mode(good_rows, cfg, models):
    cfg2 = cfg.model_copy(update={"safety": cfg.safety.model_copy(update={"pii": cfg.safety.pii.model_copy(update={"action": "reject"})})})
    res = run(good_rows + [row("How do I reach you about a delivery?", "Write to a.b@example.com for questions about a delivery, reach you about it.", rid="p")], cfg2, models)
    assert by_id(res, "p")["gate"] == "safety"


def test_toxicity_rejects_but_classification_is_allowlisted(good_rows, cfg, models):
    tox = row("Tell me what you think of my neighbour", "You are an idiot and so is your neighbour, honestly.", rid="t")
    cls = row("Classify this comment as toxic or not: you are an idiot", "This comment is toxic because it insults someone.",
              cat="classification", rid="c")
    res = run(good_rows + [tox, cls], cfg, models)
    assert by_id(res, "t")["reasons"][0]["c"] == "toxicity"
    assert by_id(res, "c")["status"] != "rejected"


def test_gate_order_and_trail(good_rows, cfg, models):
    res = run(good_rows + [row("Tell me about colours", "As an AI language model I like blue. Mail me a@b.co", rid="x")], cfg, models)
    x = by_id(res, "x")
    assert x["gate"] == "quality"
    assert "safety" not in x["checks"]  # later gates do not run on a row that was already rejected
    assert "dedupe" in x["checks"]


def test_audit_totals_reconcile(good_rows, cfg, models):
    rows = good_rows * 3 + [row("Tell me about colours", "As an AI I like blue.", rid="bad")]
    res = run(rows, cfg, models)
    a = res.audit
    assert a["rows_in"] == len(rows)
    assert a["rows_kept"] + sum(g["rejected"] for g in a["gates"]) == a["rows_in"]
    for g in a["gates"]:
        assert g["in"] - g["rejected"] == g["out"]
        assert sum(g["reasons"].values()) == g["rejected"]
    assert "# Audit report" in audit.to_markdown(a)


def test_balance_cap_invariant():
    cap = balance.category_cap({"a": 100, "b": 20, "c": 20, "d": 10}, 0.4)
    assert cap is not None
    capped = {k: min(v, cap) for k, v in {"a": 100, "b": 20, "c": 20, "d": 10}.items()}
    assert max(capped.values()) <= 0.4 * sum(capped.values()) + 1e-9
    assert balance.category_cap({"a": 10, "b": 10, "c": 10}, 0.5) is None


def test_balance_and_split_deterministic(cfg, models):
    cats = ["open_qa"] * 60 + ["closed_qa"] * 20 + ["brainstorming"] * 20
    topics = ["rivers", "mountains", "cooking", "music", "football", "chess", "gardening", "planets", "bridges", "trains"]
    rows = [row(f"Tell me something about {topics[i % 10]} number {i}", f"Here is a fact about {topics[i % 10]} with detail {i} and more words to read",
                cat=c, rid=f"r{i}") for i, c in enumerate(cats)]
    cfg2 = cfg.model_copy(update={"balance": BalanceCfg(max_share=0.4, val_fraction=0.1, min_rows=0)})
    a = run(rows, cfg2, models)
    b = run(rows, cfg2, models)
    kept = a.df[a.df["status"] == "kept"]
    assert kept["category"].value_counts().max() <= 0.4 * len(kept) + 1
    assert list(a.df["split"]) == list(b.df["split"])
    val = kept[kept["split"] == "val"]
    assert 5 <= len(val) <= 9 and val["category"].nunique() == 3  # stratified: every category is in val
    dropped = a.df[a.df["status"] == "balanced_out"]
    assert len(dropped) > 0 and dropped["reasons"].iloc[0][0]["c"] == "category_cap"


def test_export_files_and_schema(tmp_path: Path, good_rows, cfg, models):
    rows = good_rows + [row("Tell me about colours", "As an AI I like blue.", rid="bad")]
    res = pipeline.run(make_df(rows), Config(balance=BalanceCfg(enabled=False, val_fraction=0)), models)
    files = export.export(res, cfg.model_copy(update={"export": cfg.export.model_copy(update={"hf_dataset": True})}), tmp_path)
    train = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) == len(good_rows) and {"instruction", "input", "output", "category", "id"} <= train[0].keys()
    chat = json.loads((tmp_path / "chat_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant"]
    text = json.loads((tmp_path / "text_train.jsonl").read_text(encoding="utf-8").splitlines()[0])["text"]
    assert text.startswith("<|im_start|>user\n") and text.endswith("<|im_end|>\n")
    rejects = [json.loads(line) for line in (tmp_path / "rejects.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rejects] == ["bad"] and rejects[0]["reasons"][0]["c"] == "refusal_boilerplate"
    assert "Cleaned SFT set" in (tmp_path / "DATASET_CARD.md").read_text(encoding="utf-8")
    from datasets import load_from_disk

    assert len(load_from_disk(str(files["hf_dataset"]))["train"]) == len(good_rows)


def test_chunking_covers_long_text_and_toxicity_in_the_tail(good_rows, cfg, models):
    from sftforge.gates.safety import chunk_text

    chunks = chunk_text("word " * 400, 500, 5)
    assert 1 < len(chunks) <= 5 and all(len(c) <= 500 for c in chunks)
    assert chunk_text("", 500, 5) == [""]
    long_resp = "The history of rivers is long and interesting. " * 30 + "You are an idiot."
    res = run(good_rows + [row("Tell me about the history of rivers", long_resp, rid="tail")], cfg, models)
    r = by_id(res, "tail")
    assert any(x["c"] == "toxicity" for x in r["reasons"]) or r["status"] == "rejected"


def test_small_inputs_are_not_capped(good_rows, models):
    res = pipeline.run(make_df(good_rows), Config(balance=BalanceCfg(max_share=0.2, val_fraction=0)), models)
    assert (res.df["status"] == "kept").all()


def test_device_override_and_audit_records_it(monkeypatch, good_rows, models):
    from sftforge.models import resolve_device

    assert resolve_device("cpu") == "cpu"
    monkeypatch.setenv("SFTFORGE_DEVICE", "cpu")
    assert resolve_device(None) == "cpu"
    res = pipeline.run(make_df(good_rows), Config(balance=BalanceCfg(enabled=False)), models)
    assert res.audit["device"] == "stub models"
