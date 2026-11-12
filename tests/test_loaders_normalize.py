from __future__ import annotations

import json

from sftforge.loaders import infer_category, load_lines, load_text
from sftforge.normalize import normalize_text


def test_alpaca_dolly_chat_sharegpt_all_load():
    lines = [
        json.dumps({"instruction": "Add", "input": "2 and 2", "output": "4"}),
        json.dumps({"instruction": "Name a colour", "context": "", "response": "Blue", "category": "open_qa"}),
        json.dumps({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
                                 {"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye"}]}),
        json.dumps({"conversations": [{"from": "human", "value": "Q?"}, {"from": "gpt", "value": "A."}]}),
    ]
    res = load_lines(lines)
    assert res.errors == []
    assert len(res.df) == 5  # the two-turn chat yields two pairs
    assert res.fmt == {"alpaca": 1, "dolly": 1, "chat": 2, "sharegpt": 1}
    assert res.df.loc[0, "context"] == "2 and 2"
    assert list(res.df["turn"]) == [0, 0, 0, 1, 0]


def test_malformed_lines_are_reported_not_fatal():
    res = load_lines(['{"instruction": "a", "output": "b"}', "not json", '{"foo": 1}', "[1, 2]", ""])
    assert len(res.df) == 1
    assert len(res.errors) == 3
    assert res.errors[0].startswith("line 2")


def test_json_array_and_ids_are_unique():
    text = json.dumps([{"id": "x", "instruction": "a", "output": "b"}, {"id": "x", "instruction": "c", "output": "d"}])
    df = load_text(text).df
    assert df["id"].is_unique and df.loc[0, "id"] == "x"


def test_missing_category_is_inferred_and_flagged():
    df = load_lines([json.dumps({"instruction": "Summarize this article", "output": "Short."})]).df
    assert df.loc[0, "category"] == "summarization" and bool(df.loc[0, "category_inferred"])
    assert infer_category("Why is it so?") == "general_qa"


def test_normalize_strips_citations_and_whitespace():
    text, edits = normalize_text("Paris[1] is   big.\r\n\r\n\r\n\r\nYes it is[23].  ")
    assert text == "Paris is big.\n\nYes it is."
    assert {"citation_markers", "whitespace"} <= set(edits)
    assert normalize_text("clean text")[1] == []
