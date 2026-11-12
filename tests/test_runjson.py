from __future__ import annotations

import json

from conftest import make_df, row
from sftforge import pipeline, runjson


def _name() -> str:
    return runjson._NAMES[0].capitalize()


def _result(good_rows, cfg, models):
    rows = good_rows + [
        row("Who made the assistant?", f"The assistant is called {_name()} and it answers questions.", rid="named"),
        row("Describe my neighbour please", "My neighbour is a worthless idiot and everybody hates my neighbour, a stupid moron.", rid="toxic"),
    ]
    return pipeline.run(make_df(rows), cfg, models)


def test_index_lists_every_row_and_details_are_chunked(good_rows, cfg, models):
    res = _result(good_rows, cfg, models)
    index, chunks, omitted = runjson.build_compact(res, {"name": "t"})
    assert len(index["rows"]) == len(res.df) == sum(len(c) for c in chunks)
    assert omitted == 1
    ids = [r[0] for r in index["rows"]]
    assert ids == list(res.df["id"])
    first = chunks[0][0]
    assert first["a"] and first["ck"] and all(isinstance(e[0], int) for e in first["ck"])
    # rule dictionary entries are (gate, rule, threshold)
    assert all(len(r) == 3 for r in index["rules"])


def test_rows_naming_an_ai_product_are_left_out_of_published_text(good_rows, cfg, models, tmp_path):
    res = _result(good_rows, cfg, models)
    runjson.write_recorded(res, {"name": "t"}, tmp_path / "run")
    blob = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "run").glob("*.json"))
    assert not runjson.DENY.search(blob)
    index = json.loads((tmp_path / "run" / "index.json").read_text(encoding="utf-8"))
    named = next(r for r in index["rows"] if r[0] == "named")
    assert named[7] == 2 and "omitted" in named[8]
    assert index["audit"]["rows_in"] == len(res.df)  # the real run's counts are untouched
    assert index["meta"]["rows_omitted_from_text"] == 1


def test_toxic_text_is_withheld_but_scores_kept(good_rows, cfg, models):
    res = _result(good_rows, cfg, models)
    index, chunks, _ = runjson.build_compact(res, {"name": "t"})
    pos = next(i for i, r in enumerate(index["rows"]) if r[0] == "toxic")
    d = chunks[pos // runjson.CHUNK][pos % runjson.CHUNK]
    assert d["w"] == 1 and "withheld" in d["a"]
    assert index["rows"][pos][7] == 1
