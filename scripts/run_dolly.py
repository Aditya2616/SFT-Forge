"""Headline run: 5,000 Dolly rows (a slice disjoint from the benchmark slices) through the default pipeline.

Writes out/dolly (exports), results/dolly_headline.{json,md}, docs/data/dolly/ (the recorded run the
web demo replays: index.json plus detail chunks) and data_sample/ (a small cleaned sample, CC BY-SA 3.0).
"""
from __future__ import annotations

import argparse
import json

from common import ROOT, dolly_slice, write_jsonl
from sftforge import audit, export, pipeline, runjson
from sftforge.loaders import load_lines
from sftforge.models import Models
from sftforge.schema import Config

NOTICE = """The rows in dolly_cleaned_sample.jsonl are derived from databricks-dolly-15k
(https://huggingface.co/datasets/databricks/databricks-dolly-15k), (c) Databricks, Inc. and the Wikipedia
contributors whose text appears in the context fields. They are licensed under Creative Commons
Attribution-ShareAlike 3.0 Unported (https://creativecommons.org/licenses/by-sa/3.0/).
Changes: filtered, de-duplicated, PII-redacted and category-balanced by SFT Forge; text otherwise unmodified
apart from whitespace and citation-marker normalisation.

This folder is separate from the MIT-licensed code in the rest of the repository.
"""

OMITTED = """
Rows naming specific AI products (a case-insensitive deny-list: see DENY in src/sftforge/runjson.py) are left
out of the published files. Rows left out of this sample: {sample}. Rows whose text is left out of the recorded
demo runs in docs/data/dolly/: {demo} (they still appear in the stream and the statistics, without text).
The audit counts of the real runs are unchanged.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--device", default=None, help="cpu, cuda or auto")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    rows = dolly_slice("headline")[: args.rows]
    loaded = load_lines(json.dumps(r) for r in rows)
    models = Models(cfg.models, cache_dir=ROOT / ".cache", device=args.device)
    res = pipeline.run(loaded.df, cfg, models, progress=lambda s, f: print(f"\r{s} {f:.0%}", end="", flush=True))
    print()
    out = ROOT / "out" / "dolly"
    export.export(res, cfg, out, source_name="databricks-dolly-15k (5,000-row slice)")
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "dolly_headline.json").write_text(json.dumps(res.audit, indent=2), encoding="utf-8")
    (results / "dolly_headline.md").write_text(audit.to_markdown(res.audit), encoding="utf-8")
    meta = {"name": "Dolly-15k, 5,000-row slice", "kind": "run", "label": "Recorded run", "rows": len(loaded.df)}
    omitted_demo = runjson.write_recorded(res, meta, ROOT / "docs" / "data" / "dolly")
    # published sample: the first --sample kept rows (by id), skipping rows that name an AI product (runjson.DENY)
    sample, skipped = [], 0
    for r in res.df[res.df["status"] == "kept"].sort_values("id").itertuples():
        if runjson.DENY.search(f"{r.instruction} {r.context} {r.response}"):
            skipped += 1
            continue
        sample.append({"id": r.id, "instruction": r.instruction, "context": r.context, "response": r.response,
                       "category": r.category})
        if len(sample) >= args.sample:
            break
    write_jsonl(ROOT / "data_sample" / "dolly_cleaned_sample.jsonl", sample)
    (ROOT / "data_sample" / "NOTICE.md").write_text(NOTICE + OMITTED.format(sample=skipped, demo=omitted_demo), encoding="utf-8")
    a = res.audit
    print(f"kept {a['rows_kept']} / {a['rows_in']}; gates:", [(g["gate"], g["rejected"]) for g in a["gates"]])
    print("timings", a["timings_s"])


if __name__ == "__main__":
    main()
