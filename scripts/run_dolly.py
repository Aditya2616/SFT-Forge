"""Headline run: 5,000 Dolly rows (a slice disjoint from the benchmark slices) through the default pipeline.

Writes out/dolly (exports), results/dolly_headline.{json,md}, docs/data/dolly.json (the recorded run the
web demo replays) and data_sample/ (a small cleaned sample, CC BY-SA 3.0).
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
    (ROOT / "docs" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "data" / "dolly.json").write_text(
        json.dumps(runjson.build(res, meta, withhold_toxic=True), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    kept = res.df[res.df["status"] == "kept"].sort_values("id").head(args.sample)
    sample = [{"id": r.id, "instruction": r.instruction, "context": r.context, "response": r.response, "category": r.category}
              for r in kept.itertuples()]
    write_jsonl(ROOT / "data_sample" / "dolly_cleaned_sample.jsonl", sample)
    (ROOT / "data_sample" / "NOTICE.md").write_text(NOTICE, encoding="utf-8")
    a = res.audit
    print(f"kept {a['rows_kept']} / {a['rows_in']}; gates:", [(g["gate"], g["rejected"]) for g in a["gates"]])
    print("timings", a["timings_s"])


if __name__ == "__main__":
    main()
