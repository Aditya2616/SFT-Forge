"""Time the pipeline on the first N rows of the committed sample on one device (model loading included).

Writes results/throughput_<device>.json. The machine is shared with other jobs, so treat the figures as
indicative; the README quotes them with that caveat.
Usage: python scripts/throughput.py --device cpu   (or cuda)
"""
from __future__ import annotations

import argparse
import json
import os

from common import ROOT, read_jsonl
from sftforge import pipeline
from sftforge.loaders import load_lines
from sftforge.models import Models
from sftforge.schema import Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--threads", type=int, default=3, help="torch threads when running on the CPU")
    args = ap.parse_args()
    import torch

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    rows = read_jsonl(ROOT / "data_sample" / "dolly_cleaned_sample.jsonl")[: args.rows]
    cfg = Config.load(ROOT / "configs" / "default.yaml")
    models = Models(cfg.models, device=args.device)
    res = pipeline.run(load_lines(json.dumps(r) for r in rows).df, cfg, models)
    t = res.audit["timings_s"]
    out = {
        "device": res.audit["device"], "rows": len(rows), "threads": args.threads if args.device == "cpu" else 2,
        "logical_cores": os.cpu_count(), "timings_s": t, "rows_per_second": round(len(rows) / t["total"], 1),
        "rows_kept": res.audit["rows_kept"],
    }
    path = ROOT / "results" / f"throughput_{args.device}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
