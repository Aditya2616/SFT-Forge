"""Run examples/messy_sample.jsonl through the default pipeline and store the audit in results/.

The README's quickstart table is generated from results/messy_sample_audit.json (see check_numbers.py).
Usage: python scripts/record_messy.py [--device cpu]
"""
from __future__ import annotations

import argparse
import json

from common import ROOT
from sftforge import pipeline
from sftforge.loaders import load_path
from sftforge.models import Models
from sftforge.schema import Config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = Config.load(ROOT / "configs" / "default.yaml")
    res = pipeline.run(load_path(ROOT / "examples" / "messy_sample.jsonl").df, cfg, Models(cfg.models, device=args.device))
    keep = ("rows_in", "rows_kept", "rows_train", "rows_val", "gates", "redactions_in_kept", "device")
    out = {k: res.audit[k] for k in keep}
    out["rows_rejected"] = int((res.df["status"] == "rejected").sum())
    (ROOT / "results" / "messy_sample_audit.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
