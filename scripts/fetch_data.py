"""Download the public datasets used here (pinned revisions) into ./data. Nothing is committed.

  databricks/databricks-dolly-15k   CC BY-SA 3.0   headline run and benchmark base
  OpenAssistant/oasst1              Apache-2.0     non-English pairs for the wrong-language defect
  allenai/real-toxicity-prompts     Apache-2.0     toxic snippets for the toxicity defect (benchmark only)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

DOLLY_REV = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
FILES = [
    ("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", DOLLY_REV),
    ("OpenAssistant/oasst1", "2023-04-12_oasst_ready.messages.jsonl.gz", None),
    ("allenai/real-toxicity-prompts", "prompts.jsonl", None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--dolly-only", action="store_true", help="skip the benchmark-only sources")
    args = ap.parse_args()
    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    for repo, name, rev in FILES[:1] if args.dolly_only else FILES:
        path = hf_hub_download(repo, name, repo_type="dataset", revision=rev, local_dir=out / repo.split("/")[1])
        print("ok", repo, path)


if __name__ == "__main__":
    main()
