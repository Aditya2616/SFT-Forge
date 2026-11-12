"""Command line: sftforge run | serve."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import export, loaders, pipeline, runjson
from .models import Models
from .schema import Config


def _progress(stage: str, frac: float) -> None:
    print(f"\r  {stage:<8} {frac:>4.0%}", end="", file=sys.stderr, flush=True)
    if frac >= 1.0:
        print(file=sys.stderr)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    models = Models(cfg.models, device=args.device)
    loaded = loaders.load_path(args.input)
    if loaded.errors:
        print(f"skipped {len(loaded.errors)} unreadable lines (first: {loaded.errors[0]})", file=sys.stderr)
    if args.limit:
        loaded.df = loaded.df.head(args.limit)
    print(f"loaded {len(loaded.df)} rows ({loaded.fmt})", file=sys.stderr)
    result = pipeline.run(loaded.df, cfg, models, progress=_progress, load_errors=len(loaded.errors))
    export.export(result, cfg, args.out, source_name=Path(args.input).name)
    if args.recorded_dir:
        meta = {"name": args.name or Path(args.input).stem, "kind": "run", "label": "Recorded run",
                "rows": len(loaded.df)}
        omitted = runjson.write_recorded(result, meta, Path(args.recorded_dir))
        print(f"recorded run written to {args.recorded_dir} ({omitted} rows left out of the published text)", file=sys.stderr)
    a = result.audit
    print(f"\nkept {a['rows_kept']} of {a['rows_in']} ({a['keep_rate']:.1%}) -> {args.out}")
    for g in a["gates"]:
        print(f"  {g['gate']:<8} in {g['in']:>6}  rejected {g['rejected']:>5}  {g['reasons']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    if args.device:
        os.environ["SFTFORGE_DEVICE"] = args.device

    uvicorn.run("sftforge.server.app:app", host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sftforge", description="Clean instruction-tuning data.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline on a JSONL/JSON file")
    r.add_argument("input")
    r.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    r.add_argument("--out", default="out")
    r.add_argument("--limit", type=int, default=0, help="only use the first N rows")
    r.add_argument("--recorded-dir", default=None, help="also write the recorded run (index + chunks) the web demo replays")
    r.add_argument("--name", default=None)
    r.add_argument("--device", default=None, help="cpu, cuda or auto (default: auto, or $SFTFORGE_DEVICE)")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("serve", help="start the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8104)
    s.add_argument("--device", default=None, help="cpu, cuda or auto (default: auto, or $SFTFORGE_DEVICE)")
    s.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
