"""Time the pipeline on the first N rows of the committed sample on one device (model loading included).

Writes results/throughput_<device>.json, including the system-wide CPU busy fraction (and, on the GPU, its utilisation)
sampled just before the run, so the figure can be read together with the load it was taken under.
Usage: python scripts/throughput.py --device cpu   (or cuda)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import time

from common import ROOT, read_jsonl
from sftforge import pipeline
from sftforge.loaders import load_lines
from sftforge.models import Models
from sftforge.schema import Config


def _times() -> tuple[int, int, int]:
    """(idle, kernel, user) system times in 100 ns units on Windows."""

    class FT(ctypes.Structure):
        _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

    idle, kern, user = FT(), FT(), FT()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    return tuple((f.hi << 32) | f.lo for f in (idle, kern, user))  # type: ignore[return-value]


def busy_between(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    total = (b[1] - a[1]) + (b[2] - a[2])  # kernel time includes idle time
    return round(1 - (b[0] - a[0]) / total, 3) if total else 0.0


def cpu_busy(seconds: float = 3.0) -> float:
    """Fraction of all logical cores busy over `seconds` (Windows: GetSystemTimes; elsewhere: load average)."""
    if os.name != "nt":
        return round(os.getloadavg()[0] / (os.cpu_count() or 1), 3)

    a = _times()
    time.sleep(seconds)
    return busy_between(a, _times())


def gpu_busy() -> int | None:
    """GPU utilisation in percent from nvidia-smi, or None when it is not available."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        return int(out.split()[0])
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--threads", type=int, default=3, help="torch threads when running on the CPU")
    args = ap.parse_args()
    import torch

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    load = {"cpu_busy_before": cpu_busy()}
    if args.device == "cuda":
        load["gpu_busy_before_pct"] = gpu_busy()
    rows = read_jsonl(ROOT / "data_sample" / "dolly_cleaned_sample.jsonl")[: args.rows]
    cfg = Config.load(ROOT / "configs" / "default.yaml")
    models = Models(cfg.models, device=args.device)
    t_start, p_start, w_start = (_times() if os.name == "nt" else None), time.process_time(), time.perf_counter()
    res = pipeline.run(load_lines(json.dumps(r) for r in rows).df, cfg, models)
    if t_start is not None:  # system-wide busy over the run, and what this process itself accounts for
        wall = time.perf_counter() - w_start
        own = (time.process_time() - p_start) / (wall * (os.cpu_count() or 1))
        load["cpu_busy_during"] = busy_between(t_start, _times())
        load["cpu_busy_during_other"] = round(max(0.0, load["cpu_busy_during"] - own), 3)  # everything except this run
    t = res.audit["timings_s"]
    out = {
        "device": res.audit["device"], "rows": len(rows), "threads": args.threads if args.device == "cpu" else 2,
        "logical_cores": os.cpu_count(), "timings_s": t, "rows_per_second": round(len(rows) / t["total"], 1),
        "rows_kept": res.audit["rows_kept"], "load": load,
    }
    path = ROOT / "results" / f"throughput_{args.device}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
