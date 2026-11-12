"""Score the pipeline against the synthetic-contamination benchmark (known defects, known rates)."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import ROOT
from sftforge import pipeline, runjson
from sftforge.loaders import load_path
from sftforge.models import Models
from sftforge.schema import Config

# defect type -> (intended gate, reason codes that count as the right reason)
EXPECT = {
    "exact_dup": ("dedupe", {"exact_duplicate"}),
    "near_dup": ("dedupe", {"near_duplicate", "semantic_duplicate"}),
    "paraphrase_dup": ("dedupe", {"semantic_duplicate", "near_duplicate"}),
    "truncated": ("quality", {"truncated"}),
    "refusal": ("quality", {"refusal_boilerplate"}),
    "wrong_language": ("quality", {"wrong_language"}),
    "off_topic": ("quality", {"off_topic"}),
    "secret_aws": ("safety", {"secret"}),
    "secret_github": ("safety", {"secret"}),
    "secret_private_key": ("safety", {"secret"}),
    "secret_generic": ("safety", {"secret"}),
    "toxic": ("safety", {"toxicity"}),
}
PII = {"pii_email": "email", "pii_phone": "phone", "pii_card": "card", "pii_ssn": "ssn", "pii_ipv4": "ipv4",
       "pii_iban": "iban"}


def evaluate(df, truth: dict[str, dict]) -> dict:
    rows = []
    for r in df.itertuples():
        t = truth.get(r.id)
        rows.append((r, t["defect"] if t else "clean"))
    out: dict = {"per_defect": {}, "per_gate": {}}
    by_def: dict[str, list] = defaultdict(list)
    for r, d in rows:
        by_def[d].append(r)

    for d, members in sorted(by_def.items()):
        if d == "clean":
            continue
        n = len(members)
        if d in PII:
            kind = PII[d]
            red = sum(any(x["k"] == kind for x in m.redactions) and m.status in ("kept", "balanced_out") for m in members)
            rejected_other = sum(m.status == "rejected" for m in members)
            out["per_defect"][d] = {"n": n, "expected": "redacted", "caught": red, "recall": round(red / n, 3),
                                    "rejected_by_other_gate": rejected_other}
            continue
        gate, codes = EXPECT[d]
        rej = [m for m in members if m.status == "rejected"]
        right_gate = [m for m in rej if m.gate == gate]
        right_reason = [m for m in rej if any(x["c"] in codes for x in m.reasons)]
        where = Counter(m.reasons[0]["c"] for m in rej)
        out["per_defect"][d] = {
            "n": n, "expected": f"rejected at {gate}", "caught": len(rej), "recall": round(len(rej) / n, 3),
            "at_intended_gate": len(right_gate), "with_expected_reason": len(right_reason),
            "first_reasons": dict(where.most_common()),
        }

    clean = by_def["clean"]
    clean_rej = [m for m in clean if m.status == "rejected"]
    clean_bal = [m for m in clean if m.status == "balanced_out"]
    out["untouched"] = {
        "n": len(clean), "rejected": len(clean_rej), "false_reject_rate": round(len(clean_rej) / len(clean), 4),
        "balanced_out": len(clean_bal),
        "reasons": dict(Counter(m.reasons[0]["c"] for m in clean_rej).most_common()),
        "note": "includes real defects that Dolly itself contains (natural duplicates, cut-off answers); see audit sample",
    }
    for g in ("dedupe", "quality", "safety"):
        rej_g = [(m, d) for m, d in rows if m.status == "rejected" and m.gate == g]
        inj = sum(d != "clean" for _, d in rej_g)
        out["per_gate"][g] = {"rejected": len(rej_g), "injected_defects": inj,
                              "precision": round(inj / len(rej_g), 3) if rej_g else None}
    out["pii_false_redactions_on_untouched"] = sum(
        1 for m in clean if m.redactions and m.status != "rejected")
    return out


def to_md(ev: dict, title: str) -> str:
    L = [f"# {title}", "", "Synthetic contamination: defects were injected at known rates; counts below are real measurements.",
         "", f"Models ran on: {ev.get('audit', {}).get('device', 'unknown')}.", "",
         "| Defect | n | Recall | Right gate | Right reason | Notes |", "|---|---:|---:|---:|---:|---|"]
    for d, v in ev["per_defect"].items():
        note = v.get("first_reasons") or (f"rejected elsewhere: {v['rejected_by_other_gate']}" if "rejected_by_other_gate" in v else "")
        L.append(f"| {d} | {v['n']} | {v['recall']:.1%} | {v.get('at_intended_gate', '-')} | {v.get('with_expected_reason', '-')} | {note} |")
    u = ev["untouched"]
    L += ["", f"Untouched rows: {u['n']}; rejected {u['rejected']} ({u['false_reject_rate']:.2%}); reasons {u['reasons']}.",
          f"PII false redactions on untouched kept rows: {ev['pii_false_redactions_on_untouched']}.", "",
          "| Gate | Rejected | Of which injected | Precision |", "|---|---:|---:|---:|"]
    for g, v in ev["per_gate"].items():
        p = "-" if v["precision"] is None else f"{v['precision']:.1%}"
        L.append(f"| {g} | {v['rejected']} | {v['injected_defects']} | {p} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="dev", choices=["dev", "bench"])
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-json", default=None)
    ap.add_argument("--device", default=None, help="cpu, cuda or auto")
    args = ap.parse_args()
    data = Path(args.data or ROOT / "data" / f"contaminated_{args.slice}")
    cfg = Config.load(args.config)
    loaded = load_path(data / "benchmark.jsonl")
    truth = json.loads((data / "truth.json").read_text(encoding="utf-8"))
    models = Models(cfg.models, cache_dir=ROOT / ".cache", device=args.device)
    res = pipeline.run(loaded.df, cfg, models)
    ev = evaluate(res.df, truth)
    ev["audit"] = {k: res.audit[k] for k in ("rows_in", "rows_kept", "gates", "balance_cap", "timings_s", "device")}
    ev["slice"] = args.slice
    out = Path(args.out or ROOT / "results" / f"benchmark_{args.slice}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(to_md(ev, f"Contamination benchmark ({args.slice} slice)"), encoding="utf-8")
    print(to_md(ev, f"Contamination benchmark ({args.slice} slice)"))
    if args.run_json:
        meta = {"name": "Synthetic contamination benchmark", "kind": "synthetic_contamination",
                "label": "Recorded run: synthetic contamination", "rows": len(loaded.df),
                "injected": len(truth), "seed": cfg.seed,
                "benchmark": {k: v for k, v in ev.items() if k != "audit"}}
        Path(args.run_json).write_text(
            json.dumps(runjson.build(res, meta, truth={k: v["defect"] for k, v in truth.items()}, withhold_toxic=True),
                       ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    # keep the untouched rejects for a manual look
    audit_rows = [
        {"id": m.id, "gate": m.gate, "reasons": [x["m"] for x in m.reasons], "instruction": m.instruction[:200],
         "response": m.response[:300]}
        for m in res.df.itertuples() if m.status == "rejected" and m.id not in truth
    ]
    (out.parent / f"untouched_rejects_{args.slice}.json").write_text(json.dumps(audit_rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
