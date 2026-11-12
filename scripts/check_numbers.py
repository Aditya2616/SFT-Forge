"""Keep the numbers in README.md identical to the files they come from.

The README's result tables and the prose sentences that quote results sit between marker comments:

    <!-- gen:NAME -->
    ...generated text...
    <!-- /gen:NAME -->

`python scripts/check_numbers.py --write` regenerates those blocks from results/*.json, the recorded runs'
index files under docs/data and data_sample/NOTICE.md. `--check` (run by the test suite) fails when a block is
out of date. A second set of checks verifies the thresholds and example facts that README.md quotes in prose
against configs/default.yaml and the recorded runs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SLICES  # noqa: E402

README = ROOT / "README.md"
SHORT = {
    "semantic_duplicate": "semantic", "exact_duplicate": "exact", "near_duplicate": "near", "truncated": "truncated",
    "wrong_language": "wrong language", "refusal_boilerplate": "refusal", "off_topic": "off topic",
    "response_too_short": "too short", "instruction_too_short": "too short", "repetitive": "repetitive",
    "toxicity": "toxicity", "secret": "secret", "category_cap": "category cap",
}


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def n(x: int | float) -> str:
    return f"{x:,}"


def pct(a: float, b: float, digits: int = 1) -> str:
    return f"{100 * a / b:.{digits}f}%"


def reasons(d: dict[str, int], none: str = "none") -> str:
    if not d:
        return none
    merged: dict[str, int] = {}
    for k, v in d.items():
        merged[SHORT.get(k, k.replace("_", " "))] = merged.get(SHORT.get(k, k.replace("_", " ")), 0) + v
    return ", ".join(f"{k} {v}" for k, v in sorted(merged.items(), key=lambda kv: (-kv[1], kv[0])))


def gate_table(gates: list[dict], category_note: str | None = None) -> str:
    rows = ["| Gate | In | Rejected | Out | Reasons |", "|---|---:|---:|---:|---|"]
    for g in gates:
        why = reasons(g["reasons"])
        if g["gate"] == "balance" and category_note and g["rejected"]:
            why = category_note
        rows.append(f"| {g['gate']} | {n(g['in'])} | {n(g['rejected'])} | {n(g['out'])} | {why} |")
    return "\n".join(rows)


def plural(v: int, kind: str) -> str:
    noun = {"email": "email", "phone": "phone number", "card": "card number"}.get(kind, kind)
    return f"{v} {noun}" + ("" if v == 1 else "s")


def device_kind(label: str) -> str:
    return "GPU" if label.startswith("cuda") else "CPU"


# ------------------------------------------------------------------------------------------------ blocks
def block_slices(_: dict) -> str:
    d, b, h = SLICES["dev"], SLICES["bench"], SLICES["headline"]
    return (f"Slices of the shuffled Dolly-15k rows are disjoint: rows {d[0]}-{d[1]} (dev, used to tune thresholds), "
            f"{b[0]}-{b[1]} (bench, reported), {h[0]}-{h[1]} (headline run).")


def block_device(s: dict) -> str:
    dolly, bench = s["dolly"]["device"], s["bench"]["audit"]["device"]
    assert dolly == bench, "the headline run and the benchmark must be on the same device"
    return (f"The headline run and the benchmark ran on one machine (12 logical cores, one shared RTX 5060 Ti). "
            f"The models ran on the {device_kind(dolly)} in float32 (audit device: `{dolly}`). Scores can differ in "
            "the last decimals between devices; the dev-slice tuning and the bench-slice report used the same device.")


def block_messy(s: dict) -> str:
    m = s["messy"]
    assert m["device"].startswith("cpu")
    red = m["redactions_in_kept"]
    lines = ["| Gate | In | Rejected | Out | Reasons |", "|---|---:|---:|---:|---|"]
    for g in m["gates"]:
        lines.append(f"| {g['gate']} | {g['in']} | {g['rejected']} | {g['out']} | {reasons(g['reasons'])} |")
    lines += ["", f"{m['rows_kept']} of {m['rows_in']} rows are kept ({m['rows_train']} train, {m['rows_val']} validation); "
              f"{red.get('email', 0)} emails, {red.get('phone', 0)} phone numbers and {red.get('card', 0)} card number "
              f"were redacted in kept rows, and {m['rows_rejected']} rows were rejected."]
    return "\n".join(lines)


def block_dolly(s: dict) -> str:
    a = s["dolly"]
    cats = a["categories"]
    big = max(cats, key=lambda c: cats[c]["after_gates"] - cats[c]["final"])
    note = f"category cap ({big} trimmed from {n(cats[big]['after_gates'])} to {n(cats[big]['final'])})"
    gates = a["gates"]
    rejected = sum(g["rejected"] for g in gates if g["gate"] != "balance")
    capped = gates[-1]["rejected"]
    assert a["rows_in"] == a["rows_kept"] + rejected + capped
    red = a["redactions_in_kept"]
    t = a["timings_s"]
    table = gate_table(gates, note)
    text = (f"{n(a['rows_kept'])} of {n(a['rows_in'])} rows kept ({pct(a['rows_kept'], a['rows_in'])}). Dolly is already "
            f"human-curated, so the gates reject little: {rejected} rows ({pct(rejected, a['rows_in'])}) were rejected by the "
            f"three gates, and the other {n(capped)} ({pct(capped, a['rows_in'])}) were trimmed by the category cap, which is a "
            f"balancing choice and not a defect finding. {sum(red.values())} PII spans were redacted in kept rows "
            f"({', '.join(plural(v, k) for k, v in red.items())}). I did not hand-label these rejects, so there is no precision "
            f"figure for this run. Stage times on the {device_kind(a['device'])}: dedupe {t['dedupe']:.0f} s, quality "
            f"{t['quality']:.0f} s, safety {t['safety']:.0f} s, {t['total']:.0f} s in total.")
    return table + "\n\n" + text


def block_bench_intro(s: dict) -> str:
    b = s["bench"]
    inj = sum(v["n"] for v in b["per_defect"].values())
    u = b["untouched"]
    assert b["audit"]["rows_in"] == u["n"] + inj
    return (f"Bench slice: {n(b['audit']['rows_in'])} rows, {inj} of them injected defects and {n(u['n'])} left untouched. "
            "Thresholds were tuned on the dev slice only; these are bench numbers "
            "(`results/benchmark_bench.md`, dev results in `results/benchmark_dev.md`).")


def block_bench_recall(s: dict) -> str:
    pd = s["bench"]["per_defect"]
    rows = ["| Injected defect | n | Caught (any gate) | Recall |", "|---|---:|---:|---:|"]

    def line(label: str, keys: list[str], extra: str = "") -> None:
        tot = sum(pd[k]["n"] for k in keys)
        caught = sum(pd[k]["caught"] for k in keys)
        rows.append(f"| {label} | {tot} | {caught} | {round(100 * caught / tot)}%{extra} |")

    line("exact duplicate", ["exact_dup"])
    line("near duplicate", ["near_dup"])
    line("paraphrase duplicate", ["paraphrase_dup"])
    line("refusal / boilerplate", ["refusal"])
    line("wrong language", ["wrong_language"])
    line("off-topic response", ["off_topic"])
    line("truncated response", ["truncated"])
    line("toxic text", ["toxic"])
    sec = [k for k in pd if k.startswith("secret_")]
    right = sum(pd[k]["with_expected_reason"] for k in sec)
    line("secrets (AWS, GitHub, private key, generic)", sec, f" ({right} rejected for the secret itself)")
    pii = [k for k in pd if k.startswith("pii_")]
    other = sum(pd[k]["rejected_by_other_gate"] for k in pii)
    tot = sum(pd[k]["n"] for k in pii)
    caught = sum(pd[k]["caught"] for k in pii)
    assert caught + other == tot
    rows.append(f"| PII (email, phone, card, SSN, IPv4, IBAN) | {tot} | {caught} | {round(100 * caught / tot)}% redacted "
                f"({other} rejected by another gate) |")
    return "\n".join(rows)


def block_bench_gates(s: dict) -> str:
    rows = ["| Gate | Rejected | Of which injected | Precision |", "|---|---:|---:|---:|"]
    for g, v in s["bench"]["per_gate"].items():
        rows.append(f"| {g} | {v['rejected']} | {v['injected_defects']} | {100 * v['precision']:.1f}% |")
    return "\n".join(rows)


def block_bench_false(s: dict) -> str:
    b = s["bench"]
    u = b["untouched"]
    return (f"**False rejects on untouched rows:** {u['rejected']} of {n(u['n'])} ({100 * u['rejected'] / u['n']:.2f}%). "
            "These are not all mistakes: they include real defects already in Dolly (natural duplicates, cut-off answers). "
            f"{b['pii_false_redactions_on_untouched']} untouched kept row had text redacted as PII that was not PII.")


def block_demo(s: dict) -> str:
    d, c = s["idx_dolly"], s["idx_bench"]
    sizes = sorted([s["size_dolly_index"], s["size_bench_index"]])
    total = s["docs_data_bytes"]
    return (f"- `docs/data/<run>/index.json` ({round(sizes[0] / 1000)} KB for the benchmark run and "
            f"{round(sizes[1] / 1000)} KB for the Dolly run): the audit report, and one small array per row with its id, "
            "category, final status, gate, reason codes and a 60-character snippet. This is all the stream animation and "
            "the statistics need, and it covers every row.\n"
            "- `docs/data/<run>/c<N>.json`: the detail for 100 rows each (texts, rule values, scores), fetched when a sample "
            f"is opened and kept in memory. `docs/data` is {total / 1e6:.1f} MB in total.\n\n"
            "Text is clipped in the detail files (instruction 240, context 120, response 380 characters). Rows whose text "
            "names a specific AI product are left out of the published files by a small deny-list (`DENY` in "
            "`src/sftforge/runjson.py`): their text and detail are omitted, but they still appear in the stream and the "
            f"statistics. That is {d['meta']['rows_omitted_from_text']} rows in the Dolly demo, "
            f"{c['meta']['rows_omitted_from_text']} in the benchmark demo and {s['sample_skipped']} skipped for "
            "`data_sample/`. The audit counts are those of the real runs and are unchanged.")


def block_throughput(s: dict) -> str:
    c, g = s["tp_cpu"], s["tp_gpu"]
    assert c["rows"] == g["rows"]
    return (f"- CPU throughput is the limit on machines without a GPU. `scripts/throughput.py` times the first {c['rows']} "
            f"rows of the committed sample with model loading included: {c['timings_s']['total']:.0f} s on the CPU "
            f"({c['threads']} threads, {c['rows_per_second']} rows per second, safety {c['timings_s']['safety']:.0f} s) and "
            f"{g['timings_s']['total']:.0f} s on the GPU ({g['rows_per_second']} rows per second). The machine and its GPU were "
            "shared with other jobs, so these figures vary several-fold between runs and are indicative only. Live mode is "
            "capped at 500 rows for that reason.")



def block_perplexity(s: dict) -> str:
    r = s["ppl"]
    cfg = s["cfg"]["quality"]["perplexity"]
    assert cfg["enabled"] == r["enable_by_default"], "configs/default.yaml disagrees with the perplexity evidence"
    if r["chosen_max_ppl"]:
        assert cfg["max_ppl"] == r["chosen_max_ppl"]
    assert s["cfg"]["models"]["perplexity"] == r["model"]
    dev, bench = r["dev"], r["bench"]
    rows = ["| max_ppl | Recall of truncated + off-topic rows | False-reject rate on untouched rows | Rejected for high perplexity (injected) |",
            "|---:|---:|---:|---:|"]
    for k, v in dev["settings"].items():
        label = "check off" if k == "off" else k
        rows.append(f"| {label} | {100 * v['target_recall']:.1f}% ({v['target_caught']} of {v['target_n']}) | "
                    f"{100 * v['false_reject_rate']:.2f}% ({v['false_rejects']} of {v['untouched_n']}) | "
                    f"{v['high_perplexity_rejects']} ({v['high_perplexity_on_injected']}) |")
    b_off = bench["settings"]["off"]
    b_on = next(v for k, v in bench["settings"].items() if k != "off")
    rule = r["rule"]
    verdict = (f"A threshold met the rule on the dev slice (max_ppl {r['chosen_max_ppl']}), so the check is on by default."
               if r["enable_by_default"] else "No threshold met the rule on the dev slice, so the check stays off by default.")
    return (f"Dev slice ({dev['n_rows']:,} rows), tuned there: the rule, fixed before I looked at the bench slice, was that "
            f"some `max_ppl` must raise the recall of the truncated and off-topic rows by at least {rule['min_gain_pp']:g} "
            f"percentage points while the false-reject rate on untouched rows rises by at most {rule['max_extra_false_pp']:g} points.\n\n"
            + "\n".join(rows) + "\n\n"
            f"The raw score also separates the defects poorly: the ROC AUC of the response perplexity for truncated rows against "
            f"untouched rows is {dev['auc_vs_untouched']['truncated']:.2f} on dev and {bench['auc_vs_untouched']['truncated']:.2f} "
            f"on bench, and for off-topic rows {dev['auc_vs_untouched']['off_topic']:.2f} on dev and "
            f"{bench['auc_vs_untouched']['off_topic']:.2f} on bench (0.5 is chance). On the bench slice at the default "
            f"`max_ppl` {b_on['max_ppl']}, the recall of those rows is {100 * b_on['target_recall']:.1f}% against "
            f"{100 * b_off['target_recall']:.1f}% with the check off, while the false-reject rate on untouched rows rises from "
            f"{100 * b_off['false_reject_rate']:.2f}% to {100 * b_on['false_reject_rate']:.2f}%. {verdict} The untouched rows that the loosest threshold ({dev['untouched_flagged_at_loosest']['max_ppl']}) flags on the dev slice are mostly short answers: their median length is "
            f"{dev['untouched_flagged_at_loosest']['median_words']:g} words against {dev['untouched_flagged_at_loosest']['median_words_not_flagged']:g} for the rest.")


def _ft_fmt(key: str, v: float) -> str:
    return f"{100 * v:.1f}%" if key.endswith("rate") else f"{v:.3f}"


def _ft_pm(key: str, e: dict) -> str:
    return f"{_ft_fmt(key, e['mean'])} ± {_ft_fmt(key, e['sd'])}"


def block_ft_setup(s: dict) -> str:
    f = s["ft"]
    c, d = f["config"], f["data"]
    conds = f["conditions"]
    secs = [x for k in ("raw", "cleaned", "random") for x in conds[k]["train_seconds"]]
    inj_rand = conds["random"]["injected_rows"]
    defects = ", ".join(f"{v} {k.replace('_', ' ').replace('pii', 'PII')}" for k, v in sorted(d["cleaned_defects"].items(), key=lambda kv: -kv[1]))
    return (f"{c['model'].split('/')[1]} was fine-tuned in {c['precision']} on the {device_kind(f['device'])} for "
            f"{c['steps']} optimiser steps of {c['batch_size']} examples (micro-batches of {c['micro_batch']}), AdamW at a peak learning "
            f"rate of {c['lr']:g} with {c['warmup_steps']} warm-up steps and cosine decay, loss on the response tokens only, "
            f"{len(c['seeds'])} seeds per training set. A run took {min(secs) / 60:.1f} to {max(secs) / 60:.1f} minutes of training "
            f"and used at most {max(conds[k]['peak_vram_gib'] for k in ('raw', 'cleaned', 'random')):.1f} GiB of GPU memory "
            f"(reserved; capped at {c['vram_cap_gib']:g} GiB). Device: `{f['device']}`.\n\n"
            f"Training sets, all from the benchmark slice: **raw** is the whole slice ({d['raw_rows']:,} rows, {d['injected_in_raw']} "
            f"with an injected defect); **cleaned** is what SFT Forge keeps ({d['cleaned_rows']:,} rows, of which "
            f"{d['injected_in_cleaned']} still carry an injected defect: {defects}); **random** is a fresh random subsample of the raw "
            f"slice with {d['cleaned_rows']:,} rows per seed (on average {sum(inj_rand) / len(inj_rand):.0f} injected defects). "
            f"With {c['steps'] * c['batch_size']:,} examples drawn per run, the model sees the raw set {c['steps'] * c['batch_size'] / d['raw_rows']:.1f} "
            f"times and the cleaned and random sets {c['steps'] * c['batch_size'] / d['cleaned_rows']:.1f} times. "
            f"Every run is scored on {d['heldout_rows']} held-out Dolly rows (rows {SLICES['heldout'][0]}-{SLICES['heldout'][1]} "
            "of the shuffled data, minus any whose prompt appears in the training slice) that none of the training sets contain, "
            f"with greedy generations of up to {c['max_new_tokens']} new tokens.")


def block_ft_table(s: dict) -> str:
    f = s["ft"]
    conds = [k for k in ("base", "raw", "cleaned", "random") if k in f["conditions"]]
    head = {"base": "Untuned", "raw": "Raw", "cleaned": "Cleaned", "random": "Random, same size"}
    rows = ["| Measure | " + " | ".join(head[k] for k in conds) + " |", "|---|" + "---:|" * len(conds)]
    for m in f["metrics"]:
        arrow = "higher is better" if m["better"] > 0 else "lower is better"
        rows.append(f"| {m['label']} ({arrow}) | " + " | ".join(
            (_ft_fmt(m["key"], f["conditions"][k]["metrics"][m["key"]]["mean"]) if k == "base" else _ft_pm(m["key"], f["conditions"][k]["metrics"][m["key"]]))
            for k in conds) + " |")
    ppl = {k: [x for x in f["conditions"][k]["perplexity"]] for k in conds}
    rows.append("| Held-out perplexity (lower is better) | " + " | ".join(
        f"{sum(ppl[k]) / len(ppl[k]):.2f}" + ("" if k == "base" else f" ± {(sum((x - sum(ppl[k]) / len(ppl[k])) ** 2 for x in ppl[k]) / (len(ppl[k]) - 1)) ** 0.5:.2f}")
        for k in conds) + " |")
    return "\n".join(rows)


def block_ft_paired(s: dict) -> str:
    f = s["ft"]
    rows = ["| Measure | Cleaned − raw (95% interval) | Cleaned − random (95% interval) |", "|---|---:|---:|"]

    def cell(key: str, d: dict) -> str:
        def sgn(v: float) -> str:
            body = f"{100 * abs(v):.1f} pts" if key.endswith("rate") else f"{abs(v):.3f}"
            return ("+" if v >= 0 else "−") + body
        tag = "" if not d["excludes_zero"] else (" **better**" if d["favours_cleaned"] else " **worse**")
        return f"{sgn(d['mean_diff'])} [{sgn(d['ci95'][0])}, {sgn(d['ci95'][1])}]{tag}"

    for m in f["metrics"]:
        a, b = f["paired"]["cleaned_minus_raw"][m["key"]], f["paired"]["cleaned_minus_random"][m["key"]]
        rows.append(f"| {m['label']} | {cell(m['key'], a)} | {cell(m['key'], b)} |")
    return "\n".join(rows)


def block_ft_verdict(s: dict) -> str:
    f = s["ft"]
    c = f["conditions"]
    n_seeds = len(f["config"]["seeds"])

    def sgn(key: str, v: float) -> str:
        body = f"{100 * abs(v):.1f} pts" if key.endswith("rate") else f"{abs(v):.3f}"
        return ("+" if v >= 0 else "−") + body

    mean = lambda k, m: c[k]["metrics"][m]["mean"]  # noqa: E731
    head = (f"All three fine-tuned sets moved the model the same way and by about the same amount: held-out loss fell from "
            f"{mean('base', 'loss'):.3f} for the untuned model to {mean('raw', 'loss'):.3f} (raw), {mean('cleaned', 'loss'):.3f} "
            f"(cleaned) and {mean('random', 'loss'):.3f} (random), and ROUGE-L rose from {mean('base', 'rouge_l'):.3f} to "
            f"{mean('raw', 'rouge_l'):.3f}, {mean('cleaned', 'rouge_l'):.3f} and {mean('random', 'rouge_l'):.3f}.")
    parts = [head]
    for key, name in (("cleaned_minus_raw", "raw"), ("cleaned_minus_random", "the size-matched random subsample")):
        better = [f"{m['label'][0].lower() + m['label'][1:]} ({sgn(m['key'], f['paired'][key][m['key']]['mean_diff'])})" for m in f["metrics"]
                  if f["paired"][key][m["key"]]["favours_cleaned"] is True]
        worse = [f"{m['label'][0].lower() + m['label'][1:]} ({sgn(m['key'], f['paired'][key][m['key']]['mean_diff'])})" for m in f["metrics"]
                 if f["paired"][key][m["key"]]["favours_cleaned"] is False]
        if not better and not worse:
            parts.append(f"Against {name}, no measure had a paired {n_seeds}-seed 95% interval that excludes zero.")
        else:
            bits = []
            if better:
                bits.append("statistically better on " + "; ".join(better))
            if worse:
                bits.append("statistically worse on " + "; ".join(worse))
            parts.append(f"Against {name}, the cleaned set was " + " and ".join(bits) + ", meaning the {n}-seed 95% interval excludes zero.".replace("{n}", str(n_seeds)))
    return " ".join(parts)


BLOCKS = {
    "slices": block_slices, "device": block_device, "messy": block_messy, "dolly": block_dolly,
    "bench_intro": block_bench_intro, "bench_recall": block_bench_recall, "bench_gates": block_bench_gates,
    "bench_false": block_bench_false, "demo": block_demo, "throughput": block_throughput,
    "perplexity": block_perplexity, "ft_setup": block_ft_setup, "ft_table": block_ft_table,
    "ft_paired": block_ft_paired, "ft_verdict": block_ft_verdict,
}


# ------------------------------------------------------------------------------------------------ sources
def sources() -> dict:
    idx_d = load("docs/data/dolly/index.json")
    idx_b = load("docs/data/contaminated/index.json")
    notice = (ROOT / "data_sample" / "NOTICE.md").read_text(encoding="utf-8")
    skipped = int(re.search(r"left out of this sample: (\d+)", notice).group(1))
    data_bytes = sum(p.stat().st_size for p in (ROOT / "docs" / "data").rglob("*") if p.is_file())
    bench = load("results/benchmark_bench.json")
    return {
        "dolly": load("results/dolly_headline.json"), "bench": bench, "messy": load("results/messy_sample_audit.json"),
        "tp_cpu": load("results/throughput_cpu.json"), "tp_gpu": load("results/throughput_cuda.json"),
        "idx_dolly": idx_d, "idx_bench": idx_b, "sample_skipped": skipped, "docs_data_bytes": data_bytes,
        "size_dolly_index": (ROOT / "docs/data/dolly/index.json").stat().st_size,
        "size_bench_index": (ROOT / "docs/data/contaminated/index.json").stat().st_size,
        "ppl": load("results/perplexity_check.json"), "ft": load("results/finetune/results.json"),
        "cfg": yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")),
    }


def pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(<!-- gen:{name} -->)(.*?)(<!-- /gen:{name} -->)", re.S)


def render(text: str, s: dict) -> str:
    for name, fn in BLOCKS.items():
        pat = pattern(name)
        if not pat.search(text):
            raise SystemExit(f"README.md has no gen:{name} block")
        text = pat.sub(lambda m, f=fn: m.group(1) + "\n" + f(s) + "\n" + m.group(3), text)
    return text


# ------------------------------------------------------------------------------------------------ prose facts
def prose_checks(text: str, s: dict) -> list[str]:
    cfg, bad = s["cfg"], []
    b = s["bench"]
    pd = b["per_defect"]

    def need(snippet: str) -> None:
        if snippet not in text:
            bad.append(f"README.md should contain {snippet!r}")

    need(f"Jaccard {cfg['dedupe']['near']['threshold']}")
    need(f"{cfg['dedupe']['near']['shingle_words']}-word shingles")
    need(f"cosine {cfg['dedupe']['semantic']['threshold']}")
    need(f"cosine of {cfg['quality']['relevance']['min_cosine']:.2f}")
    need(f"threshold of {cfg['safety']['toxicity']['threshold']}")
    need(f"below {cfg['balance']['min_rows']} survivors")
    need("Truncation (" + str(round(100 * pd["truncated"]["recall"])) + "% recall)")
    need("Toxicity (" + str(round(100 * pd["toxic"]["recall"])) + "% recall)")
    need(f"{100 * b['untouched']['false_reject_rate']:.1f}% false-reject rate")
    need(f"{s['messy']['rows_in']} rows I wrote")
    lines = (ROOT / "examples" / "messy_sample.jsonl").read_text(encoding="utf-8").splitlines()
    if len([ln for ln in lines if ln.strip()]) != s["messy"]["rows_in"]:
        bad.append("examples/messy_sample.jsonl and results/messy_sample_audit.json disagree on the row count")
    # the README says every injected defect in the messy file was caught and no clean row was rejected;
    # the file has 21 deliberately defective rows (6 duplicates, 12 quality defects, 3 safety defects)
    if s["messy"]["rows_rejected"] != 21:
        bad.append("results/messy_sample_audit.json rejects a different number of rows than the 21 planted defects")
    if s["tp_cpu"]["logical_cores"] != 12:
        bad.append("README says 12 logical cores but throughput_cpu.json recorded a different count")
    sample = (ROOT / "data_sample" / "dolly_cleaned_sample.jsonl").read_text(encoding="utf-8").splitlines()
    need(f"{n(len(sample))} cleaned Dolly rows")
    # the off-topic example in the README must be a real row of the recorded Dolly run
    index = s["idx_dolly"]
    pos = next(i for i, r in enumerate(index["rows"]) if r[0] == "d00248")
    chunk = json.loads((ROOT / f"docs/data/dolly/c{pos // index['chunk']}.json").read_text(encoding="utf-8"))
    row = chunk[pos % index["chunk"]]
    first = row["rs"][0]
    if first["c"] != "off_topic" or abs(first["v"] - 0.083) > 1e-9:
        bad.append("the off-topic example row (d00248, cosine 0.083) no longer matches the recorded run")
    need("cosine of 0.083")
    # every gate's counts must add up
    for name, a in (("dolly", s["dolly"]), ("messy", s["messy"])):
        g = a["gates"]
        if g[0]["in"] != a["rows_in"] or g[-1]["out"] != a["rows_kept"]:
            bad.append(f"{name}: gate counts do not run from rows_in to rows_kept")
        for x, y in zip(g, g[1:], strict=False):
            if x["out"] != y["in"] or x["in"] - x["rejected"] != x["out"]:
                bad.append(f"{name}: gate {x['gate']} counts do not reconcile")
    # the recorded run index carries the same audit as the results file
    if s["idx_dolly"]["audit"]["rows_kept"] != s["dolly"]["rows_kept"]:
        bad.append("docs/data/dolly/index.json and results/dolly_headline.json disagree on rows_kept")
    if s["idx_bench"]["audit"]["rows_kept"] != b["audit"]["rows_kept"]:
        bad.append("docs/data/contaminated/index.json and results/benchmark_bench.json disagree on rows_kept")
    if (ROOT / "docs" / "data" / "finetune.json").read_text(encoding="utf-8") != (ROOT / "results" / "finetune" / "results.json").read_text(encoding="utf-8"):
        bad.append("docs/data/finetune.json differs from results/finetune/results.json")
    ft = s["ft"]
    for key in ("raw", "cleaned", "random"):
        if any(len(v["values"]) != len(ft["config"]["seeds"]) for v in ft["conditions"][key]["metrics"].values()):
            bad.append(f"fine-tune results: {key} does not have one value per seed")
    if ft["conditions"]["cleaned"]["train_rows"][0] != ft["data"]["cleaned_rows"] or ft["data"]["cleaned_rows"] != s["bench"]["audit"]["rows_kept"]:
        bad.append("the cleaned training set size differs from the rows kept in the benchmark run")
    if ft["conditions"]["random"]["train_rows"][0] != ft["data"]["cleaned_rows"]:
        bad.append("the random subsample is not the size of the cleaned set")
    if ft["data"]["raw_rows"] != s["bench"]["audit"]["rows_in"]:
        bad.append("the raw training set is not the whole benchmark slice")
    # what the UI counts from each recorded run's rows must equal that run's audit and the results file
    for name, idx, res in (("dolly", s["idx_dolly"], s["dolly"]), ("benchmark", s["idx_bench"], s["bench"]["audit"])):
        rows = idx["rows"]
        if len(rows) != idx["audit"]["rows_in"] or idx["audit"]["rows_in"] != res["rows_in"]:
            bad.append(f"{name}: index row count, its audit and the results file disagree")
        if idx["audit"]["gates"] != res["gates"]:
            bad.append(f"{name}: the index audit gates differ from the results file")
        for gi, g in enumerate(res["gates"]):
            if sum(1 for r in rows if r[3] == gi) != g["rejected"]:
                bad.append(f"{name}: rows rejected at {g['gate']} in the index differ from the audit")
        if sum(1 for r in rows if r[2] == "k") != res["rows_kept"]:
            bad.append(f"{name}: rows kept in the index differ from the audit")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the README blocks")
    mode.add_argument("--check", action="store_true", help="fail if the README is out of sync")
    args = ap.parse_args(argv)
    s = sources()
    current = README.read_text(encoding="utf-8")
    new = render(current, s)
    if args.write:
        README.write_text(new, encoding="utf-8", newline="\n")
        current = new
    elif new != current:
        import difflib

        diff = "\n".join(difflib.unified_diff(current.splitlines(), new.splitlines(), "README.md", "generated", lineterm="", n=0))
        print("README.md is out of sync with the results files:\n" + diff[:3000])
        return 1
    problems = prose_checks(current, s)
    for p in problems:
        print("MISMATCH:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
