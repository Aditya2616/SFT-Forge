"""Does cleaning with SFT Forge change what a model learns? A before/after fine-tuning comparison.

SmolLM2-135M-Instruct is fine-tuned with an identical budget (same optimiser steps, batch size, learning-rate
schedule) on three training sets built from the contaminated benchmark slice, for several seeds each:

  raw      the slice as it is (synthetic defects included)
  cleaned  the same slice after the default SFT Forge pipeline
  random   a random subsample of the raw slice with as many rows as the cleaned set (same size as `cleaned`, so
           "cleaner" is not confused with "smaller"); drawn again for every seed

and compared with the untuned model (`base`). Every run is evaluated on held-out, clean Dolly rows that no
training set contains: held-out loss, and greedy generations scored for refusal boilerplate, cut-off answers,
non-English answers, PII- or secret-shaped strings, near-verbatim copies of training responses, ROUGE-L and
MiniLM cosine similarity against the reference answers.

Steps (data from `scripts/make_benchmark.py --slice bench` must exist):
    python scripts/finetune_compare.py            # train and evaluate every missing run (resumable)
    python scripts/finetune_compare.py --summarize-only

Per-run results go to results/finetune/runs/, the summary to results/finetune/results.json and
docs/data/finetune.json (the web UI reads that copy). No weights are saved.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import subprocess
import sys
import time

from common import ROOT, dolly_slice

OUT = ROOT / "results" / "finetune"
RUNS = OUT / "runs"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
CFG = {
    "model": MODEL, "steps": 400, "batch_size": 8, "micro_batch": 1, "eval_batch": 8, "vram_cap_gib": 3.9, "max_len": 320, "lr": 2e-5, "warmup_steps": 20, "min_lr_fraction": 0.1,
    "weight_decay": 0.0, "grad_clip": 1.0, "precision": "float32", "seeds": [0, 1, 2], "eval_prompts": 300,
    "max_new_tokens": 160, "prompt_words": 120, "verbatim_ngram": 8, "verbatim_share": 0.5, "verbatim_min_words": 12,
}
CONDITIONS = ["raw", "cleaned", "random"]
# (key, label, direction): +1 means higher is better, -1 lower is better
METRICS = [
    ("loss", "Held-out loss (reference answers)", -1),
    ("refusal_rate", "Refusal or boilerplate outputs", -1),
    ("cut_rate", "Outputs that stop mid-sentence", -1),
    ("foreign_rate", "Non-English outputs", -1),
    ("pii_rate", "Outputs with PII- or secret-shaped strings", -1),
    ("verbatim_rate", "Near-verbatim copies of training rows", -1),
    ("rouge_l", "ROUGE-L F1 against the reference", 1),
    ("embed_sim", "MiniLM cosine similarity to the reference", 1),
]
WORD = re.compile(r"\w+")


# ---------------------------------------------------------------------------------------------- GPU hygiene
def free_vram_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.split(",")
        return int(out[1]) - int(out[0])
    except Exception:
        return None


def wait_for_vram(need_mib: int = 4500, tries: int = 20) -> None:
    """Start only when the shared GPU has room (the job needs under 4 GB; keep 1 GB spare)."""
    for k in range(tries):
        free = free_vram_mib()
        if free is None or free >= need_mib:
            return
        print(f"waiting for GPU memory: {free} MiB free, need {need_mib} ({k + 1}/{tries})", flush=True)
        time.sleep(60)
    raise SystemExit("not enough free GPU memory; rerun later (finished runs are kept)")


# ---------------------------------------------------------------------------------------------- data
def trim(prompt: str, words: int) -> str:
    parts = prompt.split()
    return prompt if len(parts) <= words else " ".join(parts[:words])


def build_data() -> dict:
    from sftforge import pipeline
    from sftforge.export import user_prompt
    from sftforge.loaders import load_path
    from sftforge.models import Models
    from sftforge.schema import Config

    data = ROOT / "data" / "contaminated_bench"
    truth = json.loads((data / "truth.json").read_text(encoding="utf-8"))
    df = load_path(data / "benchmark.jsonl").df
    cfg = Config.load(ROOT / "configs" / "default.yaml")
    models = Models(cfg.models, cache_dir=ROOT / ".cache")
    res = pipeline.run(df, cfg, models)
    del models
    kept = res.df[res.df["status"] == "kept"]

    def rows(frame) -> list[dict]:
        return [{"id": r.id, "prompt": trim(user_prompt(r.instruction, r.context), CFG["prompt_words"]),
                 "response": r.response, "defect": truth.get(r.id, {}).get("defect")} for r in frame.itertuples()]

    raw, cleaned = rows(df), rows(kept)
    seen = {r["prompt"].lower() for r in raw}
    held = []
    for r in dolly_slice("heldout"):
        prompt = trim(user_prompt(r["instruction"], r.get("context", "")), CFG["prompt_words"])
        if prompt.lower() in seen or len(r["response"].split()) < 5:
            continue
        held.append({"prompt": prompt, "response": r["response"], "category": r["category"]})
        if len(held) >= CFG["eval_prompts"]:
            break
    return {"raw": raw, "cleaned": cleaned, "heldout": held, "audit_device": res.audit["device"],
            "kept_by_gate": [(g["gate"], g["rejected"]) for g in res.audit["gates"]]}


# ---------------------------------------------------------------------------------------------- text metrics
def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def rouge_l(pred: str, ref: str) -> float:
    a, b = words(pred), words(ref)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


def ngram_set(texts: list[str], n: int) -> set[int]:
    out: set[int] = set()
    for t in texts:
        w = words(t)
        for i in range(len(w) - n + 1):
            out.add(hash(tuple(w[i : i + n])))
    return out


def is_verbatim(text: str, grams: set[int]) -> bool:
    w = words(text)
    n = CFG["verbatim_ngram"]
    if len(w) < CFG["verbatim_min_words"]:
        return False
    cand = [hash(tuple(w[i : i + n])) for i in range(len(w) - n + 1)]
    return sum(1 for c in cand if c in grams) / len(cand) >= CFG["verbatim_share"]


def is_foreign(text: str, lang_cfg) -> bool:
    from sftforge.detectors import language

    allowed = {a.upper() for a in lang_cfg.allowed}
    sample = language.strip_proper_nouns(text[:900])[:700]
    if sum(c.isalpha() for c in sample) < lang_cfg.min_chars:
        return False
    conf = language.language_confidences(sample)
    other = max(v for k, v in conf.items() if k not in allowed)
    mine = sum(v for k, v in conf.items() if k in allowed)
    return other >= lang_cfg.max_other_confidence and mine < lang_cfg.max_allowed_confidence


def score_outputs(outputs: list[dict], refs: list[str], train_responses: list[str]) -> dict:
    from sftforge.detectors import pii, secrets, text_rules
    from sftforge.schema import Config

    lang_cfg = Config.load(ROOT / "configs" / "default.yaml").quality.language
    grams = ngram_set(train_responses, CFG["verbatim_ngram"]) if train_responses else set()
    n = len(outputs)
    refusal = sum(1 for o in outputs if text_rules.find_refusal(o["text"]))
    cut = sum(1 for o in outputs if o["ended"] and o["text"].strip() and text_rules.truncation_signal(o["text"]))
    foreign = sum(1 for o in outputs if is_foreign(o["text"], lang_cfg))
    leak = sum(1 for o in outputs if pii.find_pii(o["text"]) or secrets.find_secrets(o["text"]))
    verb = sum(1 for o in outputs if grams and is_verbatim(o["text"], grams))
    return {
        "refusal_rate": refusal / n, "cut_rate": cut / n, "foreign_rate": foreign / n, "pii_rate": leak / n,
        "verbatim_rate": verb / n, "rouge_l": sum(rouge_l(o["text"], r) for o, r in zip(outputs, refs, strict=True)) / n,
        "empty_rate": sum(1 for o in outputs if not o["text"].strip()) / n,
        "stopped_rate": sum(1 for o in outputs if o["ended"]) / n,
        "mean_words": sum(len(words(o["text"])) for o in outputs) / n,
    }


# ---------------------------------------------------------------------------------------------- model work
def encode_example(tok, prompt: str, response: str, im_end: int, max_len: int) -> tuple[list[int], list[int]]:
    p = tok(tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False),
            add_special_tokens=False)["input_ids"]
    r = tok(response, add_special_tokens=False)["input_ids"] + [im_end]
    ids = (p + r)[:max_len]
    labels = ([-100] * len(p) + r)[:max_len]
    return ids, labels


def collate(batch, pad: int, device):
    import torch

    width = max(len(i) for i, _ in batch)
    ids = torch.tensor([i + [pad] * (width - len(i)) for i, _ in batch], device=device)
    lab = torch.tensor([lb + [-100] * (width - len(lb)) for _, lb in batch], device=device)
    att = torch.tensor([[1] * len(i) + [0] * (width - len(i)) for i, _ in batch], device=device)
    return ids, lab, att


def lr_at(step: int) -> float:
    if step < CFG["warmup_steps"]:
        return CFG["lr"] * (step + 1) / CFG["warmup_steps"]
    prog = (step - CFG["warmup_steps"]) / max(1, CFG["steps"] - CFG["warmup_steps"])
    floor = CFG["min_lr_fraction"]
    return CFG["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))


def evaluate_model(model, tok, held: list[dict], device, im_end: int) -> tuple[float, list[dict]]:
    import torch

    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(held), CFG["eval_batch"]):
            batch = [encode_example(tok, h["prompt"], h["response"], im_end, CFG["max_len"]) for h in held[s : s + CFG["eval_batch"]]]
            ids, lab, att = collate(batch, tok.pad_token_id, device)
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            tgt = lab[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                     ignore_index=-100, reduction="sum")
            tot += float(loss)
            cnt += int((tgt != -100).sum())
    outputs: list[dict] = []
    tok.padding_side = "left"
    with torch.no_grad():
        for s in range(0, len(held), 50):
            chunk = held[s : s + 50]
            texts = [tok.apply_chat_template([{"role": "user", "content": h["prompt"]}], add_generation_prompt=True,
                                             tokenize=False) for h in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            gen = model.generate(**enc, do_sample=False, max_new_tokens=CFG["max_new_tokens"], eos_token_id=im_end,
                                 pad_token_id=tok.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:].tolist():
                ended = im_end in row
                row = row[: row.index(im_end)] if ended else row
                outputs.append({"text": tok.decode(row, skip_special_tokens=True).strip(), "ended": ended})
    tok.padding_side = "right"
    return tot / max(cnt, 1), outputs


def embed_similarity(outputs: list[dict], refs: list[str], device: str) -> float:
    from sftforge.models import SbertEmbedder
    from sftforge.schema import ModelsCfg

    cfg = ModelsCfg()
    emb = SbertEmbedder(cfg.embedder, cfg.embedder_revision, device=device)
    a, b = emb.encode([o["text"] or " " for o in outputs]), emb.encode(refs)
    return float((a * b).sum(axis=1).mean())


def run_one(cond: str, seed: int, data: dict, device: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    wait_for_vram()
    if device == "cuda":  # hard cap: the allocator frees its cache and fails rather than exceed the budget
        torch.cuda.set_per_process_memory_fraction(CFG["vram_cap_gib"] * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(seed)
    if cond == "raw":
        rows = data["raw"]
    elif cond == "cleaned":
        rows = data["cleaned"]
    elif cond == "random":
        rows = rng.sample(data["raw"], len(data["cleaned"]))
    else:
        rows = []
    tok = AutoTokenizer.from_pretrained(MODEL)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    model.config.use_cache = False
    t0 = time.time()
    curve: list[float] = []
    if cond != "base":
        enc = [encode_example(tok, r["prompt"], r["response"], im_end, CFG["max_len"]) for r in rows]
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        order: list[int] = []
        step_losses: list[float] = []
        model.train()
        for step in range(CFG["steps"]):
            while len(order) < CFG["batch_size"]:
                perm = list(range(len(enc)))
                rng.shuffle(perm)
                order += perm
            idx, order = order[: CFG["batch_size"]], order[CFG["batch_size"] :]
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            # one optimiser step on 8 examples, computed in micro-batches to keep GPU memory small; the loss is the
            # token-weighted mean over the whole batch, exactly as if it had been computed in one go
            total_tokens = sum(sum(1 for x in enc[i][1][1:] if x != -100) for i in idx)
            step_loss = 0.0
            for s in range(0, len(idx), CFG["micro_batch"]):
                ids, lab, att = collate([enc[i] for i in idx[s : s + CFG["micro_batch"]]], tok.pad_token_id, device)
                logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
                part = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1),
                                                         ignore_index=-100, reduction="sum") / total_tokens
                part.backward()
                step_loss += part.detach().item()
                del logits, part
            loss = step_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            step_losses.append(loss)
            if (step + 1) % 10 == 0:
                curve.append(round(sum(step_losses[-10:]) / 10, 4))
        del opt
    train_s = time.time() - t0
    peak = torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else 0.0
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    model.config.use_cache = True
    held = data["heldout"]
    t1 = time.time()
    loss, outputs = evaluate_model(model, tok, held, device, im_end)
    eval_s = time.time() - t1
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    refs = [h["response"] for h in held]
    m = score_outputs(outputs, refs, [r["response"] for r in rows])
    m["loss"] = loss
    m["perplexity"] = math.exp(loss)
    m["embed_sim"] = embed_similarity(outputs, refs, device)
    injected = sum(1 for r in rows if r["defect"])
    label = f"cuda ({torch.cuda.get_device_name(0)}), float32" if device == "cuda" else "cpu, float32"
    return {"condition": cond, "seed": seed, "device": label, "train_rows": len(rows), "injected_rows": injected, "metrics": m, "curve_every_10_steps": curve,
            "train_seconds": round(train_s, 1), "eval_seconds": round(eval_s, 1), "peak_vram_gib": round(peak, 2), "peak_vram_allocated_gib": round(peak_alloc, 2)}


# ---------------------------------------------------------------------------------------------- summary
# two-sided 95% critical values of Student's t for 1..30 degrees of freedom (stdlib only, no scipy needed)
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
        2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t_crit(df: int) -> float:
    return T975[df - 1] if 1 <= df <= len(T975) else 1.96


def mean_sd(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def summarize(device_label: str, data_info: dict) -> dict:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*.json"))]
    by = {c: sorted([r for r in runs if r["condition"] == c], key=lambda r: r["seed"]) for c in ["base", *CONDITIONS]}
    summary: dict = {"config": CFG, "metrics": [{"key": k, "label": lb, "better": d} for k, lb, d in METRICS], "conditions": {},
                     "paired": {}, "device": device_label, "data": data_info}
    for c, rs in by.items():
        if not rs:
            continue
        entry: dict = {"n_runs": len(rs), "train_rows": [r["train_rows"] for r in rs], "injected_rows": [r["injected_rows"] for r in rs],
                       "train_seconds": [r["train_seconds"] for r in rs], "peak_vram_gib": max(r["peak_vram_gib"] for r in rs),
                       "metrics": {}}
        for k, _, _ in METRICS:
            xs = [r["metrics"][k] for r in rs]
            m, sd = mean_sd(xs)
            entry["metrics"][k] = {"mean": m, "sd": sd, "values": xs}
        entry["perplexity"] = [r["metrics"]["perplexity"] for r in rs]
        if c != "base":
            L = min(len(r["curve_every_10_steps"]) for r in rs)
            entry["curve"] = [sum(r["curve_every_10_steps"][i] for r in rs) / len(rs) for i in range(L)]
        summary["conditions"][c] = entry
    pairs = [("cleaned", "raw"), ("cleaned", "random")]
    for a, b in pairs:
        if a not in by or b not in by or len(by[a]) < 2 or len(by[a]) != len(by[b]):
            continue
        res = {}
        for k, _, direction in METRICS:
            d = [x["metrics"][k] - y["metrics"][k] for x, y in zip(by[a], by[b], strict=True)]
            m, sd = mean_sd(d)
            half = t_crit(len(d) - 1) * sd / math.sqrt(len(d))
            res[k] = {"mean_diff": m, "ci95": [m - half, m + half], "values": d,
                      "excludes_zero": (m - half > 0) or (m + half < 0),
                      "favours_cleaned": (m * direction > 0) if (m - half > 0 or m + half < 0) else None}
        summary["paired"][f"{a}_minus_{b}"] = res
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these conditions (base, raw, cleaned, random)")
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "finetune_data.json"
    if args.summarize_only:
        data = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
    else:
        from sftforge.models import resolve_device

        device = resolve_device(args.device)
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            wait_for_vram(2500) if device == "cuda" else None
            data = build_data()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data), encoding="utf-8")
        seeds = args.seeds if args.seeds is not None else CFG["seeds"]
        todo = [("base", 0)] + [(c, s) for s in seeds for c in CONDITIONS]
        for cond, seed in todo:
            if args.only and cond not in args.only:
                continue
            path = RUNS / f"{cond}_s{seed}.json"
            if path.exists():
                continue
            print(f"run {cond} seed {seed}", flush=True)
            res = run_one(cond, seed, data, device)
            path.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
            print(json.dumps({k: round(v, 4) for k, v in res["metrics"].items()}), res["train_seconds"], "s", flush=True)
    if data is None:
        raise SystemExit("no cached data; run without --summarize-only once")
    device_runs = sorted({json.loads(p.read_text(encoding="utf-8"))["device"] for p in RUNS.glob("*.json")})
    if len(device_runs) != 1:
        raise SystemExit(f"runs were made on different devices: {device_runs}")
    label = device_runs[0]
    info = {"raw_rows": len(data["raw"]), "cleaned_rows": len(data["cleaned"]), "heldout_rows": len(data["heldout"]),
            "injected_in_raw": sum(1 for r in data["raw"] if r["defect"]),
            "injected_in_cleaned": sum(1 for r in data["cleaned"] if r["defect"]),
            "cleaned_defects": {}, "gate_rejections": data["kept_by_gate"]}
    for r in data["cleaned"]:
        if r["defect"]:
            info["cleaned_defects"][r["defect"]] = info["cleaned_defects"].get(r["defect"], 0) + 1
    summary = summarize(label, info)
    text = json.dumps(summary, indent=1) + "\n"
    (OUT / "results.json").write_text(text, encoding="utf-8")
    (ROOT / "docs" / "data" / "finetune.json").write_text(text, encoding="utf-8")
    print("summary written", file=sys.stderr)


if __name__ == "__main__":
    main()
