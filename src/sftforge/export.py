"""Write the cleaned SFT set, rejects, audit report and dataset card to an output folder."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import audit as audit_mod
from . import card
from .detectors import secrets
from .pipeline import Result
from .schema import Config

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def user_prompt(instruction: str, context: str) -> str:
    return instruction if not context else f"{instruction}\n\n{context}"


def chatml(messages: list[dict[str, str]]) -> str:
    return "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)


def _formatter(template: str):
    if template == "chatml":
        return chatml
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(template)
    return lambda msgs: tok.apply_chat_template(msgs, tokenize=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _mask_text(text: str) -> str:
    for h in sorted(secrets.find_secrets(text), key=lambda h: -h.start):
        text = text[: h.start] + secrets.mask(h.text) + text[h.end :]
    return text


def export(result: Result, cfg: Config, out: str | Path, source_name: str = "input") -> dict[str, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df: pd.DataFrame = result.df
    fmt = _formatter(cfg.export.chat_template)
    files: dict[str, Path] = {}

    kept = df[df["status"] == "kept"]
    for split in ("train", "val"):
        part = kept[kept["split"] == split]
        alpaca = [
            {"id": r.id, "instruction": r.instruction, "input": r.context, "output": r.response, "category": r.category}
            for r in part.itertuples()
        ]
        chat = [
            {"id": r.id, "messages": [{"role": "user", "content": user_prompt(r.instruction, r.context)},
                                      {"role": "assistant", "content": r.response}]}
            for r in part.itertuples()
        ]
        text = [{"id": c["id"], "text": fmt(c["messages"])} for c in chat]
        for name, rows in ((f"{split}.jsonl", alpaca), (f"chat_{split}.jsonl", chat), (f"text_{split}.jsonl", text)):
            _write_jsonl(out / name, rows)
            files[name] = out / name

    rej = df[df["status"] != "kept"]
    rejects = [
        {"id": r.id, "gate": r.gate, "reasons": r.reasons, "twin": r.twin, "category": r.category,
         "instruction": _mask_text(r.instruction), "context": _mask_text(r.context), "response": _mask_text(r.response)}
        for r in rej.itertuples()
    ]
    _write_jsonl(out / "rejects.jsonl", rejects)
    files["rejects.jsonl"] = out / "rejects.jsonl"

    (out / "audit.json").write_text(json.dumps(result.audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "audit.md").write_text(audit_mod.to_markdown(result.audit), encoding="utf-8")
    (out / "DATASET_CARD.md").write_text(card.render(result.audit, cfg, source_name), encoding="utf-8")
    files.update({"audit.json": out / "audit.json", "audit.md": out / "audit.md", "DATASET_CARD.md": out / "DATASET_CARD.md"})

    if cfg.export.hf_dataset and len(kept):
        from datasets import Dataset, DatasetDict

        def ds(split: str) -> Dataset:
            part = kept[kept["split"] == split]
            return Dataset.from_dict({
                "id": part["id"].tolist(), "instruction": part["instruction"].tolist(),
                "input": part["context"].tolist(), "output": part["response"].tolist(),
                "category": part["category"].tolist(),
            })

        splits = {"train": ds("train"), "validation": ds("val")}
        dd = DatasetDict({k: v for k, v in splits.items() if len(v)})
        dd.save_to_disk(str(out / "hf_dataset"))
        files["hf_dataset"] = out / "hf_dataset"
    return files
