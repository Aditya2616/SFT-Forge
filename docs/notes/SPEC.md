# SFT Forge: spec

Tagline: clean instruction-tuning data, before it ever reaches training.

## 1. Pipeline
Input: JSONL/JSON in Alpaca (`instruction,input,output`), Dolly (`instruction,context,response,category`), or chat (`messages:[{role,content}]` / ShareGPT `conversations:[{from,value}]`). A loader normalises to a Pandas frame: `id, instruction, context, response, category, source_row`. Multi-turn chats are split into (prompt, reply) pairs with the turn index kept.

Every gate writes to per-record audit columns: `status`, `gate`, `reasons` (code, message, measured value, threshold), `checks` (every rule's measured value) and `scores`. A record is rejected at the first failing gate; later gates do not run on it, but scores from gates it passed are kept.

Gate 0, normalise: Unicode NFKC, whitespace collapse, control chars stripped, `[n]` citation markers removed from responses (logged as an edit, not a rejection).

Gate 1, dedupe
- exact: SHA-1 of normalised (instruction + context + response); the first occurrence is kept.
- near: MinHash (datasketch, 5-word shingles, 128 perms) + LSH at Jaccard 0.8 on instruction+response.
- semantic: all-MiniLM-L6-v2 embeddings on the instruction; cosine >= 0.95 within category (SemDeDup style). Threshold configurable. Reasons name the surviving twin's id and the similarity.

Gate 2, quality (transparent rules; each threshold in YAML; each rule a pure function)
- length bounds and response/instruction length ratio; alphanumeric ratio (Data-Juicer style); repetition (repeated n-gram share, repeated lines); empty/placeholder text.
- truncation (ends mid-sentence, dangling bracket or code fence); refusal/boilerplate phrase list.
- language id (lingua, subset, confidence), allowed languages in config.
- relevance: cosine(instruction, response) from MiniLM below a threshold, with per-category exemptions.
- optional perplexity from SmolLM2-135M (off by default for speed; not used in any reported run).
- combined 0-1 quality score for the UI and for balancing tie-breaks; the verdict itself comes from hard rules.

Gate 3, safety
- PII: email, phone, card (Luhn), IPv4, IBAN, SSN-like; action `redact` to typed placeholders (`[EMAIL]`) by default, or `reject` by config.
- secrets: AWS/GitHub/Slack/Stripe key patterns, private key blocks, entropy check; always reject.
- toxicity: unitary/toxic-bert, reject if max over (toxic, severe_toxic, threat, identity_hate) exceeds a threshold on instruction or response; a rule list for self-harm and similar patterns as a second signal. Reasons include class and score. Rows in the `classification` category are exempt from the toxicity gate (`allow_categories` in the config) because that task legitimately quotes toxic text; limitation stated.

Final stage, balance and export (shown as the SFT set, not a gate)
- stratified cap per category (default: no category above `max_share`, 0.20 in `configs/default.yaml`), choosing within a category by quality score; optional length-bucket balancing.
- stratified train/val split (`train_test_split(stratify_by_column="category")` after casting to ClassLabel), seeded.
- exports: `train.jsonl`/`val.jsonl` (Alpaca schema), chat-messages JSONL, chat-template text (ChatML by default, or any Hugging Face tokenizer's `apply_chat_template`), HF `save_to_disk`, `DATASET_CARD.md`, `audit.json` and `audit.md` (counts per gate, per reason, per category before/after, thresholds, model versions, input hash, seed), `rejects.jsonl` with all reasons.

## 2. Layout
```
SFT-Forge/
  pyproject.toml  requirements.txt  .env.example  LICENSE (MIT)  README.md
  .github/workflows/ci.yml
  configs/default.yaml
  src/sftforge/
    schema.py       pydantic config and record models
    loaders.py      Alpaca/Dolly/chat/ShareGPT -> DataFrame
    normalize.py
    gates/          base.py dedupe.py quality.py safety.py   (run(df, cfg, ctx) writes audit columns)
    detectors/      pii.py secrets.py language.py text_rules.py
    models.py       lazy MiniLM / toxic-bert / SmolLM2 wrappers, device selection, optional disk cache
    balance.py  export.py  card.py  audit.py  runjson.py
    pipeline.py     orchestrates the gates and returns the frame plus the audit
    cli.py          sftforge run | serve
    server/app.py   FastAPI: GET /health, GET /api/config, GET /api/example, POST /api/run (pasted JSONL),
                    POST /api/run/upload, GET /api/run/{id} (status and result, polled by the UI),
                    GET /api/run/{id}/export.zip
  docs/             the web UI (index.html, forge.js, forge.css), served by FastAPI and as the static demo;
                    docs/data/<run>/ recorded runs (index + detail chunks), docs/assets/ screenshots, docs/notes/
  examples/         messy_sample.jsonl
  data_sample/      cleaned Dolly sample (CC BY-SA 3.0)
  results/          measured outputs that the README quotes
  scripts/          fetch_data, make_benchmark, eval_benchmark, run_dolly, record_messy, throughput, check_numbers, smoke, shoot, check_ui
  tests/            fast offline tests + @pytest.mark.slow model tests
```
Models load lazily. Fast tests use deterministic stub embedder/toxicity scorers behind the same protocols (`Embedder`, `ToxicityScorer`); the slow marker runs the real models.

## 3. Data flow
loader -> normalise -> dedupe (exact, near, semantic) -> quality -> safety -> balance -> split -> export + audit. `pipeline.run()` returns the frame and the audit; `runjson` turns it into the recorded demo files (an index plus detail chunks) or, for live runs, one JSON object that the server returns when the UI polls the job.

## 4. UI (the real version of the site's scene)
One page, phone first. Header: dataset picker (recorded Dolly run, paste JSONL, upload) and Run. Centre: flow of source stream -> three gates (Dedupe, Quality, Safety) -> SFT set, vertical on phones, horizontal on desktop. A canvas draws real records as small cards; at each gate a rejected card is knocked out sideways in orange with its short reason; survivors (lime) stack in the SFT set, whose category bars fill (the balance). Below: per-gate stat strip (in, out, rejected, top reasons) and a reason histogram. Tapping any card or any row in the virtualised stream log opens an inspector sheet: the text, a per-gate timeline, each rule's measured value vs threshold, scores, redaction diff, the twin for duplicates. Controls: speed, pause, filter by gate/reason/category; threshold sliders and the export button in live mode only. Demo mode replays the files in `docs/data/<run>/`, labelled "Recorded run", with the full audit table. Design tokens follow my portfolio; transform/opacity animation only, DPR capped at 2, paused off screen, reduced-motion gives a static step-through.

## 5. Tests (fast, offline)
Loaders for each format including malformed lines; normalisation; exact and near dup on known pairs; each quality rule with positive and negative cases; Luhn and PII redaction; secrets; toxicity gating with a stub scorer; balance invariants (no category above max_share, deterministic with seed); stratified split proportions; export schema and card; audit totals reconcile (in = kept + sum of rejects); stream event order; FastAPI health, run and SSE via TestClient; CLI smoke. Slow marker: real MiniLM and toxic-bert sanity cases and an end-to-end run. CI: ruff + `pytest -m "not slow"`.

## 6. Done means
- `sftforge run data.jsonl --config configs/default.yaml --out out/` works from the README on a fresh clone with a new venv.
- Real run on a 5,000-row Dolly slice (fixed seed) reported: rows in, rejected per gate and reason, final size, category shares before/after, CPU throughput; `scripts/run_dolly.py` reproduces it.
- Thresholds checked on a hand-labelled sample of about 100 rows, per-gate precision reported, sample committed.
- UI verified by Playwright screenshots at 1440x900 and 390x844, no console errors; `docs/` demo works from a static server; paste/upload live run works.
- pytest and ruff green; README with Mermaid diagram, results, limitations (regex PII misses names; toxicity model bias; Dolly already clean; CC BY-SA obligations).

## 7. Where the build differs from this spec
- Threshold checking uses a seeded synthetic contamination benchmark (`scripts/make_benchmark.py`, `scripts/eval_benchmark.py`) with known defects and ground truth, tuned on a dev slice and reported on a separate bench slice, instead of a hand-labelled sample. There is no `label_sample` script.
- There is a single config, `configs/default.yaml`; no `strict.yaml`.
- The web UI lives in `docs/` only and is served from there by FastAPI.
- The before/after fine-tuning experiment was dropped and is not part of this project.
- Recorded demo runs are an index plus detail chunks (`docs/data/<run>/`), not one JSON file, so the first load stays small.
- Model device is chosen automatically (CUDA if available, else CPU) and recorded in the audit.
