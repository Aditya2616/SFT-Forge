# SFT Forge: research notes

Date: 2026-11-12. Everything below was checked when I wrote it; licences were read from the Hugging Face API card metadata, not from memory.

## 1. What the field does

| Project | What it contributes | What I take from it |
|---|---|---|
| Alpaca-cleaned (yahma) | Community clean-up of the 52k Alpaca set by hand and script: wrong answers, empty outputs, truncated text, "as an AI language model" boilerplate. | Rule-based gates catch a lot cheaply; refusal and truncation detectors are worth having. |
| AlpaGasus (arXiv 2307.08701) | An LLM grader scores each triplet 0-5; a 9k filtered subset beats the 52k original under GPT-4 judging. | Quality over quantity is real, but their grader is a hosted LLM. I use local, explainable signals and say so. |
| LIMA | 1,000 hand-curated examples are enough for strong behaviour. | Motivates a small, balanced final set, not "keep everything". |
| Dolly-15k (Databricks) | 15k human-written pairs across 8 InstructGPT categories, with optional `context`. | The `category` field makes stratified balancing real. Also the dataset I run on. |
| OpenHermes 2.5 | Large mixed synthetic set with dedup and filtering of refusals and boilerplate. | Same filters: refusal/boilerplate and near-duplicate removal. |
| Argilla / distilabel | Annotation UI plus pipelines that attach scores to records. | Design principle: scores and reasons live on the record, not just a keep/drop flag. |
| Lilac | Embedding clustering and concept search over datasets to find near-duplicates and clusters. | Embedding-based near-dup grouping, per-category view. |
| Cleanlab (CLEAR, arXiv 2403.12776) | Auto-Filter and Auto-Correct on instruction data, using model confidence. | Warns that over-filtering shrinks coverage, so SFT Forge reports how much each gate removes, per category. |
| Data-Juicer | Operator library: `alphanumeric_filter` (default min ratio 0.25), `flagged_words_filter` (max 0.045), `perplexity_filter` (KenLM), language-id score filter. | The heuristic set and its default thresholds are a sound starting point; every operator is a small pure function. |
| HF text-data-filtering (BigScience) | Stats-then-threshold with a visual explorer. | The "show the distribution, then choose the cut" UI idea. |
| NeMo Curator | Three dedup tiers: exact (MD5), fuzzy (MinHash+LSH, ~80% similarity), semantic (embeddings). Its fuzzy path needs a GPU/cuDF. | Same three tiers, CPU only. |
| SemDeDup (Abbas et al. 2023) | Cluster embeddings, drop pairs above a cosine threshold within a cluster. | Embedding near-dup with all-MiniLM-L6-v2 (already in the HF cache), threshold reported not hidden. |

Takeaway: the credible pattern is staged filters, each producing a per-record score and reason, with cheap exact and hash methods first and model-based ones last, plus a report. That is what the site's "three gates" shows.

## 2. Library versions here
- Installed: pandas 3.0.2, sentence-transformers 5.4.1, transformers 5.7.0, torch 2.11 (CPU), scikit-learn 1.8, fastapi 0.136.
- Missing, installable (pip index confirmed): `datasets 5.2.0`, `datasketch 2.0.0`, `lingua-language-detector 2.2.0`. These go in the project venv only.
- HF Datasets API used: `Dataset.from_pandas`, `Dataset.map/filter(with_indices=True)`, `train_test_split(stratify_by_column=...)` (requires a `ClassLabel` column, so category must be cast), `to_json(lines=True)`, `save_to_disk`.
- The pipeline works on a Pandas frame (audit columns are easy there) and converts to a `Dataset` only for split and export.

## 3. Data and models (licences from HF card metadata)

Datasets
- **databricks/databricks-dolly-15k: cc-by-sa-3.0.** Human-written, 8 categories, 13 MB. Chosen. Share-alike: the cleaned slice I publish must stay CC BY-SA 3.0 with attribution, and I will not commit the whole set. Wikipedia-derived contexts are CC BY-SA too.
- OpenAssistant/oasst1: apache-2.0. Human multilingual conversation trees. Good optional second run (language gate, tree-to-pairs).
- yahma/alpaca-cleaned: card says cc-by-4.0, but it derives from Alpaca outputs generated with OpenAI models (upstream CC BY-NC, OpenAI terms). Not clean enough in spirit; not used for the headline run.
- HuggingFaceH4/no_robots: cc-by-nc-4.0. Rejected.

Models (all CPU-friendly)
- sentence-transformers/all-MiniLM-L6-v2: apache-2.0, 22.7M params. Near-dup and instruction-response relevance.
- unitary/toxic-bert: apache-2.0, 109M params. Multi-label toxicity classifier.
- HuggingFaceTB/SmolLM2-135M: apache-2.0, 135M params. Optional perplexity signal (off by default).
- Rejected on licence: s-nlp/roberta_toxicity_classifier (openrail++), KoalaAI/Text-Moderation (openrail), facebook fasttext lid (cc-by-nc-4.0).
- Language id: `lingua-language-detector` (Apache-2.0, models bundled, no download), restricted to a language subset for memory. Fallback: `papluca/xlm-roberta-base-language-detection` (MIT, 278M), heavier.

Rule-based pieces (no model): own PII regex with validators (Luhn for cards, email, phone, IPv4, IBAN, SSN-like), secrets patterns in the style of detect-secrets (AWS keys, GitHub tokens, private key headers, high-entropy strings). I will not depend on Presidio: it needs a spaCy model of hundreds of MB for modest gain here. Limitation to state: regex PII misses names and addresses.

Download budget: dolly 13 MB + MiniLM ~90 MB + toxic-bert ~440 MB (safetensors only) + SmolLM2 ~270 MB (only if the perplexity signal is switched on), under 1 GB.

## 4. Honest expectations
Dolly is already human-curated, so reject rates will be modest (single digit to low double digit percent). That is a real result and will be reported as is. Dolly does contain true duplicates, empty or very short responses, `[1]`-style Wikipedia citation noise, some PII-like strings and a few toxic items, so each gate should fire. Thresholds will be tuned on a hand-labelled sample, not on the headline numbers.

## Sources
- AlpaGasus: https://arxiv.org/abs/2307.08701
- CLEAR (Cleanlab): https://arxiv.org/abs/2403.12776 ; https://cleanlab.ai/blog/filter-llm-tuning-data/
- NeMo Curator dedup: https://docs.nvidia.com/nemo/curator/curate-text/process-data/deduplication.md
- SemDeDup: https://arxiv.org/abs/2303.09540
- Data-Juicer filters: https://modelscope.github.io/data-juicer/en/main/data_juicer.ops.filter.html
- HF text-data-filtering: https://huggingface.co/spaces/huggingface/text-data-filtering
- Dolly-15k: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- oasst1: https://huggingface.co/datasets/OpenAssistant/oasst1
- unitary/toxic-bert: https://huggingface.co/unitary/toxic-bert
- SmolLM2-135M: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
