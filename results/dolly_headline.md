# Audit report

- Rows in: **5000**, kept: **4633** (92.7%); train 4401, validation 232
- Input SHA-256: `20e55ae3a1da8876...`, seed 42, sftforge 0.1.0
- Models ran on: cuda (NVIDIA GeForce RTX 5060 Ti), float32

## Gates

| Gate | In | Rejected | Out | Reasons |
|---|---:|---:|---:|---|
| dedupe | 5000 | 13 | 4987 | semantic_duplicate 7, exact_duplicate 3, near_duplicate 3 |
| quality | 4987 | 35 | 4952 | truncated 17, wrong_language 6, refusal_boilerplate 4, off_topic 4, response_too_short 3, repetitive 1 |
| safety | 4952 | 15 | 4937 | toxicity 15 |
| balance | 4937 | 304 | 4633 | category_cap 304 |

## Categories

| Category | Input | After gates | Final |
|---|---:|---:|---:|
| brainstorming | 615 | 605 | 605 |
| classification | 719 | 715 | 715 |
| closed_qa | 562 | 559 | 559 |
| creative_writing | 236 | 231 | 231 |
| general_qa | 711 | 699 | 699 |
| information_extraction | 487 | 484 | 484 |
| open_qa | 1253 | 1230 | 926 |
| summarization | 417 | 414 | 414 |

Per-category cap applied: 926 rows.

## Redactions in kept rows

- email: 6
- phone: 1

## Normalisation edits (not rejections)

- whitespace: 946
- unicode_nfkc: 84
- pii_redacted: 4

## Timings (seconds)

- normalise: 1.31
- dedupe: 31.68
- quality: 10.9
- safety: 24.29
- balance: 1.27
- total: 69.45
