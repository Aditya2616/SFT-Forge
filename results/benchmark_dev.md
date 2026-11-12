# Contamination benchmark (dev slice)

Synthetic contamination: defects were injected at known rates; counts below are real measurements.

Models ran on: cuda (NVIDIA GeForce RTX 5060 Ti), float32.

| Defect | n | Recall | Right gate | Right reason | Notes |
|---|---:|---:|---:|---:|---|
| exact_dup | 60 | 100.0% | 60 | 60 | {'exact_duplicate': 60} |
| near_dup | 60 | 100.0% | 60 | 60 | {'near_duplicate': 56, 'semantic_duplicate': 4} |
| off_topic | 60 | 86.7% | 41 | 37 | {'off_topic': 37, 'near_duplicate': 11, 'truncated': 3, 'wrong_language': 1} |
| paraphrase_dup | 60 | 85.0% | 51 | 51 | {'semantic_duplicate': 40, 'near_duplicate': 11} |
| pii_card | 15 | 80.0% | - | - | rejected elsewhere: 3 |
| pii_email | 15 | 100.0% | - | - | rejected elsewhere: 0 |
| pii_iban | 15 | 100.0% | - | - | rejected elsewhere: 0 |
| pii_ipv4 | 15 | 100.0% | - | - | rejected elsewhere: 0 |
| pii_phone | 15 | 100.0% | - | - | rejected elsewhere: 0 |
| pii_ssn | 15 | 100.0% | - | - | rejected elsewhere: 0 |
| refusal | 60 | 100.0% | 60 | 60 | {'refusal_boilerplate': 60} |
| secret_aws | 15 | 100.0% | 14 | 14 | {'secret': 14, 'off_topic': 1} |
| secret_generic | 15 | 100.0% | 15 | 15 | {'secret': 15} |
| secret_github | 15 | 100.0% | 12 | 12 | {'secret': 12, 'wrong_language': 2, 'near_duplicate': 1} |
| secret_private_key | 15 | 100.0% | 13 | 13 | {'secret': 13, 'off_topic': 2} |
| toxic | 45 | 55.6% | 25 | 25 | {'toxicity': 25} |
| truncated | 60 | 40.0% | 24 | 24 | {'truncated': 24} |
| wrong_language | 60 | 98.3% | 58 | 58 | {'wrong_language': 58, 'near_duplicate': 1} |

Untouched rows: 2565; rejected 45 (1.75%); reasons {'truncated': 14, 'near_duplicate': 10, 'off_topic': 5, 'toxicity': 5, 'refusal_boilerplate': 3, 'exact_duplicate': 3, 'semantic_duplicate': 2, 'wrong_language': 1, 'instruction_too_short': 1, 'repetitive': 1}.
PII false redactions on untouched kept rows: 1.

| Gate | Rejected | Of which injected | Precision |
|---|---:|---:|---:|
| dedupe | 199 | 184 | 92.5% |
| quality | 216 | 191 | 88.4% |
| safety | 84 | 79 | 94.0% |
