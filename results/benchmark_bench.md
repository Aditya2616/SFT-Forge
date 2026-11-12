# Contamination benchmark (bench slice)

Synthetic contamination: defects were injected at known rates; counts below are real measurements.

| Defect | n | Recall | Right gate | Right reason | Notes |
|---|---:|---:|---:|---:|---|
| exact_dup | 60 | 100.0% | 60 | 60 | {'exact_duplicate': 60} |
| near_dup | 60 | 100.0% | 60 | 60 | {'near_duplicate': 46, 'semantic_duplicate': 14} |
| off_topic | 60 | 86.7% | 44 | 42 | {'off_topic': 42, 'near_duplicate': 8, 'wrong_language': 2} |
| paraphrase_dup | 60 | 95.0% | 57 | 57 | {'semantic_duplicate': 48, 'near_duplicate': 9} |
| pii_card | 15 | 100.0% | - | - | rejected elsewhere: 0 |
| pii_email | 15 | 93.3% | - | - | rejected elsewhere: 1 |
| pii_iban | 15 | 100.0% | - | - | rejected elsewhere: 0 |
| pii_ipv4 | 15 | 100.0% | - | - | rejected elsewhere: 0 |
| pii_phone | 15 | 100.0% | - | - | rejected elsewhere: 0 |
| pii_ssn | 15 | 100.0% | - | - | rejected elsewhere: 0 |
| refusal | 60 | 100.0% | 60 | 60 | {'refusal_boilerplate': 60} |
| secret_aws | 15 | 100.0% | 13 | 13 | {'secret': 13, 'off_topic': 2} |
| secret_generic | 15 | 100.0% | 14 | 14 | {'secret': 14, 'off_topic': 1} |
| secret_github | 15 | 100.0% | 13 | 13 | {'secret': 13, 'wrong_language': 2} |
| secret_private_key | 15 | 100.0% | 15 | 15 | {'secret': 15} |
| toxic | 45 | 51.1% | 20 | 20 | {'toxicity': 20, 'off_topic': 2, 'truncated': 1} |
| truncated | 60 | 40.0% | 24 | 24 | {'truncated': 24} |
| wrong_language | 60 | 96.7% | 58 | 58 | {'wrong_language': 58} |

Untouched rows: 2565; rejected 42 (1.64%); reasons {'truncated': 12, 'near_duplicate': 8, 'toxicity': 7, 'semantic_duplicate': 5, 'refusal_boilerplate': 3, 'wrong_language': 3, 'response_too_long': 1, 'off_topic': 1, 'response_too_short': 1, 'exact_duplicate': 1}.
PII false redactions on untouched kept rows: 1.

| Gate | Rejected | Of which injected | Precision |
|---|---:|---:|---:|
| dedupe | 200 | 186 | 93.0% |
| quality | 215 | 194 | 90.2% |
| safety | 82 | 75 | 91.5% |
