The rows in dolly_cleaned_sample.jsonl are derived from databricks-dolly-15k
(https://huggingface.co/datasets/databricks/databricks-dolly-15k), (c) Databricks, Inc. and the Wikipedia
contributors whose text appears in the context fields. They are licensed under Creative Commons
Attribution-ShareAlike 3.0 Unported (https://creativecommons.org/licenses/by-sa/3.0/).
Changes: filtered, de-duplicated, PII-redacted and category-balanced by SFT Forge; text otherwise unmodified
apart from whitespace and citation-marker normalisation.

This folder is separate from the MIT-licensed code in the rest of the repository.

Rows naming specific AI products (a case-insensitive deny-list: see DENY in src/sftforge/runjson.py) are left
out of the published files. Rows left out of this sample: 3. Rows whose text is left out of the recorded
demo runs in docs/data/dolly/: 12 (they still appear in the stream and the statistics, without text).
The audit counts of the real runs are unchanged.
