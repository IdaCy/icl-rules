# Cross-family check

Cross-family experiment for sections 3 and 5 of the report. The plan and thresholds were set before any paid API calls for this run. The same thresholds are in scripts/analyze_cross_family.py, so the verdicts can be rebuilt from the raw cross-family JSONL files.

## Purpose

- Check whether the deconfounded-set results hold for a strong model from another provider.
- Check whether the three rule datasets depend too much on labels from one model family.

## Model and run setup

- Model: Anthropic Opus 4.8, used through the native Anthropic SDK.
- Step 1 setup: thinking off, small max tokens, and parse the True/False prefix. This matches the short-answer Step 1 setup as closely as the API allows.
- Prompt setup: replay the same logged gpt-4.1 Step 1 messages, so the model sees the same few-shot examples and held-out items.
- Main signal: compare v1 to v2 for the same model. The absolute accuracy is useful context, but the drop from v1 to v2 is the key number.

## B1: deconfounded-set check

- Datasets: word_count_geq_8 v1/v2 and second_word_capitalized v1/v2.
- Items: all held-out items, 3 contexts, 360 calls per dataset.
- Metric: pooled held-out accuracy and the v1 to v2 change.

## B1 thresholds

- word_count_geq_8 counts as replicated if v1 is at least 0.85, the v1 minus v2 drop is at least 0.10, and v2 is at most 0.80. This is the case where a model can still pick up a rough length cue after the rebuild, so v2 does not need to fall all the way to chance.
- second_word_capitalized counts as replicated if v1 is at least 0.85 and v2 is at most 0.65. This is the case where the rebuild should remove the easy capitalization cue, so a strong drop is expected.
- If v1 is below 0.85, mark the result inconclusive because the model did not first solve the original held-out set well enough.

## Confirmation arm

- Re-run the two v1 sets with thinking on at low effort.
- Use this only to separate a weak thinking-off run from a real failure on the original task.
- Report every outcome: replicate, not replicated, or inconclusive.
- One planned run. No tuning after seeing results.

## B2: outside-family label check

- For each item in the three rule pools, ask the rule's yes/no question.
- Compare the answer to the stored consensus label.
- Metric: agreement by rule.
- For physically_impossible, also report agreement by impossibility type and the both-items-correct rate within each minimal pair.

## B2 thresholds

- Agreement at least 0.90: labels do not look specific to the original model family.
- Agreement below 0.80: labels look family-dependent.
- Agreement from 0.80 to 0.90: mixed result.

## Claim for B2

This check only asks whether the stored labels look specific to one model family. High agreement does not prove human correctness; it only means a strong model from another provider gives the same labels on these items.

## Cost

- Planned: about $15 to $20.
- Actual: about $11, recorded in the cost JSON files.
