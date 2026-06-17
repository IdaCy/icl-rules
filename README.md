# Sentence Rules in In-Context Learning

This study measures if a language model can put into words a binary classification rule that it can already apply well from examples, and if the rule it states is the one responsible for its answers.

## Main findings

Across 30 rules with confound audits on gpt-4.1 and gpt-4.1-mini:

- Learning tracks word-level vocabulary. Rules whose vocabulary differs by class average 0.94 held-out accuracy, but a bag-of-words baseline solves 5 of the 6 semantic rules, so much of this is lexical lookup. A simple character rule, `all_lowercase`, is close to chance when learned from examples but reaches 0.98 when the rule is given.
- Classifying works better than stating the rule, but the classification is easier tested. Articulation has more confounding factors, like potential judges bias (which we tried to mitigate).
- When removing confounds, results become clearer. The articulation for `word_count_geq_8` moves to the new surface feature ("two adjectives", never "8 words"), and a simple character counter (0.942) beats the model (0.772). For `second_word_capitalized`, accuracy from examples drops to near chance (0.97 to 0.56, and 0.81 to 0.58) once proper-noun-ness is separated from capitalisation, even though the model still applies the rule almost perfectly when it is told it (rule-given, to 1.00).
- Grades were checked by a separate judge (gpt-4o, exact agreement 0.81 overall, weaker on a few important rules) and by compiled stated-rule checks.

## Repository layout

```
src/icl_articulation/    library: async client, prompts, multiple-choice/free-form grading,
                         faithfulness probes, stats, run logging
data/<rule>/             one directory per rule: items.jsonl + confound_report.json
                         (30 rules across surface, syntactic, semantic, positional,
                         numeric and deliberately-hard categories)
scripts/                 run_step*.py experiment runners plus local analysis
                         scripts and reproduce_no_api.py
results/                 per-run output dirs (config.json with seeds + template
                         hashes + per-run cost, responses.jsonl, metrics.json)
results/figures/         the report figures (+ captions), the deconfound +
                         confound/grade audits, and the consolidated metrics_table.csv
```

## Reproduce the results without API calls

The committed report files can be regenerated from the existing data and raw logs, with no paid API calls:

```bash
# Use a Python >=3.12 interpreter. In this workspace:
PY=.venv/bin/python

$PY -m pip install -e .

# Rebuild the corrected analyses, deconfound files, local Qwen metrics and figures.
$PY scripts/reproduce_no_api.py

# Optional: include tests.
$PY scripts/reproduce_no_api.py --include-tests
```

The no-API rebuild runs these commands in order:

```bash
$PY scripts/analyze_step3.py
$PY scripts/make_figures.py --tables-only
$PY scripts/analyze_confounds.py
$PY scripts/analyze_deconfound.py
$PY scripts/analyze_swc_deconfound.py
$PY scripts/analyze_confound_grade.py
$PY scripts/analyze_local_qwen.py
$PY scripts/make_figures.py
```

Report numbers come from the raw `results/*/responses.jsonl` logs and a small set of regenerated analysis files:

- `results/figures/metrics_table.csv|json`
- `results/figures/step3_corrected.json`
- `results/figures/confound_audit.json`
- `results/figures/confound_grade.json`
- `results/figures/deconfound_wc8.json`
- `results/figures/deconfound_swc.json`
- `results/figures/judge_agreement.json`
- `results/local-qwen-step1-metrics.json` if the Qwen limitation is retained

## Re-run paid experiments

```bash
# 1. install the package (Python >= 3.12)
PY=.venv/bin/python
$PY -m pip install -e .

# 2. set the API key (also read from a local .env)
export OPENAI_API_KEY=sk-...

# 3. Step 1 — in-context learnability (each run logs config, seed and cost to results/)
$PY scripts/run_step1.py --mode full --model gpt-4.1 --all

#    told-the-rule baseline: build the public rules-file, then run it.
#    (rule_given REQUIRES --rules-file; rule_texts.json is the canonical rule strings projected from the committed data/spec_extract.json)
$PY scripts/make_rules_file.py
$PY scripts/run_step1.py --mode rule_given --model gpt-4.1 --all --rules-file data/rule_texts.json

#    to avoid overfitting the selection, pick the strong rules from the full sweep
#    (held-out accuracy >= 0.85 for at least one model), then re-test only those on the confirmation split. (do not use --all here; that re-runs all 30 rules.)
$PY scripts/select_survivors.py                       # -> results/figures/survivors.json
$PY scripts/run_step1.py --mode confirmation --model gpt-4.1 \
    --rules "$($PY scripts/select_survivors.py --print-rules)"

# 4. Steps 2 and 3 — articulation and faithfulness (on the selected rules)
$PY scripts/run_step2_mc.py           --model gpt-4.1   # recognition (8-way multiple-choice)
$PY scripts/run_step2_freeform.py     --model gpt-4.1   # production (free-form, LLM-graded)
$PY scripts/run_step3_faithfulness.py --model gpt-4.1   # counterfactual faithfulness test

# (repeat steps 3-4 with --model gpt-4.1-mini for the second subject)

# 5. analysis + figures (all local, no API calls)
$PY scripts/analyze_step3.py       # corrected step-3 -> results/figures/step3_corrected.json
$PY scripts/make_figures.py --tables-only # metrics_table for dependent files
$PY scripts/analyze_confounds.py   # compiled predicates + confound re-audit -> confound_audit.json
$PY scripts/analyze_deconfound.py
$PY scripts/analyze_swc_deconfound.py
$PY scripts/analyze_confound_grade.py
$PY scripts/analyze_local_qwen.py
$PY scripts/make_figures.py        # consolidate metrics + render every figure
```

