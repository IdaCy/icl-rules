#!/usr/bin/env python
"""divergence / M1 — in-distribution BEHAVIORAL agreement (no API).

For each articulation-probed rule we take the model's STATED rule (c), encoded as
the compiled articulation predicate in ``step3_probes.ARTICULATION_PREDICATES``,
run it on the rule's held-out items, and compare its predictions to the MODEL'S
OWN held-out classification labels (the ``predicted`` field of the existing
Step-1 ``responses.jsonl``) — NOT the gold labels.

This measures how well the stated rule (c) reproduces the model's own
in-distribution behavior (the inferred operative rule b), at far larger n than
the hand-built Step-3 probe sets.

Important:
on in-distribution data a PROXY (c) can match the model's behavior without being
counterfactually faithful, because the proxy and the true operative rule usually
agree there. M1 is a necessary, larger-n complement to M2 — NOT proof of
faithfulness. The load-bearing test is M2 (divergence inputs).

No API. Reads only existing results/. Writes results/figures/divergence_behavioral_agreement.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from icl_articulation.rule_ids import canonical_rule_id
from icl_articulation.stats import wilson_ci
from icl_articulation.step3_probes import (
    ARTICULATION_PREDICATES,
    TARGET_RULES,
    articulation_for,
    articulation_predict,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"


def _rel_to_root(p: Any) -> str:
    """Repo-relative source path so the sidecar is portable across clones."""
    s = str(p)
    try:
        return str(Path(s).relative_to(ROOT))
    except ValueError:
        return s

# Held-out Step-1 sweeps that carry the model's OWN classification labels for the
# four articulation-probed target rules. The two OpenAI subjects are the frozen
# primary subjects from the planned run; Qwen is an optional open-model robustness
# read (the 7B is weak across the board — trust relative patterns, not levels).
RUNS: dict[str, dict[str, Any]] = {
    "gpt-4.1": {
        "kind": "sweep_dir",
        "path": RESULTS / "step1-full-gpt-4.1-20260611T000748Z",
        "subject_class": "primary",
    },
    "gpt-4.1-mini": {
        "kind": "sweep_dir",
        "path": RESULTS / "step1-full-gpt-4.1-mini-20260611T001640Z",
        "subject_class": "primary",
    },
    "qwen2.5-7b-instruct": {
        "kind": "per_rule_jsonl",
        "path_template": str(RESULTS / "local-qwen-step1-{rule}.jsonl"),
        "subject_class": "robustness",
    },
}


def _iter_rule_rows(run: dict[str, Any], rule_id: str):
    """Yield {text, predicted, true_label} rows for one rule from a run, parsing
    only held-out rows that the model actually classified (predicted in
    {True, False}); None/unparsed predictions are skipped (no behaviour signal)."""
    if run["kind"] == "sweep_dir":
        path = run["path"] / "responses.jsonl"
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("rule_id") != rule_id:
                    continue
                yield r
    elif run["kind"] == "per_rule_jsonl":
        path = Path(run["path_template"].format(rule=rule_id))
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    else:  # pragma: no cover - config error
        raise ValueError(f"unknown run kind {run['kind']!r}")


def _agree_block(matches: int, n: int) -> dict[str, Any]:
    if n == 0:
        return {"rate": None, "agree": 0, "n": 0, "ci_low": None, "ci_high": None}
    low, high = wilson_ci(matches, n)
    return {"rate": matches / n, "agree": matches, "n": n, "ci_low": low, "ci_high": high}


def analyze_rule_model(run: dict[str, Any], model: str, rule_id: str) -> dict[str, Any] | None:
    base = canonical_rule_id(rule_id)
    if base not in ARTICULATION_PREDICATES:
        return None
    try:
        articulation = articulation_for(base, model)
    except Exception:  # noqa: BLE001 — fall back to the default-model wording
        articulation = articulation_for(base)

    n_rows = 0
    n_parsed = 0  # model predicted True/False
    c_vs_behaviour = 0  # compiled-(c) prediction == model's own label
    c_vs_gold = 0  # compiled-(c) prediction == gold label
    model_vs_gold = 0  # model's own label == gold label
    c_true = 0
    behaviour_true = 0

    for r in _iter_rule_rows(run, rule_id):
        n_rows += 1
        predicted = r.get("predicted")
        if predicted not in (True, False):
            continue
        text = r["text"]
        gold = bool(r["true_label"])
        c_pred = articulation_predict(base, text)
        n_parsed += 1
        if c_pred:
            c_true += 1
        if predicted:
            behaviour_true += 1
        if c_pred == predicted:
            c_vs_behaviour += 1
        if c_pred == gold:
            c_vs_gold += 1
        if predicted == gold:
            model_vs_gold += 1

    if n_parsed == 0:
        return None

    return {
        "rule_id": base,
        "model": model,
        "subject_class": run["subject_class"],
        "articulation_c": articulation,
        "n_heldout_rows": n_rows,
        "n_parsed_behaviour": n_parsed,
        # PRIMARY M1 metric: compiled-(c) predictions vs the model's OWN labels.
        "c_vs_model_behaviour": _agree_block(c_vs_behaviour, n_parsed),
        # context: compiled-(c) vs gold, and model vs gold (the Step-1 accuracy).
        "c_vs_gold": _agree_block(c_vs_gold, n_parsed),
        "model_vs_gold": _agree_block(model_vs_gold, n_parsed),
        # class balance sanity (a constant predicate or constant behaviour inflates
        # agreement; report both base rates so the reader can judge it).
        "c_predicted_true_frac": c_true / n_parsed,
        "behaviour_true_frac": behaviour_true / n_parsed,
    }


def main() -> None:
    results: list[dict[str, Any]] = []
    for model, run in RUNS.items():
        for rule_id in TARGET_RULES:
            row = analyze_rule_model(run, model, rule_id)
            if row is not None:
                results.append(row)

    out = {
        "_measurement": "M1",
        "_what": (
            "in-distribution behavioral agreement: compiled stated-rule (c) "
            "predictions vs the MODEL'S OWN held-out Step-1 labels (inferred "
            "operative rule b), per rule x model."
        ),
        "_caveat": (
            "IN-DISTRIBUTION ONLY. On in-distribution data a proxy (c) can match "
            "the model's behaviour without being counterfactually faithful, "
            "because the proxy and the true operative rule usually agree there. "
            "M1 is a necessary larger-n complement, NOT proof of faithfulness. "
            "The discriminating test is M2 (divergence inputs)."
        ),
        "_runs": {
            m: {
                "subject_class": r["subject_class"],
                "source": _rel_to_root(r.get("path", r.get("path_template"))),
            }
            for m, r in RUNS.items()
        },
        "_predicate_source": (
            "icl_articulation.step3_probes.ARTICULATION_PREDICATES (the model's "
            "modal DIRECT step-2 articulation encoded as a pure text predicate)"
        ),
        "results": results,
    }

    FIGDIR.mkdir(parents=True, exist_ok=True)
    dest = FIGDIR / "divergence_behavioral_agreement.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # human-readable summary to stdout
    print(f"[M1] wrote {dest}  ({len(results)} rule x model rows)\n")
    hdr = f"{'rule':<26}{'model':<22}{'class':<11}{'c~behav':>9}{'c~gold':>9}{'acc':>7}{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for row in results:
        cb = row["c_vs_model_behaviour"]["rate"]
        cg = row["c_vs_gold"]["rate"]
        mg = row["model_vs_gold"]["rate"]
        print(
            f"{row['rule_id']:<26}{row['model']:<22}{row['subject_class']:<11}"
            f"{cb:>9.3f}{cg:>9.3f}{mg:>7.3f}{row['n_parsed_behaviour']:>6}"
        )


if __name__ == "__main__":
    main()
