#!/usr/bin/env python
"""Analyze the cross-family (Claude) results — no API.

Reads results/cross_family/*.jsonl and computes the
PRE-REGISTERED metrics:

  B1  per (rule, mode) pooled held-out accuracy, parse-failure rate, per-context
      spread; v1->v2 deltas for wc8 and swc; the predeclared REPLICATES /
      DOES-NOT / INCONCLUSIVE verdict against the frozen thresholds.
  B2  per-rule agreement vs the OpenAI-consensus label + parse-failure; for
      physically_impossible also per-impossibility_type agreement and
      within-minimal-pair accuracy (frame_base_id from data/).

Writes results/figures/cross_family.json and prints a summary.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CF = REPO / "results" / "cross_family"
DATA = REPO / "data"
OUT = REPO / "results" / "figures" / "cross_family.json"


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def acc_block(rows: list[dict], pred_key: str, truth_key: str) -> dict:
    parsed = [r for r in rows if r.get(pred_key) is not None]
    n = len(rows)
    np_ = len(parsed)
    correct = sum(1 for r in parsed if bool(r[pred_key]) == bool(r[truth_key]))
    by_ctx = defaultdict(lambda: [0, 0])
    for r in parsed:
        c = by_ctx[r.get("context_index")]
        c[1] += 1
        c[0] += int(bool(r[pred_key]) == bool(r[truth_key]))
    per_ctx = {str(k): round(v[0] / v[1], 4) for k, v in sorted(by_ctx.items()) if v[1]}
    return {
        "n": n, "n_parsed": np_, "parse_fail": n - np_,
        "parse_fail_rate": round((n - np_) / n, 4) if n else None,
        "accuracy": round(correct / np_, 4) if np_ else None,
        "per_context_accuracy": per_ctx,
    }


def b1() -> dict:
    out = {}
    for mode in ("think_off", "think_on"):
        rows = load(CF / f"claude-step1-{mode}.jsonl")
        by_rule = defaultdict(list)
        for r in rows:
            by_rule[r["rule"]].append(r)
        out[mode] = {rule: acc_block(rs, "predicted", "true_label") for rule, rs in sorted(by_rule.items())}
    # v1->v2 deltas + verdicts (think_off is the comparison arm)
    verdicts = {}
    off = out.get("think_off", {})
    for base, v2 in (("word_count_geq_8", "word_count_geq_8_v2"),
                     ("second_word_capitalized", "second_word_capitalized_v2")):
        a1 = off.get(base, {}).get("accuracy")
        a2 = off.get(v2, {}).get("accuracy")
        if a1 is None or a2 is None:
            verdicts[base] = {"v1": a1, "v2": a2, "verdict": "MISSING"}
            continue
        delta = round(a1 - a2, 4)
        if a1 < 0.85:
            verdict = "INCONCLUSIVE (v1<0.85: too weak / regime artefact)"
        elif base == "word_count_geq_8":
            verdict = "REPLICATES" if (delta >= 0.10 and a2 <= 0.80) else "DOES-NOT-REPLICATE"
        else:  # swc
            verdict = "REPLICATES" if a2 <= 0.65 else "DOES-NOT-REPLICATE"
        verdicts[base] = {"v1": a1, "v2": a2, "delta_v1_minus_v2": delta, "verdict": verdict}
    return {"per_dataset": out, "deconfound_verdicts": verdicts}


def b1_nocot_robustness() -> dict:
    """Reasoning-leak audit on the think_off arm: a 'clean' answer is just the label
    token; a 'reasoned' answer carries extra text despite thinking-disabled. Reports
    counts + accuracy per subset so the dissociation can be checked free of CoT."""
    rows = load(CF / "claude-step1-think_off.jsonl")
    rows = [r for r in rows if r.get("predicted") is not None]
    by_rule = defaultdict(list)
    for r in rows:
        by_rule[r["rule"]].append(r)
    out = {}

    def acc(s):
        return round(sum(int(bool(x["predicted"]) == bool(x["true_label"])) for x in s) / len(s), 4) if s else None

    for rule, rs in sorted(by_rule.items()):
        clean = [r for r in rs if (r.get("raw") or "").strip().lower() in ("true", "false")]
        reasoned = [r for r in rs if r not in clean]
        out[rule] = {
            "n": len(rs), "n_reasoned": len(reasoned),
            "reasoned_rate": round(len(reasoned) / len(rs), 4) if rs else None,
            "accuracy_overall": acc(rs), "accuracy_clean_only": acc(clean),
            "accuracy_reasoned_only": acc(reasoned),
        }
    return out


def b2() -> dict:
    rows = load(CF / "claude-validate.jsonl")
    # join PI rows with data for frame_base_id (minimal-pair grouping)
    pi_meta = {}
    for it in load(DATA / "physically_impossible" / "items.jsonl"):
        pi_meta[it["item_id"]] = it.get("slots_meta", {})
    out = {}
    by_rule = defaultdict(list)
    for r in rows:
        by_rule[r["rule"]].append(r)
    for rule, rs in sorted(by_rule.items()):
        parsed = [r for r in rs if r.get("claude_says_true") is not None]
        n, npar = len(rs), len(parsed)
        agree = sum(1 for r in parsed if bool(r["claude_says_true"]) == bool(r["true_label"]))
        block = {
            "n": n, "n_parsed": npar, "parse_fail_rate": round((n - npar) / n, 4) if n else None,
            "agreement": round(agree / npar, 4) if npar else None,
        }
        a = block["agreement"]
        block["verdict"] = ("not OpenAI-idiosyncratic (>=0.90)" if a and a >= 0.90 else
                            "family-dependent (<0.80)" if a is not None and a < 0.80 else
                            "partial (0.80-0.90)")
        if rule == "physically_impossible":
            # per impossibility_type
            by_type = defaultdict(lambda: [0, 0])
            for r in parsed:
                t = r.get("impossibility_type") or "unknown"
                by_type[t][1] += 1
                by_type[t][0] += int(bool(r["claude_says_true"]) == bool(r["true_label"]))
            block["by_impossibility_type"] = {
                k: {"agree": v[0], "n": v[1], "rate": round(v[0] / v[1], 4)}
                for k, v in sorted(by_type.items())
            }
            # within-minimal-pair: group by frame_base_id, require both twins parsed & correct
            by_frame = defaultdict(list)
            for r in parsed:
                fb = pi_meta.get(r["item_id"], {}).get("frame_base_id")
                if fb:
                    by_frame[fb].append(r)
            pairs = [g for g in by_frame.values()
                     if {bool(x["true_label"]) for x in g} == {True, False}]
            both_correct = sum(1 for g in pairs
                               if all(bool(x["claude_says_true"]) == bool(x["true_label"]) for x in g))
            block["minimal_pairs"] = {
                "n_frames_with_both_labels": len(pairs),
                "both_twins_correct": both_correct,
                "rate": round(both_correct / len(pairs), 4) if pairs else None,
            }
        out[rule] = block
    return out


def main() -> None:
    result = {"_note": "Cross-family check. Public design note: results/cross_family/NOTES.md.",
              "B1_deconfound_generality": b1(),
              "B1_nocot_robustness": b1_nocot_robustness(),
              "B2_third_validator": b2()}
    # attach cost if present
    costs = {}
    for cp in sorted(CF.glob("_cost_*.json")):
        costs[cp.stem.replace("_cost_", "")] = json.loads(cp.read_text())
    if costs:
        result["cost"] = costs
        result["cost"]["total_usd"] = round(sum(c.get("usd", 0) for c in costs.values()), 4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    print("=== B1 deconfound generality (claude-opus-4-8, think_off) ===")
    for rule, b in result["B1_deconfound_generality"]["per_dataset"].get("think_off", {}).items():
        print(f"  {rule:30s} acc={b['accuracy']} (n_parsed={b['n_parsed']}/{b['n']}, parse_fail={b['parse_fail']})  ctx={b['per_context_accuracy']}")
    print("  -- verdicts --")
    for rule, v in result["B1_deconfound_generality"]["deconfound_verdicts"].items():
        print(f"    {rule:30s} v1={v.get('v1')} v2={v.get('v2')} delta={v.get('delta_v1_minus_v2')} -> {v['verdict']}")
    print("\n=== B1 think_on confirmation arm (v1 datasets) ===")
    for rule, b in result["B1_deconfound_generality"]["per_dataset"].get("think_on", {}).items():
        print(f"  {rule:30s} acc={b['accuracy']} (n_parsed={b['n_parsed']}/{b['n']})")
    print("\n=== B1 no-CoT robustness (think_off: clean single-token vs reasoned) ===")
    for rule, b in result["B1_nocot_robustness"].items():
        print(f"  {rule:30s} reasoned={b['n_reasoned']}/{b['n']} ({b['reasoned_rate']}) | acc overall={b['accuracy_overall']} clean={b['accuracy_clean_only']} reasoned={b['accuracy_reasoned_only']}")
    print("\n=== B2 third validator (Claude vs OpenAI-consensus label) ===")
    for rule, b in result["B2_third_validator"].items():
        print(f"  {rule:30s} agreement={b['agreement']} (n_parsed={b['n_parsed']}/{b['n']}, parse_fail_rate={b['parse_fail_rate']}) -> {b['verdict']}")
        if rule == "physically_impossible":
            print("      by_impossibility_type:")
            for k, v in b["by_impossibility_type"].items():
                print(f"        {k:22s} {v['rate']} ({v['agree']}/{v['n']})")
            mp = b["minimal_pairs"]
            print(f"      minimal-pairs both-correct: {mp['rate']} ({mp['both_twins_correct']}/{mp['n_frames_with_both_labels']})")
    if costs:
        print(f"\nTOTAL cross-family API cost: ${result['cost']['total_usd']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
