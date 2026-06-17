#!/usr/bin/env python
"""Analyze the compiled-predicate results — no API.

Reads results/compiled_predicates/compiled.jsonl and
computes the PRE-REGISTERED metrics:
  - per (coder, rule): accuracy distribution over the articulations that compiled
    (min/median/max), inter-articulation spread (max-min), parse-fail + exec-fail
    counts;
  - anchor comparison vs the 4 hand-written predicate accuracies (within 0.05 of the
    compiled-predicate median for >=3/4 -> VALIDATES);
  - structural vs semantic split (median structural vs median semantic);
  - cross-coder agreement (gpt-4.1 vs claude-opus-4-8 per-rule medians).
Writes results/figures/compiled_predicates.json and prints a summary + verdicts.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results" / "compiled_predicates" / "compiled.jsonl"
COST = REPO / "results" / "compiled_predicates" / "_cost.json"
OUT = REPO / "results" / "figures" / "compiled_predicates.json"

ANCHORS = {"word_count_geq_8": 0.977, "second_word_capitalized": 0.600,
           "physically_impossible": 0.605, "food_topic": 0.902}
STRUCTURAL = {"word_count_geq_8", "second_word_capitalized", "contains_digit",
              "passive_voice", "repeated_content_word"}
SEMANTIC = {"physically_impossible", "food_topic", "mentions_animal",
            "positive_sentiment", "contains_first_name", "mentions_color"}


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    # group by coder, rule
    acc = defaultdict(lambda: defaultdict(list))      # coder -> rule -> [accuracies of compiled]
    fails = defaultdict(lambda: defaultdict(lambda: {"parse_fail": 0, "exec_fail": 0, "ok": 0, "item_errors": 0}))
    for r in rows:
        coder, rule, res = r["coder"], r["rule"], r["result"]
        f = fails[coder][rule]
        if res.get("ok"):
            f["ok"] += 1
            f["item_errors"] += res.get("n_item_errors", 0)
            acc[coder][rule].append(res["accuracy"])
        else:
            reason = res.get("reason", "")
            if reason in ("no_code_in_response", "syntax_error", "ast_reject"):
                f["parse_fail"] += 1
            else:
                f["exec_fail"] += 1

    per_coder = {}
    for coder in sorted(fails):
        rules = {}
        for rule in sorted(fails[coder]):
            a = acc[coder][rule]
            f = fails[coder][rule]
            rules[rule] = {
                "n_compiled": len(a), "parse_fail": f["parse_fail"], "exec_fail": f["exec_fail"],
                "item_errors": f["item_errors"],
                "acc_min": round(min(a), 4) if a else None,
                "acc_median": round(statistics.median(a), 4) if a else None,
                "acc_max": round(max(a), 4) if a else None,
                "spread": round(max(a) - min(a), 4) if a else None,
                "anchor": ANCHORS.get(rule),
                "bucket": "structural" if rule in STRUCTURAL else "semantic",
            }
        per_coder[coder] = rules

    # verdicts (primary coder = gpt-4.1)
    def medians(coder, bucket):
        return [per_coder[coder][r]["acc_median"] for r in per_coder.get(coder, {})
                if r in bucket and per_coder[coder][r]["acc_median"] is not None]

    verdicts = {}
    for coder in per_coder:
        anchors_ok = 0
        anchor_detail = {}
        for rule, anchor in ANCHORS.items():
            med = per_coder[coder].get(rule, {}).get("acc_median")
            within = (med is not None and abs(med - anchor) <= 0.05)
            anchor_detail[rule] = {"anchor": anchor, "model_median": med, "within_0.05": within}
            anchors_ok += int(within)
        total = sum(per_coder[coder][r]["n_compiled"] + per_coder[coder][r]["parse_fail"] + per_coder[coder][r]["exec_fail"]
                    for r in per_coder[coder])
        compiled = sum(per_coder[coder][r]["n_compiled"] for r in per_coder[coder])
        struct_med = statistics.median(medians(coder, STRUCTURAL)) if medians(coder, STRUCTURAL) else None
        sem_med = statistics.median(medians(coder, SEMANTIC)) if medians(coder, SEMANTIC) else None
        verdicts[coder] = {
            "anchors_within_0.05": anchors_ok, "anchor_detail": anchor_detail,
            "compiled_majority": compiled > total / 2, "compiled": compiled, "total": total,
            "VALIDATES": anchors_ok >= 3 and compiled > total / 2,
            "structural_median": round(struct_med, 4) if struct_med is not None else None,
            "semantic_median": round(sem_med, 4) if sem_med is not None else None,
            "REVEALS_SPLIT": (struct_med is not None and sem_med is not None and struct_med > sem_med),
        }

    # cross-coder agreement (per-rule median abs diff)
    coders = sorted(per_coder)
    cross = {}
    if len(coders) == 2:
        c1, c2 = coders
        diffs = []
        for rule in per_coder[c1]:
            m1, m2 = per_coder[c1][rule]["acc_median"], per_coder[c2].get(rule, {}).get("acc_median")
            if m1 is not None and m2 is not None:
                diffs.append(abs(m1 - m2))
        cross = {"coders": [c1, c2], "n_rules": len(diffs),
                 "mean_abs_median_diff": round(statistics.mean(diffs), 4) if diffs else None,
                 "max_abs_median_diff": round(max(diffs), 4) if diffs else None}

    result = {"_note": "Each stated rule articulation is compiled to an executable predicate and run on the data (sandboxed). "
                       "Pre-registered design. Anchors are hand-written predicate accuracies "
                       "(swc=0.600 predicate, NOT the 1.000 vocab lookup).",
              "per_coder": per_coder, "verdicts": verdicts, "cross_coder": cross}
    if COST.is_file():
        result["cost"] = json.loads(COST.read_text())
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    for coder in per_coder:
        v = verdicts[coder]
        print(f"\n=== {coder} (compiles gpt-4.1's articulations) ===")
        print(f"  compiled {v['compiled']}/{v['total']} | anchors within 0.05 of median: {v['anchors_within_0.05']}/4 "
              f"| VALIDATES={v['VALIDATES']} | struct med {v['structural_median']} vs sem med {v['semantic_median']} -> REVEALS_SPLIT={v['REVEALS_SPLIT']}")
        for rule in sorted(per_coder[coder], key=lambda r: (per_coder[coder][r]["bucket"], r)):
            b = per_coder[coder][rule]
            anch = f" anchor={b['anchor']}" if b["anchor"] is not None else ""
            print(f"    [{b['bucket'][:4]}] {rule:26s} med={b['acc_median']} [{b['acc_min']},{b['acc_max']}] spread={b['spread']} "
                  f"compiled={b['n_compiled']}/6 pf={b['parse_fail']} xf={b['exec_fail']}{anch}")
    if cross:
        print(f"\ncross-coder per-rule median |diff|: mean={cross['mean_abs_median_diff']} max={cross['max_abs_median_diff']} (n={cross['n_rules']})")
    if "cost" in result:
        print(f"\ncompiled-predicate cost: ${result['cost'].get('total_usd')}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
