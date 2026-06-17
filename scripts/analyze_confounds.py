#!/usr/bin/env python
"""Confound + representative compiled-predicate re-audit (no API). Writes
results/figures/confound_audit.json.

Two honest findings the original audit understated:

 (1) REPRESENTATIVE stated-rule predicate accuracy. The LLM judge graded several
     free-form articulations 0 or 1 for the intended rule, but a compiled
     predicate for a representative stated rule is far more accurate
     in-distribution than the grade suggests — because the dataset taught a
     confounded feature the articulation actually names. E.g.
     word_count_geq_8's "an adverb / prepositional phrase after the verb"
     (judge grade 0) scores ~0.98 on the original data.

 (2) DATASET CONFOUNDS the original report claimed were audited away. Per-token
     one-sided rate gates were hard only for the 3 LLM rules; for the 27
     programmatic rules token skew was reported but not gated, so e.g.
     word_count_geq_8 carries trailing 'by' 84 True / 0 False yet
     overall_pass=true. A trivial naive-Bayes bag-of-words baseline and even a
     char-length threshold recover most of the model's own accuracy on the
     headline rules.

This recomputes everything from the committed data/<rule>/items.jsonl (the
per-rule confound_report.json files keep the ORIGINAL audit as provenance).

Run:  python scripts/analyze_confounds.py
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.datagen.confound import tokens_missing_from_a_class
from icl_articulation.step3_probes import ARTICULATIONS, ARTICULATION_PREDICATES

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results"
OUT_PATH = RESULTS / "figures" / "confound_audit.json"
METRICS = RESULTS / "figures" / "metrics_table.csv"

TARGET_RULES = ["word_count_geq_8", "second_word_capitalized",
                "physically_impossible", "food_topic"]


def load_items(rule: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / rule / "items.jsonl").open()]


def _toks(text: str) -> set[str]:
    return {w.strip(".,!?;:").lower() for w in text.split()}


def naive_bayes_heldout(rule: str) -> tuple[float | None, int]:
    """Multinomial NB bag-of-words: train on the few-shot pool, test on held-out
    (the same in-distribution split the model saw). Laplace-smoothed."""
    rows = load_items(rule)
    train = [r for r in rows if r["split"] in ("few_shot_pool", "few_shot")]
    test = [r for r in rows if r["split"] == "held_out"]
    if not train or not test:
        return None, 0
    ct = {True: collections.Counter(), False: collections.Counter()}
    nc = {True: 0, False: 0}
    vocab: set[str] = set()
    for r in train:
        lab = bool(r["label"])
        nc[lab] += 1
        for w in _toks(r["text"]):
            ct[lab][w] += 1
            vocab.add(w)
    v = len(vocab)
    tot = {lab: sum(ct[lab].values()) for lab in (True, False)}
    n_train = sum(nc.values())

    def logp(text: str, lab: bool) -> float:
        lp = math.log(nc[lab] / n_train) if nc[lab] else -1e9
        for w in _toks(text):
            lp += math.log((ct[lab][w] + 1) / (tot[lab] + v))
        return lp

    correct = sum(1 for r in test
                  if (logp(r["text"], True) > logp(r["text"], False)) == bool(r["label"]))
    return correct / len(test), len(test)


def best_single_token(rule: str) -> dict:
    """The single token whose presence best predicts the label (in-distribution)."""
    rows = load_items(rule)
    n = len(rows)
    present_true: collections.Counter = collections.Counter()
    present: collections.Counter = collections.Counter()
    for r in rows:
        for w in _toks(r["text"]):
            present[w] += 1
            if r["label"]:
                present_true[w] += 1
    n_true = sum(1 for r in rows if r["label"])
    best = {"token": None, "accuracy": 0.0}
    for w, cnt in present.items():
        if cnt < 0.05 * n:
            continue
        # predict True iff token present; accuracy over the dataset
        tp = present_true[w]
        fp = cnt - tp
        # present->True, absent->False
        acc = (tp + (n - n_true - fp)) / n
        acc = max(acc, 1 - acc)
        if acc > best["accuracy"]:
            best = {"token": w, "accuracy": acc, "true_count": tp, "false_count": fp}
    return best


def char_length_best_threshold(rule: str) -> float:
    rows = load_items(rule)
    n = len(rows)
    lens = sorted({len(r["text"]) for r in rows})
    best = 0.0
    for th in lens:
        acc = sum(1 for r in rows if (len(r["text"]) < th) == bool(r["label"])) / n
        best = max(best, acc, 1 - acc)
    return best


def extensional_grade(rule: str) -> dict:
    """In-distribution accuracy of a representative compiled stated-rule
    predicate on the rule's own dataset. For second_word_capitalized the shallow
    30-name predicate undercounts, so we also report the accuracy of 'the second
    word is a proper noun' using the dataset's own proper-noun vocabulary
    (slots_meta.proper), which are independently-recognizable names/cities/months."""
    rows = load_items(rule)
    n = len(rows)
    pred = ARTICULATION_PREDICATES[rule]
    correct = sum(1 for r in rows if pred(r["text"]) == bool(r["label"]))
    out = {
        "articulation_gpt_4_1": ARTICULATIONS[rule]["gpt-4.1"],
        "stated_rule_predicate_accuracy": correct / n,
        "n": n,
    }
    if rule == "second_word_capitalized":
        proper_vocab = {(r["slots_meta"].get("proper") or "").lower() for r in rows} - {""}

        def second(r: dict) -> str:
            t = r["text"].split()
            return t[1].lower() if len(t) > 1 else ""

        acc = sum(1 for r in rows if (second(r) in proper_vocab) == bool(r["label"])) / n
        out["proper_noun_vocab_accuracy"] = acc
        out["proper_noun_vocab_size"] = len(proper_vocab)
        out["note"] = ("shallow 30-name predicate undercounts; with the dataset's "
                       "full proper-noun vocabulary the 'second word is a proper "
                       "noun' reading scores 1.0 — equivalent to the "
                       "true capitalization rule in-distribution")
    return out


def mentions_animal_main_noun() -> dict:
    """How often is the animal the sentence's grammatical SUBJECT (head of the
    opening noun phrase) vs an object/oblique, in the True items? Stated
    definition: subject iff the generator frame opens 'The/A/An [adj] {X}'. The
    point: the model's 'the main noun refers to an animal' articulation is
    imperfect — the animal is the main noun in fewer than half the True items —
    yet it classifies at 0.99."""
    import re
    rows = load_items("mentions_animal")
    trues = [r for r in rows if r["label"]]
    subj_re = re.compile(r"^\s*(the|a|an)\s+(\w+\s+){0,2}\{X\}(\s|$)", re.I)
    subj = sum(1 for r in trues if subj_re.match(r["slots_meta"].get("frame", "")))
    n = len(trues)
    return {
        "definition": "animal is the grammatical subject iff its frame opens 'The/A/An [adj] {X}'",
        "n_true": n,
        "animal_is_subject": subj,
        "animal_is_subject_frac": subj / n if n else None,
        "animal_is_object_or_oblique_frac": (n - subj) / n if n else None,
    }


def swc_vocab_disjointness() -> dict:
    rows = load_items("second_word_capitalized")
    true_sw, false_sw = set(), set()
    for r in rows:
        t = r["text"].split()
        if len(t) > 1:
            (true_sw if r["label"] else false_sw).add(t[1].lower())
    return {
        "true_second_words_distinct": len(true_sw),
        "false_second_words_distinct": len(false_sw),
        "overlap": len(true_sw & false_sw),
        "false_second_words": sorted(false_sw),
    }


def judge_grades() -> dict[str, float]:
    """gpt-4.1 free-form median direct grade per rule, from metrics_table.csv."""
    import csv
    out: dict[str, float] = {}
    if not METRICS.is_file():
        return out
    for row in csv.DictReader(METRICS.open()):
        if row["model"] == "gpt-4.1" and row["freeform_median_direct"]:
            out[row["rule_id"]] = float(row["freeform_median_direct"])
    return out


def model_heldout_acc() -> dict[str, float]:
    import csv
    out: dict[str, float] = {}
    if not METRICS.is_file():
        return out
    for row in csv.DictReader(METRICS.open()):
        if row["model"] == "gpt-4.1" and row["step1_heldout_acc"]:
            out[row["rule_id"]] = float(row["step1_heldout_acc"])
    return out


def main() -> int:
    grades = judge_grades()
    heldout = model_heldout_acc()

    extensional = {}
    baselines = {}
    for rule in TARGET_RULES:
        eg = extensional_grade(rule)
        eg["judge_grade_median_direct_gpt_4_1"] = grades.get(rule)
        extensional[rule] = eg
        nb, nb_n = naive_bayes_heldout(rule)
        baselines[rule] = {
            "model_heldout_acc_gpt_4_1": heldout.get(rule),
            "naive_bayes_bow_heldout_acc": nb,
            "naive_bayes_n": nb_n,
            "best_single_token": best_single_token(rule),
            "char_length_best_threshold_acc": char_length_best_threshold(rule),
        }

    # one-sided high-frequency tokens across the 30 BASE rules (honest cross-rule
    # audit). The *_v2 deconfound rebuilds are excluded: they are corrected
    # versions of two existing rules, not additional rules, and the reported
    # "16 of 30" denominator is over the base set.
    one_sided = {}
    n_base_rules = 0
    for d in sorted(DATA.iterdir()):
        if not (d / "items.jsonl").is_file():
            continue
        if d.name.endswith("_v2"):
            continue
        n_base_rules += 1
        items = load_items(d.name)
        flagged = tokens_missing_from_a_class(items)
        if flagged:
            one_sided[d.name] = {
                "n_flagged": len(flagged),
                "tokens": flagged[:8],
            }

    out = {
        "_note": "Recomputed from committed data/<rule>/items.jsonl; the per-rule "
                 "confound_report.json files preserve the ORIGINAL audit.",
        "extensional_articulation_grades": extensional,
        "confound_baselines": baselines,
        "second_word_capitalized_vocab": swc_vocab_disjointness(),
        "mentions_animal_main_noun": mentions_animal_main_noun(),
        "one_sided_high_freq_tokens_by_rule": one_sided,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # readable summary
    print("=== extensional articulation accuracy vs judge grade (gpt-4.1) ===")
    for rule, eg in extensional.items():
        acc = eg.get("proper_noun_vocab_accuracy", eg["stated_rule_predicate_accuracy"])
        print(f"  {rule:26s} stated-rule extensional acc {acc:.3f}  "
              f"judge grade {eg['judge_grade_median_direct_gpt_4_1']}")
    print("\n=== confound baselines (headline rules) ===")
    for rule, b in baselines.items():
        nb = b["naive_bayes_bow_heldout_acc"]
        print(f"  {rule:26s} model {b['model_heldout_acc_gpt_4_1']:.3f} | "
              f"naive-Bayes BoW {nb:.3f} | char-len {b['char_length_best_threshold_acc']:.3f} | "
              f"best token '{b['best_single_token']['token']}' {b['best_single_token']['accuracy']:.3f}")
    print(f"\n=== one-sided high-freq tokens: {len(one_sided)}/{n_base_rules} base rules flagged ===")
    for rule in TARGET_RULES:
        if rule in one_sided:
            toks = ", ".join(f"{t['token']}({t['true_count']}/{t['false_count']})"
                             for t in one_sided[rule]["tokens"][:5])
            print(f"  {rule:26s} {one_sided[rule]['n_flagged']} flagged: {toks}")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
