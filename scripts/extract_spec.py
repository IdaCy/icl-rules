#!/usr/bin/env python
"""Emit data/spec_extract.json — the COMMITTED public projection of the private
rule spec (not included in this repository).

The private rule spec is internal research material and is not in this
repository. The dataset-generation framework, the step-2 multiple-choice builder, and the
free-form grading rubric all need a SMALL, stable, public subset of it: the
per-rule canonical articulation, equivalence class, banned distractors, multiple-choice
distractor seeds, step-3 edge ideas, battery exemptions, and the equiv_keys
exemption mapping (translated to battery.py predicate keys), plus the structured
generation facts (mode + banks used). This script extracts ONLY those and writes
them as JSON.

It deliberately does NOT copy ambiguity_notes, expected_difficulty,
distribution_guards prose, the generation.recipe prose, the changelog, or any
other internal prose.

Run:  .venv/bin/python scripts/extract_spec.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SPEC_PATH = Path(os.environ.get("RULE_SPEC_PATH", REPO / "private_specs" / "rule-specs.yaml"))
OUT_PATH = REPO / "data" / "spec_extract.json"

# allowlist of per-rule keys to copy VERBATIM (canonical, public-facing)
RULE_VERBATIM_KEYS = (
    "canonical_articulation",
    "equivalence_class",
    "banned_distractors",
    "mc_distractor_seeds",
    "step3_edge_ideas",
)

# spec predicate phrasing (equiv_keys.predicate / battery_exemptions.predicate)
# -> battery.py Predicate.key. Only the predicates the spec actually references
# in those two blocks need a mapping; anything unmapped is reported and skipped
# (loud), so a future spec edit cannot silently drop an exemption.
SPEC_PREDICATE_TO_BATTERY_KEY = {
    "all lowercase": "all_lowercase",
    "all lowercase (complement: contains an uppercase letter somewhere)": "all_lowercase",
    "any word beyond position 1 capitalized": "nonfirst_word_capitalized",
    "contains a digit": "contains_digit",
    "contains a comma": "contains_comma",
    "count('the') >= 2": "count_the>=2",
    "word_count >= 8": "word_count>=8",
    "word_count >= 8 (complement: fewer than 8 words)": "word_count>=8",
    "first word starts with a vowel letter": "first_starts_vowel",
    "first word starts with a consonant letter": "first_starts_consonant",
    "last word ends with a vowel letter": "last_ends_vowel",
    "last word ends with a consonant letter": "last_ends_consonant",
    "first-word POS = verb": "first_word_pos=verb",
    "first-word POS = determiner": "first_word_pos=determiner",
    "first-word POS = pronoun": "first_word_pos=pronoun",
    "first-word POS = noun": "first_word_pos=noun",
    "first-word POS = adjective": "first_word_pos=adjective",
    "first-word POS = adverb": "first_word_pos=adverb",
}

# relations that grant a grade-2 / battery exemption via the equivalence class.
# 'banned_not_equivalent' and 'none' do NOT (they note a banned distractor or a
# coincidental ~50%); they are dropped from the exemption map.
EXEMPTING_RELATIONS = frozenset(
    {"instantiates", "complement", "complement_of_rule", "complement_on_data"}
)


def _length_match_exempt_plan_numbers(spec: dict) -> frozenset[int]:
    """The plan_numbers the length_matching policy exempts from the
    |mean_wc(T) - mean_wc(F)| <= tol class-conditional word-count match.

    The policy text (globals.length_matching.policy) reads "For every rule
    EXCEPT 23 and 25 (where word count is the rule)..." -- a word-count
    THRESHOLD/parity rule cannot satisfy the 0.2 match by definition, so the
    spec carves these two out. Parsed from the policy prose (rather than
    hard-coded) so a future spec edit to the exempt set flows through, and LOUD
    if the sentinel phrasing ever changes so it cannot silently drop the carve-
    out."""
    policy = spec["globals"]["length_matching"]["policy"]
    m = re.search(r"every rule EXCEPT\s+([\d,\s]+?and\s+\d+)\b", policy)
    if not m:
        raise SystemExit(
            "length_matching.policy no longer matches the 'every rule EXCEPT "
            "<ns>' sentinel; update _length_match_exempt_plan_numbers"
        )
    nums = frozenset(int(n) for n in re.findall(r"\d+", m.group(1)))
    if not nums:
        raise SystemExit("length_matching.policy named no exempt rule numbers")
    return nums


def extract(spec: dict) -> dict:
    if "rules" not in spec or "globals" not in spec:
        raise SystemExit("spec missing 'rules' or 'globals' — wrong file?")

    # equiv_keys: group by rule_id, translate predicate -> battery key, keep only
    # exempting relations, store the verbatim equiv_strings.
    equiv_rows = spec["globals"]["generic_probe_battery"]["equiv_keys"]
    per_rule_equiv: dict[str, dict[str, list[str]]] = {}
    unmapped: set[str] = set()
    for row in equiv_rows:
        pred = row["predicate"]
        rule = row["rule"]
        relation = row.get("relation")
        if relation not in EXEMPTING_RELATIONS:
            continue
        key = SPEC_PREDICATE_TO_BATTERY_KEY.get(pred)
        if key is None:
            unmapped.add(pred)
            continue
        strings = list(row.get("equiv_strings", []))
        per_rule_equiv.setdefault(rule, {}).setdefault(key, [])
        for s in strings:
            if s not in per_rule_equiv[rule][key]:
                per_rule_equiv[rule][key].append(s)
    if unmapped:
        raise SystemExit(
            "equiv_keys references predicates with no battery-key mapping "
            f"(add them to SPEC_PREDICATE_TO_BATTERY_KEY): {sorted(unmapped)}"
        )

    # plan_numbers the length_matching policy exempts from the word-count match
    # (the word-count rules, where matching means is unsatisfiable by definition).
    lm_exempt_plans = _length_match_exempt_plan_numbers(spec)

    rules_out: dict[str, dict] = {}
    for r in spec["rules"]:
        rid = r["id"]
        entry: dict = {"plan_number": r.get("plan_number"), "category": r.get("category")}
        # length-match exemption (globals.length_matching.policy "EXCEPT 23/25"):
        # True for the word-count rules, whose class-conditional word-count means
        # MUST differ by construction, so Gate D skips the |mean_T-mean_F| assert
        # for them (the diff is still computed and reported in the audit).
        entry["length_match_exempt"] = r.get("plan_number") in lm_exempt_plans
        for k in RULE_VERBATIM_KEYS:
            if k in r:
                entry[k] = r[k]
        # battery_exemptions: list of battery predicate keys (translated). The
        # spec stores per-exemption arithmetic prose; the public extract keeps
        # only the predicate KEY (the machine-usable part) — the arithmetic is
        # internal justification and stays private.
        exemptions: list[str] = []
        for be in r.get("battery_exemptions", []) or []:
            pred = be["predicate"]
            key = SPEC_PREDICATE_TO_BATTERY_KEY.get(pred)
            if key is None:
                raise SystemExit(
                    f"rule {rid} battery_exemptions predicate {pred!r} has no "
                    "battery-key mapping (add it to SPEC_PREDICATE_TO_BATTERY_KEY)"
                )
            exemptions.append(key)
        entry["battery_exemptions"] = exemptions
        entry["equiv_keys"] = per_rule_equiv.get(rid, {})
        # structured generation facts (NOT the recipe prose): the mode and the
        # banks the rule draws from — the public 'per-rule quota' surface a
        # generator needs to know which banks to load.
        gen = r.get("generation", {}) or {}
        entry["generation"] = {
            "mode": gen.get("mode"),
            "banks": list(gen.get("banks", []) or []),
        }
        rules_out[rid] = entry

    return {
        "_note": "Public projection of the private rule-specs.yaml; see scripts/extract_spec.py.",
        "battery_pass_threshold_inclusive": 0.75,
        "rules": rules_out,
    }


def main() -> int:
    if not SPEC_PATH.is_file():
        raise SystemExit(f"private spec not found: {SPEC_PATH}")
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    extract_data = extract(spec)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(extract_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(extract_data['rules'])} rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
