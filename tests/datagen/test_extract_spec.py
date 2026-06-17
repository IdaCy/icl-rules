"""scripts/extract_spec.py tests: the committed data/spec_extract.json has the
expected shape, the equiv_keys/exemptions translate to battery keys, and NO
private prose keys leaked. The extractor is also re-run on the live spec so the
committed file can be regenerated reproducibly."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXTRACT_JSON = REPO / "data" / "spec_extract.json"
SCRIPT = REPO / "scripts" / "extract_spec.py"
SPEC = Path(os.environ.get("RULE_SPEC_PATH", REPO / "private_specs" / "rule-specs.yaml"))

# keys that must NEVER appear in the public extract (private prose)
FORBIDDEN_RULE_KEYS = frozenset(
    {"ambiguity_notes", "expected_difficulty", "distribution_guards"}
)

# POSITIVE allowlist: the complete, closed set of per-rule keys the extractor is
# permitted to emit. Every per-rule entry's keys MUST be a SUBSET of this set —
# this regression-locks the extractor's closed allowlist so a future private spec
# key cannot silently leak (the negative FORBIDDEN_RULE_KEYS list only catches
# the three prose keys we know about today).
ALLOWED_RULE_KEYS = frozenset(
    {
        "plan_number",
        "category",
        "canonical_articulation",
        "equivalence_class",
        "equiv_keys",
        "battery_exemptions",
        "length_match_exempt",
        "banned_distractors",
        "mc_distractor_seeds",
        "step3_edge_ideas",
        "generation",
    }
)

# generation sub-dict is fully closed: exactly the public mode + banks, nothing
# else (no recipe prose, no distribution_guards bleed-through).
ALLOWED_GENERATION_KEYS = frozenset({"mode", "banks"})


def _load_extract_module():
    spec = importlib.util.spec_from_file_location("extract_spec", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_extract_exists_and_shape() -> None:
    data = json.loads(EXTRACT_JSON.read_text())
    assert data["battery_pass_threshold_inclusive"] == 0.75
    assert len(data["rules"]) == 30
    for rid, entry in data["rules"].items():
        assert "canonical_articulation" in entry
        assert "equivalence_class" in entry
        assert "battery_exemptions" in entry
        assert "equiv_keys" in entry
        assert isinstance(entry["equiv_keys"], dict)
        assert set(entry) - FORBIDDEN_RULE_KEYS == set(entry)  # no forbidden keys


def test_no_private_prose_keys() -> None:
    data = json.loads(EXTRACT_JSON.read_text())
    for rid, entry in data["rules"].items():
        leaked = FORBIDDEN_RULE_KEYS & set(entry)
        assert not leaked, f"rule {rid} leaked private keys {leaked}"


def test_rule_keys_subset_of_allowlist() -> None:
    """Closed-allowlist regression lock: a future spec key cannot silently leak.

    Every per-rule entry's keys must be a SUBSET of the known-public allowlist,
    and entry['generation'].keys() must be exactly {mode, banks}."""
    data = json.loads(EXTRACT_JSON.read_text())
    for rid, entry in data["rules"].items():
        unknown = set(entry) - ALLOWED_RULE_KEYS
        assert not unknown, f"rule {rid} has non-allowlisted keys {unknown}"
        gen_keys = set(entry["generation"].keys())
        assert gen_keys == ALLOWED_GENERATION_KEYS, (
            f"rule {rid} generation keys {gen_keys} != {set(ALLOWED_GENERATION_KEYS)}"
        )


def test_battery_exemptions_are_battery_keys() -> None:
    from icl_articulation.datagen.battery import PREDICATES_BY_KEY

    data = json.loads(EXTRACT_JSON.read_text())
    imp = data["rules"]["imperative"]
    assert set(imp["battery_exemptions"]) == {"first_word_pos=verb", "first_word_pos=pronoun"}
    # every exemption + equiv_key references a real battery predicate key
    for rid, entry in data["rules"].items():
        for k in entry["battery_exemptions"]:
            assert k in PREDICATES_BY_KEY, f"{rid}: {k} not a battery key"
        for k in entry["equiv_keys"]:
            assert k in PREDICATES_BY_KEY, f"{rid}: equiv_keys key {k} not a battery key"


def test_length_match_exempt_only_word_count_rules() -> None:
    """globals.length_matching.policy exempts exactly the two word-count rules
    (plan_numbers 23 and 25) from the |mean_wc(T)-mean_wc(F)| match; every other
    rule must carry length_match_exempt == False."""
    data = json.loads(EXTRACT_JSON.read_text())
    exempt = {rid for rid, e in data["rules"].items() if e.get("length_match_exempt")}
    assert exempt == {"word_count_geq_8", "even_word_count"}, exempt
    # the flag is present (and a bool) on every rule, not just the exempt ones
    for rid, entry in data["rules"].items():
        assert isinstance(entry["length_match_exempt"], bool), rid


def test_equiv_keys_translate_correctly() -> None:
    data = json.loads(EXTRACT_JSON.read_text())
    # all_lowercase: the 'all_lowercase' predicate instantiates the lowercase member
    al = data["rules"]["all_lowercase"]["equiv_keys"]
    assert "all_lowercase" in al
    assert "the sentence is written entirely in lowercase" in al["all_lowercase"]
    # question_word_order: verb instantiates + determiner is the complement
    qwo = data["rules"]["question_word_order"]["equiv_keys"]
    assert "first_word_pos=verb" in qwo
    assert "first_word_pos=determiner" in qwo


def test_extractor_reruns_reproducibly() -> None:
    pytest.importorskip("yaml")
    if not SPEC.is_file():
        pytest.skip("private spec not present")
    import yaml

    mod = _load_extract_module()
    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    fresh = mod.extract(spec)
    committed = json.loads(EXTRACT_JSON.read_text())
    assert fresh == committed, "data/spec_extract.json is stale; rerun scripts/extract_spec.py"
