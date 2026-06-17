"""groundtruth.py tests: the programmatic ground-truth verifier (quality-bar #4).

The core guarantee is the MUTATION TEST: a correctly-labeled item set passes
assert_labels_correct, and flipping ONE label makes it raise GroundTruthError.
A representative predicate-backed rule of EACH category is exercised that way,
plus the validator-derived provenance path, plus the registry-completeness
check (all 30 rule ids covered, predicate or sentinel).

Everything runs offline; no nltk / network dependency (the verifier's
recompute predicates are pure string/bank functions — only the optional bank
membership predicates touch banks, which are committed, not downloaded)."""

from __future__ import annotations

import copy

import pytest

from icl_articulation.datagen.groundtruth import (
    Backing,
    GroundTruthError,
    RULE_PREDICATES,
    VALIDATED_FLAG,
    assert_labels_correct,
    label_of,
    verify_dataset,
)
from icl_articulation.datagen.schema import write_items


# --- helpers ------------------------------------------------------------------


def _item(rule_id, label, text, **meta):
    """One schema-shaped item dict (label normalized at validation time)."""
    return {
        "item_id": f"{rule_id}-{'T' if label else 'F'}-{abs(hash(text)) % 10_000}",
        "base_id": f"b{abs(hash(text)) % 10_000}",
        "rule_id": rule_id,
        "label": label,
        "text": text,
        "slots_meta": dict(meta),
        "split": "few_shot_pool",
    }


def _flip_one_and_expect_raise(rule_id, items):
    """The mutation test: flip the first item's label, assert it now raises."""
    mutated = copy.deepcopy(items)
    mutated[0]["label"] = not bool(mutated[0]["label"])
    with pytest.raises(GroundTruthError):
        assert_labels_correct(rule_id, mutated)


# --- representative predicate-backed rule per category ------------------------
# Each builds a tiny CORRECTLY-labeled set, asserts it PASSES, then flips one
# label and asserts it RAISES.


def test_surface_rule_all_lowercase_passes_then_mutation_raises():
    rid = "all_lowercase"
    items = [
        _item(rid, True, "the dog ran home quickly"),
        _item(rid, True, "7 cats slept on the warm mat"),
        _item(rid, False, "The dog ran home quickly"),
        _item(rid, False, "Cats Slept on the Mat"),
    ]
    assert_labels_correct(rid, items)
    _flip_one_and_expect_raise(rid, items)


def test_surface_rule_contains_letter_z_bank_membership_passes_then_raises():
    # rule 5: char-level (bank-INDEPENDENT) membership-class entry
    rid = "contains_letter_z"
    items = [
        _item(rid, True, "the lazy fox slept"),
        _item(rid, True, "they ate pizza at noon"),
        _item(rid, False, "the happy fox slept"),
        _item(rid, False, "they ate bread at noon"),
    ]
    assert_labels_correct(rid, items)
    _flip_one_and_expect_raise(rid, items)


def test_syntactic_rule_question_word_order_passes_then_raises():
    rid = "question_word_order"
    items = [
        _item(rid, True, "Can the dog chase the ball today"),
        _item(rid, True, "Is the soup warm enough"),
        _item(rid, False, "The dog can chase the ball today"),
        _item(rid, False, "The soup is warm enough"),
    ]
    assert_labels_correct(rid, items)
    _flip_one_and_expect_raise(rid, items)


def test_semantic_bank_membership_rule_mentions_animal_passes_then_raises():
    # rule 13: BANK_MEMBERSHIP against the live ANIMALS bank
    rid = "mentions_animal"
    items = [
        _item(rid, True, "They found a horse behind the shed"),
        _item(rid, True, "The cat appeared in the photo"),
        _item(rid, False, "They found a table behind the shed"),
        _item(rid, False, "The chair appeared in the photo"),
    ]
    assert_labels_correct(rid, items)
    _flip_one_and_expect_raise(rid, items)


def test_semantic_first_name_excludes_other_proper_nouns():
    # rule 17: a FIRST_NAMES token is True; a NONNAME_PROPER (city) is False
    rid = "contains_first_name"
    items = [
        _item(rid, True, "The letter from Anna arrived this morning"),
        _item(rid, True, "Everyone talked about David during lunch"),
        _item(rid, False, "The letter from Madrid arrived this morning"),
        _item(rid, False, "Everyone talked about Toyota during lunch"),
    ]
    assert_labels_correct(rid, items)
    _flip_one_and_expect_raise(rid, items)


def test_positional_rule_first_word_longer_than_last_passes_then_raises():
    rid = "first_word_longer_than_last"
    items = [
        _item(rid, True, "Strawberries near the cat"),
        _item(rid, True, "Architects met the fox"),
        _item(rid, False, "Cats near the strawberries"),
        _item(rid, False, "Drivers waited near the airport"),  # 7 vs 7 tie -> False
    ]
    assert_labels_correct(rid, items)
    _flip_one_and_expect_raise(rid, items)


def test_numeric_rule_exactly_two_commas_passes_then_raises():
    rid = "exactly_two_commas"
    items = [
        _item(rid, True, "apples, pears, and plums"),
        _item(rid, True, "My neighbor, a retired teacher, waters the garden"),
        _item(rid, False, "apples, pears and plums"),  # 1 comma
        _item(rid, False, "first, second, third, fourth"),  # 3 commas
    ]
    assert_labels_correct(rid, items)
    _flip_one_and_expect_raise(rid, items)


def test_hard_rule_double_letter_word_passes_then_raises():
    rid = "double_letter_word"
    items = [
        _item(rid, True, "the coffee was warm this morning"),
        _item(rid, True, "they sat near the tall wall"),
        _item(rid, False, "the window was open this morning"),  # non-adjacent repeat
        _item(rid, False, "they sat near the old fence"),
    ]
    assert_labels_correct(rid, items)
    _flip_one_and_expect_raise(rid, items)


def test_best_effort_imperative_handles_pronoun_object_mix():
    """Regression: the recipe plants pronoun/bare-plural OBJECTS in 40% of items
    in BOTH classes ('Close it before lunch'). The object pronoun must NOT be
    read as a declarative subject, or a valid True imperative would be flagged.
    Declaratives are recognized only by a SUBJECT pronoun in subject position."""
    rid = "imperative"
    items = [
        _item(rid, True, "Close it before lunch today"),  # pronoun object
        _item(rid, True, "Wash windows every single morning"),  # bare-plural object
        _item(rid, True, "Quickly close it before the meeting"),  # adverb + verb + obj
        _item(rid, False, "She closes it before lunch today"),  # subject + object pronoun
        _item(rid, False, "Usually they wash windows every morning"),  # adverb + subject
    ]
    assert_labels_correct(rid, items)
    _flip_one_and_expect_raise(rid, items)


def test_best_effort_rule_passive_voice_passes_then_raises():
    # rule 9 is BEST_EFFORT, but EXACT on the recipe's emitted surface shape
    rid = "passive_voice"
    items = [
        _item(rid, True, "The meal was cooked by the chef"),
        _item(rid, True, "The roof was repaired yesterday"),
        _item(rid, False, "The chef was cooking the meal"),  # progressive active
        _item(rid, False, "The chef cooked the meal slowly"),  # simple active
    ]
    assert_labels_correct(rid, items)
    _flip_one_and_expect_raise(rid, items)


# --- exhaustive per-category recompute coverage -------------------------------


def test_every_recomputable_rule_round_trips_a_minimal_pair():
    """For every PREDICATE / BANK / BEST_EFFORT rule, a hand-built True item and
    False item must verify, AND flipping either must raise. Catches a predicate
    whose canonical recompute silently disagrees with its own examples."""
    samples = {
        "all_lowercase": ("the dog ran home", "The dog ran home"),
        "contains_digit": ("they moved 7 boxes", "they moved seven boxes"),
        "contains_exclamation": ("the team won the final today!", "the team won the final today"),
        "title_case": ("The Dog Ran Home", "The Dog ran Home"),
        "contains_letter_z": ("the lazy dog slept", "the happy dog slept"),
        "repeated_content_word": ("the dog watched the dog run", "the dog watched the cat run"),
        "starts_with_vowel": ("Apples fell from the tree", "Bananas fell from the tree"),
        "past_tense": ("The cook cleaned the kitchen quietly", "The cook cleans the kitchen quietly"),
        "passive_voice": ("The meal was cooked by the chef", "The chef cooked the meal slowly"),
        "imperative": ("Close the window before lunch", "She closes the window before lunch"),
        "question_word_order": ("Can the dog chase the ball", "The dog can chase the ball"),
        "contains_comparative": ("The cat is taller than the dog", "The cat is small near the dog"),
        "mentions_animal": ("They found a horse near the shed", "They found a table near the shed"),
        "mentions_color": ("She bought a blue chair", "She bought a small chair"),
        "contains_first_name": ("The note from Anna arrived", "The note from Madrid arrived"),
        "first_word_longer_than_last": ("Strawberries near the cat", "Cats near the strawberries"),
        "last_word_ends_with_vowel": ("They saw the sofa", "They saw the cat"),
        "the_appears_twice": ("The dog saw the cat", "The dog saw a cat"),
        "second_word_capitalized": ("Apparently Maria forgot the keys", "Apparently nobody forgot keys"),
        "word_count_geq_8": ("one two three four five six seven eight", "one two three four five six seven"),
        "contains_number_gt_50": ("they moved 80 boxes", "they moved 40 boxes"),
        "even_word_count": ("the dog ran fast", "the dog ran"),
        "exactly_two_commas": ("apples, pears, and plums", "apples, pears and plums"),
        "first_last_same_letter": ("Dogs ran downtown", "Dogs ran home"),
        "double_letter_word": ("the coffee was warm", "the window was open"),
        "first_two_words_alphabetical": ("Apple boxes arrived", "Tall boxes arrived"),
        "all_words_longer_than_3": ("Hungry wolves chased frightened rabbits", "Hungry wolves chased two rabbits"),
    }
    recomputable = {
        rid for rid, e in RULE_PREDICATES.items() if e.recomputable
    }
    # every recomputable rule must have a sample here (and vice versa)
    assert set(samples) == recomputable, (
        f"sample set != recomputable rule set: "
        f"missing={recomputable - set(samples)} extra={set(samples) - recomputable}"
    )
    for rid, (true_text, false_text) in samples.items():
        items = [_item(rid, True, true_text), _item(rid, False, false_text)]
        assert_labels_correct(rid, items)  # must pass
        _flip_one_and_expect_raise(rid, items)  # mutation must raise


# --- validator-derived rules: provenance required, never recomputed -----------


@pytest.mark.parametrize(
    "rid", ["positive_sentiment", "food_topic", "physically_impossible"]
)
def test_validator_derived_requires_agreement_provenance(rid):
    # carries the two-validator agreement provenance -> passes (no recompute)
    ok = [
        _item(rid, True, "the food was wonderful tonight", **{VALIDATED_FLAG: True}),
        _item(rid, False, "the service was slow and rude", **{VALIDATED_FLAG: False}),
    ]
    assert_labels_correct(rid, ok)

    # missing provenance -> raises
    missing = [_item(rid, True, "the food was wonderful tonight")]
    with pytest.raises(GroundTruthError):
        assert_labels_correct(rid, missing)

    # provenance disagrees with the stored label -> raises
    disagree = [
        _item(rid, True, "the food was wonderful tonight", **{VALIDATED_FLAG: False})
    ]
    with pytest.raises(GroundTruthError):
        assert_labels_correct(rid, disagree)


def test_validator_derived_label_of_is_sentinel_not_a_predicate():
    for rid in ("positive_sentiment", "food_topic", "physically_impossible"):
        entry = RULE_PREDICATES[rid]
        assert entry.backing is Backing.VALIDATOR_DERIVED
        assert entry.label_of is None
        assert entry.recomputable is False


# --- registry completeness ----------------------------------------------------


def test_registry_covers_all_30_rule_ids():
    assert len(RULE_PREDICATES) == 30
    # every entry is either recomputable (predicate / bank / best_effort) or a
    # validator-derived sentinel; none is half-specified.
    for rid, entry in RULE_PREDICATES.items():
        assert entry.rule_id == rid
        if entry.recomputable:
            assert callable(entry.label_of)
        else:
            assert entry.label_of is None
        assert entry.ruling  # every ruling is documented inline


def test_registry_matches_committed_spec_extract():
    """The registry's rule ids must equal the committed spec extract's, so a
    new rule cannot be added to the spec without a ground-truth entry here."""
    import json
    from pathlib import Path

    extract = (
        Path(__file__).resolve().parents[2] / "data" / "spec_extract.json"
    )
    data = json.loads(extract.read_text(encoding="utf-8"))
    assert set(RULE_PREDICATES) == set(data["rules"])


def test_backing_breakdown_is_as_documented():
    counts = {b: 0 for b in Backing}
    for entry in RULE_PREDICATES.values():
        counts[entry.backing] += 1
    # 19 pure predicates, 4 bank-membership (5,13,14,17), 4 best-effort
    # (8,9,10,12), 3 validator-derived (15,16,18)
    assert counts[Backing.PREDICATE] == 19
    assert counts[Backing.BANK_MEMBERSHIP] == 4
    assert counts[Backing.BEST_EFFORT] == 4
    assert counts[Backing.VALIDATOR_DERIVED] == 3


def test_unknown_rule_id_raises():
    with pytest.raises(GroundTruthError):
        assert_labels_correct("not_a_rule", [])


def test_item_with_wrong_rule_id_raises():
    items = [_item("contains_digit", True, "they moved 7 boxes")]
    items[0]["rule_id"] = "all_lowercase"
    with pytest.raises(GroundTruthError):
        assert_labels_correct("contains_digit", items)


def test_variant_rule_id_resolves_to_base_predicate():
    rid = "contains_digit_deconfounded"
    items = [
        _item(rid, True, "they moved 7 boxes before noon"),
        _item(rid, False, "they moved seven boxes before noon"),
    ]
    assert label_of(rid, "they moved 7 boxes before noon") is True
    assert_labels_correct(rid, items)

    items[0]["rule_id"] = "all_lowercase_deconfounded"
    with pytest.raises(GroundTruthError):
        assert_labels_correct(rid, items)


# --- verify_dataset (jsonl round-trip) ----------------------------------------


def test_verify_dataset_reads_jsonl_and_checks(tmp_path):
    rid = "contains_digit"
    items = [
        _item(rid, True, "they moved 7 boxes before noon"),
        _item(rid, True, "the crew counted 12 crates on the dock"),
        _item(rid, False, "they moved seven boxes before noon"),
        _item(rid, False, "the crew counted twelve crates on the dock"),
    ]
    path = tmp_path / "items.jsonl"
    write_items(items, path)
    assert verify_dataset(rid, path) == 4

    # corrupt one stored label on disk and verify it raises
    bad = copy.deepcopy(items)
    bad[0]["label"] = False
    bad_path = tmp_path / "bad.jsonl"
    write_items(bad, bad_path)
    with pytest.raises(GroundTruthError):
        verify_dataset(rid, bad_path)
