#!/usr/bin/env python
"""Independent no-API verifier for Deconfounded rebuilt datasets.

This is intentionally separate from ``datagen.groundtruth``. It reimplements the
mechanical predicates used by Deconfounded variants and checks every row in every split.
Validator-derived semantic rules cannot be recomputed locally; for those, this
script verifies the required validation provenance flag.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.contexts import load_items
from icl_articulation.datagen import banks
from icl_articulation.datagen.groundtruth import VALIDATED_FLAG
from icl_articulation.datagen.schema import words
from icl_articulation.rule_ids import canonical_rule_id

VOWELS = set("aeiou")
PUNCT = ".,!?;:\"'()[]"
PARTICIPLE_RE = re.compile(r"[a-z]{2,}(ed|en)$")


def alpha_len(token: str) -> int:
    return sum(1 for ch in token if ch.isalpha())


def first_alpha(token: str) -> str:
    for ch in token:
        if ch.isalpha():
            return ch.lower()
    return ""


def last_alpha(token: str) -> str:
    for ch in reversed(token):
        if ch.isalpha():
            return ch.lower()
    return ""


def low_words(text: str) -> list[str]:
    return [w.lower().strip(PUNCT) for w in words(text)]


def contains_bank_word(bank_name: str) -> Callable[[str], bool]:
    bank_words = {w.lower() for w in banks.get_bank(bank_name).words()}
    return lambda text: any(tok in bank_words for tok in low_words(text))


FIRST_NAMES = {w.lower() for w in banks.get_bank("FIRST_NAMES").words()}


def p_contains_first_name(text: str) -> bool:
    return any(tok in FIRST_NAMES for tok in low_words(text))


def p_starts_with_vowel(text: str) -> bool:
    toks = words(text)
    return bool(toks) and first_alpha(toks[0]) in VOWELS


def p_last_word_ends_with_vowel(text: str) -> bool:
    toks = words(text)
    return bool(toks) and last_alpha(toks[-1]) in VOWELS


def p_even_word_count(text: str) -> bool:
    return len(words(text)) % 2 == 0


def p_passive_voice(text: str) -> bool:
    toks = low_words(text)
    for i, tok in enumerate(toks[:-1]):
        nxt = toks[i + 1]
        if tok in {"was", "were"} and not nxt.endswith("ing") and PARTICIPLE_RE.search(nxt):
            return True
    return False


def p_the_appears_twice(text: str) -> bool:
    return sum(1 for tok in low_words(text) if tok == "the") >= 2


def p_first_word_longer_than_last(text: str) -> bool:
    toks = words(text)
    return bool(toks) and alpha_len(toks[0]) > alpha_len(toks[-1])


def p_all_words_longer_than_3(text: str) -> bool:
    toks = words(text)
    return bool(toks) and all(alpha_len(tok) > 3 for tok in toks)


def p_first_two_words_alphabetical(text: str) -> bool:
    toks = low_words(text)
    return len(toks) >= 2 and toks[0] < toks[1]


def p_word_count_geq_8(text: str) -> bool:
    return len(words(text)) >= 8


def p_second_word_capitalized(text: str) -> bool:
    toks = words(text)
    return len(toks) >= 2 and toks[1][:1].isupper()


PREDICATES: dict[str, Callable[[str], bool]] = {
    "mentions_color": contains_bank_word("COLORS"),
    "mentions_animal": contains_bank_word("ANIMALS"),
    "contains_first_name": p_contains_first_name,
    "starts_with_vowel": p_starts_with_vowel,
    "last_word_ends_with_vowel": p_last_word_ends_with_vowel,
    "even_word_count": p_even_word_count,
    "passive_voice": p_passive_voice,
    "the_appears_twice": p_the_appears_twice,
    "first_word_longer_than_last": p_first_word_longer_than_last,
    "all_words_longer_than_3": p_all_words_longer_than_3,
    "first_two_words_alphabetical": p_first_two_words_alphabetical,
    "word_count_geq_8": p_word_count_geq_8,
    "second_word_capitalized": p_second_word_capitalized,
}

VALIDATOR_RULES = {"food_topic", "positive_sentiment", "physically_impossible"}


def verify_rule(rule_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    base = canonical_rule_id(rule_id)
    ids = [str(it.get("item_id")) for it in items]
    texts = [str(it.get("text")) for it in items]
    base_by_split: dict[str, set[str]] = {}
    for it in items:
        base_by_split.setdefault(str(it.get("split")), set()).add(str(it.get("base_id")))
    train_bases = base_by_split.get("few_shot_pool", set())
    eval_bases = base_by_split.get("held_out", set()) | base_by_split.get("confirmation", set())

    split_counts: dict[str, dict[str, int]] = {}
    for split in sorted({str(it.get("split")) for it in items}):
        group = [it for it in items if it.get("split") == split]
        split_counts[split] = {
            "n": len(group),
            "true": sum(1 for it in group if bool(it.get("label"))),
            "false": sum(1 for it in group if not bool(it.get("label"))),
        }

    if base in PREDICATES:
        pred = PREDICATES[base]
        mismatches = [
            {
                "item_id": str(it.get("item_id")),
                "label": bool(it.get("label")),
                "text": str(it.get("text")),
                "predicate": pred(str(it.get("text"))),
            }
            for it in items
            if pred(str(it.get("text"))) != bool(it.get("label"))
        ]
        mode = "predicate"
    elif base in VALIDATOR_RULES:
        mismatches = [
            {
                "item_id": str(it.get("item_id")),
                "reason": "missing or disagreeing validator agreement provenance",
            }
            for it in items
            if not isinstance(it.get("slots_meta"), dict)
            or VALIDATED_FLAG not in it["slots_meta"]
            or bool(it["slots_meta"][VALIDATED_FLAG]) != bool(it.get("label"))
        ]
        mode = "validator_provenance"
    else:
        mismatches = [
            {"item_id": str(it.get("item_id")), "reason": f"no verifier for {base}"}
        ]
        mode = "missing"

    duplicate_ids = [k for k, v in Counter(ids).items() if v > 1]
    duplicate_texts = [k for k, v in Counter(texts).items() if v > 1]
    return {
        "rule_id": rule_id,
        "canonical_rule_id": base,
        "mode": mode,
        "n_items": len(items),
        "split_counts": split_counts,
        "n_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "duplicate_item_ids": duplicate_ids[:20],
        "duplicate_texts": duplicate_texts[:20],
        "base_overlap_train_eval": len(train_bases & eval_bases),
        "passes": (
            not mismatches
            and not duplicate_ids
            and not duplicate_texts
            and len(train_bases & eval_bases) == 0
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--rules", help="comma-separated rule ids; default verifies every directory")
    p.add_argument("--output", default="deconfounded_independent_verify.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    if args.rules:
        rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    else:
        rules = sorted(p.name for p in data_dir.iterdir() if (p / "items.jsonl").is_file())
    results: dict[str, Any] = {}
    ok = True
    for rule_id in rules:
        items = load_items(data_dir / rule_id / "items.jsonl")
        result = verify_rule(rule_id, items)
        results[rule_id] = result
        ok = ok and bool(result["passes"])
        status = "PASS" if result["passes"] else "FAIL"
        print(f"{status:4} {rule_id:35} mismatches={result['n_mismatches']}")
    payload = {"data_dir": str(data_dir), "rules": results, "overall_pass": ok}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
