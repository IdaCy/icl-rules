"""Rule-id helpers for dataset variants.

Variant datasets such as ``word_count_geq_8_deconfounded`` preserve the same underlying
rule as ``word_count_geq_8`` while living in a different data directory. These
helpers keep loading paths and reported run ids variant-specific, but resolve
rubrics, predicates, and probe definitions against the canonical base rule.
"""

from __future__ import annotations

DECONFOUNDED_SUFFIXES = (
    "_deconfounded",
    "_deconfounded_b",
    "_deconfounded_c",
    "_deconfounded_d",
    "_deconfounded_e",
)

ALIASES = {
    "word_count_geq_8_v2": "word_count_geq_8",
    "second_word_capitalized_v2": "second_word_capitalized",
}


def canonical_rule_id(rule_id: str) -> str:
    """Return the base rule id whose semantics ``rule_id`` preserves."""
    for suffix in sorted(DECONFOUNDED_SUFFIXES, key=len, reverse=True):
        if rule_id.endswith(suffix):
            rule_id = rule_id[: -len(suffix)]
            break
    return ALIASES.get(rule_id, rule_id)


def is_rule_variant(rule_id: str) -> bool:
    return canonical_rule_id(rule_id) != rule_id
