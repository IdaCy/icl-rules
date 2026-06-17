#!/usr/bin/env python
"""Build a deconfounded Deconfounded dataset for ``first_two_words_alphabetical``.

Writes ``data/first_two_words_alphabetical_deconfounded_b/items.jsonl`` (the original
``..._deconfounded`` is never touched).

The confound in ``..._deconfounded``: the first two words are drawn from an
alphabet-skewed adjective/noun bank, so the IDENTITY of the first two words
predicts the label (a word-identity proxy scores ~0.66 on held_out). The point
of the rule is that the model must COMPARE the two words alphabetically, not
recognise particular words.

The fix here -- BOTH-ORDERINGS construction. For each unordered pair of distinct
words {A, B} (with A < B alphabetically), emit BOTH orderings sharing the same
verb + adjunct tail:

    "A B <verb> <tail>"  -> label True   (first word precedes second)
    "B A <verb> <tail>"  -> label False  (first word follows second)

So every first/second-position word appears equally often in True and False, and
the ONLY thing distinguishing the classes is the order. Any single-word or
word-identity proxy is therefore ~0.50 by construction; only the alphabetical
RELATIONSHIP carries label signal.

OOD eval: the word inventory is partitioned into a TRAIN word set (few_shot_pool)
and a disjoint EVAL word set (held_out / confirmation / spare). No word used in
the first-two positions of an eval pair appears in any few_shot_pool pair, and no
text is duplicated pool<->eval.

Schema fields per item: item_id, base_id, rule_id, label, text, slots_meta, split.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.datagen.banks import _regular_verb_forms, get_bank
from icl_articulation.datagen.genutils import Gen, base_id as make_base_id, to_sentence_case

SEED = 20260614
RULE_ID = "first_two_words_alphabetical_deconfounded_b"
OUT = Path("data") / "first_two_words_alphabetical_deconfounded_b" / "items.jsonl"

# Split sizes mirror data/first_two_words_alphabetical_deconfounded exactly:
#   few_shot_pool 200 (100 base pairs x both orderings) -> 100 T / 100 F
#   held_out      120 (60 base pairs)                   ->  60 T /  60 F
#   confirmation  100 (50 base pairs)                   ->  50 T /  50 F
#   spare          20 (10 base pairs)                   ->  10 T /  10 F
N_FEWSHOT_PAIRS = 100
N_HELD_PAIRS = 60
N_CONF_PAIRS = 50
N_SPARE_PAIRS = 10

WC_TARGETS = (5, 6, 7, 8, 9)


def _word_inventory() -> list[str]:
    """Uniform-across-the-alphabet word inventory for the first two positions.

    Combines the two banks the rule already uses (adjectives + plural nouns),
    each covering the 14 letters {b,c,d,f,g,h,l,m,n,p,r,s,t,w} with 5 entries,
    giving 10 words per letter. Words are used purely as orderable tokens; both
    orderings of every pair are emitted so identity carries no label signal.
    """
    adj = [w.lower() for w in get_bank("ADJ_BY_LETTER").words()]
    noun = [w.lower() for w in get_bank("NOUN_PLURAL_BY_LETTER").words()]
    return sorted(set(adj) | set(noun))


def _build_tail(gen: Gen, verbs_past: list[str], adverbs: list[str], target: int) -> tuple[str, str]:
    """A shared (verb_past, tail) bringing 'W1 W2 verb [tail]' to ``target`` words.

    Core is 3 words (W1 W2 verb); tail adds (target-3) adverb-place words. Every
    adverb is a single word, so this is exact and label-independent.
    """
    verb_past = gen.choice(verbs_past)
    n_tail = target - 3
    tail_words = [gen.choice(adverbs) for _ in range(n_tail)]
    return verb_past, " ".join(tail_words)


def _emit_pair(w_lo: str, w_hi: str, verb_past: str, tail: str) -> tuple[dict, dict]:
    """Given W_lo < W_hi alphabetically, build the True and False text/meta."""
    core_true = f"{w_lo} {w_hi} {verb_past}"
    core_false = f"{w_hi} {w_lo} {verb_past}"
    text_true = to_sentence_case((core_true + " " + tail).strip())
    text_false = to_sentence_case((core_false + " " + tail).strip())
    return (
        {"first": w_lo, "second": w_hi, "text": text_true},
        {"first": w_hi, "second": w_lo, "text": text_false},
    )


def main() -> None:
    gen = Gen(SEED)
    inv = _word_inventory()

    # Disjoint TRAIN / EVAL word partition (OOD eval). 55% to train.
    pgen = gen.derive("partition")
    shuffled = list(inv)
    pgen.shuffle(shuffled)
    cut = round(len(shuffled) * 0.55)
    train_words = sorted(shuffled[:cut])
    eval_words = sorted(shuffled[cut:])

    verb_bases = [w.lower() for w in get_bank("VERB_REGULAR").words()]
    verbs_past = [_regular_verb_forms(v)[2] for v in verb_bases]
    adverbs = [w.lower() for w in get_bank("ADVERB_PLACE").words()]

    def balanced_pairs(words: list[str], n_pairs: int, rng: Gen) -> list[tuple[str, str]]:
        """Pick ``n_pairs`` distinct unordered pairs (w_lo, w_hi), w_lo < w_hi,
        emitted as LETTER-MIRRORED quartets so that, in the both-orderings
        dataset, neither the first-word identity NOR its initial letter predicts
        the label.

        A word is first-position-True exactly when it is the ``lo`` of its pair
        and first-position-False when it is the ``hi``. A given initial letter is
        ``lo`` when paired with a larger letter and ``hi`` when paired with a
        smaller one, so to keep both the FIRST-word and SECOND-word initial-letter
        marginals symmetric across the two classes, every letter must be ``lo``
        (paired up) and ``hi`` (paired down) at MATCHED rates. We greedily pick
        each pair to minimise the running per-LETTER (lo_count - hi_count)
        imbalance, AND simultaneously minimise the per-WORD imbalance, so neither
        word identity nor initial letter predicts the label -> both proxies near
        0.50. Distinct lowercase words never tie, so labels are unambiguous.
        """
        all_pairs = [
            (a, b)
            for i, a in enumerate(words)
            for b in words[i + 1:]
            if a < b
        ]
        rng.shuffle(all_pairs)
        word_imb: dict[str, int] = collections.defaultdict(int)   # lo - hi per word
        letter_imb: dict[str, int] = collections.defaultdict(int)  # lo - hi per letter
        used: set[int] = set()
        chosen: list[tuple[str, str]] = []
        for _ in range(n_pairs):
            best_idx = None
            best_key = None
            for idx, (lo, hi) in enumerate(all_pairs):
                if idx in used:
                    continue
                # resulting letter imbalance (primary) and word imbalance (tie-break)
                l_cost = abs(letter_imb[lo[0]] + 1) + abs(letter_imb[hi[0]] - 1)
                w_cost = abs(word_imb[lo] + 1) + abs(word_imb[hi] - 1)
                key = (l_cost, w_cost)
                if best_key is None or key < best_key:
                    best_key, best_idx = key, idx
                    if key == (0, 0):
                        break
            assert best_idx is not None, f"ran out of pairs for {n_pairs}"
            lo, hi = all_pairs[best_idx]
            used.add(best_idx)
            word_imb[lo] += 1
            word_imb[hi] -= 1
            letter_imb[lo[0]] += 1
            letter_imb[hi[0]] -= 1
            chosen.append((lo, hi))

        # Repair pass: swap a chosen pair for an unused one when doing so reduces
        # the total per-letter |lo - hi| imbalance. Early letters tend to be stuck
        # as ``lo`` and late letters as ``hi`` after the forward greedy; targeted
        # swaps pull the initial-letter marginals back toward symmetry.
        def total_letter_imb() -> int:
            return sum(abs(v) for v in letter_imb.values())

        unused = [i for i in range(len(all_pairs)) if i not in used]
        improved = True
        while improved:
            improved = False
            base = total_letter_imb()
            for ci, (clo, chi) in enumerate(chosen):
                for ui in unused:
                    nlo, nhi = all_pairs[ui]
                    # apply swap to a copy of letter_imb
                    letter_imb[clo[0]] -= 1
                    letter_imb[chi[0]] += 1
                    letter_imb[nlo[0]] += 1
                    letter_imb[nhi[0]] -= 1
                    new = total_letter_imb()
                    if new < base:
                        chosen[ci] = (nlo, nhi)
                        word_imb[clo] -= 1
                        word_imb[chi] += 1
                        word_imb[nlo] += 1
                        word_imb[nhi] -= 1
                        used.discard(all_pairs.index((clo, chi)))
                        used.add(ui)
                        unused.remove(ui)
                        unused.append(all_pairs.index((clo, chi)))
                        improved = True
                        break
                    else:
                        # revert
                        letter_imb[clo[0]] += 1
                        letter_imb[chi[0]] -= 1
                        letter_imb[nlo[0]] -= 1
                        letter_imb[nhi[0]] += 1
                if improved:
                    break
        return chosen

    items: list[dict] = []
    seen_text: set[str] = set()

    def build_split(split: str, n_pairs: int, words: list[str]) -> None:
        sgen = gen.derive(f"split:{split}")
        chosen = balanced_pairs(words, n_pairs, sgen)
        for w_lo, w_hi in chosen:
            target = sgen.choice(list(WC_TARGETS))
            verb_past, tail = _build_tail(sgen, verbs_past, adverbs, target)
            t_meta, f_meta = _emit_pair(w_lo, w_hi, verb_past, tail)
            base_id = make_base_id("b8b", split, w_lo, w_hi, verb_past, tail)
            for label, m in ((True, t_meta), (False, f_meta)):
                text = m["text"]
                assert text not in seen_text, f"duplicate text: {text!r}"
                seen_text.add(text)
                slots_meta = {
                    "seed": SEED,
                    "deconfounded_source_rule": "first_two_words_alphabetical",
                    "construction": "both_orderings",
                    "first_word": m["first"],
                    "second_word": m["second"],
                    "verb_past": verb_past,
                    "tail": tail,
                    "first_initial": m["first"][0],
                    "second_initial": m["second"][0],
                    "word_count": target,
                }
                items.append(
                    {
                        "item_id": f"{base_id}-{'T' if label else 'F'}",
                        "base_id": base_id,
                        "rule_id": RULE_ID,
                        "label": label,
                        "text": text,
                        "slots_meta": slots_meta,
                        "split": split,
                    }
                )

    build_split("few_shot_pool", N_FEWSHOT_PAIRS, train_words)
    build_split("held_out", N_HELD_PAIRS, eval_words)
    build_split("confirmation", N_CONF_PAIRS, eval_words)
    build_split("spare", N_SPARE_PAIRS, eval_words)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} items to {OUT}")


if __name__ == "__main__":
    main()
