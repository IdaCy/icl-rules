#!/usr/bin/env python
"""Build data/second_word_capitalized_v2/ — a DECONFOUNDED rebuild of
second_word_capitalized (insight-pass multi-rule deconfound). LOCAL, no API.
NEVER overwrites the original dataset.

The original made True items by seating a PROPER NOUN at position 2 (the subject:
'Apparently Maria opened the door') and False items a lowercase common-noun
subject ('Apparently someone opened the door'). So position-2-capitalization
matched, in-distribution, "the second word is a proper noun" under the dataset
vocabulary:
the True/False second-word vocabularies were 100% disjoint and a bag-of-words
baseline reached 0.98. The model accordingly articulated "the subject is a proper
noun" (a SEMANTIC reading), graded 1, never the abstract POSITIONAL rule "the
second word is capitalized".

This v2 BREAKS that confound with a minimal-casing design:

  * Position 2 is drawn from a SHARED pool of common-noun subjects AND proper
    nouns (names / places). The SAME word-type appears CAPITALIZED in True items
    and LOWERCASE in False items, so after case-folding the token distribution is
    identical across classes (zero one-sided content tokens) and "the second word
    is a proper noun" is decorrelated from the label (proper nouns are ~50/50
    across classes by construction).
  * The ONLY feature that predicts the label is the capitalization of word 2 —
    i.e. the abstract positional rule itself.
  * Everything else (opener, verb, object noun, trailing adjunct) is drawn from
    shared pools by round-robin, so no frame word is one-sided. Word count is
    fixed at 7 and char-length is ~matched (the two classes differ only in the
    case of one letter), so length/word-count predicates sit at 0.5.

Pre-specified readings (both reportable):
  (a) RELOCATION / LEARNING-GAP: if in-context accuracy collapses toward chance
      (like all_lowercase) or the model articulates a NEW non-positional proxy,
      then original swc was classified via the proper-noun proxy, NOT the
      positional/orthographic rule — the model never learned the abstract rule.
  (b) REVEAL: if accuracy stays high AND the model now articulates "the second
      word is capitalized", deconfounding revealed the abstract rule (the
      surface≈abstract contrast case).

Audit criterion: ZERO one-sided high-frequency tokens (case-folded), proper-noun
rate matched across classes, ground-truth exact, balanced splits. char-length is
reported (it is ~identical by construction).

Run:  python scripts/gen_swc_v2.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from icl_articulation.datagen import banks, groundtruth

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "second_word_capitalized_v2" / "items.jsonl"
SEED = 20260612

# --- banks (shared across BOTH classes) --------------------------------------
# common-noun SUBJECTS that read naturally at position 2 after a sentence-initial
# adverb and before a past-tense transitive verb.
COMMON_SUBJ = ["workers", "students", "neighbors", "travelers", "soldiers",
               "farmers", "sailors", "painters", "dancers", "singers",
               "someone", "people", "everyone", "children", "visitors"]
# the past-tense regular transitive verbs (shared by both classes).
VERB = ["opened", "closed", "cleaned", "painted", "watched", "pushed", "pulled",
        "washed", "carried", "moved", "counted", "collected", "shared",
        "removed", "ordered", "guarded", "polished", "checked", "mended", "raised"]
NOUN = ["gate", "box", "coat", "table", "window", "door", "basket", "lamp",
        "fence", "bottle", "ladder", "kettle", "drawer", "mirror", "bucket",
        "shelf", "crate", "barrel", "cart", "stool"]
# 2-word, comma-free, lowercase trailing adjuncts (so every frame is 7 words and
# the last word is never capitalized).
TAIL = ["by night", "at dawn", "at noon", "in town", "by day", "at dusk",
        "in spring", "by hand", "on foot", "at home", "near home", "by noon"]


def _cap(w: str) -> str:
    return w[0].upper() + w[1:]


def _build(opener: str, w2: str, verb: str, noun: str, tail: str, want_true: bool) -> str:
    """'{Opener} {W2} {verb} the {noun} {tail(2w)}' — 7 words.

    Word 1 (opener) is always sentence-cased. Word 2 (w2) is CAPITALIZED for the
    True variant and LOWERCASE for the False variant — the only difference."""
    word2 = _cap(w2) if want_true else w2.lower()
    return " ".join([_cap(opener), word2, verb, "the", noun, tail])


def _cycler(items, rng: random.Random):
    pool: list = []
    while True:
        if not pool:
            pool = list(items)
            rng.shuffle(pool)
        yield pool.pop()


def main() -> int:
    rng = random.Random(SEED)
    openers = banks.get_bank("ADVERB_SENT_INITIAL").words()
    names = banks.get_bank("FIRST_NAMES").words()
    places = banks.get_bank("NONNAME_PROPER").words()
    # the position-2 pool: common nouns and proper nouns (names + places). We draw
    # word 2 as 50% common / 50% proper, and reuse the SAME word for the True and
    # False item at each index, so that:
    #  (a) every W2 word appears capitalized (True) and lowercase (False) the same
    #      number of times -> after case-folding the token distribution is balanced
    #      (zero one-sided tokens), and
    #  (b) proper-noun-ness is ~0.5 in BOTH classes and "naturalness" is matched
    #      (each class is 50% naturally cased [cap proper / lower common] and 50%
    #      oddly cased [cap common / lower proper]).
    # So neither proper-noun-ness nor casing-naturalness predicts the label — ONLY
    # the capitalization of word 2 does.
    proper_set = {w.lower() for w in (names + places)}
    c_common = _cycler([w.lower() for w in COMMON_SUBJ], rng)
    c_proper = _cycler([w.lower() for w in (names + places)], rng)
    c_op = _cycler([o.lower() for o in openers], rng)
    c_vb = _cycler(VERB, rng)
    c_nn = _cycler(NOUN, rng)
    c_tl = _cycler(TAIL, rng)

    seen: set[str] = set()
    items = []
    bi = 0

    def emit(w2: str, want_true: bool, split: str):
        nonlocal bi
        for _ in range(10000):
            text = _build(next(c_op), w2, next(c_vb), next(c_nn), next(c_tl), want_true)
            if text in seen:
                continue
            if groundtruth.label_of("second_word_capitalized", text) is not want_true:
                continue
            seen.add(text)
            base = f"swcv2-{bi:05d}"; bi += 1
            return {"item_id": base, "base_id": base,
                    "rule_id": "second_word_capitalized_v2", "label": want_true,
                    "text": text,
                    "slots_meta": {"seed": SEED, "w2": w2,
                                   "w2_is_proper": w2 in proper_set},
                    "split": split}
        raise RuntimeError("could not emit a unique item")

    SPLITS = [("few_shot_pool", 100), ("held_out", 60), ("confirmation", 50), ("spare", 30)]
    # for each index draw ONE W2 word (alternating common/proper for a 50/50 mix)
    # and emit BOTH a True (W2 capitalized) and a False (W2 lowercase) item from it.
    idx = 0
    for split, n in SPLITS:
        for _ in range(n):
            w2 = next(c_common) if idx % 2 == 0 else next(c_proper)
            idx += 1
            items.append(emit(w2, True, split))
            items.append(emit(w2, False, split))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    from icl_articulation.contexts import validate_dataset
    validate_dataset(items)
    _audit(items, proper_set)
    print(f"\nwrote {OUT} ({len(items)} items) — validate_dataset OK")
    return 0


def _audit(items, proper_set):
    trues = [it for it in items if it["label"]]
    falses = [it for it in items if not it["label"]]
    bad = sum(1 for it in items
              if groundtruth.label_of("second_word_capitalized", it["text"]) != it["label"])

    def toks(t):  # case-folded, like the confound auditor
        return {w.strip(".,!?;:").lower() for w in t.split()}
    tc, fc = Counter(), Counter()
    for it in trues:
        for w in toks(it["text"]):
            tc[w] += 1
    for it in falses:
        for w in toks(it["text"]):
            fc[w] += 1
    total = len(trues) + len(falses)
    one_sided = [(w, tc[w], fc[w]) for w in set(tc) | set(fc)
                 if (tc[w] + fc[w]) >= 0.03 * total and (tc[w] == 0 or fc[w] == 0)]

    # proper-noun rate of word 2, per class (should be ~equal -> decorrelated)
    def w2_proper(it):
        return it["slots_meta"]["w2_is_proper"]
    pt = sum(1 for it in trues if w2_proper(it)) / len(trues)
    pf = sum(1 for it in falses if w2_proper(it)) / len(falses)
    # "second word is a proper noun" extensional accuracy (the OLD proxy)
    proxy_acc = sum(1 for it in items if (it["slots_meta"]["w2"] in proper_set) == bool(it["label"])) / len(items)

    import statistics
    tcl = statistics.mean(len(it["text"]) for it in trues)
    fcl = statistics.mean(len(it["text"]) for it in falses)
    print("=== AUDIT (second_word_capitalized_v2) ===")
    print(f"  ground-truth mismatches: {bad} (must be 0)")
    print(f"  one-sided high-freq tokens (case-folded): {len(one_sided)} -> {one_sided[:8]}")
    print(f"  word-2 proper-noun rate: True {pt:.2f}  False {pf:.2f}  (matched -> decorrelated)")
    print(f"  OLD proxy 'word 2 is a proper noun' extensional acc: {proxy_acc:.3f} (was ~1.0; target ~0.5)")
    print(f"  char-length mean: True {tcl:.1f}  False {fcl:.1f}  (differ only by case of 1 letter)")
    print(f"  example True : {trues[0]['text']!r}")
    print(f"  example False: {falses[0]['text']!r}")


if __name__ == "__main__":
    raise SystemExit(main())
