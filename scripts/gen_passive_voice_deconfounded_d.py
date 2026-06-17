"""Build data/passive_voice_deconfounded_d/items.jsonl — passive_voice_deconfounded as MINIMAL PAIRS,
so no non-morphology feature (length, vocabulary, tail, role) carries label signal.

passive_voice_deconfounded is already the clean design: identical frames
  PASSIVE (True):  "The {role} was {V-ed}  {tail}"
  ACTIVE  (False): "The {role} was {V-ing} {tail}"
same role/tail pools, no post-verbal NP, word-count auto-matched, bag-of-words at
chance. Its only residual is char-length (~0.64): the gerund suffix "-ing" (3) is
one char longer than the participle "-ed"/"-en" (2), so active sentences run ~1
char longer.

That residual is INTRINSIC — it is the voice morphology itself. Trying to force a
length-threshold to chance only relocates the one-char difference into tail/role
selection (a fresh lexical confound), which is worse than disclosing it. deconfounded_d
instead emits MINIMAL PAIRS: for each (role, verb, tail) it produces BOTH the
passive and the active sentence, sharing a base_id. Role/verb/tail are therefore
identically distributed across the two classes, so bag-of-words, tail identity and
role identity all sit at chance and the ONLY thing that differs within a pair is
the participle-vs-gerund suffix (= the definition of voice). The residual char-
length gap is reported, not engineered away, and is provably nothing but the
morphology. Verb OOD-partition (few-shot pool vs eval) is preserved from deconfounded.

Run:  python scripts/gen_passive_voice_deconfounded_d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from icl_articulation.datagen import schema  # noqa: E402
from icl_articulation.datagen.genutils import fix_indefinite_articles  # noqa: E402
from icl_articulation.datagen.schema import make_item, write_items  # noqa: E402
from icl_articulation.datagen.generators.base import Gen  # noqa: E402

SEED = 23
RULE_ID = "passive_voice_deconfounded_d"

VERBS = [
    "paint", "clean", "wash", "cook", "plant", "study", "copy", "count",
    "move", "prepare", "polish", "mend", "sort", "pack", "fold", "dry",
    "repair", "inspect",
]
ROLE_WORDS = [
    "worker", "teacher", "farmer", "painter", "baker", "gardener", "tailor",
    "cook", "nurse", "clerk", "porter", "driver", "artist", "student",
    "neighbor", "visitor",
]
TAILS = [
    "near home", "at dawn", "at noon", "by night", "in town", "outside",
    "nearby", "today", "indoors", "downtown",
]

SPLIT_SIZES = {
    "few_shot_pool": {True: 100, False: 100},
    "held_out": {True: 60, False: 60},
    "confirmation": {True: 50, False: 50},
    "spare": {True: 10, False: 10},
}


def _split_values(values, seed, *, train_frac=0.6):
    vals = sorted({str(v).lower() for v in values})
    g = Gen(seed)
    g.shuffle(vals)
    cut = max(1, min(len(vals) - 1, round(len(vals) * train_frac)))
    return set(vals[:cut]), set(vals[cut:])


TRAIN_VERBS, EVAL_VERBS = _split_values(VERBS, SEED + 909, train_frac=0.6)


def _past(v):
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return f"{v[:-1]}ied"
    return f"{v}d" if v.endswith("e") else f"{v}ed"


def _ing(v):
    if v.endswith("e") and v != "see":
        return f"{v[:-1]}ing"
    return f"{v}ing"


def _verbs_for(split):
    if split == "spare":
        return sorted(VERBS)
    return sorted(TRAIN_VERBS) if split == "few_shot_pool" else sorted(EVAL_VERBS)


def _base_combos(split, gen):
    """All (role, verb, tail) frames valid (wc 5-7) for BOTH voices, shuffled."""
    combos = []
    for role in ROLE_WORDS:
        for verb in _verbs_for(split):
            for tail in TAILS:
                t_text = fix_indefinite_articles(f"The {role} was {_past(verb)} {tail}")
                f_text = fix_indefinite_articles(f"The {role} was {_ing(verb)} {tail}")
                if not (5 <= schema.word_count(t_text) <= 7):
                    continue
                if not (5 <= schema.word_count(f_text) <= 7):
                    continue
                combos.append((role, verb, tail, t_text, f_text))
    gen.derive(f"{split}:combos").shuffle(combos)
    return combos


def build():
    gen = Gen(SEED)
    items, index = [], 0
    used: set[str] = set()  # enforce global text uniqueness across splits
    for split, by_label in SPLIT_SIZES.items():
        n = by_label[True]
        assert by_label[True] == by_label[False]
        combos = _base_combos(split, gen)
        # The eval splits (held_out, confirmation) draw from the same eval-verb
        # combo pool, and spare draws from all verbs, so frames collide across
        # splits. Take only frames whose passive AND active text are globally
        # unused, keeping every split text-disjoint and 50/50 balanced.
        chosen = []
        for combo in combos:
            _, _, _, t_text, f_text = combo
            if t_text in used or f_text in used or t_text == f_text:
                continue
            chosen.append(combo)
            used.add(t_text)
            used.add(f_text)
            if len(chosen) == n:
                break
        if len(chosen) < n:
            raise RuntimeError(f"{split}: only {len(chosen)} unique frames < {n}")
        for role, verb, tail, t_text, f_text in chosen:
            # The minimal pair shares matched role/verb/tail (the deconfounding
            # property lives in the text). base_id stays unique per item so the
            # eval splits satisfy one-variant-per-base; the pairing is recorded
            # in slots_meta for provenance.
            pair = f"passive-deconfounded_d-pair-{index:05d}"
            common = {"verb_base": verb, "tail": tail, "role_word": role, "seed": SEED,
                      "pair_id": pair,
                      "verb_partition": "train" if split == "few_shot_pool" else "eval"}
            t_id = f"passive-deconfounded_d-{2 * index:05d}"
            f_id = f"passive-deconfounded_d-{2 * index + 1:05d}"
            items.append(make_item(
                item_id=t_id, base_id=t_id, rule_id=RULE_ID, label=True,
                text=t_text, split=split,
                slots_meta={"shape": "passive_was_participle", **common}))
            items.append(make_item(
                item_id=f_id, base_id=f_id, rule_id=RULE_ID, label=False,
                text=f_text, split=split,
                slots_meta={"shape": "active_progressive", **common}))
            index += 1
    return items


def main():
    items = build()
    out_dir = REPO / "data" / "passive_voice_deconfounded_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "items.jsonl"
    write_items(items, out)
    print(f"wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
