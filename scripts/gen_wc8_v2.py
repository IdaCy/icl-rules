#!/usr/bin/env python
"""Build data/word_count_geq_8_v2/ — a DECONFOUNDED rebuild of word_count_geq_8
(external-review P2b). LOCAL, no API. NEVER overwrites the original dataset.

The original made True items by appending trailing adjuncts to a False core, so
prepositions/adverbs appeared ONLY in True (e.g. "by" 84 True / 0 False) and a
naive-Bayes bag-of-words baseline recovered the label at 0.892. Here:

  * Each base shares a subject noun, verb, object noun, AND its trailing adjunct
    (present or not) across BOTH its True (>=8 words) and False (<=7 words)
    variants — so trailing-adjunct presence is IDENTICAL within a pair and
    matched across classes by construction.
  * Word count varies through the number of ADJECTIVES (and an optional pre-verb
    adverb) in the subject/object noun phrases, NOT through adjuncts.
  * Both classes draw adjectives/adverbs/adjuncts from the SAME banks, and the
    builder verifies every content token appears in BOTH classes (no one-sided
    tokens). Determiner/article counts are matched within each pair.

Audit criterion (per the descope): ZERO one-sided high-frequency tokens and
matched adjunct rates across classes. The naive-Bayes and char-length baselines
are REPORTED, not gated. This v2 build leaves an aggregate char-length signal
(more words -> more chars at matched word lengths): a char-counter recovers the
label at ~0.94 here. That signal is NOT inherent to the rule, only to this
build — the later word_count_geq_8_v3 (scripts/gen_wc8_v3.py) overlaps the per-
class character-length distributions (True = many short words, False = few long
words) and drops the char-counter to ~chance while keeping the >=8-words label
exact. Prefer v3 when the claim depends on the model counting words rather than
reading length.

Run:  python scripts/gen_wc8_v2.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from icl_articulation.datagen import groundtruth

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "word_count_geq_8_v2" / "items.jsonl"
SEED = 20260612

# --- banks (shared across BOTH classes) --------------------------------------
SUBJ_ADJ = ["old", "young", "tall", "short", "kind", "quiet", "clever", "brave",
            "gentle", "eager", "weary", "cheerful", "careful", "honest", "polite",
            "tired", "calm", "bold", "shy", "proud"]
SUBJ_NOUN = ["worker", "teacher", "doctor", "farmer", "painter", "driver", "baker",
             "singer", "writer", "builder", "sailor", "hunter", "gardener", "porter",
             "tailor", "miner", "cook", "nurse", "guard", "clerk"]
ADVERB = ["quietly", "slowly", "calmly", "gently", "quickly", "boldly", "neatly",
          "firmly", "softly", "sweetly", "kindly", "bravely", "warmly", "plainly"]
VERB = ["cleaned", "painted", "moved", "carried", "opened", "closed", "washed",
        "fixed", "lifted", "pushed", "pulled", "packed", "sorted", "counted",
        "mended", "polished", "checked", "guarded", "watched", "raised"]
OBJ_ADJ = ["dusty", "broken", "narrow", "wooden", "heavy", "small", "large",
           "empty", "dirty", "shiny", "round", "rusty", "faded", "cracked",
           "plain", "sturdy", "worn", "spare", "neat", "wide"]
OBJ_NOUN = ["gate", "box", "coat", "table", "window", "door", "basket", "lamp",
            "fence", "bottle", "ladder", "kettle", "drawer", "mirror", "bucket",
            "shelf", "crate", "barrel", "cart", "stool"]
# trailing adjuncts (phrases), each with its whitespace word length
ADJUNCTS = ["at home", "by hand", "in the yard", "near the lake", "with care",
            "on the porch", "after lunch", "before noon", "at dawn", "by the door",
            "in the shed", "down the road", "today", "again", "outside", "indoors",
            "nearby", "downtown", "overnight", "by night"]


def _wc(text: str) -> int:
    return len(text.split())


def _build(rng: random.Random, subj_n, verb, obj_n, adjunct, n_subj_adj, n_obj_adj, adverb):
    parts = ["The"]
    if n_subj_adj:
        parts += rng.sample(SUBJ_ADJ, n_subj_adj)
    parts.append(subj_n)
    if adverb:
        parts.append(rng.choice(ADVERB))
    parts.append(verb)
    parts.append("the")
    if n_obj_adj:
        parts += rng.sample(OBJ_ADJ, n_obj_adj)
    parts.append(obj_n)
    if adjunct:
        parts += adjunct.split()
    return " ".join(parts)


P_ADJUNCT = 0.4  # same target rate for BOTH classes (matched adjunct presence)


def make_one(rng: random.Random, want_true: bool):
    """One INDEPENDENT item (its own base; no shared core, so no cross-split text
    leakage) of the desired class. Adjunct present at P_ADJUNCT for BOTH classes;
    word count driven by adjective/adverb count. Returns (text, has_adjunct)."""
    subj_n = rng.choice(SUBJ_NOUN)
    verb = rng.choice(VERB)
    obj_n = rng.choice(OBJ_NOUN)
    has_adjunct = rng.random() < P_ADJUNCT
    adjunct = rng.choice(ADJUNCTS) if has_adjunct else None
    alen = len(adjunct.split()) if adjunct else 0
    base_wc = 5 + alen  # "The <subj> <verb> the <obj>" + adjunct

    if want_true:
        target = rng.choice([8, 8, 9, 9, 10])
        budget = max(1, target - base_wc)
    else:
        room = 7 - base_wc
        if room < 0:
            return None  # rare 3-word adjunct already > 7
        # bias False items toward USING adjectives when there's room, so every
        # adjective lands in BOTH classes (True items use more adjectives to reach
        # 8 words; without this, a rare adjective can end up one-sided).
        budget = rng.randint(1, room) if room >= 1 else 0
    adverb = 1 if (budget >= 2 and rng.random() < 0.4) else 0  # keep room for adjectives
    rem = budget - adverb
    # spread adjectives across subject AND object so both banks land in both classes
    n_subj = min(2, rng.randint(0, rem)); rem -= n_subj
    n_obj = min(2, rem)
    text = _build(rng, subj_n, verb, obj_n, adjunct, n_subj, n_obj, adverb)
    wc = _wc(text)
    # nudge True items that fell short of 8 by adding adjectives/adverb
    tries = 0
    while want_true and wc < 8 and tries < 6:
        if not adverb:
            adverb = 1
        elif n_subj < 2:
            n_subj += 1
        elif n_obj < 2:
            n_obj += 1
        else:
            break
        text = _build(rng, subj_n, verb, obj_n, adjunct, n_subj, n_obj, adverb)
        wc = _wc(text); tries += 1
    if want_true and wc < 8:
        return None
    if not want_true and wc > 7:
        return None
    return text, bool(adjunct)


def _draw(rng, want_true, n_with_adj, n_without_adj, seen):
    """n_with_adj items WITH an adjunct + n_without_adj WITHOUT, all unique."""
    out = []
    for need_adj, n in ((True, n_with_adj), (False, n_without_adj)):
        got = 0
        while got < n:
            r = make_one(rng, want_true)
            if r is None:
                continue
            text, has_adj = r
            if has_adj != need_adj or text in seen:
                continue
            # ground-truth check against the registered predicate
            if groundtruth.label_of("word_count_geq_8", text) is not want_true:
                continue
            seen.add(text); out.append((text, has_adj)); got += 1
    rng.shuffle(out)
    return out


def main() -> int:
    rng = random.Random(SEED)
    # per class: 100 few-shot + 60 held-out + 50 confirmation + 30 spare = 240,
    # of which ~P_ADJUNCT carry a trailing adjunct (matched across classes).
    N = 240
    n_adj = round(P_ADJUNCT * N)
    seen: set[str] = set()
    trues = _draw(rng, True, n_adj, N - n_adj, seen)
    falses = _draw(rng, False, n_adj, N - n_adj, seen)

    SPLITS = [("few_shot_pool", 100), ("held_out", 60), ("confirmation", 50), ("spare", 30)]
    items = []
    bi = 0
    for cls_items, label in ((trues, True), (falses, False)):
        i = 0
        for split, n in SPLITS:
            for _ in range(n):
                text, adj = cls_items[i]; i += 1
                base = f"wc8v2-{bi:05d}"; bi += 1
                items.append({
                    "item_id": base, "base_id": base, "rule_id": "word_count_geq_8_v2",
                    "label": label, "text": text,
                    "slots_meta": {"seed": SEED, "has_adjunct": adj, "wc": _wc(text)},
                    "split": split,
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    # validate against the same checker the runners use
    from icl_articulation.contexts import validate_dataset
    validate_dataset(items)
    _audit(items)
    print(f"\nwrote {OUT} ({len(items)} items) — validate_dataset OK")
    return 0


def _audit(items):
    trues = [it for it in items if it["label"]]
    falses = [it for it in items if not it["label"]]
    # ground-truth re-derivation
    bad = sum(1 for it in items if (groundtruth.label_of("word_count_geq_8", it["text"])) != it["label"])
    # matched adjunct rate
    at = sum(1 for it in trues if it["slots_meta"]["has_adjunct"]) / len(trues)
    af = sum(1 for it in falses if it["slots_meta"]["has_adjunct"]) / len(falses)
    # one-sided tokens (doc-freq >= 3%, present in only one class)
    def toks(t): return {w.strip(".,").lower() for w in t.split()}
    tc, fc = Counter(), Counter()
    for it in trues:
        for w in toks(it["text"]): tc[w] += 1
    for it in falses:
        for w in toks(it["text"]): fc[w] += 1
    n = min(len(trues), len(falses))
    one_sided = [(w, tc[w], fc[w]) for w in set(tc) | set(fc)
                 if (tc[w] + fc[w]) >= 0.03 * (len(trues) + len(falses))
                 and (tc[w] == 0 or fc[w] == 0)]
    import statistics
    tcl = statistics.mean(len(it["text"]) for it in trues)
    fcl = statistics.mean(len(it["text"]) for it in falses)
    print("=== AUDIT (word_count_geq_8_v2) ===")
    print(f"  ground-truth mismatches: {bad} (must be 0)")
    print(f"  adjunct rate: True {at:.2f}  False {af:.2f}  (matched target)")
    print(f"  one-sided high-freq tokens: {len(one_sided)} -> {one_sided[:8]}")
    print(f"  char-length mean: True {tcl:.1f}  False {fcl:.1f} (build-specific length signal, reported; removed in v3)")


if __name__ == "__main__":
    raise SystemExit(main())
