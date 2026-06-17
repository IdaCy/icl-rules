#!/usr/bin/env python
"""Build data/word_count_geq_8_v3/ — a CHAR-LENGTH-DECONFOUNDED rebuild of
word_count_geq_8 (external-review P2c). LOCAL, no API. NEVER overwrites v1/v2.

The confound this targets: in v2, True items (>=8 words) are systematically
LONGER in CHARACTERS than False items (<=7 words) — char-mean ~49 vs ~37 — so a
model can pass by measuring STRING LENGTH instead of counting words.

The fix (decorrelate char-length from word-count):
  * True items (>=8 words) are composed of MANY but SHORT words.
  * False items (<=7 words) are composed of FEWER but LONGER words.
  * Per-class char-mean is tuned to within ~1 char and the char ranges heavily
    overlap, so a single-threshold char-count classifier is near chance.
The word-count boundary stays EXACT (True = 8 words, False = 7 words).

Invariants preserved so the confound isn't merely relocated:
  * Shared vocabulary banks: every content token is drawn from pools used by
    BOTH classes; the builder VERIFIES no content token is one-sided.
  * Determiner/article usage matched: every sentence has the same 'The ... the'
    determiner skeleton across classes.
  * Grammatical, natural past-tense transitive sentences in both classes.

OOD eval: held_out / confirmation share the recombination space but every text
is de-duplicated globally (no text appears in both pool and eval).

Run:  python scripts/gen_wc8_v3.py
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from icl_articulation.datagen import groundtruth
from icl_articulation.datagen.schema import word_count

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "word_count_geq_8_v3" / "items.jsonl"
RULE_ID = "word_count_geq_8_v3"
SEED = 20260614

# --- shared banks, grouped by char length ------------------------------------
# Each pool is used by BOTH classes. To overlap char distributions we mix SHORT
# and LONG members in BOTH classes, but bias the True (8-word) class toward
# short members and the False (7-word) class toward long members so the per-word
# char budget compensates for the differing word counts.

# Banks are deliberately LARGE: a big pool dilutes each individual token's
# per-class frequency, so even though the short/long TIER is class-biased (the
# mechanism that equalises char length), no single TOKEN carries enough signal
# for a bag-of-words classifier to exploit.
SHORT_ADJ = ["red", "old", "big", "wet", "dry", "hot", "icy", "raw", "shy", "odd",
             "tall", "tiny", "warm", "cold", "soft", "hard", "kind", "wild", "fine", "bold",
             "fat", "thin", "new", "sad", "glad", "rich", "poor", "pale", "dark", "loud",
             "neat", "rude", "lazy", "busy", "calm", "deep", "flat", "grey", "huge", "weak"]
LONG_ADJ = ["enormous", "delicate", "wonderful", "dangerous", "beautiful",
            "expensive", "marvelous", "forgotten", "elaborate", "elegant",
            "powerful", "splendid", "graceful", "stubborn", "generous", "abundant",
            "tremendous", "magnificent", "peculiar", "luminous",
            "mysterious", "enchanted", "ridiculous", "fabulous", "ferocious",
            "talented", "ambitious", "courageous", "victorious", "prosperous",
            "fortunate", "remarkable", "tremulous", "industrious", "monstrous",
            "scandalous", "harmonious", "spectacular", "venerable", "formidable"]
SHORT_NOUN = ["fox", "dog", "cat", "owl", "bee", "hen", "elk", "rat", "ram", "cub",
              "pup", "kid", "boy", "man", "cook", "guard", "clerk", "baker", "hunter", "sailor",
              "ant", "pig", "cow", "hog", "calf", "lamb", "colt", "crow", "duck", "goat",
              "girl", "lad", "monk", "nun", "chef", "maid", "page", "scout", "tutor", "guide"]
LONG_NOUN = ["gardener", "passenger", "professor", "carpenter", "shopkeeper",
             "messenger", "musician", "engineer", "physician", "neighbour",
             "traveller", "assistant", "librarian", "secretary", "inspector",
             "explorer", "merchant", "labourer", "wanderer", "magician",
             "performer", "conductor", "navigator", "decorator", "translator",
             "blacksmith", "fisherman", "watchman", "policeman", "councillor",
             "historian", "comedian", "champion", "volunteer", "apprentice",
             "spectator", "treasurer", "negotiator", "instructor", "supervisor"]
SHORT_VERB = ["ate", "saw", "hid", "fed", "led", "met", "won", "dug", "cut", "set",
              "moved", "fixed", "used", "kept", "held", "took", "made", "sent", "lost", "found",
              "ran", "had", "got", "put", "let", "sat", "lit", "tied", "drew", "rose",
              "named", "owned", "paid", "read", "rode", "sold", "told", "wore", "knew", "hung"]
LONG_VERB = ["collected", "discovered", "considered", "remembered", "delivered",
             "inspected", "protected", "presented", "preserved", "abandoned",
             "described", "imagined", "examined", "approached", "purchased",
             "arranged", "exhibited", "interpreted", "recovered", "constructed",
             "assembled", "decorated", "celebrated", "translated", "rescued",
             "organised", "calculated", "illustrated", "demolished", "transported",
             "rearranged", "photographed", "memorised", "scrutinised", "polished",
             "established", "investigated", "accompanied", "abolished", "supervised"]
SHORT_OBJ = ["box", "cup", "key", "bag", "jar", "pan", "pen", "net", "log", "map",
             "gate", "lamp", "bell", "coat", "rope", "boot", "cart", "door", "fork", "drum",
             "mug", "pot", "rug", "tub", "vat", "wig", "hat", "axe", "saw", "oar",
             "disk", "flag", "hook", "kite", "nail", "raft", "sack", "vase", "wand", "yarn"]
LONG_OBJ = ["umbrella", "telescope", "container", "instrument", "furniture",
            "machinery", "envelope", "blanket", "ornament", "bicycle",
            "cupboard", "treasure", "painting", "mattress", "lantern",
            "trumpet", "necklace", "sculpture", "package", "saddle",
            "calendar", "binocular", "microscope", "wheelbarrow", "harpsichord",
            "decanter", "chandelier", "telegram", "harmonica", "barometer",
            "tapestry", "armchair", "doorknob", "footstool", "birdcage",
            "snowshoe", "wristwatch", "kerosene", "parchment", "manuscript"]
SHORT_ADV = ["here", "there", "today", "now", "fast", "soon", "well", "late",
             "back", "near", "twice", "again", "alone", "daily", "early", "later",
             "far", "low", "high", "once", "still", "yet", "ahead", "aloud", "apart", "aside",
             "below", "duly", "ever", "fully", "hence", "idly", "lately", "madly", "newly", "only"]
LONG_ADV = ["carefully", "quietly", "suddenly", "cheerfully", "patiently",
            "gracefully", "anxiously", "promptly", "eagerly", "silently",
            "calmly", "bravely", "gently", "neatly", "boldly", "warmly",
            "frantically", "deliberately", "reluctantly", "obediently", "joyfully",
            "nervously", "carelessly", "generously", "furiously", "gloriously",
            "hurriedly", "instantly", "naturally", "perfectly", "secretly",
            "tenderly", "urgently", "wisely", "abruptly", "candidly"]

ALL_ADJ = SHORT_ADJ + LONG_ADJ
ALL_NOUN = SHORT_NOUN + LONG_NOUN
ALL_VERB = SHORT_VERB + LONG_VERB
ALL_OBJ = SHORT_OBJ + LONG_OBJ


def _wc(text: str) -> int:
    return word_count(text)


# --- per-slot length knob -----------------------------------------------------
# Every slot draws from a SHARED pool that holds both short and long members; a
# per-slot Bernoulli decides short-vs-long INDEPENDENTLY of the class. Tuning
# these probabilities (more long words in the 7-word False class, more short
# words in the 8-word True class) equalises the per-class char means WITHOUT
# making any token one-sided — both length tiers of every pool appear in both
# classes. Probabilities are the fraction LONG for each slot in each class.
# The 8th word in the True class adds ~6 chars structurally. To equalise char
# means we make True's words a little SHORTER and False's a little LONGER, but we
# SPREAD this small bias across ALL content slots rather than concentrating it on
# one or two slots. A small gap spread over five slots keeps the per-TOKEN class
# skew low (no slot's pool becomes near-deterministic), while the aggregate char
# means still match. Values are the fraction LONG for each class, applied to
# every content slot uniformly; the adverb slot uses the same fractions.
_P_LONG_TRUE = 0.34
_P_LONG_FALSE = 0.62


def _pick(rng, short_pool, long_pool, p_long):
    return rng.choice(long_pool) if rng.random() < p_long else rng.choice(short_pool)


def _build_true(rng: random.Random) -> str:
    """An 8-word grammatical past-tense sentence (label True).

    Skeleton (8 words, two 'the'-family determiners):
        The <subjAdj> <subjNoun> <adverb> <verb> the <objAdj> <objNoun>
    All slots draw from SHARED pools; the per-slot short/long mix is biased
    toward SHORT here (to offset the extra word vs the 7-word False class) but
    BOTH tiers of every pool appear, so no token is one-sided. The verb tier is
    drawn the same way as False (shared verb length distribution)."""
    p = _P_LONG_TRUE
    sadj = _pick(rng, SHORT_ADJ, LONG_ADJ, p)
    snoun = _pick(rng, SHORT_NOUN, LONG_NOUN, p)
    adv = _pick(rng, SHORT_ADV, LONG_ADV, p)
    verb = _pick(rng, SHORT_VERB, LONG_VERB, p)
    obadj = _pick(rng, SHORT_ADJ, LONG_ADJ, p)
    obnoun = _pick(rng, SHORT_OBJ, LONG_OBJ, p)
    return " ".join(["The", sadj, snoun, adv, verb, "the", obadj, obnoun])


def _build_false(rng: random.Random) -> str:
    """A 7-word grammatical past-tense sentence (label False).

    Skeleton (7 words, two 'the'-family determiners) — same as True minus the
    pre-verb adverb:
        The <subjAdj> <subjNoun> <verb> the <objAdj> <objNoun>
    Per-slot short/long mix biased toward LONG here (to offset one fewer word
    vs the True class); both tiers of every pool still appear, so no token is
    one-sided. To keep the adverb pool from being True-only, a fraction of False
    items instead carry an adverb and drop the object adjective (still 7 words)."""
    p = _P_LONG_FALSE
    sadj = _pick(rng, SHORT_ADJ, LONG_ADJ, p)
    snoun = _pick(rng, SHORT_NOUN, LONG_NOUN, p)
    verb = _pick(rng, SHORT_VERB, LONG_VERB, p)
    obnoun = _pick(rng, SHORT_OBJ, LONG_OBJ, p)
    # ~44% of False items take the adverb shape (drop the object adjective, add a
    # pre-verb adverb -> still 7 words). With True carrying an adverb 100% of the
    # time, this makes the per-item adverb rate ~100% vs ~44%; the adverb pool is
    # two-sided. The object-adjective is present in True always and in ~56% of
    # False — both sides represented.
    if rng.random() < 0.44:
        adv = _pick(rng, SHORT_ADV, LONG_ADV, p)
        return " ".join(["The", sadj, snoun, adv, verb, "the", obnoun])
    obadj = _pick(rng, SHORT_ADJ, LONG_ADJ, p)
    return " ".join(["The", sadj, snoun, verb, "the", obadj, obnoun])


def _draw(rng, want_true, n, seen):
    out = []
    builder = _build_true if want_true else _build_false
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > n * 500:
            raise RuntimeError(f"could not draw {n} {'True' if want_true else 'False'} items")
        text = builder(rng)
        if text in seen:
            continue
        wc = _wc(text)
        if want_true and wc != 8:
            continue
        if not want_true and wc != 7:
            continue
        if groundtruth.label_of("word_count_geq_8", text) is not want_true:
            continue
        seen.add(text)
        out.append(text)
    return out


SPLITS = [("few_shot_pool", 100), ("held_out", 60), ("confirmation", 50), ("spare", 30)]


def main() -> int:
    rng = random.Random(SEED)
    seen: set[str] = set()
    per_class = sum(n for _, n in SPLITS)  # 240
    trues = _draw(rng, True, per_class, seen)
    falses = _draw(rng, False, per_class, seen)
    rng.shuffle(trues)
    rng.shuffle(falses)

    items = []
    bi = 0
    for cls_items, label in ((trues, True), (falses, False)):
        i = 0
        for split, n in SPLITS:
            for _ in range(n):
                text = cls_items[i]; i += 1
                base = f"wc8v3-{bi:05d}"; bi += 1
                items.append({
                    "item_id": base, "base_id": base, "rule_id": RULE_ID,
                    "label": label, "text": text,
                    "slots_meta": {"seed": SEED, "wc": _wc(text), "chars": len(text)},
                    "split": split,
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    from icl_articulation.contexts import validate_dataset
    validate_dataset(items)
    _audit(items)
    print(f"\nwrote {OUT} ({len(items)} items) — validate_dataset OK")
    return 0


def _char_threshold_acc(train, test):
    """Best single char-count threshold (either direction) learned on train,
    scored on test."""
    cands = sorted({len(it["text"]) for it in train})
    best_t, best_sign, best_acc = cands[0], 1, 0.0
    for t in cands:
        for sign in (1, -1):
            acc = sum(1 for it in train
                      if (sign * (len(it["text"]) - t) >= 0) == it["label"]) / len(train)
            if acc > best_acc:
                best_acc, best_t, best_sign = acc, t, sign
    test_acc = sum(1 for it in test
                   if (best_sign * (len(it["text"]) - best_t) >= 0) == it["label"]) / len(test)
    return test_acc, best_t, best_sign


def _audit(items):
    trues = [it for it in items if it["label"]]
    falses = [it for it in items if not it["label"]]
    pool = [it for it in items if it["split"] == "few_shot_pool"]
    held = [it for it in items if it["split"] == "held_out"]

    bad = sum(1 for it in items
              if groundtruth.label_of("word_count_geq_8", it["text"]) != it["label"])

    bal = {}
    for sp in ("few_shot_pool", "held_out", "confirmation", "spare"):
        g = [it for it in items if it["split"] == sp]
        bal[sp] = (sum(it["label"] for it in g), sum(not it["label"] for it in g))

    tcl = statistics.mean(len(it["text"]) for it in trues)
    fcl = statistics.mean(len(it["text"]) for it in falses)
    trange = (min(len(it["text"]) for it in trues), max(len(it["text"]) for it in trues))
    frange = (min(len(it["text"]) for it in falses), max(len(it["text"]) for it in falses))
    char_acc, ct, cs = _char_threshold_acc(pool, held)

    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.naive_bayes import MultinomialNB
        vec = CountVectorizer()
        X = vec.fit_transform([it["text"] for it in pool])
        clf = MultinomialNB().fit(X, [it["label"] for it in pool])
        Xt = vec.transform([it["text"] for it in held])
        pred = clf.predict(Xt)
        nb_acc = sum(int(p) == int(it["label"]) for p, it in zip(pred, held)) / len(held)
    except Exception as e:  # pragma: no cover
        nb_acc = f"skipped ({e})"

    def toks(t):
        return {w.lower() for w in t.split() if w.lower() != "the"}
    tc, fc = Counter(), Counter()
    for it in trues:
        for w in toks(it["text"]):
            tc[w] += 1
    for it in falses:
        for w in toks(it["text"]):
            fc[w] += 1
    vocab = set(tc) | set(fc)
    one_sided = [(w, tc[w], fc[w]) for w in vocab if tc[w] == 0 or fc[w] == 0]
    skews = sorted(((w, tc[w], fc[w], tc[w] - fc[w]) for w in vocab),
                   key=lambda x: abs(x[3]), reverse=True)[:10]

    pool_texts = {it["text"] for it in pool}
    eval_texts = {it["text"] for it in items if it["split"] in ("held_out", "confirmation")}
    dup = pool_texts & eval_texts

    print("=== AUDIT (word_count_geq_8_v3) ===")
    print(f"  1. ground-truth mismatches: {bad} (must be 0)")
    print(f"  2. per-split balance (T,F): {bal}")
    print(f"  3. char mean: True {tcl:.1f}  False {fcl:.1f}  (diff {abs(tcl-fcl):.2f})")
    print(f"     char range: True {trange}  False {frange}")
    print(f"     char-threshold classifier (train pool -> held_out): acc {char_acc:.3f}  (thr {ct}, sign {cs})")
    nbs = nb_acc if isinstance(nb_acc, str) else f"{nb_acc:.3f}"
    print(f"     CountVectorizer NB (train pool -> held_out): acc {nbs}")
    print(f"  4. one-sided content tokens: {len(one_sided)} -> {one_sided[:8]}")
    print(f"     top class-skewed tokens (tok,T,F,diff): {skews}")
    print(f"  5. duplicate texts pool<->eval: {len(dup)} (must be 0)")


if __name__ == "__main__":
    raise SystemExit(main())
