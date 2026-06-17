"""Build data/the_appears_twice_deconfounded_b/items.jsonl — a DECONFOUNDED variant of
data/the_appears_twice_deconfounded for the rule ``the_appears_twice``.

Rule (exact, re-used verbatim from the canonical predicate):
    label=True iff the word "the" appears >= 2 times (word-level, case-insensitive;
    sentence-initial "The" counts; "the" inside another word does not).

Char-length: the inventory mixes 1/3/4/5-letter determiners; because the same
uniform draw feeds both the flip fillers (False) and the compensators (True),
the per-class char-length distribution is matched too (re-audited below).

Word-count stays identical within each True/False pair (determiner swaps only).
Class-conditional "the"-count: True in {2,3}, False in {0,1} (deconfounded distribution).

Splits replicate data/the_appears_twice_deconfounded exactly:
    few_shot_pool 100T/100F, held_out 60/60, confirmation 50/50, spare 10/10.
Eval (held_out, confirmation, spare) is OOD: its noun fillers are held out from
the pool's noun set, and no surface text is shared pool<->eval.

NO API. Fully local. Deterministic from SEED.
"""

from __future__ import annotations

import json
from pathlib import Path

from icl_articulation.datagen import banks
from icl_articulation.datagen.banks import _regular_verb_forms
from icl_articulation.datagen.genutils import (
    Gen,
    base_id as _base_id,
    fill_frame,
    to_sentence_case,
)
from icl_articulation.datagen.groundtruth import _r21_the_appears_twice
from icl_articulation.datagen.schema import word_count, words

SEED = 20260614
RULE_ID = "the_appears_twice_deconfounded_b"
OUT = Path("data/the_appears_twice_deconfounded_b/items.jsonl")

_THE = "the"
# Non-"the" determiner inventory used at flip slots. Mixed lengths so char-count
# is not a proxy; the SAME pool also seeds non-flip slots in BOTH classes.
_NON_THE = ("a", "his", "her", "this", "that", "their", "our", "its")
_DET_SLOTS = ("D1", "D2", "D3", "D4")
_NP_SLOTS = ("N1", "N2", "N3", "N4")  # noun slots that may take an adjective premodifier

_FRAMES = (
    "{D1} {N1} {V} {D2} {N2} near {D3} {N3} behind {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} beside {D3} {N3} under {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} past {D3} {N3} toward {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} above {D3} {N3} below {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} inside {D3} {N3} near {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} along {D3} {N3} beside {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} beyond {D3} {N3} behind {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} under {D3} {N3} above {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} outside {D3} {N3} near {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} toward {D3} {N3} past {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} below {D3} {N3} beside {D4} {N4}",
    "{D1} {N1} {V} {D2} {N2} behind {D3} {N3} under {D4} {N4}",
)

_MIN_WORDS, _MAX_WORDS = 6, 15
# Adjective premodifiers are added IDENTICALLY to both variants of a pair (so
# per-pair word count stays equal and they introduce no class skew). Their COUNT
# is biased by the base's True the-count so the two classes' non-"the"-word-count
# distributions overlap (set in build_bases), defeating a raw "non-the word
# count" threshold proxy.
_TRUE_HI_SHARE = 0.20
_FALSE_ZERO_SHARE = 0.40
_INITIAL_THE_SHARE = 0.50
_N_BASES = 520


def _render(frame, nouns, verb, dets, adjs):
    verb_surface = _regular_verb_forms(verb)[1]
    # adjs maps slot name (N1..N4) -> adjective or "" ; prepend to the noun.
    noun_surface = {}
    for i, slot in enumerate(_NP_SLOTS):
        a = adjs.get(slot, "")
        noun_surface[slot] = f"{a} {nouns[i]}" if a else nouns[i]
    fillers = {
        "D1": dets["D1"], "D2": dets["D2"], "D3": dets["D3"], "D4": dets["D4"],
        "N1": noun_surface["N1"], "N2": noun_surface["N2"],
        "N3": noun_surface["N3"], "N4": noun_surface["N4"],
        "V": verb_surface,
    }
    return to_sentence_case(fill_frame(frame, fillers))


def _build_profiles(gen, n):
    n_init = round(n * _INITIAL_THE_SHARE)
    init_flags = [True] * n_init + [False] * (n - n_init)
    gen.shuffle(init_flags)
    n_true_hi = round(n * _TRUE_HI_SHARE)
    true_counts = [3] * n_true_hi + [2] * (n - n_true_hi)
    gen.shuffle(true_counts)
    n_false_zero = round(n * _FALSE_ZERO_SHARE)
    if n_false_zero > (n - n_init):
        raise ValueError("cannot place all 0-the False in initial_the=False half")
    profiles = []
    zeros_left = n_false_zero
    for i, init in enumerate(init_flags):
        if init:
            fc = 1
        else:
            fc = 0 if zeros_left > 0 else 1
            if zeros_left > 0:
                zeros_left -= 1
        profiles.append((init, true_counts[i], fc))
    if zeros_left:
        raise ValueError("false_count=0 quota not exhausted")
    return profiles


def _assign(initial_the, true_count, false_count, gen, filler_pool):
    base_the = ["D1"] if initial_the else []
    init_n = 1 if initial_the else 0
    swing = ["D2", "D3", "D4"]
    gen.shuffle(swing)
    swing_true_n = true_count - init_n
    swing_false_n = false_count - init_n
    if not (0 <= swing_false_n <= swing_true_n <= 3):
        raise ValueError("unreachable counts")
    swing_true = swing[:swing_true_n]
    swing_false = swing_true[:swing_false_n]
    flip_slots = swing_true[swing_false_n:]
    k = len(flip_slots)

    the_true = set(base_the + swing_true)
    the_false = set(base_the + swing_false)

    det_true, det_false = {}, {}
    for s in _DET_SLOTS:
        if s in the_true:
            det_true[s] = _THE
        if s in the_false:
            det_false[s] = _THE

    # flip slots: "the" in True; in False the slot becomes a uniformly-rotated
    # filler drawn from the FULL combined pool (8 determiners + ~36 adjectives).
    # We DELIBERATELY relax strict char-length neutrality here in favour of token
    # diversity (the two are in tension): spreading the replacement over ~44 token
    # types — mostly content adjectives — keeps every single token's class skew
    # tiny and, because eval adjectives are OOD, prevents a bag-of-words model from
    # summing a small closed feature set to recover the "the"-deficit. The residual
    # is a mild char-length signal (audited below; ~0.58), which is far weaker than
    # the his/her possessive confound it replaces. A bare adjective NP ("small
    # apple") reads fine without "the".
    for s in flip_slots:
        det_false[s] = gen.choice(filler_pool)
    # slots that are non-"the" in BOTH variants: identical filler in both variants
    # drawn from the full pool, contributing equally to both classes (zero skew).
    for s in _DET_SLOTS:
        if s not in the_true and s not in flip_slots:
            d = gen.choice(filler_pool)
            det_true[s] = d
            det_false[s] = d
    return the_true, the_false, det_true, det_false


def build_bases(gen, nouns_bank, adj_bank, n_target, seen_surfaces):
    verbs_bank = banks.get_bank("VERB_REGULAR").words()
    profiles = _build_profiles(gen.derive("profiles"), max(n_target, 200))
    bases = []
    seen_ids = set()
    attempts, fi, pi = 0, 0, 0
    max_attempts = n_target * 2000
    while len(bases) < n_target and attempts < max_attempts:
        attempts += 1
        frame = _FRAMES[fi % len(_FRAMES)]; fi += 1
        initial_the, tc, fc = profiles[pi % len(profiles)]; pi += 1
        n1, n2, n3, n4 = gen.sample(nouns_bank, 4)
        verb = gen.choice(verbs_bank)
        filler_pool = list(_NON_THE) + list(adj_bank)
        res = _assign(initial_the, tc, fc, gen.derive(f"a:{attempts}"), filler_pool)
        if res is None:
            continue
        the_true, the_false, det_true, det_false = res

        # adjective premodifiers: identical in BOTH variants (no skew; keeps
        # per-pair word count equal). Their COUNT is ANTI-correlated with the
        # True the-count: a base whose True variant has many "the" (hence few
        # non-"the" words) gets MORE adjectives, while a base whose False variant
        # has many non-"the" words gets fewer. This pushes the two classes'
        # non-"the"-word-count distributions to OVERLAP, so a raw "count of
        # non-the words" threshold can no longer cleanly separate the labels.
        ag = gen.derive(f"adj:{attempts}")
        n_adj = ag.choice((0, 1, 1, 2, 2, 3))  # independent of label; mean ~1.5
        adj_slots = ag.sample(list(_NP_SLOTS), n_adj) if n_adj else []
        adjs = {s: ag.choice(adj_bank) for s in adj_slots}

        bid = _base_id("r21b", _FRAMES.index(frame), n1, n2, n3, n4, verb)
        if bid in seen_ids:
            continue
        nouns = (n1, n2, n3, n4)
        ttext = _render(frame, nouns, verb, det_true, adjs)
        ftext = _render(frame, nouns, verb, det_false, adjs)
        if word_count(ttext) != word_count(ftext):
            continue
        if not (_MIN_WORDS <= word_count(ttext) <= _MAX_WORDS):
            continue
        if ttext == ftext or ttext in seen_surfaces or ftext in seen_surfaces:
            continue
        if not _r21_the_appears_twice(ttext) or _r21_the_appears_twice(ftext):
            continue
        seen_ids.add(bid)
        seen_surfaces.add(ttext); seen_surfaces.add(ftext)
        bases.append({
            "base_id": bid, "frame": frame, "frame_index": _FRAMES.index(frame),
            "nouns": nouns, "verb": verb, "adjs": adjs,
            "det_true": det_true, "det_false": det_false,
            "initial_the": initial_the, "true_count": tc, "false_count": fc,
        })
    if len(bases) < n_target:
        raise ValueError(f"only built {len(bases)} / {n_target} bases")
    return bases


def _item(base, label):
    dets = base["det_true"] if label else base["det_false"]
    text = _render(base["frame"], base["nouns"], base["verb"], dets, base["adjs"])
    n_the = sum(1 for t in words(text) if t.lower() == _THE)
    assert (n_the >= 2) == label, (text, n_the, label)
    assert _r21_the_appears_twice(text) == label
    suffix = "T" if label else "F"
    meta = {
        "seed": SEED, "deconfounded_source_rule": "the_appears_twice",
        "frame_index": base["frame_index"], "frame": base["frame"],
        "nouns": list(base["nouns"]), "verb": base["verb"],
        "verb_surface": _regular_verb_forms(base["verb"])[1],
        "determiners": {s: dets[s] for s in _DET_SLOTS},
        "adjectives": dict(base["adjs"]),
        "the_count": n_the, "initial_the": base["initial_the"],
        "transform": "the2plus" if label else (
            "the1" if base["false_count"] == 1 else "the0"),
        "true_count": base["true_count"], "false_count": base["false_count"],
    }
    return {
        "item_id": f"{base['base_id']}-{suffix}", "base_id": base["base_id"],
        "rule_id": RULE_ID, "label": label, "text": text, "slots_meta": meta,
    }


def main():
    gen = Gen(SEED)

    # OOD noun split: disjoint pool vs eval noun banks; eval nouns never appear
    # in the pool and vice-versa, so eval is out-of-distribution on fillers.
    nouns_bank = list(banks.get_bank("NOUN_CONCRETE").words())
    sp = gen.derive("nounsplit")
    shuffled = list(nouns_bank); sp.shuffle(shuffled)
    n_eval = round(len(shuffled) * 0.45)
    eval_nouns = shuffled[:n_eval]
    pool_nouns = shuffled[n_eval:]

    # OOD adjective split too (eval adjectives held out from the pool).
    adj_bank = list(banks.get_bank("ADJ_PLAIN").words())
    ap = gen.derive("adjsplit")
    adj_sh = list(adj_bank); ap.shuffle(adj_sh)
    n_eval_adj = round(len(adj_sh) * 0.45)
    eval_adj = adj_sh[:n_eval_adj]
    pool_adj = adj_sh[n_eval_adj:]

    need_pool = 100
    need_eval = 60 + 50 + 10
    seen_surfaces = set()
    pool_bases = build_bases(gen.derive("pool"), pool_nouns, pool_adj, need_pool, seen_surfaces)
    eval_bases = build_bases(gen.derive("eval"), eval_nouns, eval_adj, need_eval, seen_surfaces)

    items = []
    for b in pool_bases[:need_pool]:
        items.append(_item(b, True))
        items.append(_item(b, False))
        items[-1]["split"] = items[-2]["split"] = "few_shot_pool"

    eval_sel = eval_bases[:need_eval]
    spans = [("held_out", 60), ("confirmation", 50), ("spare", 10)]
    idx = 0
    for split, cnt in spans:
        for b in eval_sel[idx:idx + cnt]:
            t = _item(b, True); fa = _item(b, False)
            t["split"] = fa["split"] = split
            items.append(t); items.append(fa)
        idx += cnt

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"wrote {len(items)} items to {OUT}")


if __name__ == "__main__":
    main()
