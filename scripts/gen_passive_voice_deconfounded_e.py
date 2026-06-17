"""Build data/passive_voice_deconfounded_e/items.jsonl — passive_voice as an ABSTRACTION,
with NO single surface cue (suffix, auxiliary, "by", object-NP, subject animacy)
separating the classes.

passive_voice_deconfounded / deconfounded_d contrast "was {V-ed}" with "was {V-ing}", so the
participle suffix alone wins. A first maximal-diversity attempt re-introduced two
cheap cues: matching the object-NP rate forced every NP-bearing passive to be a
"by"-phrase (so "by" predicted passive at 0.75), and transitive objects pulled
length apart.

The structural fact: a passive's only post-verbal NP IS a by-phrase, so balancing
object-NP and suppressing "by" conflict. deconfounded_e drops transitive objects and
by-phrases entirely and two-sides the "-ed" cue a different way:

  PASSIVE (True), agentless, animate role subject (the patient):
    was/were + pp, got + pp, is being + pp, has been + pp, modal be + pp,
    is + pp + freq.            e.g. "The worker was inspected at dawn"
  ACTIVE (False), intransitive, same animate role subjects (the agent):
    simple past (-ed), progressive (-ing), present (-s), present perfect
    ("has been V-ing"), modal.  e.g. "The worker rested at dawn"

No item has an object NP or a "by"-phrase; both classes draw subjects from the
same animate role pool. So object-NP, "by", and subject animacy are all absent or
balanced. "-ed" is two-sided (passive participle vs active intransitive past),
"-ing" two-sided (active progressive vs passive "being"), was/were/is/has/been and
modals all appear in both classes. Selection matched-samples on word-count so
length carries no signal. The only reliable signal left is the be/get +
past-participle construction itself = the definition of voice. Transitive verbs
(passive) and intransitive verbs (active) are disjoint lexical sets, but both are
OOD-partitioned (eval verbs unseen), so the model cannot pass by memorizing verb
identity — it must recognise the construction.

Run:  python scripts/gen_passive_voice_deconfounded_e.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from icl_articulation.datagen import schema  # noqa: E402
from icl_articulation.datagen.genutils import fix_indefinite_articles  # noqa: E402
from icl_articulation.datagen.schema import make_item, write_items  # noqa: E402
from icl_articulation.datagen.generators.base import Gen  # noqa: E402

SEED = 53
RULE_ID = "passive_voice_deconfounded_e"

# transitive verbs that take an ANIMATE patient (so a role can be passivised):
# (pres3sg, past_participle, pres_participle)
TRANS = {
    "inspect":   ("inspects",   "inspected",   "inspecting"),
    "train":     ("trains",     "trained",     "training"),
    "hire":      ("hires",      "hired",       "hiring"),
    "praise":    ("praises",    "praised",     "praising"),
    "examine":   ("examines",   "examined",    "examining"),
    "promote":   ("promotes",   "promoted",    "promoting"),
    "question":  ("questions",  "questioned",  "questioning"),
    "interview": ("interviews", "interviewed", "interviewing"),
    "reward":    ("rewards",    "rewarded",    "rewarding"),
    "escort":    ("escorts",    "escorted",    "escorting"),
    "greet":     ("greets",     "greeted",     "greeting"),
    "assist":    ("assists",    "assisted",    "assisting"),
    "pay":       ("pays",       "paid",        "paying"),
    "teach":     ("teaches",    "taught",      "teaching"),
    "catch":     ("catches",    "caught",      "catching"),
}
# intransitive verbs (active only): (pres3sg, past, pres_participle)
INTRANS = {
    "rest":    ("rests",    "rested",    "resting"),
    "arrive":  ("arrives",  "arrived",   "arriving"),
    "work":    ("works",    "worked",    "working"),
    "wait":    ("waits",    "waited",    "waiting"),
    "travel":  ("travels",  "traveled",  "traveling"),
    "depart":  ("departs",  "departed",  "departing"),
    "pause":   ("pauses",   "paused",    "pausing"),
    "wander":  ("wanders",  "wandered",  "wandering"),
    "return":  ("returns",  "returned",  "returning"),
    "recover": ("recovers", "recovered", "recovering"),
    "proceed": ("proceeds", "proceeded", "proceeding"),
    "hesitate":("hesitates","hesitated", "hesitating"),
}
ROLES = [
    "worker", "teacher", "farmer", "painter", "baker", "gardener", "tailor",
    "cook", "nurse", "clerk", "porter", "driver", "artist", "student",
    "neighbor", "visitor",
]
TAILS = [
    "at dawn", "at noon", "by night", "in town", "near home", "outside",
    "nearby", "today", "indoors", "downtown", "by hand", "this week",
]
FREQ = ["daily", "weekly", "each day", "every week"]
MODALS = ["must", "should", "will", "can"]


def _split(values, seed, train_frac=0.6):
    vals = sorted({str(x).lower() for x in values})
    Gen(seed).shuffle(vals)
    cut = max(1, min(len(vals) - 1, round(len(vals) * train_frac)))
    return set(vals[:cut]), set(vals[cut:])


TR_TRAIN, TR_EVAL = _split(list(TRANS), SEED + 909)
IN_TRAIN, IN_EVAL = _split(list(INTRANS), SEED + 137)


def _trans_for(split):
    if split == "spare":
        return sorted(TRANS)
    return sorted(TR_TRAIN) if split == "few_shot_pool" else sorted(TR_EVAL)


def _intrans_for(split):
    if split == "spare":
        return sorted(INTRANS)
    return sorted(IN_TRAIN) if split == "few_shot_pool" else sorted(IN_EVAL)


SPLIT_SIZES = {
    "few_shot_pool": {True: 100, False: 100},
    "held_out": {True: 60, False: 60},
    "confirmation": {True: 50, False: 50},
    "spare": {True: 10, False: 10},
}


def _candidates(split, label):
    out, seen = [], set()
    g = Gen(SEED).derive(f"{split}:{label}")

    def add(text, family):
        text = fix_indefinite_articles(text)
        if text in seen:
            return
        wc = schema.word_count(text)
        if wc < 5 or wc > 8:
            return
        seen.add(text)
        out.append({"text": text, "family": family, "wc": wc, "cl": len(text)})

    if label:  # agentless PASSIVE
        for r in ROLES:
            for v in _trans_for(split):
                p3, pp, presp = TRANS[v]
                for tail in TAILS:
                    add(f"The {r} was {pp} {tail}", "pv_was")
                    add(f"The {r}s were {pp} {tail}", "pv_were")
                    add(f"The {r} got {pp} {tail}", "pv_got")
                    add(f"The {r} is being {pp} {tail}", "pv_being")
                    add(f"The {r} has been {pp} {tail}", "pv_perfect")
                    add(f"The {r} {g.choice(MODALS)} be {pp} {tail}", "pv_modal")
                add(f"The {r} is {pp} {g.choice(FREQ)}", "pv_present")
    else:  # intransitive ACTIVE
        for r in ROLES:
            for v in _intrans_for(split):
                p3, past, presp = INTRANS[v]
                for tail in TAILS:
                    # -ed-bearing active families (two-side the participle suffix):
                    # simple past (short) AND present-perfect "has V-ed" (aux + length).
                    add(f"The {r} {past} {tail}", "ac_past")
                    add(f"The {r} has {past} {tail}", "ac_perfect_ed")
                    add(f"The {r} was {presp} {tail}", "ac_prog")
                    add(f"The {r} is {presp} {tail}", "ac_presprog")
                    add(f"The {r}s were {presp} {tail}", "ac_prog_pl")
                    add(f"The {r} {g.choice(MODALS)} {v} {tail}", "ac_modal")
                add(f"The {r} {p3} {g.choice(FREQ)}", "ac_present")
    g.shuffle(out)
    return out


# family weights: bias the round-robin so the participle suffix (-ed) and the
# auxiliaries are ~two-sided rather than letting passive's all-participle families
# dominate. -ed-bearing active families get extra weight.
FAMILY_WEIGHT = {
    "ac_past": 3, "ac_perfect_ed": 2, "ac_prog": 1, "ac_presprog": 1,
    "ac_prog_pl": 1, "ac_modal": 1, "ac_present": 1,
    "pv_was": 2, "pv_got": 1, "pv_were": 1, "pv_being": 1, "pv_perfect": 2,
    "pv_modal": 1, "pv_present": 1,
}


def _pick(cands, n, gen):
    by = defaultdict(list)
    for x in cands:
        by[x["family"]].append(x)
    for k in by:
        gen.derive(k).shuffle(by[k])
    order = []
    for k in sorted(by):
        order += [k] * FAMILY_WEIGHT.get(k, 1)
    gen.derive("order").shuffle(order)
    picked, pos = [], defaultdict(int)
    while len(picked) < n:
        moved = False
        for k in order:
            if len(picked) >= n:
                break
            if pos[k] < len(by[k]):
                picked.append(by[k][pos[k]])
                pos[k] += 1
                moved = True
        if not moved:
            break
    return picked


def _select(split, n, gen, used):
    # Drop any candidate already emitted in an earlier split (spare draws from
    # all verbs and the eval splits share the eval pool, so texts collide across
    # splits). Filtering here keeps every split globally text-disjoint.
    cT = [x for x in _candidates(split, True) if x["text"] not in used]
    cF = [x for x in _candidates(split, False) if x["text"] not in used]
    bT, bF = defaultdict(list), defaultdict(list)
    for x in cT:
        bT[x["wc"]].append(x)
    for x in cF:
        bF[x["wc"]].append(x)
    wcs = sorted(set(bT) & set(bF))
    cap = {w: min(len(bT[w]), len(bF[w])) for w in wcs}
    total = sum(cap.values())
    if total < n:
        raise RuntimeError(f"{split}: matched wc capacity {total} < {n}")
    alloc = {w: min(cap[w], (cap[w] * n) // total) for w in wcs}
    short = n - sum(alloc.values())
    for w in sorted(wcs, key=lambda w: cap[w] - alloc[w], reverse=True):
        if short <= 0:
            break
        add = min(cap[w] - alloc[w], short)
        alloc[w] += add
        short -= add
    chT, chF = [], []
    for w in wcs:
        if alloc[w]:
            chT += _pick(bT[w], alloc[w], gen.derive(f"{split}:{w}:T"))
            chF += _pick(bF[w], alloc[w], gen.derive(f"{split}:{w}:F"))
    return chT, chF


def build():
    gen = Gen(SEED)
    items, index = [], 0
    used: set[str] = set()  # enforce global text uniqueness across splits
    for split, by_label in SPLIT_SIZES.items():
        chT, chF = _select(split, by_label[True], gen.derive(split), used)
        for label, chosen in ((True, chT), (False, chF)):
            for x in chosen:
                used.add(x["text"])
                bid = f"passive-deconfounded_e-{index:05d}"
                items.append(make_item(
                    item_id=bid, base_id=bid, rule_id=RULE_ID, label=label,
                    text=x["text"], split=split,
                    slots_meta={"family": x["family"], "has_postverbal_the_np": False,
                                "seed": SEED,
                                "verb_partition": "train" if split == "few_shot_pool" else "eval"},
                ))
                index += 1
    return items


def main():
    items = build()
    out_dir = REPO / "data" / "passive_voice_deconfounded_e"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "items.jsonl"
    write_items(items, out)
    print(f"wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
