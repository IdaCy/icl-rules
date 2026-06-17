"""Build data/passive_voice_deconfounded_c/items.jsonl — passive_voice WITHOUT the structural
object/auxiliary shortcut that contaminated passive_voice_deconfounded_b.

THE BUG IN deconfounded_b: every ACTIVE clause was transitive with an overt post-verbal
"the {object}" NP, and almost no PASSIVE had a post-verbal "the"-NP. So a shallow
non-voice cue ("is there a 'the {noun}' after the verb? -> active") scored ~0.92
on held_out, and combined with passive-only auxiliaries (got/being/been) the
classes were ~perfectly separable by surface STRUCTURE without voice reasoning.

THE FIX HERE: balance the post-verbal "the {noun}" NP and diversify auxiliaries
so that NO non-voice surface feature predicts the label. The residual signal is
be/get + past-participle MORPHOLOGY ONLY (which IS the definition of voice).

  ACTIVE (False), ~50/50:
    * transitive  : "The worker repaired the visitor in town"  (post-verbal the-NP)
    * intransitive: "The worker rested at dawn"                (NO post-verbal NP)
  PASSIVE (True), ~50/50:
    * agentless   : "The worker was repaired at dawn"          (NO post-verbal the-NP)
    * by-phrase   : "The worker was repaired by the visitor"   (post-verbal the-NP)

  => fraction with a post-verbal "the {noun}" NP is ~equal across True/False, so a
     "post-verbal-the -> active" classifier ~= 0.50.

  Auxiliaries are spread two-sided / diluted: actives use simple-past, present-s,
  past-progressive ("was V-ing"), present-perfect ("has V-ed"); passives use
  was/were V-ed, got V-ed, is being V-ed, has been V-ed. "was" appears in both
  classes (active past-progressive vs passive be-passive); no single token
  (was/were/by/got/being/been) is a strong one-sided marker.

Run:  python scripts/gen_passive_voice_deconfounded_c.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from icl_articulation.datagen import schema  # noqa: E402
from icl_articulation.datagen.genutils import fix_indefinite_articles  # noqa: E402
from icl_articulation.datagen.schema import make_item, write_items  # noqa: E402
from icl_articulation.datagen.generators.base import Gen  # noqa: E402


def _split_values(values, seed, *, train_frac=0.6):
    vals = sorted({str(v).lower() for v in values})
    if len(vals) < 2:
        raise RuntimeError(f"need at least two values to partition, got {vals}")
    g = Gen(seed)
    g.shuffle(vals)
    cut = max(1, min(len(vals) - 1, round(len(vals) * train_frac)))
    return set(vals[:cut]), set(vals[cut:])


SEED = 7
RULE_ID = "passive_voice_deconfounded_c"

# --- TRANSITIVE verbs: usable as active-transitive AND passive (be/get V-ed). ---
TRANS_REGULAR = [
    "paint", "clean", "wash", "cook", "plant", "polish", "mend", "pack",
    "repair", "inspect", "move", "prepare",
]
# verb -> (present_3sg, simple_past, past_participle, pres_part)
TRANS_IRREGULAR = {
    "build":  ("builds",  "built",  "built",   "building"),
    "write":  ("writes",  "wrote",  "written", "writing"),
    "break":  ("breaks",  "broke",  "broken",  "breaking"),
    "take":   ("takes",   "took",   "taken",   "taking"),
    "give":   ("gives",   "gave",   "given",   "giving"),
    "throw":  ("throws",  "threw",  "thrown",  "throwing"),
    "drive":  ("drives",  "drove",  "driven",  "driving"),
    "draw":   ("draws",   "drew",   "drawn",   "drawing"),
    "sell":   ("sells",   "sold",   "sold",    "selling"),
    "hide":   ("hides",   "hid",    "hidden",  "hiding"),
    "choose": ("chooses", "chose",  "chosen",  "choosing"),
    "carry":  ("carries", "carried", "carried", "carrying"),
}
TRANS_IRREGULAR_VERBS = list(TRANS_IRREGULAR.keys())
ALL_TRANS = TRANS_REGULAR + TRANS_IRREGULAR_VERBS

# --- INTRANSITIVE verbs: used ONLY in intransitive ACTIVE frames (no object;
# cannot be passivized, so voice stays unambiguous).
# verb -> (pres_3sg, past, pres_part, past_participle)  [participle for "has V-ed"]
INTRANS = {
    "rest":    ("rests",    "rested",   "resting",   "rested"),
    "arrive":  ("arrives",  "arrived",  "arriving",  "arrived"),
    "work":    ("works",    "worked",   "working",   "worked"),
    "wait":    ("waits",    "waited",   "waiting",   "waited"),
    "travel":  ("travels",  "traveled", "traveling", "traveled"),
    "depart":  ("departs",  "departed", "departing", "departed"),
    "smile":   ("smiles",   "smiled",   "smiling",   "smiled"),
    "pause":   ("pauses",   "paused",   "pausing",   "paused"),
    "sleep":   ("sleeps",   "slept",    "sleeping",  "slept"),
    "speak":   ("speaks",   "spoke",    "speaking",  "spoken"),
    "sit":     ("sits",     "sat",      "sitting",   "sat"),
    "stay":    ("stays",    "stayed",   "staying",   "stayed"),
    "wander":  ("wanders",  "wandered", "wandering", "wandered"),
    "vanish":  ("vanishes", "vanished", "vanishing", "vanished"),
}
INTRANS_VERBS = list(INTRANS.keys())

ROLE_WORDS = [
    "worker", "teacher", "farmer", "painter", "baker", "gardener", "tailor",
    "cook", "nurse", "clerk", "porter", "driver", "artist", "student",
    "neighbor", "visitor",
]
ROLE_PLURALS = {r: r + "s" for r in ROLE_WORDS}

TAILS = [
    "near home", "at dawn", "at noon", "by night", "in town", "outside",
    "nearby", "today", "indoors", "downtown",
]


# --- regular-verb morphology helpers -------------------------------------------
def _past(v: str) -> str:
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return f"{v[:-1]}ied"
    return f"{v}d" if v.endswith("e") else f"{v}ed"


def _ing(v: str) -> str:
    if v.endswith("e") and v != "see":
        return f"{v[:-1]}ing"
    return f"{v}ing"


def _pres_s(v: str) -> str:
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return f"{v[:-1]}ies"
    if v.endswith(("s", "x", "z", "ch", "sh")):
        return f"{v}es"
    return f"{v}s"


# transitive verb-form accessors
def vt_present(v):
    return TRANS_IRREGULAR[v][0] if v in TRANS_IRREGULAR else _pres_s(v)


def vt_past(v):
    return TRANS_IRREGULAR[v][1] if v in TRANS_IRREGULAR else _past(v)


def vt_partic(v):
    return TRANS_IRREGULAR[v][2] if v in TRANS_IRREGULAR else _past(v)


def vt_ing(v):
    return TRANS_IRREGULAR[v][3] if v in TRANS_IRREGULAR else _ing(v)


# intransitive verb-form accessors
def vi_present(v):
    return INTRANS[v][0]


def vi_past(v):
    return INTRANS[v][1]


def vi_ing(v):
    return INTRANS[v][2]


def vi_partic(v):
    return INTRANS[v][3]


# --- templates -----------------------------------------------------------------
# Each returns (text, shape, has_postverbal_the_np).

# PASSIVE (True) — agentless (NO post-verbal the-NP). TRANS verbs.
def p_was_passive(subj, subj_pl, other, verb, tail):
    return f"The {subj} was {vt_partic(verb)} {tail}", "passive_was", False


def p_were_passive(subj, subj_pl, other, verb, tail):
    return f"The {subj_pl} were {vt_partic(verb)} {tail}", "passive_were", False


def p_got_passive(subj, subj_pl, other, verb, tail):
    return f"The {subj} got {vt_partic(verb)} {tail}", "passive_got", False


def p_being_passive(subj, subj_pl, other, verb, tail):
    return f"The {subj} is being {vt_partic(verb)} {tail}", "passive_being", False


def p_has_been_passive(subj, subj_pl, other, verb, tail):
    return f"The {subj} has been {vt_partic(verb)} {tail}", "passive_has_been", False


# PASSIVE (True) — WITH by-phrase (HAS post-verbal "by the {other}" the-NP).
def p_was_by(subj, subj_pl, other, verb, tail):
    return f"The {subj} was {vt_partic(verb)} by the {other}", "passive_was_by", True


def p_were_by(subj, subj_pl, other, verb, tail):
    return f"The {subj_pl} were {vt_partic(verb)} by the {other}", "passive_were_by", True


def p_got_by(subj, subj_pl, other, verb, tail):
    return f"The {subj} got {vt_partic(verb)} by the {other}", "passive_got_by", True


def p_being_by(subj, subj_pl, other, verb, tail):
    return f"The {subj} is being {vt_partic(verb)} by the {other}", "passive_being_by", True


def p_has_been_by(subj, subj_pl, other, verb, tail):
    return f"The {subj} has been {vt_partic(verb)} by the {other}", "passive_has_been_by", True


# ACTIVE (False) — transitive (HAS post-verbal "the {other}" the-NP). TRANS verbs.
def a_past_obj(subj, subj_pl, other, verb, tail):
    return f"The {subj} {vt_past(verb)} the {other} {tail}", "active_past_obj", True


def a_pres_obj(subj, subj_pl, other, verb, tail):
    return f"The {subj} {vt_present(verb)} the {other} {tail}", "active_pres_obj", True


def a_was_ing_obj(subj, subj_pl, other, verb, tail):
    return f"The {subj} was {vt_ing(verb)} the {other} {tail}", "active_was_ing_obj", True


def a_were_ing_obj(subj, subj_pl, other, verb, tail):
    return f"The {subj_pl} were {vt_ing(verb)} the {other}", "active_were_ing_obj", True


def a_has_obj(subj, subj_pl, other, verb, tail):
    return f"The {subj} has {vt_partic(verb)} the {other} {tail}", "active_has_obj", True


# ACTIVE (False) — intransitive (NO post-verbal NP). INTRANS verbs only.
def a_past_intr(subj, subj_pl, other, verb, tail):
    return f"The {subj} {vi_past(verb)} {tail}", "active_past_intr", False


def a_pres_intr(subj, subj_pl, other, verb, tail):
    return f"The {subj} {vi_present(verb)} {tail}", "active_pres_intr", False


def a_was_ing_intr(subj, subj_pl, other, verb, tail):
    return f"The {subj} was {vi_ing(verb)} {tail}", "active_was_ing_intr", False


def a_were_ing_intr(subj, subj_pl, other, verb, tail):
    return f"The {subj_pl} were {vi_ing(verb)} {tail}", "active_were_ing_intr", False


def a_has_intr(subj, subj_pl, other, verb, tail):
    # present perfect intransitive: "The {subj} has {V-pp} {tail}"
    return f"The {subj} has {vi_partic(verb)} {tail}", "active_has_intr", False


PASSIVE_AGENTLESS = [p_was_passive, p_were_passive, p_got_passive, p_being_passive, p_has_been_passive]
PASSIVE_BYPHRASE = [p_was_by, p_were_by, p_got_by, p_being_by, p_has_been_by]
ACTIVE_TRANS = [a_past_obj, a_pres_obj, a_was_ing_obj, a_were_ing_obj, a_has_obj]
ACTIVE_INTRANS = [a_past_intr, a_pres_intr, a_was_ing_intr, a_were_ing_intr, a_has_intr]


def _interleave(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(x)
        out.append(y)
    return out


# Interleaving agentless<->by-phrase and trans<->intrans makes the post-verbal
# the-NP ~50/50 within each class, and matches each aux form across both sub-groups.
PASSIVE_TEMPLATES = _interleave(PASSIVE_AGENTLESS, PASSIVE_BYPHRASE)
ACTIVE_TEMPLATES = _interleave(ACTIVE_TRANS, ACTIVE_INTRANS)
INTRANS_TEMPLATES = {t.__name__ for t in ACTIVE_INTRANS}

# OOD partition: held_out/confirmation use eval verbs+roles only.
_tr_reg, _ev_reg = _split_values(TRANS_REGULAR, SEED + 909, train_frac=0.6)
_tr_irr, _ev_irr = _split_values(TRANS_IRREGULAR_VERBS, SEED + 451, train_frac=0.6)
_tr_int, _ev_int = _split_values(INTRANS_VERBS, SEED + 137, train_frac=0.6)
train_trans = _tr_reg | _tr_irr
eval_trans = _ev_reg | _ev_irr
train_intrans = _tr_int
eval_intrans = _ev_int
train_roles, eval_roles = _split_values(ROLE_WORDS, SEED + 313, train_frac=0.6)

SPLIT_SIZES = {
    "few_shot_pool": {True: 100, False: 100},
    "held_out": {True: 60, False: 60},
    "confirmation": {True: 50, False: 50},
    "spare": {True: 10, False: 10},
}


def build():
    items = []
    used_texts: set[str] = set()
    index = 0
    tmpl_counter: dict[tuple, int] = {}
    cand_iters: dict[tuple, list] = {}
    cand_pos: dict[tuple, int] = {}

    def trans_verbs(split):
        if split == "spare":
            return sorted(ALL_TRANS)
        return sorted(train_trans) if split == "few_shot_pool" else sorted(eval_trans)

    def intrans_verbs(split):
        if split == "spare":
            return sorted(INTRANS_VERBS)
        return sorted(train_intrans) if split == "few_shot_pool" else sorted(eval_intrans)

    def roles(split):
        if split == "spare":
            return sorted(ROLE_WORDS)
        return sorted(train_roles) if split == "few_shot_pool" else sorted(eval_roles)

    def candidates(split, tmpl_name):
        ck = (split, tmpl_name)
        if ck not in cand_iters:
            verbs = intrans_verbs(split) if tmpl_name in INTRANS_TEMPLATES else trans_verbs(split)
            rls = roles(split)
            combos = [
                (subj, verb, tail, other)
                for subj in rls
                for verb in verbs
                for tail in TAILS
                for other in rls
                if other != subj
            ]
            Gen(SEED).derive(f"{split}:{tmpl_name}").shuffle(combos)
            cand_iters[ck] = combos
            cand_pos[ck] = 0
        return ck

    def make_text(label, split):
        templates = PASSIVE_TEMPLATES if label else ACTIVE_TEMPLATES
        key = (split, label)
        t_idx = tmpl_counter.get(key, 0)
        tmpl = templates[t_idx % len(templates)]
        tmpl_counter[key] = t_idx + 1
        ck = candidates(split, tmpl.__name__)
        combos = cand_iters[ck]
        while cand_pos[ck] < len(combos):
            subj, verb, tail, other = combos[cand_pos[ck]]
            cand_pos[ck] += 1
            subj_pl = ROLE_PLURALS[subj]
            text, shape, has_np = tmpl(subj, subj_pl, other, verb, tail)
            text = fix_indefinite_articles(text)
            if text in used_texts:
                continue
            wc = schema.word_count(text)
            if wc < 5 or wc > 8:
                continue
            used_texts.add(text)
            transitivity = (
                "intransitive" if tmpl.__name__ in INTRANS_TEMPLATES
                else ("passive" if label else "transitive")
            )
            return text, {
                "shape": shape,
                "template": tmpl.__name__,
                "verb_base": verb,
                "tail": tail,
                "subject": subj,
                "other": other,
                "has_postverbal_the_np": has_np,
                "transitivity": transitivity,
                "seed": SEED,
                "verb_partition": "train" if split == "few_shot_pool" else "eval",
                "role_partition": "train" if split == "few_shot_pool" else "eval",
            }
        raise RuntimeError(f"ran out of unique items: {split} label={label} tmpl={tmpl.__name__}")

    for split, by_label in SPLIT_SIZES.items():
        for label in (True, False):
            for _ in range(by_label[label]):
                text, meta = make_text(label, split)
                bid = f"passive-deconfounded_c-{index:05d}"
                items.append(
                    make_item(
                        item_id=bid,
                        base_id=bid,
                        rule_id=RULE_ID,
                        label=label,
                        text=text,
                        slots_meta=meta,
                        split=split,
                    )
                )
                index += 1
    return items


def main():
    items = build()
    out_dir = REPO / "data" / "passive_voice_deconfounded_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "items.jsonl"
    write_items(items, out)
    print(f"wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
