"""Build data/passive_voice_deconfounded_b/items.jsonl — a DECONFOUNDED passive_voice dataset.

The old passive_voice_deconfounded made every True "The {role} was {V}ed {tail}" and every
False "The {role} was {V}ing {tail}", so the -ed/-ing suffix (and "was") perfectly
separated the classes. Here we diversify surface forms so that:

  * simple-past ACTIVE clauses end in -ed (puts -ed into the False class),
  * get-passives are passive WITHOUT "was",
  * progressive passives ("is being cleaned") put -ing into the True class,
  * past-progressive ACTIVE ("was ...ing the X") keeps "was"/-ing in False.

Result: neither the verb suffix nor "was" separates True from False. Voice is the
ONLY thing that determines the label; every template is unambiguous.

Run:  python scripts/gen_passive_voice_deconfounded_b.py
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


# Inlined from scripts/gen_deconfounded_variants.py (importing it pulls in modules absent
# from this checkout). Identical behaviour, so the OOD split matches the family.
def _split_values(values, seed, *, train_frac=0.55):
    vals = sorted({str(v).lower() for v in values})
    if len(vals) < 2:
        raise RuntimeError(f"need at least two values to partition, got {vals}")
    g = Gen(seed)
    g.shuffle(vals)
    cut = max(1, min(len(vals) - 1, round(len(vals) * train_frac)))
    return set(vals[:cut]), set(vals[cut:])


def _take_cycle(pool, index):
    if not pool:
        raise RuntimeError("empty pool")
    return pool[index % len(pool)]

SEED = 7
RULE_ID = "passive_voice_deconfounded_b"

# --- shared content banks (reused from _build_passive_voice_deconfounded) ---------------
# Regular verbs (-ed past AND -ed past-participle) reused from the original bank,
# plus IRREGULAR verbs whose past/participle do NOT end in -ed. Mixing irregulars
# into BOTH classes is what truly breaks the -ed/-ing suffix confound: a passive
# can read "was written"/"got broken" (no -ed) and an active "wrote"/"broke".
VERBS = [
    "paint", "clean", "wash", "cook", "plant", "study", "copy", "count",
    "move", "prepare", "polish", "mend", "sort", "pack", "fold", "dry",
    "repair", "inspect",
]

# verb -> (present_3sg, simple_past, past_participle, present_participle)
IRREGULAR = {
    "build":  ("builds",  "built",   "built",    "building"),
    "write":  ("writes",  "wrote",   "written",  "writing"),
    "break":  ("breaks",  "broke",   "broken",   "breaking"),
    "take":   ("takes",   "took",    "taken",    "taking"),
    "make":   ("makes",   "made",    "made",     "making"),
    "give":   ("gives",   "gave",    "given",    "giving"),
    "throw":  ("throws",  "threw",   "thrown",   "throwing"),
    "drive":  ("drives",  "drove",   "driven",   "driving"),
    "draw":   ("draws",   "drew",    "drawn",    "drawing"),
    "sell":   ("sells",   "sold",    "sold",     "selling"),
    "hide":   ("hides",   "hid",     "hidden",   "hiding"),
    "choose": ("chooses", "chose",   "chosen",   "choosing"),
}
IRREGULAR_VERBS = list(IRREGULAR.keys())
ALL_VERBS = VERBS + IRREGULAR_VERBS
ROLE_WORDS = [
    "worker", "teacher", "farmer", "painter", "baker", "gardener", "tailor",
    "cook", "nurse", "clerk", "porter", "driver", "artist", "student",
    "neighbor", "visitor",
]
ROLE_PLURALS = {
    "worker": "workers", "teacher": "teachers", "farmer": "farmers",
    "painter": "painters", "baker": "bakers", "gardener": "gardeners",
    "tailor": "tailors", "cook": "cooks", "nurse": "nurses", "clerk": "clerks",
    "porter": "porters", "driver": "drivers", "artist": "artists",
    "student": "students", "neighbor": "neighbors", "visitor": "visitors",
}
TAILS = [
    "near home", "at dawn", "at noon", "by night", "in town", "outside",
    "nearby", "today", "indoors", "downtown",
]
# object/thing nouns (shared bank). These are the PASSIVE SUBJECT (the thing that
# undergoes the action) AND the ACTIVE OBJECT, so they appear in BOTH classes —
# nothing about "table/fence/..." can mark a class. Likewise ROLE_WORDS are the
# ACTIVE SUBJECT and the PASSIVE AGENT ("by the worker"), so they too are
# two-sided. This removes the structural "active has an object noun" confound.
OBJECTS = [
    "table", "fence", "shirt", "wall", "floor", "garden", "report", "engine",
    "window", "letter", "chair", "roof", "boat", "kitchen", "shelf", "lamp",
]
OBJECT_PLURALS = {
    "table": "tables", "fence": "fences", "shirt": "shirts", "wall": "walls",
    "floor": "floors", "garden": "gardens", "report": "reports",
    "engine": "engines", "window": "windows", "letter": "letters",
    "chair": "chairs", "roof": "roofs", "boat": "boats", "kitchen": "kitchens",
    "shelf": "shelves", "lamp": "lamps",
}


def past(v: str) -> str:
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return f"{v[:-1]}ied"
    return f"{v}d" if v.endswith("e") else f"{v}ed"


def ing(v: str) -> str:
    if v.endswith("e") and v != "see":
        return f"{v[:-1]}ing"
    return f"{v}ing"


def pres_s(v: str) -> str:
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return f"{v[:-1]}ies"
    if v.endswith(("s", "x", "z", "ch", "sh")):
        return f"{v}es"
    return f"{v}s"


# --- unified verb-form accessors (regular fall back to -ed rules) --------------
def vf_present(v: str) -> str:
    return IRREGULAR[v][0] if v in IRREGULAR else pres_s(v)


def vf_past(v: str) -> str:  # simple past (active)
    return IRREGULAR[v][1] if v in IRREGULAR else past(v)


def vf_partic(v: str) -> str:  # past participle (passive)
    return IRREGULAR[v][2] if v in IRREGULAR else past(v)


def vf_ing(v: str) -> str:
    return IRREGULAR[v][3] if v in IRREGULAR else ing(v)


# --- templates ----------------------------------------------------------------
# CRITICAL for deconfounding: the grammatical SUBJECT must not reveal the class.
# So each template takes a pre-chosen `subj` / `subj_pl` and a second noun `other`
# (the active object or the passive agent). Both `subj` and `other` are drawn from
# the SAME shared subject bank (people + things), independent of class. A "table"
# is as likely to head a True item as a False one, and "the worker" appears as
# subject in both classes too. Thus neither the subject noun, the second noun, nor
# the verb suffix marks the class — only the voice construction does.
#
# PASSIVE (True): subject undergoes the action; optional agent "by the {other}".
def t_be_passive_by(subj, subj_pl, other, verb, tail):
    # "The {subj} was {V-pp} by the {other}"
    return f"The {subj} was {vf_partic(verb)} by the {other}", "passive_was_by"


def t_plural_be_passive(subj, subj_pl, other, verb, tail):
    # "The {subjs} were {V-pp} {tail}" (no "was")
    return f"The {subj_pl} were {vf_partic(verb)} {tail}", "passive_were_tail"


def t_get_passive(subj, subj_pl, other, verb, tail):
    # "The {subj} got {V-pp} {tail}" (passive WITHOUT "was")
    return f"The {subj} got {vf_partic(verb)} {tail}", "passive_got"


def t_progressive_passive(subj, subj_pl, other, verb, tail):
    # "The {subj} is being {V-pp} {tail}" (-ing in "being")
    return f"The {subj} is being {vf_partic(verb)} {tail}", "passive_progressive"


def t_has_been_passive(subj, subj_pl, other, verb, tail):
    # perfect passive: "The {subj} has been {V-pp} {tail}" (no "was")
    return f"The {subj} has been {vf_partic(verb)} {tail}", "passive_has_been"


def t_plural_get_passive(subj, subj_pl, other, verb, tail):
    # plural get-passive: "The {subjs} got {V-pp} {tail}"
    return f"The {subj_pl} got {vf_partic(verb)} {tail}", "passive_plural_got"


# ACTIVE (False): subject performs the action on the {other}. Mixed verb forms.
def f_simple_past(subj, subj_pl, other, verb, tail):
    # "The {subj} {V-past} the {other} {tail}"
    return f"The {subj} {vf_past(verb)} the {other} {tail}", "active_simple_past"


def f_simple_present(subj, subj_pl, other, verb, tail):
    # "The {subj} {V-s} the {other} {tail}"
    return f"The {subj} {vf_present(verb)} the {other} {tail}", "active_simple_present"


def f_past_progressive(subj, subj_pl, other, verb, tail):
    # "The {subj} was {V-ing} the {other} {tail}"
    return f"The {subj} was {vf_ing(verb)} the {other} {tail}", "active_past_progressive"


def f_present_progressive(subj, subj_pl, other, verb, tail):
    # "The {subj} is {V-ing} the {other} {tail}"
    return f"The {subj} is {vf_ing(verb)} the {other} {tail}", "active_present_progressive"


def f_plural_past_progressive(subj, subj_pl, other, verb, tail):
    # plural active past progressive: "The {subjs} were {V-ing} the {other}"
    # NOTE: puts "were" into the ACTIVE class (it is active because of the object),
    # so "were" no longer marks passive on its own.
    return f"The {subj_pl} were {vf_ing(verb)} the {other}", "active_plural_past_prog"


def f_plural_simple_past(subj, subj_pl, other, verb, tail):
    # plural active simple past: "The {subjs} {V-past} the {other} {tail}"
    return f"The {subj_pl} {vf_past(verb)} the {other} {tail}", "active_plural_simple_past"


def f_present_perfect(subj, subj_pl, other, verb, tail):
    # present-perfect ACTIVE: "The {subj} has {V-pp} the {other} {tail}"
    # puts "has" + the participle into the ACTIVE class (it is active: it has a
    # direct object), so "has"/the participle no longer mark passive.
    return f"The {subj} has {vf_partic(verb)} the {other} {tail}", "active_present_perfect"


# 6 templates per class; round-robin keeps each at ~1/6 of the class. "were"/"is"
# now occur in BOTH classes; only "got"/"being" remain passive-only (genuine,
# non-spurious voice markers) and at a diluted ~1/6 rate each.
PASSIVE_TEMPLATES = [
    t_be_passive_by, t_plural_be_passive, t_get_passive, t_progressive_passive,
    t_has_been_passive, t_plural_get_passive,
]
ACTIVE_TEMPLATES = [
    f_simple_past, f_simple_present, f_past_progressive, f_present_progressive,
    f_plural_past_progressive, f_plural_simple_past, f_present_perfect,
]

# OOD partition: held_out/confirmation use eval verbs+roles only. Partition the
# regular and irregular verb banks separately so BOTH appear in train and eval,
# and partition the (shared) ROLE bank so eval subjects/agents are unseen too.
_tr_reg, _ev_reg = _split_values(VERBS, SEED + 909, train_frac=0.6)
_tr_irr, _ev_irr = _split_values(IRREGULAR_VERBS, SEED + 451, train_frac=0.6)
train_verbs = _tr_reg | _tr_irr
eval_verbs = _ev_reg | _ev_irr
train_roles, eval_roles = _split_values(ROLE_WORDS, SEED + 313, train_frac=0.6)

SPLIT_SIZES = {
    "few_shot_pool": {True: 100, False: 100},
    "held_out": {True: 60, False: 60},
    "confirmation": {True: 50, False: 50},
    "spare": {True: 10, False: 10},
}


def build():
    gen = Gen(SEED).derive(RULE_ID)
    for pool in (ALL_VERBS, ROLE_WORDS, TAILS):
        gen.shuffle(pool)

    items = []
    used_texts: set[str] = set()
    index = 0
    # round-robin template counters per (split, label) so all templates within a
    # class are used at equal rates (prevents any single voice marker like
    # "got"/"were"/"being" from dominating one class via uneven sampling).
    tmpl_counter: dict[tuple, int] = {}
    # per-(split, label, template) shuffled candidate iterators built from the full
    # slot product, so every combination is reachable and we never run dry early.
    cand_iters: dict[tuple, "list[tuple]"] = {}
    cand_pos: dict[tuple, int] = {}

    import itertools

    def allowed_verbs(split):
        if split == "spare":
            return sorted(ALL_VERBS)
        return sorted(train_verbs) if split == "few_shot_pool" else sorted(eval_verbs)

    def allowed_roles(split):
        if split == "spare":
            return sorted(ROLE_WORDS)
        return sorted(train_roles) if split == "few_shot_pool" else sorted(eval_roles)

    def candidates(split, tmpl_name):
        ck = (split, tmpl_name)
        if ck not in cand_iters:
            verbs = allowed_verbs(split)
            roles = allowed_roles(split)
            combos = [
                (subj, verb, tail, other)
                for subj in roles
                for verb in verbs
                for tail in TAILS
                for other in roles
                if other != subj
            ]
            Gen(SEED).derive(f"{split}:{tmpl_name}").shuffle(combos)
            cand_iters[ck] = combos
            cand_pos[ck] = 0
        return ck

    def make_text(label, split, salt):
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
            text, shape = tmpl(subj, subj_pl, other, verb, tail)
            text = fix_indefinite_articles(text)
            if text in used_texts:
                continue
            wc = schema.word_count(text)
            if wc < 5 or wc > 8:
                continue
            used_texts.add(text)
            return text, {
                "shape": shape,
                "template": tmpl.__name__,
                "verb_base": verb,
                "tail": tail,
                "subject": subj,
                "other": other,
                "seed": SEED,
                "verb_partition": "train" if split == "few_shot_pool" else "eval",
                "role_partition": "train" if split == "few_shot_pool" else "eval",
            }
        raise RuntimeError(f"ran out of unique items for {split} label={label} tmpl={tmpl.__name__}")

    for split, by_label in SPLIT_SIZES.items():
        for label in (True, False):
            for _ in range(by_label[label]):
                text, meta = make_text(label, split, index)
                bid = f"passive-deconfounded_b-{index:05d}"
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
    out_dir = REPO / "data" / "passive_voice_deconfounded_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "items.jsonl"
    write_items(items, out)
    print(f"wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
