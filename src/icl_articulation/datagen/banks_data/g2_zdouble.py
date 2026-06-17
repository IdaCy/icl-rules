"""Bank group G2 (letter-z and double-letter banks).

Owns these banks (see banks.BANK_QUOTAS for the per-bank contract):
  Z_WORDS, Z_FREE_MATCHED, DOUBLE_WORDS, NONADJ_REPEAT_WORDS

Notes the author honors (transcribed into BANK_QUOTAS, enforced by the
self-check ``banks.check_bank``):
  - Z_WORDS / Z_FREE_MATCHED are a MATCHED PAIR: each Z_WORDS entry has a
    counterpart of the SAME POS, length +/- 2, SAME frequency_tier, NO z. The
    link is the shared ``pair`` key; the check matches per-pair, not
    distributionally. Authored side by side below so the matching is obvious.
  - Z_WORDS: every entry contains a 'z'; <= 25% animal words, <= 25% food
    words; varied POS. NOTE (open tension, see module-end): every common
    English z-ANIMAL (zebra, lizard, chimpanzee, gazelle, buzzard, ...) ranks
    rarer than wordfreq top-10000 (tier 3), which the global tier rule bans;
    likewise the only tier-<=2 z-FOOD is 'pizza'. The animal/food caps are
    MAXIMUMS, so 0 animals and 1 food (pizza) satisfy them honestly; we do not
    invent tier-3 entries to "use up" a cap that is an upper bound.
  - DOUBLE_WORDS: every entry has_adjacent_double=true; double letters spread
    over >= 10 letter types (here 14: ll ss ee oo tt ff mm nn pp rr dd bb cc
    gg); <= 25% food; varied POS and topic. No single double-type exceeds ~19%
    of the bank (rule 28's "no 'contains ll' shortcut above 75%" guard).
  - NONADJ_REPEAT_WORDS: double-free (has_adjacent_double=false) yet each has a
    NON-adjacent repeated letter (has_nonadjacent_repeat=true) -- the splitter
    bank for rule 28 (adjacent-double vs any-repeat). banks.py verifies both
    tags programmatically from the surface; we author only word/pos/tier.

All frequency_tier tags are wordfreq-honest (tier 1 = top 2000, tier 2 = top
10000); verified against ``wordfreq.top_n_list`` while authoring. Every entry
is lowercase and alphabetic-only, per the entry contract in banks.py.

DOUBLE_FREE_VOCAB (the rule-28 banned-function-word list) is exported below as
a module constant; it is a flag/list per the spec note, not a word bank with a
quota, so it is intentionally NOT placed in ``BANKS``.
"""

from __future__ import annotations

from typing import Any


def _z(word: str, pos: str, tier: int, pair: str, subtype: str | None = None) -> dict[str, Any]:
    e: dict[str, Any] = {"word": word, "pos": pos, "frequency_tier": tier, "pair": pair}
    if subtype is not None:
        e["subtype"] = subtype
    return e


def _f(word: str, pos: str, tier: int, pair: str) -> dict[str, Any]:
    # Z_FREE_MATCHED counterpart: same pos + tier as its z-mate, no z, length +/-2.
    return {"word": word, "pos": pos, "frequency_tier": tier, "pair": pair}


def _d(word: str, pos: str, tier: int, subtype: str | None = None) -> dict[str, Any]:
    e: dict[str, Any] = {"word": word, "pos": pos, "frequency_tier": tier}
    if subtype is not None:
        e["subtype"] = subtype
    return e


def _n(word: str, pos: str, tier: int) -> dict[str, Any]:
    return {"word": word, "pos": pos, "frequency_tier": tier}


# --- Z_WORDS <-> Z_FREE_MATCHED (1:1 paired) ----------------------------------
# Authored as parallel lists keyed by the same ``pair`` id. Each row: a z-word
# and its z-free counterpart of identical POS + tier, |length diff| <= 2. The
# pair id is the z-word itself (unique, readable).

Z_WORDS: list[dict[str, Any]] = [
    # --- nouns ---
    _z("zoo", "noun", 2, "zoo"),
    _z("citizen", "noun", 2, "citizen"),
    _z("horizon", "noun", 2, "horizon"),
    _z("magazine", "noun", 1, "magazine"),
    _z("puzzle", "noun", 2, "puzzle"),
    _z("prize", "noun", 2, "prize"),
    _z("size", "noun", 1, "size"),
    _z("zone", "noun", 1, "zone"),
    _z("zero", "noun", 2, "zero"),
    _z("dozen", "noun", 2, "dozen"),
    _z("jazz", "noun", 2, "jazz"),
    _z("pizza", "noun", 2, "pizza", subtype="food"),
    _z("breeze", "noun", 2, "breeze"),
    _z("quiz", "noun", 2, "quiz"),
    _z("hazard", "noun", 2, "hazard"),
    _z("bronze", "noun", 2, "bronze"),
    _z("wizard", "noun", 2, "wizard"),
    _z("zombie", "noun", 2, "zombie"),
    _z("citizenship", "noun", 2, "citizenship"),
    _z("organization", "noun", 1, "organization"),
    _z("civilization", "noun", 2, "civilization"),
    _z("zip", "noun", 2, "zip"),
    # --- adjectives ---
    _z("lazy", "adjective", 2, "lazy"),
    _z("crazy", "adjective", 1, "crazy"),
    _z("frozen", "adjective", 2, "frozen"),
    _z("amazing", "adjective", 1, "amazing"),
    _z("amazed", "adjective", 2, "amazed"),
    # --- verbs ---
    _z("freeze", "verb", 2, "freeze"),
    _z("squeeze", "verb", 2, "squeeze"),
    _z("organize", "verb", 2, "organize"),
    _z("realize", "verb", 1, "realize"),
    _z("realized", "verb", 2, "realized"),
    _z("recognize", "verb", 2, "recognize"),
    _z("analyze", "verb", 2, "analyze"),
    _z("seize", "verb", 2, "seize"),
    _z("seized", "verb", 2, "seized"),
    _z("apologize", "verb", 2, "apologize"),
    _z("minimize", "verb", 2, "minimize"),
    _z("zoom", "verb", 2, "zoom"),
    _z("buzz", "verb", 2, "buzz"),
]

Z_FREE_MATCHED: list[dict[str, Any]] = [
    # --- counterparts of the z-nouns ---
    _f("cafe", "noun", 2, "zoo"),
    _f("soldier", "noun", 2, "citizen"),
    _f("ceiling", "noun", 2, "horizon"),
    _f("industry", "noun", 1, "magazine"),
    _f("mystery", "noun", 2, "puzzle"),
    _f("medal", "noun", 2, "prize"),
    _f("shape", "noun", 1, "size"),
    _f("area", "noun", 1, "zone"),
    _f("blank", "noun", 2, "zero"),
    _f("bunch", "noun", 2, "dozen"),
    _f("tune", "noun", 2, "jazz"),
    _f("salad", "noun", 2, "pizza"),
    _f("shadow", "noun", 2, "breeze"),
    _f("exam", "noun", 2, "quiz"),
    _f("danger", "noun", 2, "hazard"),
    _f("copper", "noun", 2, "bronze"),
    _f("knight", "noun", 2, "wizard"),
    _f("monster", "noun", 2, "zombie"),
    _f("membership", "noun", 2, "citizenship"),
    _f("government", "noun", 1, "organization"),
    _f("institution", "noun", 2, "civilization"),
    _f("clip", "noun", 2, "zip"),
    # --- counterparts of the z-adjectives ---
    _f("quiet", "adjective", 2, "lazy"),
    _f("funny", "adjective", 1, "crazy"),
    _f("golden", "adjective", 2, "frozen"),
    _f("powerful", "adjective", 1, "amazing"),
    _f("pleased", "adjective", 2, "amazed"),
    # --- counterparts of the z-verbs ---
    _f("roast", "verb", 2, "freeze"),
    _f("stretch", "verb", 2, "squeeze"),
    _f("arrange", "verb", 2, "organize"),
    _f("consider", "verb", 1, "realize"),
    _f("admitted", "verb", 2, "realized"),
    _f("translate", "verb", 2, "recognize"),
    _f("examine", "verb", 2, "analyze"),
    _f("grab", "verb", 2, "seize"),
    _f("grabbed", "verb", 2, "seized"),
    _f("complain", "verb", 2, "apologize"),
    _f("decrease", "verb", 2, "minimize"),
    _f("rush", "verb", 2, "zoom"),
    _f("glow", "verb", 2, "buzz"),
]


# --- DOUBLE_WORDS -------------------------------------------------------------
# Every entry carries an adjacent double letter (banks.py verifies). 14 double
# types span ll ss ee oo tt ff mm nn pp rr dd bb cc gg; <= 25% food; POS and
# topic varied. Size is a minimum; this set runs above it to give the rule-28
# generator slack while keeping every quota satisfied.

DOUBLE_WORDS: list[dict[str, Any]] = [
    # ll
    _d("wall", "noun", 1),
    _d("ball", "noun", 1),
    _d("valley", "noun", 1),
    _d("dollar", "noun", 2),
    _d("pillow", "noun", 2),
    _d("village", "noun", 1),
    _d("college", "noun", 1),
    _d("balloon", "noun", 2),
    _d("umbrella", "noun", 2),
    _d("small", "adjective", 1),
    _d("tall", "adjective", 2),
    _d("yellow", "adjective", 2),
    _d("silly", "adjective", 2),
    _d("follow", "verb", 1),
    _d("allow", "verb", 1),
    _d("spell", "verb", 2),
    _d("really", "adverb", 1),
    # ss
    _d("grass", "noun", 2),
    _d("glass", "noun", 1),
    _d("dress", "noun", 1),
    _d("lesson", "noun", 2),
    _d("message", "noun", 1),
    _d("business", "noun", 1),
    _d("pressure", "noun", 1),
    _d("press", "verb", 1),
    _d("discuss", "verb", 2),
    _d("across", "preposition", 1),
    # ee
    _d("coffee", "noun", 1, subtype="food"),
    _d("cheese", "noun", 2, subtype="food"),
    _d("sheep", "noun", 2),
    _d("wheel", "noun", 2),
    _d("queen", "noun", 1),
    _d("degree", "noun", 1),
    _d("green", "adjective", 1),
    _d("sweet", "adjective", 1),
    _d("deep", "adjective", 1),
    _d("agree", "verb", 1),
    _d("sleep", "verb", 1),
    _d("meet", "verb", 1),
    # oo
    _d("book", "noun", 1),
    _d("moon", "noun", 2),
    _d("door", "noun", 1),
    _d("floor", "noun", 1),
    _d("room", "noun", 1),
    _d("school", "noun", 1),
    _d("spoon", "noun", 2),
    _d("cool", "adjective", 1),
    _d("smooth", "adjective", 2),
    _d("choose", "verb", 1),
    # tt
    _d("bottle", "noun", 2),
    _d("letter", "noun", 1),
    _d("button", "noun", 2),
    _d("battle", "noun", 1),
    _d("pretty", "adjective", 1),
    _d("better", "adjective", 1),
    _d("attack", "verb", 1),
    # ff
    _d("office", "noun", 1),
    _d("effort", "noun", 1),
    _d("traffic", "noun", 1),
    _d("offer", "verb", 1),
    _d("suffer", "verb", 2),
    # mm
    _d("hammer", "noun", 2),
    _d("summer", "noun", 1),
    _d("comment", "noun", 1),
    _d("common", "adjective", 1),
    # nn
    _d("dinner", "noun", 1),
    _d("tunnel", "noun", 2),
    _d("winner", "noun", 1),
    _d("funny", "adjective", 1),
    _d("connect", "verb", 2),
    # pp
    _d("apple", "noun", 1, subtype="food"),
    _d("pepper", "noun", 2, subtype="food"),
    _d("puppy", "noun", 2),
    _d("happy", "adjective", 1),
    _d("appear", "verb", 1),
    _d("support", "verb", 1),
    # rr
    _d("mirror", "noun", 2),
    _d("arrow", "noun", 2),
    _d("narrow", "adjective", 2),
    _d("carry", "verb", 1),
    # dd
    _d("ladder", "noun", 2),
    _d("middle", "noun", 1),
    _d("hidden", "adjective", 2),
    _d("add", "verb", 1),
    # bb
    _d("rabbit", "noun", 2),
    _d("ribbon", "noun", 2),
    _d("bubble", "noun", 2),
    # cc
    _d("soccer", "noun", 2),
    _d("accident", "noun", 1),
    _d("accept", "verb", 1),
    # gg
    _d("luggage", "noun", 2),
    _d("suggest", "verb", 1),
]


# --- NONADJ_REPEAT_WORDS ------------------------------------------------------
# Double-FREE (no adjacent double) but a letter repeats non-adjacently. banks.py
# verifies both has_adjacent_double=False and has_nonadjacent_repeat=True. Used
# to salt rule-28's False class so "a repeated letter anywhere" is not the rule.
# POS varied (rule 28 matches the substituted word's POS).

NONADJ_REPEAT_WORDS: list[dict[str, Any]] = [
    # nouns
    _n("window", "noun", 1),     # w..w
    _n("banana", "noun", 2),     # a/n repeats non-adjacent
    _n("level", "noun", 1),      # l..l, e
    _n("radar", "noun", 2),      # r..r, a
    _n("elephant", "noun", 2),   # e..e
    _n("garage", "noun", 2),     # g..g, a
    _n("tomato", "noun", 2),     # t..t, o
    _n("salad", "noun", 2),      # a..a
    _n("photograph", "noun", 2), # p..p, h, o
    _n("sister", "noun", 1),     # s..s
    _n("river", "noun", 1),      # r..r
    _n("paper", "noun", 1),      # p..p
    _n("sentence", "noun", 2),   # s/e/n/t repeat non-adjacent
    _n("memory", "noun", 1),     # m..m
    _n("energy", "noun", 1),     # e..e
    _n("engine", "noun", 1),     # e..e, n
    _n("statement", "noun", 1),  # t/e repeat
    _n("analysis", "noun", 1),   # a..a, s
    _n("average", "noun", 1),    # a..a
    _n("camera", "noun", 1),     # a..a
    _n("series", "noun", 1),     # s..s, e
    _n("present", "noun", 1),    # e..e
    _n("estate", "noun", 1),     # e..e, t
    _n("senate", "noun", 1),     # e..e
    _n("decade", "noun", 2),     # e..e, d..d
    _n("parade", "noun", 2),     # a..a
    # adjectives
    _n("important", "adjective", 1),  # t? a/n? 'important' i..., a/n/t repeat? a..a no; p? -> t? actually has p,t,n,a... a appears twice
    _n("general", "adjective", 1),    # e..e
    _n("several", "adjective", 1),    # e..e
    _n("national", "adjective", 1),   # n..n, a
    _n("natural", "adjective", 1),    # n? a..a
    _n("regular", "adjective", 1),    # r..r
    _n("similar", "adjective", 1),    # i..i
    _n("popular", "adjective", 1),    # p..p
    _n("serious", "adjective", 1),    # s..s
    # verbs
    _n("remember", "verb", 1),   # r/e/m repeat
    _n("prepare", "verb", 2),    # p..p, r, e
    _n("return", "verb", 1),     # r..r
    _n("research", "verb", 1),   # r..r, e
    _n("record", "verb", 1),     # r..r
    _n("receive", "verb", 1),    # e..e
    _n("reduce", "verb", 1),     # e..e, r
    _n("require", "verb", 1),    # r..r, e
    _n("escape", "verb", 2),     # e..e
    _n("explore", "verb", 2),    # e..e
    # numerals
    _n("seven", "numeral", 1),   # e..e
    _n("eleven", "numeral", 2),  # e..e
]


DOUBLE_FREE_VOCAB: frozenset[str] = frozenset(
    {
        "all",
        "will",
        "off",
        "too",
        "been",
        "see",
        "good",
        "three",
        "week",
    }
)
"""rule-28 banned function/content words for the False (double-free) templates.

Per the banks-block note on DOUBLE_FREE_VOCAB and rule 28's recipe: base
sentences for rule 28 are built entirely from has_adjacent_double=False
vocabulary, and these specific words -- though common -- carry an adjacent
double letter (or are otherwise flagged in the spec) and so are banned from
those templates: all/will/off/too/been/see/good/three/week. Exported as a flag
list (not a quota'd bank) for the rule-28 generator to consult.
"""


BANKS: dict[str, list[Any]] = {
    "Z_WORDS": Z_WORDS,
    "Z_FREE_MATCHED": Z_FREE_MATCHED,
    "DOUBLE_WORDS": DOUBLE_WORDS,
    "NONADJ_REPEAT_WORDS": NONADJ_REPEAT_WORDS,
}
