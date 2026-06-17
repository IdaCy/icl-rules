"""Rule 26: exactly_two_commas (category numeric).

Canonical articulation: True iff the input contains EXACTLY two commas (raw-string
char rule; every ',' counts regardless of function).

The whole job of this generator is to make the COMMA COUNT the only signal that
separates the two classes — comma PRESENCE, list length, word count, char count,
first/last word, and the/a/and rates must all sit at ~50% agreement. The recipe's
machinery that earns this:

  * Two structure families, balanced 50/50 across the dataset AND across the two
    classes (so 'family' is uncorrelated with the label):

    (a) LIST family with the OXFORD-COMMA TRICK. The True and False variants of a
        list base are WORD-IDENTICAL (same list items, same opener, same 'and');
        they differ ONLY by whether the comma before 'and' is present:
          3-item list  Oxford   'a, b, and c'      = 2 commas  -> True
                       no-Oxford 'a, b and c'       = 1 comma   -> False  (false_mode=one)
          4-item list  no-Oxford 'a, b, c and d'    = 2 commas  -> True
                       Oxford    'a, b, c, and d'    = 3 commas  -> False  (false_mode=three)
        So list LENGTH (3 vs 4) and word count appear on BOTH sides of every comma
        count, and the True/False pair of a base is character-identical except for
        one comma -> every comma-blind predicate is per-base identical across the
        pair (agreement exactly 0.5 on the few_shot_pool, where both variants are
        emitted).

    (b) APPOSITIVE/ADJUNCT family. Both variants of a base LEAD WITH THE SAME
        SUBJECT (no leading temporal that would split the first-word distribution
        by class), and the appositive phrase is ARTICLE-FREE (no a/the/and) so
        dropping it never moves the the/a/and battery predicates:
          True  (2 commas): 'S, AP, PR'                e.g. 'our neighbor, retired teacher, watered the garden'
          False=three (3) : 'S, AP, PR, ADV'           + a trailing comma + 1-word place adverb
          False=one   (1) : 'S PR, ADV'                drop the appositive, keep a 1-comma 2-word place adjunct

  * MARGINAL LENGTH MATCHING by construction: EVERY emitted item is equalized to a
    single fixed word count (``_TARGET_WORDS``) by appending AT MOST ONE comma-free
    ADVERB_PLACE adjunct at the end (a single 1-3 word place phrase — NEVER a stack
    of trailing single-word adverbs). The padding is comma-free, so it never
    perturbs the comma count; word count is therefore constant across the whole
    dataset (length-matching is exactly 0.0, well under the 0.2 tolerance), every
    word_count>=k / char_count>=k predicate is constant -> agreement ~= P(label) ~=
    0.5, and the LAST word of every item is drawn from the shared place pool ->
    last-word predicates ~= 0.5. Each appositive False variant carries its sole
    place adjunct INSIDE its core (sized so the core already hits the target), so it
    receives no further padding and never ends in a run of stacked adverbs.

  * COMMA PRESENCE is CONSTANT: every item (both classes) contains >= 1 comma, so
    the 'contains a comma' predicate sits at ~0.5 (it is True everywhere; agreement
    == P(True)). There are no 0-comma items.

base_id = the content (opener + list items, or subject + appositive + predicate)
plus the family/false_mode tag; both variants of a base share it.

This module conforms to the GENERATOR INTERFACE documented in ``base``.

REGISTRATION: this rule is run through the shared gated pipeline directly via
``base.emit_rule`` (see ``scripts``/the module ``__main__`` shim at the bottom of
this file's package) to avoid a write-conflict on the shared ``registry._REGISTRY``
table. The module exposes the two interface callables ``build_bases`` and
``instantiate`` (plus the optional ``STYLE_RULE_ID`` hook), so a one-line
``_REGISTRY`` entry can be added later with zero changes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import (
    Gen,
    GenError,
    adjunct_word_lengths,
)
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

# Spec banks: list items + appositive nouns from NOUN_CONCRETE; predicate verbs
# from VERB_REGULAR; appositive/descriptor adjectives from ADJ_PLAIN. ADVERB_PLACE
# supplies BOTH the comma-free word-count padding and the trailing place adjunct
# that carries the appositive family's extra comma. Every padding/trailing adjunct
# is a SINGLE 1-3 word ADVERB_PLACE phrase; the word-count equalizer never appends
# a second adjunct, so no item ever ends in a stack of trailing adverbs.
_NOUN_BANK = "NOUN_CONCRETE"
_VERB_BANK = "VERB_REGULAR"
_ADJ_BANK = "ADJ_PLAIN"
_ADJUNCT_BANK = "ADVERB_PLACE"

# rule-spec style policy id (commas ARE the rule; the style validator self-selects
# the comma-allowing policy from this rule_id).
STYLE_RULE_ID = "exactly_two_commas"

# Comfortably over the 340-base floor so the by-base split (100 + 120 + 100 +
# >= 20 spare = >= 340) has headroom. Built distinct, with family balanced 50/50
# and the False comma-count (1 vs 3) balanced 50/50.
_N_BASES = 440

# Every item is equalized to EXACTLY this many words with comma-free padding.
# 7 sits inside the recipe's 6-11 word band and the global [4,14] cap. It is the
# value that lets the core->target DEFICIT of EVERY variant be closed by a SINGLE
# 1-3 word ADVERB_PLACE phrase, with NO manner-adverb stacking:
#   list 3-item core = 6 -> deficit 1 -> one 1-word place adverb
#   list 4-item core = 7 -> deficit 0 -> no padding
#   app_true     core = 6 -> deficit 1 -> one 1-word place adverb
#   app_false_three core 'S, AP, PR, ADV' carries a 1-word place adverb -> 7, no pad
#   app_false_one   core 'S PR, ADV'      carries a 2-word place phrase -> 7, no pad
# Every place adjunct is comma-free, so the comma count the core fixed is exact;
# every place adjunct avoids "the" (the no-the ADVERB_PLACE pool tops out at 2
# words, which is why the one-comma core needs the 2-word phrase and the target is
# 7, not 8). Char counts stay matched ACROSS classes (word count is constant), so
# every char_count>=k battery predicate sits at ~0.5 agreement.
_TARGET_WORDS = 7

# List openers: plural / pronoun subjects + a verb, SHARED verbatim between the
# True and False variant of a list base (the Oxford trick changes only one comma),
# so every first-word / opener-token predicate is per-base identical across the
# pair. First words spread across letter buckets (we/they/she/he/...).
_LIST_OPENERS: tuple[str, ...] = (
    "we found",
    "they moved",
    "she packed",
    "he stacked",
    "we counted",
    "they cleaned",
    "she painted",
    "he carried",
    "we sorted",
    "they washed",
)

# Appositive subjects: a possessive determiner + a person noun. Article-free; the
# determiner spreads first letters across buckets (our/their/his/her). SHARED
# between both variants of an appositive base (both lead with the subject).
_APP_SUBJECTS: tuple[str, ...] = (
    "our neighbor",
    "their teacher",
    "his cousin",
    "her mentor",
    "our captain",
    "their landlord",
    "his partner",
    "her trainer",
)

# Appositive descriptor nouns (the AP head): a person/role noun, ARTICLE-FREE.
_APP_ROLES: tuple[str, ...] = (
    "teacher",
    "doctor",
    "farmer",
    "painter",
    "singer",
    "builder",
    "dancer",
    "writer",
    "trainer",
    "captain",
    "gardener",
    "ranger",
)

# Predicate frames: a regular past-tense verb + 'the {object}'. The object noun is
# the per-base content slot. SHARED between both variants. '{V}' is filled from a
# small curated set of regular past verbs (no commas, no banned punctuation).
_PRED_VERBS: tuple[str, ...] = (
    "watered",
    "painted",
    "cleaned",
    "opened",
    "fixed",
    "moved",
    "counted",
    "washed",
    "carried",
    "guarded",
    "repaired",
    "rented",
)


def _safe_plural(word: str) -> str | None:
    """Naive English plural, returning None for endings where +s is wrong.

    Used so list items read as a plain plural list ('tables, chairs, and doors')
    without an article per item. Conservative: skips s/x/z/sh/ch endings and
    consonant+y (which would need -ies); those nouns are simply not used as list
    items (119 of the 131 NOUN_CONCRETE words remain, ample for distinct bases)."""
    if word.endswith(("s", "x", "z", "sh", "ch")):
        return None
    if word.endswith("y") and (len(word) < 2 or word[-2] not in "aeiou"):
        return None
    return word + "s"


@dataclass(frozen=True)
class Base:
    """One exactly_two_commas base spec carrying everything to instantiate BOTH
    its 2-comma True variant and its (1- or 3-comma) False variant.

    ``family`` is 'list' or 'appositive'; ``false_mode`` is 'one' or 'three' (the
    comma count of the False variant). The True variant always has exactly 2
    commas. ``base_id`` mixes the content + family + false_mode so it is stable and
    distinct; both variants share it."""

    base_id: str
    family: str            # 'list' | 'appositive'
    false_mode: str        # 'one' | 'three'
    # list family
    opener: str = ""
    items: tuple[str, ...] = ()
    # appositive family
    subject: str = ""
    appositive: str = ""
    predicate: str = ""
    # the trailing place adjunct (comma-free, no "the") that BOTH carries the
    # appositive False variant's extra comma AND sizes its core to _TARGET_WORDS:
    # 1 word for false_mode='three' ('S, AP, PR, ADV' = 7), 2 words for
    # false_mode='one' ('S PR, ADV' = 7). Empty for True/list bases.
    place_adverb: str = ""


def _list_core(opener: str, items: tuple[str, ...], oxford: bool) -> str:
    """Render a list clause. ``oxford`` toggles the comma before 'and'.

    3 items: a, b[,] and c   -> 2 commas (oxford) | 1 comma (no oxford)
    4 items: a, b, c[,] and d -> 3 commas (oxford) | 2 commas (no oxford)
    The words are identical regardless of ``oxford``; only the pre-'and' comma
    moves. The clause has no trailing punctuation."""
    *head, last = items
    # the head items are comma-joined: 'a, b' (3-item) or 'a, b, c' (4-item)
    head_str = ", ".join(head)
    sep = ", and " if oxford else " and "
    return f"{opener} {head_str}{sep}{last}"


def _pad_at_most_one_place(
    text: str,
    target: int,
    place_by_len: dict[int, list[str]],
    gen: Gen,
) -> tuple[str, tuple[str, ...]]:
    """Pad ``text`` to exactly ``target`` words by appending AT MOST ONE place phrase.

    The whole deficit is closed by a SINGLE comma-free ADVERB_PLACE phrase whose
    word count EQUALS the deficit (a 1-, 2-, or 3-word phrase) — never a second
    adjunct and never a stack of trailing single-word adverbs (the word-salad the
    old manner-padding produced). A zero deficit appends nothing (the appositive
    False cores already hit the target inside their core). The appended phrase is
    comma-free, so the comma count the core fixed is preserved exactly; the draw is
    deterministic. Raises (loud) if no single place phrase of the exact deficit
    length exists, so a stacked fallback can never silently reappear.
    Returns (padded_text, appended_tokens)."""
    deficit = target - word_count(text)
    if deficit < 0:
        raise GenError(f"text already over target {target}: {text!r}")
    if deficit == 0:
        return text, ()
    if deficit not in place_by_len:
        raise GenError(
            f"exactly_two_commas: deficit {deficit} has no single ADVERB_PLACE phrase "
            f"of that length (have {sorted(place_by_len)}); refusing to stack adjuncts: {text!r}"
        )
    phrase = gen.choice(place_by_len[deficit])
    out = f"{text} {phrase}"
    if word_count(out) != target:
        raise GenError(
            f"exactly_two_commas: padded to {word_count(out)} words, expected {target}: {out!r}"
        )
    return out, tuple(out.split()[word_count(text):])


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (the GENERATOR INTERFACE).

    Half the bases are LIST family, half APPOSITIVE; within each family half are
    false_mode='one' and half 'three', so globally families are 50/50 across the
    two classes and the False class is 50% 1-comma / 50% 3-comma. Deterministic
    given ``gen``. base_ids (and so the eventual surface strings) are kept
    distinct. Raises (loud) if it cannot reach the floor with the exact splits."""
    nouns_all = banks.get_bank(_NOUN_BANK).words()
    list_nouns = [n for n in nouns_all if _safe_plural(n)]
    plural = {n: _safe_plural(n) for n in list_nouns}

    g_struct = gen.derive("structure")
    g_list = gen.derive("list")
    g_app = gen.derive("appositive")

    # plural list-item pool, shuffled (deterministic)
    pool = [plural[n] for n in list_nouns]
    g_list.shuffle(pool)

    bases: list[Base] = []
    seen_ids: set[str] = set()

    n_list = _N_BASES // 2
    n_app = _N_BASES - n_list

    # --- LIST family ----------------------------------------------------------
    # false_mode='one'  -> 3-item lists (Oxford True=2, no-Oxford False=1)
    # false_mode='three'-> 4-item lists (no-Oxford True=2, Oxford False=3)
    list_built = 0
    cursor = 0
    idx = 0
    while list_built < n_list:
        false_mode = "one" if (list_built % 2 == 0) else "three"
        k = 3 if false_mode == "one" else 4
        if cursor + k > len(pool):
            # reshuffle a fresh permutation to keep drawing distinct combinations
            g_list.shuffle(pool)
            cursor = 0
            idx += 1
            if idx > 50:  # pragmatic guard against an impossible request
                break
        items = tuple(pool[cursor : cursor + k])
        cursor += k
        opener = _LIST_OPENERS[list_built % len(_LIST_OPENERS)]
        bid = f"list|{false_mode}|{opener}|{'+'.join(items)}"
        if bid in seen_ids:
            continue
        seen_ids.add(bid)
        bases.append(Base(base_id=bid, family="list", false_mode=false_mode, opener=opener, items=items))
        list_built += 1

    # --- APPOSITIVE family ----------------------------------------------------
    # subject (shared), article-free appositive head, predicate verb + 'the {obj}'.
    obj_pool = list(list_nouns)  # singular concrete nouns for 'the {object}'
    g_app.shuffle(obj_pool)
    # The appositive False core carries its sole trailing place adjunct, sized so
    # the core already hits _TARGET_WORDS (7) and so receives NO further padding:
    #   false_mode='three' core 'S, AP, PR, ADV' (6+wc) -> wc 1  ('S, AP, PR, downtown')
    #   false_mode='one'   core 'S PR, ADV'      (5+wc) -> wc 2  ('S PR, at work')
    # Both must be "the"-free (the predicate already carries the lone "the"), so we
    # draw from the no-the ADVERB_PLACE pool (1- and 2-word phrases only).
    no_the_place_all = [
        p for p in banks.get_bank(_ADJUNCT_BANK).words()
        if "the" not in p.lower().split()
    ]
    place_pool_by_mode = {
        "three": [p for p in no_the_place_all if word_count(p) == 1],
        "one": [p for p in no_the_place_all if word_count(p) == 2],
    }
    for mode, pl in place_pool_by_mode.items():
        if not pl:
            raise ValueError(
                f"exactly_two_commas: no no-the ADVERB_PLACE phrase for mode {mode!r}"
            )

    app_built = 0
    oi = 0
    while app_built < n_app:
        false_mode = "one" if (app_built % 2 == 0) else "three"
        subject = _APP_SUBJECTS[app_built % len(_APP_SUBJECTS)]
        role = _APP_ROLES[app_built % len(_APP_ROLES)]
        verb = _PRED_VERBS[app_built % len(_PRED_VERBS)]
        if oi >= len(obj_pool):
            g_app.shuffle(obj_pool)
            oi = 0
        obj = obj_pool[oi]
        oi += 1
        place_pool = place_pool_by_mode[false_mode]
        place = place_pool[app_built % len(place_pool)]
        appositive = role
        predicate = f"{verb} the {obj}"
        bid = f"app|{false_mode}|{subject}|{appositive}|{predicate}|{place}"
        if bid in seen_ids:
            continue
        seen_ids.add(bid)
        bases.append(
            Base(
                base_id=bid,
                family="appositive",
                false_mode=false_mode,
                subject=subject,
                appositive=appositive,
                predicate=predicate,
                place_adverb=place,
            )
        )
        app_built += 1

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"exactly_two_commas: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    # exact family + false_mode balance (loud if drift)
    n_list_b = sum(1 for b in bases if b.family == "list")
    n_one = sum(1 for b in bases if b.false_mode == "one")
    if n_list_b * 2 != len(bases):
        raise ValueError(
            f"exactly_two_commas: family not 50/50 ({n_list_b}/{len(bases)} list)"
        )
    if n_one * 2 != len(bases):
        raise ValueError(
            f"exactly_two_commas: false_mode not 50/50 ({n_one}/{len(bases)} one-comma)"
        )
    return bases


def _build_core(spec: Base, label: bool) -> tuple[str, str]:
    """Return (core_text, transform_tag) for one variant of ``spec``.

    The core carries exactly its commas (True -> 2; False -> spec.false_mode). All
    padding is added later (comma-free), so the core fixes the comma count."""
    if spec.family == "list":
        if label:
            # True = 2 commas: 3-item Oxford OR 4-item no-Oxford
            oxford = spec.false_mode == "one"  # 3-item -> Oxford gives 2
            core = _list_core(spec.opener, spec.items, oxford=oxford)
            return core, f"list_true_{'oxford' if oxford else 'no_oxford'}"
        # False: 3-item no-Oxford (1 comma) | 4-item Oxford (3 commas)
        oxford = spec.false_mode == "three"  # 4-item -> Oxford gives 3
        core = _list_core(spec.opener, spec.items, oxford=oxford)
        return core, f"list_false_{spec.false_mode}"

    # appositive family
    if label:
        # True = 2 commas: 'S, AP, PR'
        core = f"{spec.subject}, {spec.appositive}, {spec.predicate}"
        return core, "app_true"
    if spec.false_mode == "three":
        # 3 commas: 'S, AP, PR, ADV' — ADV is a 1-word place adverb, so the core is
        # already _TARGET_WORDS (7) and takes no further padding.
        core = f"{spec.subject}, {spec.appositive}, {spec.predicate}, {spec.place_adverb}"
        return core, "app_false_three"
    # false_mode == 'one': 1 comma: 'S PR, ADV' (drop the article-free appositive).
    # ADV is a 2-word place phrase, so the core is already _TARGET_WORDS (7).
    core = f"{spec.subject} {spec.predicate}, {spec.place_adverb}"
    return core, "app_false_one"


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    Builds the comma-bearing core for ``label`` (True -> exactly 2 commas; False ->
    spec.false_mode commas), then pads with comma-free ADVERB_PLACE adjuncts to
    exactly _TARGET_WORDS. Deterministic given (spec, label, gen)."""
    # place-padding pool excludes any phrase containing "the" so padding never
    # adds a "the": the only "the" tokens then come from the cores, which are
    # per-base symmetric across the two variants (list cores have none; appositive
    # predicates carry exactly one "the" in BOTH variants) -> count('the')>=2 stays
    # ~constant-False (~0.5) across classes.
    no_the_place = [p for p in banks.get_bank(_ADJUNCT_BANK).words() if "the" not in p.lower().split()]
    place_by_len = adjunct_word_lengths(no_the_place)
    core, transform = _build_core(spec, label)
    # pad to the fixed target with a SINGLE comma-free place phrase (per-variant
    # seeded stream). The appositive False cores already hit _TARGET_WORDS via the
    # place adjunct baked into the core, so they get a zero-deficit no-op here and
    # can never end up with a second trailing adjunct.
    pad_gen = gen.derive(f"{spec.base_id}|{'T' if label else 'F'}")
    text, appended = _pad_at_most_one_place(core, _TARGET_WORDS, place_by_len, pad_gen)

    expected = 2 if label else (1 if spec.false_mode == "one" else 3)
    # belt-and-braces: the core fixed the comma count and padding is comma-free
    if text.count(",") != expected:
        raise ValueError(
            f"exactly_two_commas: built {text.count(',')} commas, expected "
            f"{expected} for {transform}: {text!r}"
        )
    meta = {
        "family": spec.family,
        "false_mode": spec.false_mode,
        "transform": transform,
        "core": core,
        "comma_count": text.count(","),
        "padding_adjuncts": list(appended),
        "target_words": _TARGET_WORDS,
    }
    return text, meta
