"""Rule 12: contains_comparative (category: syntactic).

Canonical articulation (FORM-based): True iff the input contains a word in
comparative form -- an ``-er`` comparative ("taller") or an analytic comparative
("more careful") -- used as an adjective or an adverb. On the emitted data only
comparative ADJECTIVES appear, so the groundtruth verifier ``_r12`` (an ``-er``
word, EXCLUDING the ER_NONCOMPARATIVE salt nouns, OR ``more {adj}``) recomputes
every label exactly.

Construction (recipe, rule-specs id: contains_comparative)
----------------------------------------------------------
Two frame families, each base fixed to ONE family so its True and its False
variant SHARE every non-adjective filler (the only difference between the two
variants is the adjective slot + the count-equalizing material):

  P  predicative WITH ``than`` (True) / plain copula (False):
       True  : "The {N1} is {COMP} than the {N2}"
       False : "The {N1} is {adj} {PREP} the {N3}"      (+ a 1-word EXTRA
               attributive adjective inside the final NP -- "the {adj2} {N3}" --
               in the ``more`` case so the word counts match the True variant)
       where {PREP} is a per-base locative preposition drawn from a pool
       (in/on/near/by/under/...). The preposition is INCIDENTAL padding, so
       spreading it across the pool keeps every single preposition token well
       under the 0.75 battery threshold (an always-"in" fill made 'in' a 0.767
       single-token False cue -- the confound this spread removes).
  A  attributive WITHOUT ``than``:
       True  : "The team wants a {COMP} {N} for the {N2}"
       False : "The team wants a {adj} {N} for the {N2}" (+ a 1-word EXTRA
               attributive adjective stacked inside the object NP -- "a {adj2}
               {adj} {N}" -- in the ``more`` case)

The ``more``-case word-count equalizer is a SECOND plain attributive adjective
placed GRAMMATICALLY inside a noun phrase (family A: "a {adj2} {adj} {N_obj}";
family P: "{PREP} the {adj2} {N3}"), NOT a place adverb wedged between an NP and
a following preposition. (An earlier version inserted a 1-word ADVERB_PLACE
adjunct mid-phrase -- "a national sofa overhead for the barn", "the market is
correct abroad inside the shed" -- which was ungrammatical; the second-adjective
fill keeps the noun phrase intact and the last word a noun.) The extra adjective
is drawn from ADJ_PLAIN excluding that base's own {adj}, so the two stacked
adjectives differ; ADJ_PLAIN holds no ``-er`` ender and introduces no ``more``
token, so the count-equalizer stays groundtruth-inert.

``COMP`` is 70% an ``-er`` form (1 word: "taller") and 30% an analytic
``more {adj}`` (2 words). The recipe's count note -- "'more {adj}' is 2 words
where '-er' is 1" -- is exactly why the False variant's equalizing material is
chosen per base from the True variant's word count.

Word counts (so the word_count battery predicates sit at 50%):
  P / -er  : True  "The N1 is COMP than the N2"          = 7  -> False = 7
  P / more : True  "The N1 is more A than the N2"         = 8  -> False = 8
  A / -er  : True  "The team wants a COMP N for the N2"   = 9  -> False = 9
  A / more : True  "The team wants a more A N for the N2" = 10 -> False = 10
Every variant of a base has the SAME word count, so the 6 word_count predicates
are exactly 50% by construction.

The LAST word of every text is always a NOUN_CONCRETE noun (N2 in P-True / A;
N3 in P-False), drawn from one shared pool for both classes, so last_ends_* and
last_word_len cannot separate the classes. The FIRST word is always "The", so
the first_* predicates are constant -> 50%. char_count is balanced by the
count-equalizer carrying the residual length, and verified by Gate D.

ER SALT: ~30% of False items carry an ER_NONCOMPARATIVE noun (teacher, corner,
...) in a noun slot PRESENT in that family's False text -- the subject N1 in
family P, the object noun N_obj in family A (family A-False has no N1 slot, so
salting N1 there was a silent no-op that had halved the effective rate to ~17%).
These nouns end in ``-er`` but are excluded from the comparative form check, so
they stay correctly False while making the (non-battery) "word ending in -er"
distractor disagree on ~30% -- the recipe's salt trick.

base_id = base_id(family, comp_subtype, the non-adjective fillers) -- shared by
both variants, distinct across bases (the schema split assigner rejects dups).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, base_id, equalize_word_count, word_count
from ...schema import PROGRAMMATIC_N_BASES_MIN

# --- banks (rule-specs generation.banks) --------------------------------------
_ADJ_COMPARABLE = "ADJ_COMPARABLE"
_ADJ_PLAIN = "ADJ_PLAIN"
_ER_NONCOMPARATIVE = "ER_NONCOMPARATIVE"
_NOUN_CONCRETE = "NOUN_CONCRETE"
_ADVERB_PLACE = "ADVERB_PLACE"

# Build comfortably more than the 340-base floor (100 + 120 + 100 + >= 20 spare).
_N_BASES = 420

# recipe proportions, applied across bases (deterministic, seeded).
_FRAC_ATTRIBUTIVE = 0.50   # 50% family A (attributive, no 'than'); rest family P
_FRAC_MORE = 0.30          # 30% analytic 'more {adj}'; rest -er forms
_FRAC_SALT = 0.30          # 30% of False items carry an ER_NONCOMPARATIVE noun

# CVC-doubling -er adjectives in the frozen ADJ_COMPARABLE 'er' subtype. The rest
# either drop a silent 'e' (append 'r') or just append 'er'. (Enumerated against
# the frozen bank; _check_er_forms asserts at build time that every produced
# form matches the groundtruth -er regex.)
_ER_DOUBLE = frozenset({"big", "thin", "wet"})

# Family-P False padding fills the 3-word gap left by dropping "than the {N2}"
# with a "{PREP} the {N3}" PP whose last word stays a shared NOUN_CONCRETE noun.
# The PP's PREPOSITION is INCIDENTAL (pure padding, not rule-bearing), so it must
# NOT become a single-token class cue. We therefore spread it across this pool of
# locative prepositions (each grammatical with "the {concrete noun}"), assigned
# per base, so no single preposition token sits anywhere near the 0.75 battery
# threshold. (Earlier versions always used "in the {N3}", which made 'in' a
# 0.767 single-token False cue -- the confound this fix removes.) None of these
# prepositions end in '-er', so the groundtruth comparative check is untouched.
# NB: every preposition here must be groundtruth-inert -- in particular none may
# end in '-er' (e.g. "under"), or the comparative-form check would fire on the
# padding and flip a False item to True. _check_false_preps asserts this at build.
_FALSE_PREPS = (
    "in",
    "on",
    "near",
    "by",
    "behind",
    "beside",
    "beyond",
    "inside",
    "atop",
    "below",
)


def _comparative_er(adj: str) -> str:
    """The synthetic ``-er`` comparative of a base ADJ_COMPARABLE 'er' adjective."""
    if adj in _ER_DOUBLE:
        return f"{adj}{adj[-1]}er"
    if adj.endswith("e"):
        return f"{adj}r"
    return f"{adj}er"


@dataclass(frozen=True)
class Base:
    """A rule-12 base: a frame family + a fixed comparative subtype + all the
    non-adjective fillers (shared by both variants) + the True/False adjectives
    + the per-base False-only salt plan. ``base_id`` mixes the frame family,
    the comp subtype and the non-adjective fillers (the recipe's base_id)."""

    base_id: str
    family: str            # "P" (predicative, with 'than') | "A" (attributive)
    comp_subtype: str      # "er" | "more"
    comp_form: str         # the True adjective surface (e.g. "taller", "careful")
    plain_adj: str         # the False adjective surface (plain ADJ_PLAIN)
    n1: str
    n2: str
    n3: str                # P-False tail noun ("{PREP} the {N3}"); unused in fam A
    n_obj: str             # the object noun in family A ("a {COMP} {N_obj}")
    false_prep: str        # the incidental P-False PP preposition (family P only)
    salt: bool             # does the False variant carry an ER_NONCOMPARATIVE noun?
    salt_word: str         # the salt noun (empty if not salted)
    salt_slot: str         # which noun slot the salt replaces in the False text


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (GENERATOR INTERFACE).

    Deterministic given ``gen``: draw the per-base frame family, comparative
    subtype, adjectives, nouns and salt plan from seeded streams; reject duplicate
    base_ids; raise (loud) if the floor cannot be reached."""
    comp_bank = banks.get_bank(_ADJ_COMPARABLE)
    er_adjs = [e.word for e in comp_bank.entries if e.subtype == "er"]
    # 'more {adj}' analytic comparatives: the groundtruth check treats
    # 'more {word ending in -s}' as a plural-noun quantifier ('more chairs'),
    # NOT a comparative, so an s-ending adjective ('famous') would make the
    # True variant recompute to False. Restrict the 'more' pool to non-s
    # adjectives so every analytic comparative is groundtruth-True.
    more_adjs = [
        e.word
        for e in comp_bank.entries
        if e.subtype == "more" and not e.word.endswith("s")
    ]
    if not more_adjs:
        raise ValueError(
            "contains_comparative: no non-'-s' 'more' adjectives available"
        )
    plain_adjs = [e.word for e in banks.get_bank(_ADJ_PLAIN).entries]
    salt_nouns = [e.word for e in banks.get_bank(_ER_NONCOMPARATIVE).entries]

    # filler nouns: NOUN_CONCRETE minus any '-er' ender (an '-er' noun in a slot
    # would trip the groundtruth comparative check and is not in the salt list).
    nouns = [
        e.word
        for e in banks.get_bank(_NOUN_CONCRETE).entries
        if not _ends_er(e.word)
    ]
    if len(nouns) < 4:
        raise ValueError("contains_comparative: too few non-'-er' filler nouns")

    _check_er_forms(er_adjs)
    _check_false_preps()
    _check_extra_adjs(plain_adjs)

    g_fam = gen.derive("family")
    g_sub = gen.derive("subtype")
    g_adj = gen.derive("adj")
    g_noun = gen.derive("noun")
    g_salt = gen.derive("salt")
    g_prep = gen.derive("false_prep")

    bases: list[Base] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = _N_BASES * 60
    while len(bases) < _N_BASES and attempts < max_attempts:
        attempts += 1
        family = "A" if g_fam.rng.random() < _FRAC_ATTRIBUTIVE else "P"
        comp_subtype = "more" if g_sub.rng.random() < _FRAC_MORE else "er"
        if comp_subtype == "er":
            adj = g_adj.choice(er_adjs)
            comp_form = _comparative_er(adj)
        else:
            adj = g_adj.choice(more_adjs)
            comp_form = adj  # surfaced as "more {adj}" in instantiate
        plain_adj = g_adj.choice(plain_adjs)

        # four distinct nouns (n1, n2, n3, n_obj) so no text repeats a noun
        # within itself (keeps each surface natural and the dedup clean).
        n1, n2, n3, n_obj = g_noun.sample(nouns, 4)

        # incidental P-False PP preposition (spread across _FALSE_PREPS so no
        # single preposition token becomes a class cue). Family A does not use it.
        false_prep = g_prep.choice(_FALSE_PREPS)

        # per-base salt plan (False-only). Salt replaces a noun slot that is
        # actually PRESENT in the False text of this family with an
        # ER_NONCOMPARATIVE noun. Family P-False renders n1 ("The {n1} is ...");
        # family A-False has NO n1 slot ("The team wants a {adj} {n_obj} for the
        # {n2}"), so for family A we salt the object noun n_obj instead -- both
        # are stable, always-present slots in their family's False text. (Salting
        # n1 in family A was a silent no-op, which had dropped the effective salt
        # rate to ~17%; salting a present slot restores the recipe's ~30%.)
        salt = g_salt.rng.random() < _FRAC_SALT
        salt_word = ""
        salt_slot = ""
        if salt:
            salt_word = g_salt.choice(salt_nouns)
            salt_slot = "n1" if family == "P" else "n_obj"

        bid = base_id(
            "contains_comparative", family, comp_subtype, n1, n2, n3, n_obj,
            comp_form, plain_adj, false_prep, int(salt), salt_word,
        )
        if bid in seen:
            continue
        seen.add(bid)
        bases.append(
            Base(
                base_id=bid,
                family=family,
                comp_subtype=comp_subtype,
                comp_form=comp_form,
                plain_adj=plain_adj,
                n1=n1,
                n2=n2,
                n3=n3,
                n_obj=n_obj,
                false_prep=false_prep,
                salt=salt,
                salt_word=salt_word,
                salt_slot=salt_slot,
            )
        )

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"contains_comparative: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (GENERATOR INTERFACE).

    True  = the comparative-form sentence (groundtruth labels True).
    False = the plain-adjective sentence, count-equalized to the True variant's
    word count with ADVERB_PLACE adjuncts, optionally carrying an
    ER_NONCOMPARATIVE salt noun (groundtruth labels False).
    Deterministic given (spec, label, gen)."""
    if label:
        text = _true_text(spec)
        transform = "comparative"
        used_salt = False
    else:
        text, used_salt = _false_text(spec, gen)
        transform = "plain_adj"

    meta = {
        "family": spec.family,
        "comp_subtype": spec.comp_subtype,
        "comp_form": spec.comp_form,
        "plain_adj": spec.plain_adj,
        "n1": spec.n1,
        "n2": spec.n2,
        "n3": spec.n3,
        "n_obj": spec.n_obj,
        "false_prep": spec.false_prep if (not label and spec.family == "P") else "",
        "salt": used_salt,
        "salt_word": spec.salt_word if used_salt else "",
        "transform": transform,
        "target_word_count": _target_word_count(spec),
    }
    return text, meta


# --- variant surface builders -------------------------------------------------


def _comp_surface(spec: Base) -> str:
    """The True adjective phrase: 'taller' (er) or 'more careful' (more)."""
    if spec.comp_subtype == "more":
        return f"more {spec.comp_form}"
    return spec.comp_form


def _true_text(spec: Base) -> str:
    comp = _comp_surface(spec)
    if spec.family == "P":
        return f"The {spec.n1} is {comp} than the {spec.n2}"
    return f"The team wants a {comp} {spec.n_obj} for the {spec.n2}"


def _target_word_count(spec: Base) -> int:
    """The shared word count of both variants (= the True variant's count)."""
    return word_count(_true_text(spec))


def _false_text(spec: Base, gen: Gen) -> tuple[str, bool]:
    """The plain-adjective False variant, count-equalized to the True count.

    Returns (text, used_salt). The last word stays a NOUN_CONCRETE noun (N2 in
    family A, N3 in family P). In the 'more' case a single EXTRA plain attributive
    adjective is placed GRAMMATICALLY inside a noun phrase to equalize the word
    count to the True variant (family A: "a {adj2} {adj} {N_obj}"; family P:
    "{PREP} the {adj2} {N3}") -- the NP stays intact and the last word stays a
    noun. The family-P PP preposition is the per-base ``false_prep`` (spread
    across _FALSE_PREPS so it is never a single-token class cue). Salt (if
    planned) replaces a present noun slot -- N1 in family P, the object noun
    N_obj in family A -- with an ER_NONCOMPARATIVE noun, still groundtruth-False
    since the salt list is excluded from the comparative check. The ADVERB_PLACE
    adjunct pool is retained only for the defensive ``equalize_word_count``
    safety net below (a no-op by construction)."""
    target = _target_word_count(spec)
    adjuncts_by_len = _adjuncts_by_len()

    if spec.family == "P":
        # "The {N1} is {adj} {PREP} the {N3}" stem (7 words) ends in N3, with the
        # incidental preposition drawn per base from _FALSE_PREPS. For the 'more'
        # case (target 8) a 1-word EXTRA attributive adjective is placed inside
        # the final NP ("{PREP} the {adj2} {N3}") so the last word stays N3 and
        # the phrase stays grammatical. Salt (family P) replaces the subject N1.
        n1 = spec.salt_word if (spec.salt and spec.salt_slot == "n1") else spec.n1
        used_salt = bool(spec.salt and spec.salt_slot == "n1" and spec.salt_word)
        if spec.comp_subtype == "er":
            pp = f"{spec.false_prep} the {spec.n3}"
            text = f"The {n1} is {spec.plain_adj} {pp}"
        else:
            # target 8: 4-word core ("The N1 is adj") + a 4-word PP that carries
            # an extra adjective inside its NP ("PREP the adj2 N3") -> last word
            # N3, total 8, "the {adj2} {N3}" grammatical.
            adj2 = _extra_adj(spec, gen)
            pp = f"{spec.false_prep} the {adj2} {spec.n3}"
            text = f"The {n1} is {spec.plain_adj} {pp}"
    else:
        # family A. "The team wants a {adj} {N_obj} for the {N2}" = 9 words (= 'er'
        # target). For 'more' (target 10) an extra attributive adjective is
        # STACKED before {adj} ("a {adj2} {adj} {N_obj}") so the last word stays
        # N2 and the object NP stays grammatical. Salt (family A) replaces N_obj
        # (family A-False has no N1 slot to salt).
        n_obj = (
            spec.salt_word if (spec.salt and spec.salt_slot == "n_obj") else spec.n_obj
        )
        used_salt = bool(spec.salt and spec.salt_slot == "n_obj" and spec.salt_word)
        if spec.comp_subtype == "er":
            text = f"The team wants a {spec.plain_adj} {n_obj} for the {spec.n2}"
        else:
            adj2 = _extra_adj(spec, gen)
            text = (
                f"The team wants a {adj2} {spec.plain_adj} {n_obj} "
                f"for the {spec.n2}"
            )

    # Defensive: close any residual gap exactly (the count solver). By design the
    # builders above already hit the target, so this is a no-op, but it guarantees
    # exact equality even if a frame changes.
    if word_count(text) != target:
        text = equalize_word_count(text, target, adjuncts_by_len, gen)
    return text, used_salt


# --- helpers ------------------------------------------------------------------


def _extra_adj(spec: Base, gen: Gen) -> str:
    """A 1-word EXTRA plain attributive adjective for the 'more'-case word-count
    equalizer, drawn from ADJ_PLAIN excluding this base's own ``plain_adj`` so the
    two stacked adjectives differ (grammatical: "a small national sofa", not "a
    national national sofa"). ADJ_PLAIN carries no '-er' ender and no 'more'
    token (asserted in build_bases via the bank), so this fill is groundtruth-
    inert: it cannot flip the False label to True."""
    pool = [a for a in _plain_adjs() if a != spec.plain_adj]
    return gen.choice(pool)


def _plain_adjs() -> list[str]:
    """The ADJ_PLAIN surface words (the source of both the False {adj} and the
    'more'-case extra equalizer adjective)."""
    return [e.word for e in banks.get_bank(_ADJ_PLAIN).entries]


def _ends_er(word: str) -> bool:
    """True iff a (>=3-letter) lowercase word ends in '-er' (the groundtruth
    comparative-form regex shape)."""
    w = word.lower()
    return len(w) >= 3 and w.endswith("er")


def _adjuncts_by_len() -> dict[int, list[str]]:
    """ADVERB_PLACE phrases grouped by word count {1: [...], 2: [...], 3: [...]}.

    Excludes any phrase whose last token ends in '-er' (e.g. 'around the corner')
    so an adjunct can never change a sentence's last word into an '-er' token --
    keeping the groundtruth check independent of the count-equalizer."""
    by_len: dict[int, list[str]] = {}
    for e in banks.get_bank(_ADVERB_PLACE).entries:
        phrase = e.word
        toks = phrase.split()
        if _ends_er(toks[-1]):
            continue
        by_len.setdefault(len(toks), []).append(phrase)
    if 1 not in by_len:
        raise ValueError("contains_comparative: no 1-word ADVERB_PLACE adjunct")
    return by_len


def _check_false_preps() -> None:
    """Assert no incidental P-False preposition ends in '-er' (which the
    groundtruth comparative regex [a-z]{3,}er$ would match, flipping a False item
    to True) -- a build-time guard mirroring _check_er_forms for the padding."""
    bad = [p for p in _FALSE_PREPS if _ends_er(p)]
    if bad:
        raise ValueError(
            f"contains_comparative: P-False prepositions ending in '-er' would "
            f"trip the comparative-form check: {bad}"
        )


def _check_extra_adjs(plain_adjs: list[str]) -> None:
    """Assert the ADJ_PLAIN pool can supply the 'more'-case extra equalizer
    adjective groundtruth-inertly: at least 2 entries (so excluding the base's
    own plain_adj still leaves a choice), and NO entry ending in '-er' (which the
    comparative regex [a-z]{3,}er$ would match, flipping a False item to True).
    Mirrors _check_false_preps / _check_er_forms for the second-adjective fill."""
    if len(plain_adjs) < 2:
        raise ValueError(
            "contains_comparative: ADJ_PLAIN needs >= 2 entries for the "
            "'more'-case extra equalizer adjective"
        )
    bad = [a for a in plain_adjs if _ends_er(a)]
    if bad:
        raise ValueError(
            f"contains_comparative: ADJ_PLAIN adjectives ending in '-er' would "
            f"trip the comparative-form check as the equalizer fill: {bad}"
        )


def _check_er_forms(er_adjs: list[str]) -> None:
    """Assert every synthetic '-er' form matches the groundtruth -er regex
    ([a-z]{3,}er$) -- a build-time guard so a bank edit can't silently produce a
    comparative the verifier would not recognise."""
    import re

    rx = re.compile(r"[a-z]{3,}er$")
    bad = [a for a in er_adjs if not rx.search(_comparative_er(a))]
    if bad:
        raise ValueError(
            f"contains_comparative: '-er' forms not matching groundtruth regex "
            f"for base adjectives {bad}"
        )
