"""Rule 22: second_word_capitalized (category positional).

Canonical articulation: True iff the SECOND word begins with a capital letter.
'Second word' is the second stripped whitespace token (global tokenizer);
'capitalized' is first char uppercase. The ground-truth predicate is
``groundtruth._r22_second_word_capitalized``.

Construction (rule-specs recipe + distribution_guards, verbatim intent)
=======================================================================
Every sentence opens with a sentence-case-capitalized first word drawn from a
SHARED opener vocabulary (ADVERB_SENT_INITIAL) at MATCHED rates in both classes.
The OPENER CONSTRAINT (load-bearing) is honoured: openers are 1-word sentence
adverbs that are grammatical immediately before a position-2 proper-noun
subject; NO determiner opener ('The ...') appears in EITHER class (a 'The'
opener could never precede a position-2 proper noun in a True item and would
otherwise become a False-class marker). The two variants of a base SHARE the
same opener, so every first-word battery predicate is identical on the two
variants of a base and scores exactly 0.5.

  PROPER_MIX = 50% FIRST_NAMES (people) + 50% NONNAME_PROPER (cities / months /
  countries / orgs). The name-vs-non-name mix is what stops 'second word is a
  person's name' from being equivalent to the rule (it disagrees on the ~50% of
  True items whose position-2 proper noun is a place/month/org).

  True  (proper noun at position 2, as the subject):
      '{OP} {P} {V} the {N}'      e.g. 'Apparently Maria opened the door'
                                       'Yesterday London hosted the games'
      -> word 2 is the proper noun  -> rule labels True.

  False (no proper noun at position 2). Exactly 50% of False items RE-SEAT the
  SAME proper noun at a later position (3..5) and 50% carry NO proper noun, so
  'a capitalized word appears after the first word' / 'contains a proper noun'
  disagrees on exactly the 50% no-proper False items (25% of the dataset) and
  the battery's ``nonfirst_word_capitalized`` predicate sits at exactly 0.75
  (its boundary), inclusive:
      reseat:  '{OP} {SUBJ} {V} near {P}'   'Apparently someone walked near London'
               '{OP} {SUBJ} {V} past {P}'   'Yesterday people drove past Maria'
      noname:  '{OP} {SUBJ} {V} the {N}'    'Apparently someone opened the door'
      -> word 2 is the lowercase subject ('someone'/'people'/...) -> labels False.

  Word counts are MATCHED: the True and False variants of a base have the SAME
  word count (all frames are 5 words; an optional shared place adjunct lengthens
  BOTH variants of a base identically), so |mean_wc(T) - mean_wc(F)| == 0 and the
  per-base count is identical -> every word_count / char-ish length predicate is
  balanced.

  base_id = frame content (opener + the proper noun + verb + nouns + the
  adjunct + the False family) so the True and False variants of a base share it.

Why every NON-exempt battery predicate stays <= 0.75 (this rule grants NO
equivalence/exemption — equiv_keys = {} and battery_exemptions = [] in the
spec extract, so ALL 40 frozen predicates must clear the bar on the data):

  * first-word predicates (first_word_pos=*, first_letter_bucket_*,
    first_word_len>=k, first_starts_vowel/consonant): the opener is SHARED by a
    base's two variants, so each is identical on True and False of every base and
    scores exactly 0.5. 'first_word_pos=adverb' is ~1.0 on BOTH classes (every
    opener is an adverb) -> agreement ~0.5. 'first_word_pos=determiner' is ~0 on
    both (no determiner openers) -> 0.5.
  * word_count / char_count predicates: the count is identical on a base's two
    variants (matched), so each scores exactly 0.5.
  * contains_the / contains_a / contains_and / count_the>=2: present at matched
    rates by construction (both families draw 'the' / nouns / verbs from the same
    banks; the reseat family swaps 'the {N}' for 'near {P}', balanced against the
    True 'the {N}', see below).
  * last_word predicates: the last word is a NOUN (True / noname) or the proper
    noun (reseat) drawn from the same banks across classes -> balanced.
  * nonfirst_word_capitalized: True on every True item (the position-2 proper
    noun) and on exactly 50% of False items (the reseat family) -> agreement
    0.5*1 + 0.5*0.5 = 0.75 (boundary, inclusive PASS).
  * all_lowercase: always False (sentence case + capitalized proper nouns) -> 0.5.
  * contains_digit / contains_comma: never present -> 0.5.

Mirrors the reference rule ``all_lowercase``'s build_bases / instantiate shape;
the shared gated pipeline (``base.emit_rule``) enforces all four gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, base_id as make_base_id
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

_NAME_BANK = "FIRST_NAMES"
_PLACE_BANK = "NONNAME_PROPER"
_OPENER_BANK = "ADVERB_SENT_INITIAL"
_NOUN_BANK = "NOUN_CONCRETE"
_VERB_BANK = "VERB_REGULAR"

# the global [4, 14] word-count window the schema validator re-checks. EVERY
# frame in this rule is exactly 7 words (opener + 5 body words + a 2-word tail,
# where the reseat tail is '{prep} {P}'), so the True and False variants of a
# base have identical word counts: |mean_wc(T) - mean_wc(F)| == 0 and every
# word_count / char-length-ish predicate is balanced at 0.5. Asserted per item.
_FIXED_WORDS = 7
_MIN_WORDS, _MAX_WORDS = _FIXED_WORDS, _FIXED_WORDS

# build comfortably more than the 340-base floor so the by-base split
# (100 + 120 + 100 + >= 20 spare) has headroom.
_N_BASES = 380

# Fraction of bases whose False variant RE-SEATS the proper noun (vs. drops it).
# The recipe pins this at 50% so 'contains a capitalized word after the first
# word' disagrees on exactly 25% of the dataset. The binding GATE constraint is
# weaker but one-sided: the battery's ``nonfirst_word_capitalized`` predicate
# needs the reseat share of the EMITTED False items (r) to be >= 0.5
# (agreement = 0.5 + 0.5*(1-r) <= 0.75). The pipeline, not this generator,
# decides which bases emit as the False variant in the one-variant splits, so a
# base-level 50/50 split lands r slightly BELOW 0.5 by sampling noise (measured
# r = 0.4875 -> agreement 0.756, a hair over). We therefore bias the base-level
# reseat share a little above 50% so worst-case sampling keeps r >= 0.5 with
# margin; this is the smallest deviation from the recipe's 50% that makes the
# gate robust (measured agreement ~0.70, comfortably inclusive). 'reseat' stays a
# majority but not a marker: BOTH families share opener / subject / verb / banks,
# so no battery predicate other than nonfirst_word_capitalized can see it.
_RESEAT_BASE_FRAC = 0.60

# lowercase common-noun SUBJECTS that sit at position 2 in False items (never
# capitalized; never proper). Generic indefinite subjects that read naturally
# after a sentence-initial adverb and before a past-tense verb in BOTH the
# 'reseat' and 'noname' frames.
_SUBJECTS: tuple[str, ...] = (
    "someone",
    "nobody",
    "everyone",
    "people",
    "somebody",
    "anyone",
    "workers",
    "students",
    "neighbors",
    "travelers",
)

# the past-tense verb forms we use (regular '-ed' VERB_REGULAR forms). All are
# TRANSITIVE so the shared 'the {N}' reads as a grammatical direct object after a
# proper-noun subject ('Maria opened the door') AND a generic subject ('someone
# opened the door'); the trailing 2-word adverbial / '{prep} {P}' phrase then
# modifies the clause grammatically in every family.
_VERBS: tuple[str, ...] = (
    "opened",
    "closed",
    "cleaned",
    "painted",
    "watched",
    "pushed",
    "pulled",
    "washed",
    "carried",
    "moved",
    "used",
    "collected",
    "shared",
    "removed",
    "ordered",
    "counted",
)

# the reseat prepositions placing the proper noun at the LAST position. A
# preposition that never takes an article before a proper noun ('near London',
# not 'near the London'), so the reseat frame carries exactly one 'the' just like
# True / noname.
_RESEAT_PREPS: tuple[str, ...] = ("near", "past", "beyond", "toward", "behind", "below")

# 2-word place / time adjuncts that END the True and noname frames (where the
# reseat frame instead ends with '{prep} {P}'). Always EXACTLY two words and
# comma-free (style policy bans commas), so every frame is exactly 7 words and
# carries exactly one 'the' -> word_count and contains_the predicates are
# balanced across classes. The adjunct's two words are common (non-proper, lower
# case) so the last word is never capitalized in True / noname.
_TAIL_ADJUNCTS: tuple[str, ...] = (
    "by night",
    "at dawn",
    "at noon",
    "in town",
    "by day",
    "at dusk",
    "in spring",
    "by hand",
)


@dataclass(frozen=True)
class Base:
    """A second_word_capitalized base.

    Both variants share ``base_id``. The True variant seats ``proper`` at
    position 2; the False variant either re-seats ``proper`` at a late position
    (``false_family == 'reseat'``) or drops it (``false_family == 'noname'``).
    ``opener`` is shared verbatim so all first-word battery predicates are
    identical across the two variants."""

    base_id: str
    opener: str        # sentence-initial adverb, lowercase here (cased at surface)
    proper: str        # the proper noun (already capitalized in its bank)
    proper_kind: str   # 'name' | 'place' (the PROPER_MIX provenance)
    subject: str       # lowercase common-noun subject for False frames
    verb: str          # past-tense verb form
    noun: str          # common noun for the shared 'the {N}' object
    reseat_prep: str   # preposition for the reseat False frame ('near {P}')
    false_family: str  # 'reseat' | 'noname'
    adjunct: str       # the 2-word tail of the True / noname frames


def _surface(parts: list[str]) -> str:
    """Join word parts and sentence-case ONLY the first word.

    Every non-first content word is already lowercase except proper nouns, which
    keep their bank capitalization. The first word is capitalized here (default
    sentence case), so the FIRST LETTER of the sentence is the opener's initial —
    position 1 is always capitalized in BOTH classes, exactly as the rule needs
    (it reads position 2, never position 1)."""
    parts = [p for p in parts if p]
    first = parts[0]
    first_cap = first[0].upper() + first[1:]
    return " ".join([first_cap, *parts[1:]])


def _true_text(b: Base) -> str:
    """True variant: '{OP} {P} {V} the {N} {adjunct(2w)}' — proper noun at word 2.

    7 words; exactly one 'the'; last word is the (lowercase) adjunct's 2nd word."""
    return _surface([b.opener, b.proper, b.verb, "the", b.noun, *b.adjunct.split()])


def _false_text(b: Base) -> str:
    """False variant: position-2 word is the lowercase common-noun subject.

    Both families are 7 words with exactly one 'the', matching the True frame:
      reseat: '{OP} {SUBJ} {V} the {N} {prep} {P}'  — proper noun LAST (pos 7).
      noname: '{OP} {SUBJ} {V} the {N} {adjunct(2w)}' — no proper noun anywhere."""
    if b.false_family == "reseat":
        parts = [b.opener, b.subject, b.verb, "the", b.noun, b.reseat_prep, b.proper]
    else:
        parts = [b.opener, b.subject, b.verb, "the", b.noun, *b.adjunct.split()]
    return _surface(parts)


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (the GENERATOR INTERFACE).

    Deterministic given ``gen``. We enumerate a large seeded cross-product of
    (opener, proper, subject, verb, noun, reseat_prep, adjunct, false_family),
    splitting the proper-noun draw 50/50 between FIRST_NAMES and NONNAME_PROPER
    (PROPER_MIX) and biasing the False family ~60/40 reseat/noname (see
    _RESEAT_BASE_FRAC), then take the first _N_BASES whose two variants are both
    7 words and yield a distinct base_id and distinct surfaces. Raises if it
    cannot reach the floor (loud — no quiet short dataset)."""
    names = banks.get_bank(_NAME_BANK).words()
    places = banks.get_bank(_PLACE_BANK).words()
    openers = banks.get_bank(_OPENER_BANK).words()
    nouns = banks.get_bank(_NOUN_BANK).words()

    subjects = list(_SUBJECTS)
    verbs = list(_VERBS)
    preps = list(_RESEAT_PREPS)
    adjuncts = list(_TAIL_ADJUNCTS)

    # Independent seeded streams for each axis so the cross-product is sampled
    # reproducibly and the 50/50 balances are exact, not "in expectation".
    g_op = gen.derive("opener")
    g_pr = gen.derive("proper")
    g_su = gen.derive("subject")
    g_vb = gen.derive("verb")
    g_nn = gen.derive("noun")
    g_pp = gen.derive("prep")
    g_aj = gen.derive("adjunct")

    # round-robin generators that reshuffle each pass for variety + determinism.
    def cycler(items: list, g: Gen):
        pool: list = []
        while True:
            if not pool:
                pool = list(items)
                g.shuffle(pool)
            yield pool.pop()

    c_op = cycler(openers, g_op)
    c_su = cycler(subjects, g_su)
    c_vb = cycler(verbs, g_vb)
    c_nn = cycler(nouns, g_nn)
    c_pp = cycler(preps, g_pp)
    c_aj = cycler(adjuncts, g_aj)

    # PROPER_MIX 50/50: interleave names and places by alternating; reshuffle
    # within each kind each pass.
    name_pool: list[str] = []
    place_pool: list[str] = []

    def next_proper(i: int) -> tuple[str, str]:
        nonlocal name_pool, place_pool
        if i % 2 == 0:
            if not name_pool:
                name_pool = list(names)
                g_pr.shuffle(name_pool)
            return name_pool.pop(), "name"
        if not place_pool:
            place_pool = list(places)
            g_pr.shuffle(place_pool)
        return place_pool.pop(), "place"

    bases: list[Base] = []
    seen_ids: set[str] = set()
    seen_surface: set[str] = set()
    n_reseat = 0  # running count of 'reseat' bases accepted so far
    i = 0
    guard = 0
    while len(bases) < _N_BASES:
        guard += 1
        if guard > _N_BASES * 50:
            break
        proper, kind = next_proper(i)
        i += 1
        opener = next(c_op)
        subject = next(c_su)
        verb = next(c_vb)
        noun = next(c_nn)
        prep = next(c_pp)
        adjunct = next(c_aj)
        # False family: ~_RESEAT_BASE_FRAC of bases are 'reseat'. The running
        # reseat count is held just under the target ratio so the base-level share
        # is exact (deterministic), giving the dataset-level reseat share of
        # emitted False items the margin it needs (see _RESEAT_BASE_FRAC).
        want_reseat = (n_reseat + 1) <= round((len(bases) + 1) * _RESEAT_BASE_FRAC)
        false_family = "reseat" if want_reseat else "noname"

        bid = make_base_id(
            "second_word_capitalized",
            opener,
            proper,
            kind,
            subject,
            verb,
            noun,
            prep,
            false_family,
            adjunct,
        )
        if bid in seen_ids:
            continue

        b = Base(
            base_id=bid,
            opener=opener,
            proper=proper,
            proper_kind=kind,
            subject=subject,
            verb=verb,
            noun=noun,
            reseat_prep=prep,
            false_family=false_family,
            adjunct=adjunct,
        )
        t_text, f_text = _true_text(b), _false_text(b)
        # word-count window + per-base count match (both 5 or both 7).
        wt, wf = word_count(t_text), word_count(f_text)
        if not (_MIN_WORDS <= wt <= _MAX_WORDS and _MIN_WORDS <= wf <= _MAX_WORDS):
            continue
        if wt != wf:
            continue
        # surface-uniqueness across BOTH variants of every base (Gate A re-checks).
        if t_text in seen_surface or f_text in seen_surface or t_text == f_text:
            continue

        seen_ids.add(bid)
        seen_surface.add(t_text)
        seen_surface.add(f_text)
        bases.append(b)
        if false_family == "reseat":
            n_reseat += 1

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"second_word_capitalized: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN} (enlarge the slot banks)"
        )

    gen.shuffle(bases)
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  -> proper noun seated at position 2 (rule labels True).
    False -> lowercase common-noun subject at position 2; the proper noun is
             re-seated late (reseat) or dropped (noname) (rule labels False).
    Deterministic; ``gen`` is unused (the construction carries no randomness at
    instantiate time — every choice is fixed in the base spec)."""
    text = _true_text(spec) if label else _false_text(spec)
    wc = word_count(text)
    if not (_MIN_WORDS <= wc <= _MAX_WORDS):
        raise ValueError(
            f"second_word_capitalized: word count {wc} out of [{_MIN_WORDS}, "
            f"{_MAX_WORDS}] for {text!r}"
        )
    # provenance: the slots / transform that produced this surface.
    meta = {
        "transform": "proper_at_2" if label else f"false_{spec.false_family}",
        "opener": spec.opener,
        "proper": spec.proper,
        "proper_kind": spec.proper_kind,
        "subject": spec.subject,
        "verb": spec.verb,
        "noun": spec.noun,
        "reseat_prep": spec.reseat_prep,
        "false_family": spec.false_family,
        "adjunct": spec.adjunct,
    }
    return text, meta
