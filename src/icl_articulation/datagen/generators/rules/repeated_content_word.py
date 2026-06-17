"""Rule 6: repeated_content_word (category surface).

Canonical articulation (groundtruth._r6): True iff some CONTENT word — a word not
on the frozen globals.stopword list — occurs >= 2 times, matched
case-insensitively after stripping punctuation. No lemmatization ('dog' != 'dogs').

CONSTRUCTION (rule-specs generation.recipe + distribution_guards, verbatim
constraints honored):

  * Coordination / two-clause frames, 8-12 words, >= 12 frames covering
    NOUN-, VERB-, and ADJECTIVE-repeats. Each frame starts with a determiner
    ("the"/"a") and is built so that the SINGLE manipulated content slot ``{R}``
    is INTERIOR (never the first or last word). A frame names, statically:
        - the bank that fills ``{R}`` (and so the POS of the repeat),
        - the "source" slot whose word the True variant copies into ``{R}``.

  * base_id = frame + filler set. BOTH variants of a base use the SAME frame and
    the SAME non-repeated fillers (distribution_guards: "near-identical
    vocabulary across classes; True items reuse one filler instead of drawing a
    new one"). The two variants differ in EXACTLY ONE token (the ``{R}`` slot):
        False -> {R} = its own distinct filler  -> all content slots distinct,
                 and the generator VERIFIES no accidental content repeat (-> the
                 rule labels it False).
        True  -> {R} = a copy of the source slot's word -> that content word now
                 occurs twice (-> the rule labels it True).
    Because the variants differ in one interior token, every positional /
    casing / frame-level battery predicate (first-/last-word features, the-count,
    contains-and, first-word POS, ...) takes the SAME value on both variants of a
    base — they cannot separate the classes. The only feature that moves is the
    char count of that one token; we draw the False filler from the SAME bank as
    the copied source word, so the per-base char delta is mean-zero and the
    char-count predicates stay near 50% (the confound report length-matches on
    WORD count, which is identical per base by construction -> |dT-dF| = 0).

  * STOPWORD-REPEAT REQUIREMENT (>= 60% of items in BOTH classes contain a
    repeated stopword): every frame repeats "the" (or "a") by design, so 100% of
    items in BOTH classes carry a stopword repeat. This blocks the "some word
    appears twice (any word)" distractor (False items DO repeat a stopword) and
    keeps the English un-stilted (review B rule-6 fix).

  * REPEATED POS VARIES (>= 30% non-noun repeats): the frame set is split across
    noun / verb / adjective repeat frames; build_bases enforces that >= 30% of
    bases are non-noun-repeat frames.

A FIXED seed threads the whole build (logged in slots_meta by the pipeline and
again here in ``transform``/provenance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks as banks_mod
from ...genutils import Gen, fill_frame, to_sentence_case
from ...groundtruth import STOPWORDS, _r6_repeated_content_word
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count, words

# Style policy: rule 6 falls through to the strict global default (no terminal,
# no comma) — no alias needed, so no STYLE_RULE_ID export.

_NOUN = "NOUN_CONCRETE"
_VERB = "VERB_REGULAR"
_ADJ = "ADJ_PLAIN"

# Build comfortably above the 340-base floor (100 + 120 + 100 + >= 20 spare).
_N_BASES = 360

# spec'd frame window
_MIN_WORDS, _MAX_WORDS = 8, 12

# >= 30% of bases must be non-noun-repeat frames (distribution_guards).
_MIN_NONNOUN_SHARE = 0.30


@dataclass(frozen=True)
class _Frame:
    """One coordination / two-clause frame.

    ``template`` has named {slots}; ``slot_banks`` maps each slot to its bank
    (the POS the slot carries). ``r_slot`` is the single INTERIOR manipulated
    slot; ``src_slot`` is the slot whose word the True variant copies into
    ``r_slot``. ``r_pos`` is the coarse POS the repeat carries (for the
    distribution guard). r_slot and src_slot must draw from the SAME bank so the
    char-delta of the swap is mean-zero."""

    name: str
    template: str
    slot_banks: dict[str, str]
    r_slot: str
    src_slot: str
    r_pos: str  # "noun" | "verb" | "adjective"


# -- the frames. Every frame: starts with "the"/"a"; r_slot & src_slot interior
# (never the first or last token); r_slot & src_slot share a bank; NO hard-coded
# content word is repeated inside a frame (so the False, all-distinct variant has
# ZERO content repeats and the True variant's ONLY repeat is the R/SRC copy).
# Every frame repeats the stopword "the" or "a" by design (>= 60% stopword-repeat
# requirement -> 100% here). Word counts are asserted into [8, 12] at build time.
_FRAMES: list[_Frame] = [
    # ---- NOUN repeats (r_slot & src_slot are nouns) -----------------------
    # the two coordinated clauses use DIFFERENT hard-coded verbs/preps so the
    # only content repeat is the R/SRC noun.
    _Frame(
        "n_watch", "the {SRC} watched the {R} and the {C} guarded the {D}",
        {"SRC": _NOUN, "R": _NOUN, "C": _NOUN, "D": _NOUN}, "R", "SRC", "noun",
    ),
    _Frame(
        "n_near", "the {SRC} stood near the {R} and the {C} watched the {D}",
        {"SRC": _NOUN, "R": _NOUN, "C": _NOUN, "D": _NOUN}, "R", "SRC", "noun",
    ),
    _Frame(
        "n_carry", "the {SRC} held the {R} as the {C} lifted the {D}",
        {"SRC": _NOUN, "R": _NOUN, "C": _NOUN, "D": _NOUN}, "R", "SRC", "noun",
    ),
    _Frame(
        "n_beside", "a {SRC} rested by a {R} and a {C} guarded a {D}",
        {"SRC": _NOUN, "R": _NOUN, "C": _NOUN, "D": _NOUN}, "R", "SRC", "noun",
    ),
    _Frame(
        "n_under", "the {SRC} fell under the {R} and the {C} passed the {D}",
        {"SRC": _NOUN, "R": _NOUN, "C": _NOUN, "D": _NOUN}, "R", "SRC", "noun",
    ),
    _Frame(
        "n_above", "the {SRC} hung above the {R} and the {C} hid the {D}",
        {"SRC": _NOUN, "R": _NOUN, "C": _NOUN, "D": _NOUN}, "R", "SRC", "noun",
    ),
    # ---- VERB repeats (r_slot & src_slot are verbs) -----------------------
    # verbs sit interior; subjects are nouns. "to {SRC}" / "to {R}" keeps the verb
    # forms BARE (no inflection) so the case-insensitive exact match is clean and
    # no -ed/-s morphology is introduced. The two hard-coded matrix verbs differ.
    _Frame(
        "v_tried", "the workers tried to {SRC} and the helpers wanted to {R} today",
        {"SRC": _VERB, "R": _VERB}, "R", "SRC", "verb",
    ),
    _Frame(
        "v_began", "a student hoped to {SRC} before a teacher chose to {R} again",
        {"SRC": _VERB, "R": _VERB}, "R", "SRC", "verb",
    ),
    _Frame(
        "v_wanted", "the farmer agreed to {SRC} and the driver refused to {R} later",
        {"SRC": _VERB, "R": _VERB}, "R", "SRC", "verb",
    ),
    _Frame(
        "v_chose", "the captain learned to {SRC} while the sailor failed to {R} instead",
        {"SRC": _VERB, "R": _VERB}, "R", "SRC", "verb",
    ),
    _Frame(
        "v_hoped", "a singer offered to {SRC} and a dancer chose to {R} today",
        {"SRC": _VERB, "R": _VERB}, "R", "SRC", "verb",
    ),
    # ---- ADJECTIVE repeats (r_slot & src_slot are adjectives) -------------
    # the two hard-coded nouns differ; no hard-coded content word repeats.
    _Frame(
        "a_two", "the {SRC} table and the {R} chair stood in the room",
        {"SRC": _ADJ, "R": _ADJ}, "R", "SRC", "adjective",
    ),
    _Frame(
        "a_box", "a {SRC} box and a {R} basket sat near the gate",
        {"SRC": _ADJ, "R": _ADJ}, "R", "SRC", "adjective",
    ),
    _Frame(
        "a_house", "the {SRC} house and the {R} barn stood beside the river",
        {"SRC": _ADJ, "R": _ADJ}, "R", "SRC", "adjective",
    ),
    _Frame(
        "a_road", "a {SRC} road and a {R} bridge crossed the wide valley",
        {"SRC": _ADJ, "R": _ADJ}, "R", "SRC", "adjective",
    ),
    _Frame(
        "a_garden", "the {SRC} garden and the {R} field lay behind the barn",
        {"SRC": _ADJ, "R": _ADJ}, "R", "SRC", "adjective",
    ),
]


def _frame_fixed_content_words(fr: _Frame) -> set[str]:
    """Lowercased CONTENT words hard-coded into a frame's template (not slots).

    These count toward the rule's repeat check, so the generator must guarantee
    no slot filler collides with them (else an accidental content repeat would
    flip a False variant's ground-truth label)."""
    # tokens that are literal in the template (slots removed), minus stopwords
    bare = fr.template
    for slot in fr.slot_banks:
        bare = bare.replace("{" + slot + "}", " ")
    return {w.lower() for w in words(bare) if w.lower() not in STOPWORDS}


@dataclass(frozen=True)
class Base:
    """A repeated_content_word base spec.

    ``fillers`` is the shared, all-distinct filler set for the frame's slots
    (the False variant). ``r_word`` is the distinct filler the False variant puts
    in the manipulated slot; the True variant overwrites it with the source
    slot's word. ``base_id`` is hashed from (frame, sorted fillers)."""

    base_id: str
    frame_name: str
    fillers: tuple[tuple[str, str], ...]  # (slot, word) for ALL slots incl. r_slot
    r_pos: str


def _frame_by_name(name: str) -> _Frame:
    for fr in _FRAMES:
        if fr.name == name:
            return fr
    raise KeyError(name)


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct base specs (the GENERATOR INTERFACE).

    For each base: pick a frame, draw DISTINCT content fillers for all of its
    slots (so the False variant has no content repeat), drawing the manipulated
    slot ``{R}``'s filler from the SAME bank as the source slot (mean-zero char
    delta on the True swap). Guarantees: all content slots distinct from each
    other AND from the frame's hard-coded content words; word count in [8, 12];
    distinct base_id; BOTH the True and the False surface globally unique (the
    True variant overwrites {R} with {SRC}, so two bases differing only in their
    {R} filler would share a True surface — deduped here on both surfaces);
    >= 30% non-noun-repeat frames."""
    bank_words = {
        name: banks_mod.get_bank(name).words()
        for name in (_NOUN, _VERB, _ADJ)
    }

    bases: list[Base] = []
    seen_ids: set[str] = set()
    seen_surfaces: set[str] = set()  # every emitted True AND False surface
    nonnoun = 0

    # round-robin over frames so the noun/verb/adj mix is even; seeded draws per
    # base make the filler sets reproducible.
    attempts = 0
    max_attempts = _N_BASES * 400
    fi = 0
    while len(bases) < _N_BASES and attempts < max_attempts:
        attempts += 1
        fr = _FRAMES[fi % len(_FRAMES)]
        fi += 1

        fixed = _frame_fixed_content_words(fr)
        # draw distinct fillers for every slot, avoiding the frame's fixed words
        used: set[str] = set(fixed)
        chosen: dict[str, str] = {}
        ok = True
        for slot, bank in fr.slot_banks.items():
            pool = [w for w in bank_words[bank] if w.lower() not in used]
            if not pool:
                ok = False
                break
            w = gen.choice(pool)
            chosen[slot] = w
            used.add(w.lower())
        if not ok:
            continue

        # base_id = frame + filler SET (sorted slot->word pairs), hashed
        filler_pairs = tuple(sorted(chosen.items()))
        bid = _make_base_id(fr.name, filler_pairs)
        if bid in seen_ids:
            continue

        # both variant surfaces (False = distinct fillers; True = {R}<-{SRC})
        false_text = to_sentence_case(fill_frame(fr.template, chosen))
        true_fillers = {**chosen, fr.r_slot: chosen[fr.src_slot]}
        true_text = to_sentence_case(fill_frame(fr.template, true_fillers))

        wc = word_count(false_text)
        if not (_MIN_WORDS <= wc <= _MAX_WORDS):
            continue
        if false_text in seen_surfaces or true_text in seen_surfaces:
            continue
        if false_text == true_text:  # cannot happen (R!=SRC) but defensive
            continue

        seen_ids.add(bid)
        seen_surfaces.add(false_text)
        seen_surfaces.add(true_text)
        bases.append(
            Base(
                base_id=bid,
                frame_name=fr.name,
                fillers=filler_pairs,
                r_pos=fr.r_pos,
            )
        )
        if fr.r_pos != "noun":
            nonnoun += 1

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"repeated_content_word: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    share = nonnoun / len(bases)
    if share < _MIN_NONNOUN_SHARE:
        raise ValueError(
            f"repeated_content_word: only {share:.2%} non-noun repeats, need "
            f">= {_MIN_NONNOUN_SHARE:.0%}"
        )
    return bases


def _make_base_id(frame_name: str, filler_pairs: tuple[tuple[str, str], ...]) -> str:
    from ...genutils import base_id as _bid

    return _bid("r6", frame_name, *(f"{s}={w}" for s, w in filler_pairs))


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    False: fill the frame with the base's distinct fillers (no content repeat).
    True:  copy the source slot's word into the manipulated slot (one content
    word now occurs twice). Deterministic; ``gen`` is unused (no randomness at
    instantiate time — all draws happened in build_bases)."""
    fr = _frame_by_name(spec.frame_name)
    fillers = {s: w for s, w in spec.fillers}

    if label:
        src_word = fillers[fr.src_slot]
        fillers = {**fillers, fr.r_slot: src_word}
        transform = "repeat_source_into_R"
        repeated_word = src_word
    else:
        transform = "all_distinct"
        repeated_word = None

    text = to_sentence_case(fill_frame(fr.template, fillers))

    # ground-truth self-check (defensive; the pipeline's GATE B re-checks this).
    if _r6_repeated_content_word(text) != label:
        raise ValueError(
            f"repeated_content_word: built text whose rule label != intended "
            f"{label}: {text!r} (frame {fr.name})"
        )

    meta = {
        "frame": fr.name,
        "r_slot": fr.r_slot,
        "src_slot": fr.src_slot,
        "r_pos": fr.r_pos,
        "transform": transform,
        "repeated_word": repeated_word,
        "fillers": {s: w for s, w in spec.fillers},
    }
    return text, meta
