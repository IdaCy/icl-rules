"""Rule 4: title_case (plan rule 4, category surface).

Canonical articulation: True iff EVERY word begins with an uppercase letter
(the first alphabetic char of every stripped token is uppercase; tokens with no
alphabetic char are ignored by the quantifier — see ambiguity_notes). This is
'every word capitalized', NOT editorial title case (stopwords are capitalized
too).

GENERATION (rule-specs id: title_case, generation.recipe):
    Bases 5-10 words, common nouns only, sentence case.
      True  variant: capitalize the first alphabetic char of EVERY word.
      False variant: default sentence case, but ~50% of False items get ONE
                     capitalized proper noun from NONNAME_PROPER substituted
                     into a noun slot at a uniformly random position 2..6
                     ('The train to Madrid left early'). To keep vocabulary
                     symmetric, the SAME proper-noun substitution is applied to
                     the paired True variant of that base (which is then fully
                     capitalized anyway).
      base_id = base.

The two variants of a base are WORD-IDENTICAL except for casing (and the salted
proper noun, which is present in BOTH variants of a salted base). So every
generic battery predicate that is not a casing feature has IDENTICAL value on
the two variants of a base and contributes exactly 50% agreement in the
few_shot_pool split (both variants emitted), and stays balanced elsewhere. The
ONLY casing-sensitive predicates the frozen battery carries are:

    all_lowercase             — False on BOTH classes (sentence case and title
                                case both start with a capital), so it sits at
                                the True/False label balance (~50%). PASS.
    nonfirst_word_capitalized — True on every True item (all words capitalized)
                                and on every SALTED False item (mid-sentence
                                proper-noun capital at position 2..6); False on
                                non-salted False items. The proper-noun salt is
                                exactly what keeps this predicate (and the
                                'contains >= 2 capitals' / 'mid-sentence capital'
                                family) under the 0.75 agreement ceiling.

DISTRIBUTION GUARD / SALT RATE.  The recipe pins '~50% of False items'. The
pipeline emits BOTH variants per base in few_shot_pool but only ONE balanced
variant per base in held_out/confirmation/spare, so a per-base 50% salt yields
only ~46% of the *emitted* False items salted, which pushes
nonfirst_word_capitalized to ~0.77 (> 0.75). We therefore salt a slightly higher
per-base fraction (``_SALT_RATE``) so that ~50-55% of the EMITTED False items
carry the salt and the predicate lands comfortably under the ceiling with a few
points of margin, while staying faithful to the recipe's 'about half the False
items carry the mid-sentence proper-noun' intent. The salt is ALWAYS mirrored to
the base's True variant (vocabulary symmetry: 'contains a city/brand' stays
class-balanced), and is NEVER at position 1 or the last position, so first-word
and last-word battery predicates are untouched by the salt.

NO digits, NO commas, NO hyphens, NO terminal punctuation, no 'I' (global
sentence_style; title_case is not in RULE_STYLE_POLICY so it gets the strict
default policy, which the schema gate re-checks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, fill_frame, frame_slots, to_title_case
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count, words

_NOUN_BANK = "NOUN_CONCRETE"
_ADJ_BANK = "ADJ_PLAIN"
_VERB_BANK = "VERB_REGULAR"
_PROPER_BANK = "NONNAME_PROPER"

# Build comfortably above the 340-base floor so the by-base split
# (100 few_shot + 120 held_out + 100 confirmation + >= 20 spare) has headroom.
_N_BASES = 380

# base sentences are 5-10 words by frame design (rule-specs recipe); the schema
# validator re-checks the global [4, 14] window.
_MIN_WORDS, _MAX_WORDS = 5, 10

# Per-BASE salt rate. See module docstring: tuned a touch above the recipe's 50%
# so that ~50-55% of the *emitted* False items carry the salt (the few_shot
# both-variants / one-variant-split asymmetry otherwise drops the emitted-False
# salt rate below 50% and lifts nonfirst_word_capitalized over 0.75).
_SALT_RATE = 0.58

# --- frame templates ----------------------------------------------------------
# Every frame is a common-noun / adjective / regular-verb sentence in sentence
# case once instantiated, 5-10 words. {P} is the SALT-ABLE noun slot: it sits at
# a fixed word position in 2..6 (asserted at build time), follows a preposition,
# and reads naturally for BOTH a common place-noun and a NONNAME_PROPER token
# (city / country / month / brand), e.g. 'the bus to {P}' -> 'the bus to Madrid'.
# {N} are other common-noun slots, {A} adjective slots. Verbs are literal regular
# past-tense forms in the frame so the sentence is grammatical without inflection
# bookkeeping. The first and last words are NEVER a {P} slot (so the salt cannot
# touch first/last-word battery predicates and is always at position 2..6).
#
# Each tuple is (frame, p_slot_word_position_1indexed). The p position is the
# 1-indexed word position of the {P} token in the instantiated sentence (the
# build-time invariant check re-derives it from the global tokenizer and rejects
# any mismatch, so a frame edit cannot silently drift the salt off 2..6). The
# salt slot follows a preposition or is a leading modifier so it reads naturally
# for BOTH a common place-noun and a NONNAME_PROPER token, e.g.
# 'the bus to {P}' -> 'the bus to Madrid'. Positions 2..6 are all covered so the
# salt position is ~uniform over 2..6 (recipe: 'uniformly random position 2..6').
_FRAMES: list[tuple[str, int]] = [
    # pos 2: proper noun as a leading modifier ('The Japan office reopened ...')
    ("the {P} office reopened last spring quietly", 2),
    ("the {P} factory expanded across the region", 2),
    # pos 3: plural-noun subject + preposition ('Trains from Madrid arrived ...')
    ("trains from {P} arrived after midnight", 3),
    ("flights toward {P} departed without delay", 3),
    ("roads near {P} stayed closed today", 3),
    # pos 4: 'the {N} to {P} ...'
    ("the {N} to {P} departed before noon", 4),
    ("the {N} from {P} returned without delay", 4),
    ("the {N} near {P} remained closed today", 4),
    # pos 5: 'the {A} {N} to {P} ...'
    ("the {A} {N} to {P} departed quietly", 5),
    ("the {A} {N} from {P} arrived early", 5),
    ("the {A} {N} near {P} waited patiently", 5),
    # pos 6: 'the {A} {N} of the {P} ...'
    ("the {A} {N} of the {P} closed yesterday", 6),
    ("the {A} {N} beside the {P} stayed shut", 6),
]


@dataclass(frozen=True)
class Base:
    """A title_case base spec.

    ``base_id`` is the base sentence itself (rule-spec: 'base_id = base').
    ``frame`` / ``fillers`` / ``p_pos`` record provenance and the salt slot, and
    ``salted`` / ``proper`` carry the (mirrored) proper-noun salt decision so both
    variants of the base instantiate the SAME word sequence."""

    base_id: str          # == sentence (sentence-case surface string, salted form)
    sentence: str
    frame: str
    fillers: dict[str, str]
    p_pos: int            # 1-indexed word position of the salt slot (2..6)
    salted: bool
    proper: str           # the proper noun substituted (empty if not salted)
    common_noun: str      # the common noun the {P} slot held before salting


def _proper_pos(words_list: list[str], target: int) -> bool:
    """True iff ``target`` (1-indexed) is a non-first, non-last word position."""
    return 2 <= target <= 6 and target < len(words_list)


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct sentence-case base sentences (GENERATOR INTERFACE).

    Deterministic given ``gen``. Two phases:
      1. Enumerate (frame, common-noun {P}, other fillers) candidates, seeded-
         shuffle, and accept the first _N_BASES whose sentence is 5-10 words,
         surface-distinct, and whose {P} token sits at the frame's pinned word
         position in 2..6 (non-first, non-last). At this phase EVERY {P} slot
         holds a COMMON noun (no salt yet) so the accepted set is one homogeneous
         common-noun population.
      2. SALT exactly ``round(_SALT_RATE * n)`` of the accepted bases (seeded
         choice): replace the {P} common noun with a NONNAME_PROPER token of the
         SAME word count (the bank's propers are all single tokens). The salt is
         a base property, so BOTH variants of a salted base share it (vocabulary
         symmetry). The {P} position is unchanged, so it stays in 2..6.

    Raises if it cannot reach the 340 floor (loud — no quiet short dataset)."""
    nouns = banks.get_bank(_NOUN_BANK).words()
    adjs = banks.get_bank(_ADJ_BANK).words()
    propers = banks.get_bank(_PROPER_BANK).words()

    fill_gen = gen.derive("fill")
    salt_gen = gen.derive("salt")

    # --- phase 1: enumerate common-noun candidates --------------------------
    candidates: list[tuple[str, int, dict[str, str]]] = []
    for frame, p_pos in _FRAMES:
        slots = set(frame_slots(frame))
        for _ in range(160):
            fillers: dict[str, str] = {"P": fill_gen.choice(nouns)}
            if "N" in slots:
                fillers["N"] = fill_gen.choice(nouns)
            if "A" in slots:
                fillers["A"] = fill_gen.choice(adjs)
            candidates.append((frame, p_pos, fillers))
    salt_gen.shuffle(candidates)

    accepted: list[Base] = []
    seen: set[str] = set()
    for frame, p_pos, fillers in candidates:
        if len(accepted) >= _N_BASES:
            break
        sentence = _to_sentence_case(fill_frame(frame, fillers))
        if sentence in seen:
            continue
        toks = words(sentence)
        if not (_MIN_WORDS <= len(toks) <= _MAX_WORDS):
            continue
        if not _proper_pos(toks, p_pos):
            continue
        # the {P} token must actually sit at p_pos (1-indexed) — guards against a
        # frame edit silently drifting the salt slot off positions 2..6.
        if toks[p_pos - 1].lower() != fillers["P"].lower():
            continue
        seen.add(sentence)
        accepted.append(
            Base(
                base_id=sentence,
                sentence=sentence,
                frame=frame,
                fillers=dict(fillers),
                p_pos=p_pos,
                salted=False,
                proper="",
                common_noun=fillers["P"],
            )
        )

    if len(accepted) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"title_case: only built {len(accepted)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )

    # --- phase 2: salt exactly round(_SALT_RATE * n) accepted bases ---------
    # ``final_seen`` is the single distinctness authority over the FINAL surface
    # strings (common + salted). We walk a seeded-shuffled order and salt bases
    # until we have hit the target count; a base whose every proper-noun option
    # would collide with an already-emitted surface is left common and the next
    # base is salted instead, so the target salt count is still reached while no
    # duplicate surface ever escapes (the schema gate also re-checks).
    n = len(accepted)
    n_salt = round(_SALT_RATE * n)

    order = list(range(n))
    salt_gen.shuffle(order)

    final_seen: set[str] = set()
    salted_flag: dict[int, Base] = {}
    n_done = 0
    for idx in order:
        if n_done >= n_salt:
            break
        b = accepted[idx]
        # try proper nouns (seeded shuffle) until one yields a globally-distinct
        # salted surface; skip this base if none does (rare; diversity is ample).
        opts = list(propers)
        salt_gen.shuffle(opts)
        chosen: Base | None = None
        for proper in opts:
            f = dict(b.fillers)
            f["P"] = proper
            salted_sentence = _to_sentence_case(fill_frame(b.frame, f))
            if salted_sentence in final_seen or salted_sentence == b.sentence:
                continue
            toks = words(salted_sentence)
            # salt preserves the word-count / position invariants (propers are
            # single tokens); assert loud if a bank change ever breaks that.
            if (
                len(toks) != word_count(b.sentence)
                or not _proper_pos(toks, b.p_pos)
                or toks[b.p_pos - 1] != proper
            ):
                raise ValueError(
                    f"title_case: salt broke an invariant on {b.sentence!r} "
                    f"-> {salted_sentence!r}"
                )
            chosen = Base(
                base_id=salted_sentence,
                sentence=salted_sentence,
                frame=b.frame,
                fillers=f,
                p_pos=b.p_pos,
                salted=True,
                proper=proper,
                common_noun=b.common_noun,
            )
            break
        if chosen is None:
            continue  # leave this base common; salt a later one instead
        final_seen.add(chosen.sentence)
        salted_flag[idx] = chosen
        n_done += 1

    if n_done < n_salt:
        raise ValueError(
            f"title_case: could only salt {n_done} of {n_salt} target bases "
            "(insufficient proper-noun / frame diversity)"
        )

    # assemble final bases in the original accepted order (stable), using the
    # salted replacement where one was chosen, else the common base. Add common
    # surfaces to final_seen too and drop any (none expected) that collide.
    bases: list[Base] = []
    for idx, b in enumerate(accepted):
        if idx in salted_flag:
            bases.append(salted_flag[idx])
        else:
            if b.sentence in final_seen:
                # extraordinarily unlikely (common vs salted collision); skip to
                # preserve distinctness — the count floor still holds (we have
                # _N_BASES - duplicates >= 340 by construction headroom).
                continue
            final_seen.add(b.sentence)
            bases.append(b)

    ids = [b.base_id for b in bases]
    if len(set(ids)) != len(ids):
        raise ValueError("title_case: produced duplicate base sentences after salting")
    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"title_case: only {len(bases)} distinct bases after salting, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (GENERATOR INTERFACE).

    True  variant = to_title_case(base) — every word's first alphabetic char is
                    uppercase -> rule labels True.
    False variant = the sentence-case base (only the first word capitalized,
                    plus the salted proper noun if this base is salted) -> rule
                    labels False (a non-first common word is lowercase, OR the
                    base is short; either way some word does not start with a
                    capital). The salted proper noun is the SAME in both variants
                    (vocabulary symmetry) — in the True variant it is capitalized
                    anyway. Deterministic; ``gen`` is unused (no instantiate-time
                    randomness)."""
    if label:
        text = to_title_case(spec.sentence)
        transform = "title_case"
    else:
        text = spec.sentence  # sentence case (salted form already baked in)
        transform = "sentence_case"
    meta = {
        "frame": spec.frame,
        "fillers": spec.fillers,
        "p_pos": spec.p_pos,
        "salted": spec.salted,
        "proper": spec.proper,
        "transform": transform,
        "base_sentence": spec.sentence,
    }
    return text, meta


# --- local sentence-case helper -----------------------------------------------
# genutils.to_sentence_case LOWERCASES the whole string first, which would
# DESTROY a salted proper noun's interior capital (e.g. 'madrid'). For this rule
# the False/base form must keep the proper noun capitalized while making the rest
# of the sentence sentence-case. So we capitalize only the FIRST alphabetic char
# and leave the already-correctly-cased proper noun token intact: our frames are
# authored in lowercase and the only capital we want to preserve is the salt.


def _to_sentence_case(text: str) -> str:
    """Sentence case that PRESERVES an interior proper-noun capital.

    Input ``text`` is an all-lowercase frame instantiation EXCEPT the salted
    proper noun (which the bank supplies capitalized). We only uppercase the
    first alphabetic character of the whole string; every other character is left
    as-is, so the proper noun keeps its capital and all common words stay
    lowercase. (The frames contain no other capitals.)"""
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1 :]
    return text
