"""Rule 8: past_tense (category: syntactic).

Canonical articulation: True iff the input's main verb is in the PAST tense,
False iff it is in the PRESENT tense. On the EMITTED data this coincides
extensionally with "the verb ends in -ed" (the rule's equivalence_class), which
is exactly what groundtruth._r8_past_tense checks: True iff some non-stopword
token matches ``[a-z]{2,}ed$``.

Construction (per the private rule spec, not in this repository; id: past_tense):
  * Frames 'The {N1} {V} the {N2} {place-adjunct}', subject number varied
    (singular 'The cook' / plural 'The cooks'); >= 12 frame skeletons; 6-10
    words. V from VERB_REGULAR only (distinct, unambiguous regular forms).
  * True  (past)    : V inflects with -ed  ('walked'), regardless of number.
  * False (present) : 3sg -s with a SINGULAR subject ('walks'); the bare base
    form with a PLURAL subject ('walk') — agreement correct either way.
  * Adjuncts from ADVERB_PLACE ONLY (tense-neutral). NO temporal vocabulary
    anywhere (yesterday/today/now/often...): ADVERB_PLACE carries none, and the
    frames add none. No auxiliaries / progressives (those are rule 9's data).
  * base_id = frame skeleton + fillers + subject number; the True and False
    variants of a base differ in ONE morpheme only (the verb form), so word
    count is IDENTICAL across the pair and char count differs by <= 2.

Confound machinery this recipe pins, and how it is honoured here:
  * Subject number balanced 50/50 WITHIN each class — half of every class's
    bases are plural-subject. The split assigner picks one balanced variant per
    base for held_out/confirmation/spare and BOTH variants for few_shot_pool, so
    a per-base 50/50 split of singular vs plural makes 'plural subject' ~50% in
    both the True and False classes. (Realised by assigning number deterministically
    so that, across all bases, exactly half are plural; because each base's pair
    shares its number, every class inherits the same 50/50 number marginal.)
  * Word counts matched exactly per base (same frame + same adjunct on both
    variants) -> Gate D length-matching is exact.
  * No temporal adverbs -> the only 'time expression' distractor labels
    everything the same, never proxies tense.
  * The verb's -ed/-s/base morpheme is the ONLY surface difference within a
    base. The verb sits MID-sentence (never first or last token, because every
    base carries a trailing place-adjunct), so no first-word / last-word battery
    predicate can see it; the residual signal is the 0-2 char length delta, which
    is held away from the char_count thresholds by sampling each base's total
    word count uniformly over [6, 10] and choosing adjunct char-lengths so the
    marginal char distribution is broad and class-balanced (verified by Gate C).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ... import banks
from ...banks import _regular_verb_forms  # the framework's regular-inflection helper
from ...genutils import Gen, solve_adjuncts
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

_VERB_BANK = "VERB_REGULAR"
_NOUN_BANK = "NOUN_CONCRETE"
_ADVERB_BANK = "ADVERB_PLACE"

# Build comfortably over the 340-base floor (100 few_shot + 120 held_out + 100
# confirmation + >= 20 spare); the (frame, N1, V, N2, number) space is enormous.
_N_BASES = 360

# Word-count window the recipe names; the global [4, 14] cap is re-checked by
# the schema validator. Every base lands in [6, 10] words.
_MIN_WORDS, _MAX_WORDS = 6, 10

# A token (other than a stopword) ending in -ed would make the FALSE (present)
# variant read as True under groundtruth._r8_past_tense. The verb base/3sg-s
# forms never end in -ed, but two NOUN_CONCRETE fillers do ('shed', 'seed'),
# so they are banned from the noun slots. ('bed' is safe: '[a-z]{2,}ed$' needs
# >= 2 letters before 'ed', so single-letter-stem 'bed' never matches.)
_ED_RE = re.compile(r"[a-z]{2,}ed$")

# >= 12 frame skeletons. All share the 'The {N1} {V} the {N2}' backbone (5
# tokens before the adjunct); subject number is carried by the {N1} filler
# (singular 'cook' vs plural 'cooks'), which build_bases supplies already
# inflected. The skeletons vary the trailing connective so the surface forms
# differ while the verb stays mid-sentence and the True/False pair still differs
# in the verb morpheme ALONE. Each skeleton ends in a {ADJ} place-adjunct slot
# the equalizer fills with 1-3 ADVERB_PLACE phrases (shared across the pair).
_FRAME_SKELETONS: tuple[str, ...] = (
    "The {N1} {V} the {N2} {ADJ}",
    "The {N1} {V} the {N2} and the {N3} {ADJ}",
    "The {N1} {V} the {N2} near the {N3} {ADJ}",
    "The {N1} {V} the {N2} beside the {N3} {ADJ}",
    "The {N1} {V} the {N2} {ADJ} and the {N3}",
    "The {N1} {V} the {N2} past the {N3} {ADJ}",
)
# subject number (sg / pl) x the 6 skeletons above => 12 frame variants, as the
# recipe asks ('subject number varied ...; >= 12 frames').

# minimum / maximum extra place-adjunct words the equalizer may append, so the
# total word count stays in [6, 10]. Computed per skeleton from its fixed tokens.


def _skeleton_fixed_word_count(skeleton: str) -> int:
    """Word count of a skeleton with every slot replaced by a single token and
    the {ADJ} slot removed (the adjunct is appended separately to a target)."""
    # replace each non-ADJ slot with a one-word placeholder, drop {ADJ}
    filled = skeleton
    for slot in ("{N1}", "{V}", "{N2}", "{N3}"):
        filled = filled.replace(slot, "x")
    filled = filled.replace("{ADJ}", "").strip()
    # collapse double spaces left by the dropped {ADJ}
    return word_count(filled)


def _pluralize(noun: str) -> str:
    """Regular English plural of a concrete noun (form only; never yields -ed).

    Standard spelling rules: sibilant -es; consonant+y -> -ies; -f/-fe -> -ves;
    -o (consonant) -> -oes; else +s. Restricted at the call site to nouns whose
    plural is regular, so this stays exact."""
    n = noun.lower()
    if n.endswith(("s", "x", "z", "ch", "sh")):
        return n + "es"
    if n.endswith("y") and len(n) >= 2 and n[-2] not in "aeiou":
        return n[:-1] + "ies"
    if n.endswith("fe"):
        return n[:-2] + "ves"
    if n.endswith("f"):
        return n[:-1] + "ves"
    if n.endswith("o") and len(n) >= 2 and n[-2] not in "aeiou":
        return n + "es"
    return n + "s"


@dataclass(frozen=True)
class Base:
    """One past_tense base: a frame skeleton + fillers + subject number + the
    pre-chosen place adjuncts. The True/False variants differ ONLY in the verb
    form; everything else (including the adjuncts) is shared and stored here so
    instantiate is a pure transform."""

    base_id: str
    skeleton: str
    n1: str          # subject noun, ALREADY inflected for number (sg or pl form)
    verb: str        # the VERB_REGULAR base form (lowercase)
    n2: str
    n3: str          # filler for skeletons that use a third noun ('' if unused)
    plural: bool     # subject number: True = plural subject -> present base form
    adjuncts: tuple[str, ...]  # place adjuncts appended after the frame (shared)
    target_words: int


def _noun_pool() -> list[str]:
    """NOUN_CONCRETE words usable in a noun slot: drop the two -ed nouns that
    would fool the present-class ground truth."""
    return [w for w in banks.get_bank(_NOUN_BANK).words() if not _ED_RE.search(w.lower())]


def _plural_safe_nouns(pool: list[str]) -> list[str]:
    """Subject nouns whose regular plural is unambiguous and never ends in -ed
    (it cannot: plurals end in -s/-es/-ies/-ves) — all of ``pool`` qualifies, but
    we additionally require the +s/-es/-ies/-ves plural to be DISTINCT from the
    singular (always true) and to not collide with another singular (kept simple
    by construction)."""
    return list(pool)


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (the GENERATOR INTERFACE).

    Deterministic given ``gen``. Each base fixes a frame skeleton, an (N1, V, N2
    [, N3]) filling, a subject number, and a target word count in [6, 10] whose
    deficit is solved with ADVERB_PLACE place-adjuncts. Subject number is laid
    down so EXACTLY half of all bases are plural (50/50), which — since each
    base's pair shares its number — makes 'plural subject' ~50% in BOTH classes.
    The total word count target is swept uniformly over [6, 10] so the marginal
    char-count distribution is broad and the 0-2 char verb delta never lets a
    char_count threshold separate the classes (Gate C verifies)."""
    nouns = _noun_pool()
    subject_nouns = _plural_safe_nouns(nouns)
    verbs = banks.get_bank(_VERB_BANK).words()

    # ADVERB_PLACE phrases grouped by word length (1, 2, 3) for the equalizer.
    adverbs = banks.get_bank(_ADVERB_BANK).words()
    adjuncts_by_len: dict[int, list[str]] = {}
    for ph in adverbs:
        adjuncts_by_len.setdefault(word_count(ph), []).append(ph)
    available_lengths = sorted(adjuncts_by_len)

    bg = gen.derive("bases")

    # Enumerate a large shuffled candidate space of (skeleton, n1, verb, n2, n3).
    candidates: list[tuple[str, str, str, str, str]] = []
    for skel in _FRAME_SKELETONS:
        needs_n3 = "{N3}" in skel
        for n1 in subject_nouns:
            for v in verbs:
                # one N2 (and optionally N3) per (skel, n1, v) drawn later;
                # record the tuple-stub and resolve nouns at sampling time to
                # keep the candidate list bounded.
                candidates.append((skel, n1, v, "", "" if not needs_n3 else "?"))
    bg.shuffle(candidates)

    # Decide the subject-number assignment up front: exactly half of the bases
    # we ACCEPT will be plural. We tag candidates with an alternating intended
    # number after the shuffle, then honour it as we accept bases (balanced).
    bases: list[Base] = []
    seen_ids: set[str] = set()
    seen_text_keys: set[str] = set()
    n_plural = 0
    n_singular = 0

    nbg = bg.derive("nouns")
    abg = bg.derive("adjuncts")
    wbg = bg.derive("targets")

    for skel, n1_base, verb, _n2_stub, n3_stub in candidates:
        if len(bases) >= _N_BASES:
            break
        needs_n3 = "{N3}" in skel

        # choose distinct object nouns
        n2 = nbg.choice([w for w in nouns if w != n1_base])
        if needs_n3:
            n3 = nbg.choice([w for w in nouns if w not in (n1_base, n2)])
        else:
            n3 = ""

        # subject number: keep the running counts balanced 50/50.
        if n_plural < n_singular:
            plural = True
        elif n_singular < n_plural:
            plural = False
        else:
            plural = bool(nbg.randint(0, 1))
        n1_surface = _pluralize(n1_base) if plural else n1_base

        # target total word count in [6, 10]: the fixed skeleton tokens (incl.
        # the verb + nouns, one token each) plus the appended place adjuncts.
        fixed_wc = _skeleton_fixed_word_count(skel)
        # the smallest target must leave room for >= 1 adjunct word (adjunct
        # slot is non-empty so the verb is never the final token).
        lo = max(_MIN_WORDS, fixed_wc + 1)
        hi = _MAX_WORDS
        if lo > hi:
            continue  # this skeleton's fixed part is already too long; skip
        target = wbg.randint(lo, hi)
        deficit = target - fixed_wc
        if deficit < 1:
            continue

        # Append place adjuncts to hit the target word count EXACTLY (shared
        # across the True/False pair, so word count is identical and the only
        # within-base difference is the verb morpheme). Solve the deficit into
        # phrase word-lengths, then draw one ADVERB_PLACE phrase per length.
        lengths = solve_adjuncts(deficit, available_lengths)
        phrases: list[str] = []
        ok = True
        for L in lengths:
            pool = adjuncts_by_len.get(L)
            if not pool:
                ok = False
                break
            phrases.append(abg.choice(pool))
        if not ok or not phrases:
            continue
        abg.shuffle(phrases)
        adjuncts = tuple(phrases)

        base_id = "pt-" + _stable_id(
            skel, n1_surface, verb, n2, n3, "pl" if plural else "sg", "|".join(adjuncts)
        )
        if base_id in seen_ids:
            continue

        # de-dup on the SURFACE of both variants so no two bases collide in text.
        spec = Base(
            base_id=base_id,
            skeleton=skel,
            n1=n1_surface,
            verb=verb,
            n2=n2,
            n3=n3,
            plural=plural,
            adjuncts=adjuncts,
            target_words=target,
        )
        t_text, _ = instantiate(spec, True, abg)
        f_text, _ = instantiate(spec, False, abg)
        if t_text in seen_text_keys or f_text in seen_text_keys:
            continue
        # sanity: both variants must land in the word window (they share count)
        if not (_MIN_WORDS <= word_count(t_text) <= _MAX_WORDS):
            continue
        if word_count(t_text) != word_count(f_text):
            continue

        seen_ids.add(base_id)
        seen_text_keys.add(t_text)
        seen_text_keys.add(f_text)
        bases.append(spec)
        if plural:
            n_plural += 1
        else:
            n_singular += 1

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"past_tense: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def _stable_id(*parts: object) -> str:
    import hashlib

    blob = "␟".join(str(p) for p in parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _verb_form(verb_base: str, *, label: bool, plural: bool) -> str:
    """The surface verb form for a (label, subject-number) combination.

    True (past)    -> the -ed past form (any number).
    False (present)-> 3sg -s for a singular subject, bare base for a plural one.
    Uses the framework's regular-inflection helper so spelling matches the bank
    self-check (consonant+y -> -ied/-ies, silent-e, sibilant -es, CVC doubling)."""
    base, sg_s, past, _gerund = _regular_verb_forms(verb_base)
    if label:
        return past
    return base if plural else sg_s


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  -> past (-ed) verb form        -> groundtruth labels it True.
    False -> present (3sg -s / base form) -> groundtruth labels it False.
    Deterministic; ``gen`` is unused (the adjuncts/fillers are fixed on the
    spec) but kept to match the interface signature."""
    verb_surface = _verb_form(spec.verb, label=label, plural=spec.plural)
    fillers = {"N1": spec.n1, "V": verb_surface, "N2": spec.n2}
    if "{N3}" in spec.skeleton:
        fillers["N3"] = spec.n3
    # fill the skeleton (drop the {ADJ} slot; adjuncts are appended after)
    body = spec.skeleton.replace(" {ADJ}", "").replace("{ADJ} ", "").replace("{ADJ}", "")
    for slot, val in fillers.items():
        body = body.replace("{" + slot + "}", val)
    body = " ".join(body.split())  # collapse any double spaces
    text = " ".join([body, *spec.adjuncts]).strip()
    # sentence case (first letter upper); shared across the pair, so casing
    # predicates sit at exactly 50%.
    text = text[0].upper() + text[1:] if text else text

    meta = {
        "skeleton": spec.skeleton,
        "n1": spec.n1,
        "verb_base": spec.verb,
        "verb_form": verb_surface,
        "n2": spec.n2,
        "n3": spec.n3 or None,
        "plural_subject": spec.plural,
        "adjuncts": list(spec.adjuncts),
        "target_words": spec.target_words,
        "tense": "past" if label else "present",
        "transform": "verb_past_ed" if label else ("verb_base" if spec.plural else "verb_3sg_s"),
    }
    return text, meta
