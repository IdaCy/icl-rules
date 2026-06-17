"""Rule 28: double_letter_word (category: hard_articulation, substitution).

Canonical articulation: True iff the input contains at least one word with the
SAME LETTER TWICE IN A ROW (an adjacent double, as in "coffee" or "wall").
Ground truth (groundtruth._r28_double_letter_word) is: any stripped word has an
adjacent doubled letter.

CONSTRUCTION (rule-specs id: double_letter_word, generation.recipe +
distribution_guards). Each base is a (frame, double-word) pair; the two variants
are SURFACE-IDENTICAL except for ONE interior content noun (the {X} slot):

    base sentence  = a frame whose FIXED vocabulary is ENTIRELY double-free
                     (no word has an adjacent doubled letter, incl. function
                     words). The {X} slot is NEVER first or last word.
    False variant  = {X} filled with a double-FREE noun (matched length/tier to
                     the base's double word)                          -> label False
    True  variant  = {X} filled with the base's DOUBLE_WORDS noun
                     (one adjacent doubled letter)                     -> label True
    base_id        = frame + double word  (both variants share it)

Because the two variants differ in exactly ONE single-token noun, every base's
True and False surface have the SAME word count -> the length-matching gate is
satisfied by construction (|mean_wc(T) - mean_wc(F)| == 0). The False filler is
chosen length-matched (in characters) to the base's double word, so per-item
char counts track and no char_count battery bucket separates the classes. The
{X} slot is always interior, so first-word / last-word POS / length / letter
predicates can never key on the substituted word.

Distribution_guards honored:
  * Pairwise POS / length / frequency matching: every True double word is a
    NOUN; its False counterpart is a NOUN of length +/- 2 and the SAME
    frequency tier, picked closest in character length (so char counts match).
  * Topic cap on DOUBLE_WORDS (<= 25% food): the DOUBLE_WORDS bank enforces
    subtype food <= 0.25 at the bank level; we additionally draw nouns only, and
    the noun set has just two food entries (coffee, apple), well under the cap.
  * Double-letter type spread (no 'contains ll' shortcut above 75%): the 55
    double NOUNS span 14 distinct doubled-letter types (l s e o t f m n p r d b
    c g); seeded round-robin over a shuffled type order keeps any single type
    (e.g. 'll') far under 75% of True items.
  * Nonadjacent salt: >= 55% of bases use a NONADJ_REPEAT_WORDS noun (window,
    banana, garage, river, paper, camera, ...) as their False filler, so >= 50%
    of False items contain a word with a repeated letter SOMEWHERE-BUT-NOT-
    ADJACENT. The 'a word has a repeated letter anywhere' reading therefore
    disagrees with the true label on >= 25% of the data (the adjacent-vs-anywhere
    splitter the step-3 probes exploit).
  * Exactly ONE double word per True item: the frame fixed vocabulary is
    double-free and the {X} filler is the ONLY double word in a True item.
  * Substitution position varies: frames place {X} at positions 3, 4, and 5.

This module exposes the GENERATOR INTERFACE (build_bases / instantiate) and is
dispatched by registry.run through the shared gated pipeline (base.emit_rule).
The single generation seed is threaded by the pipeline and recorded in every
item's slots_meta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, fill_frame, to_sentence_case
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count, words

_DOUBLE_BANK = "DOUBLE_WORDS"
_NONADJ_BANK = "NONADJ_REPEAT_WORDS"
_NOUN_BANK = "NOUN_CONCRETE"

# Frames whose FIXED vocabulary is entirely double-free (asserted in build_bases
# so a stray doubled letter can never silently slip into a False item). The {X}
# slot is always interior (never the first or last token) so the substituted
# noun is invisible to every first-word / last-word battery predicate. Slot
# position and frame length vary (5-10 words) for word-count spread.
_FRAMES: tuple[str, ...] = (
    # slot at position 4 (8 words)
    "They placed the {X} near the garden gate",
    "She found a {X} beside the quiet house",
    "We noticed the {X} inside the old shed",
    "He kept the {X} under the kitchen sink",
    "They moved the {X} toward the open gate",
    "She left the {X} behind the broken fence",
    "We saw the {X} beyond the quiet river",
    "He painted the {X} above the front porch",
    "She placed the {X} upon the dusty desk",
    "We hid the {X} among the tangled vines",
    "He held the {X} above the calm lake",
    "She bought the {X} at the local market",
    "He set the {X} beneath the wide window",
    "They left the {X} outside the iron gate",
    "He saw the {X} near the river bank",
    "We placed the {X} under the kitchen table",
    "He moved the {X} toward the open field",
    # slot at position 3
    "She kept a {X} inside the metal box",
    "They saw the {X} from the high tower",
    "We found a {X} beneath the front porch",
    # shorter frames (slot interior, 5-7 words)
    "She found the {X} near the river",
    "They kept the {X} in the barn",
    "We saw a {X} beside the gate",
    "He left the {X} on the desk",
    "She placed a {X} by the window",
    "We hid the {X} under the bench",
    # slot at position 5 (longer, 10 words)
    "Early today she found the {X} near the river bank",
    "After lunch they kept the {X} inside the metal box",
    "Last night we saw the {X} beside the garden gate",
    "Every morning he placed the {X} upon the wide desk",
)

# Fraction of bases whose FALSE filler is a NONADJ_REPEAT noun (the salt). Set
# above 0.5 so that, even after the seeded balanced one-variant split assignment
# hides some False variants, >= 50% of the emitted False items still carry salt.
_SALT_BASE_FRAC = 0.60

# build well past the 340-base floor: 30 frames x 55 double nouns = 1650 pairs.
_N_BASES = 480

_MIN_WORDS, _MAX_WORDS = 5, 10  # the spec's per-frame window (within global [4,14])

_SLOT = "{X}"


@dataclass(frozen=True)
class Base:
    """A double_letter_word base: a (frame, double-noun) pair plus the pre-chosen
    double-free FALSE filler (length/tier matched). ``base_id`` == frame + double
    word (rule-spec: 'base_id = base sentence', here the frame+double pair that
    fully determines both variant surfaces)."""

    base_id: str
    frame: str
    double_word: str    # DOUBLE_WORDS noun -> True variant filler
    free_word: str      # double-free noun -> False variant filler
    free_bank: str      # NONADJ_REPEAT_WORDS (salt) or NOUN_CONCRETE
    double_type: str    # the doubled-letter type(s) in double_word (for provenance)


def _doubled_types(word: str) -> str:
    """The set of letters that appear twice in a row in ``word`` (sorted str)."""
    a = "".join(ch for ch in word.lower() if ch.isalpha())
    return "".join(sorted({a[i] for i in range(len(a) - 1) if a[i] == a[i + 1]}))


def _matched_candidates(target_len: int, pool: list[str], lens: dict[str, int]) -> list[str]:
    """The pool words within +/- 2 characters of ``target_len`` (the spec's
    length-match window), ordered by closeness then by the pool's seeded order.

    Returning a LIST (not just the single nearest) lets the caller cycle through
    several equally-good length matches so two bases sharing a frame + tier do not
    collapse to the same False surface (the dedup gate would reject that)."""
    cands = [w for w in pool if abs(lens[w] - target_len) <= 2]
    if not cands:  # fall back to the whole pool if nothing is within +/- 2
        cands = list(pool)
    cands.sort(key=lambda w: (abs(lens[w] - target_len), pool.index(w)))
    return cands


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct (frame, double-noun) bases (the GENERATOR INTERFACE).

    Deterministic given ``gen``. Asserts every frame's fixed vocabulary is
    double-free (loud — a doubled letter in a frame would mislabel a False item),
    then enumerates (frame, double-noun) pairs, seeded-shuffles them, and walks
    the shuffled order. Double nouns are assigned in a seeded, type-spread
    round-robin so no single doubled-letter type dominates the True class. Each
    base gets a length/tier-matched double-FREE FALSE filler; >= _SALT_BASE_FRAC
    of bases draw that filler from NONADJ_REPEAT_WORDS (the salt)."""
    # --- frame vocabulary guard (loud) ---------------------------------------
    for fr in _FRAMES:
        fixed = [w for w in words(fr) if w != _SLOT]
        bad = [w for w in fixed if banks.has_adjacent_double(w)]
        if bad:
            raise ValueError(
                f"double_letter_word: frame {fr!r} has double-lettered fixed "
                f"word(s) {bad} (would mislabel its False variant True)"
            )

    double_nouns = [e for e in banks.get_bank(_DOUBLE_BANK).entries if e.pos == "noun"]
    if not double_nouns:
        raise ValueError("double_letter_word: no double NOUNS in DOUBLE_WORDS")

    free_nouns = [e for e in banks.get_bank(_NOUN_BANK).entries if not e.has_adjacent_double]
    nonadj_nouns = [e for e in banks.get_bank(_NONADJ_BANK).entries if e.pos == "noun"]
    # plain double-free nouns that are ALSO free of any nonadjacent repeat (the
    # 'no salt' filler pool, so the salted vs unsalted choice is clean).
    plain_nouns = [e for e in free_nouns if not e.has_nonadjacent_repeat]

    free_len = {e.word: e.length for e in free_nouns}
    nonadj_len = {e.word: e.length for e in nonadj_nouns}
    plain_len = {e.word: e.length for e in plain_nouns}

    # by frequency tier, for tier-matched FALSE fillers
    nonadj_by_tier = {1: [e.word for e in nonadj_nouns if e.frequency_tier == 1],
                      2: [e.word for e in nonadj_nouns if e.frequency_tier == 2]}
    plain_by_tier = {1: [e.word for e in plain_nouns if e.frequency_tier == 1],
                     2: [e.word for e in plain_nouns if e.frequency_tier == 2]}
    for pool in (*nonadj_by_tier.values(), *plain_by_tier.values()):
        gen.shuffle(pool)

    # seeded, type-spread double-noun order: bucket nouns by doubled-letter type,
    # shuffle within each bucket and shuffle the bucket order, then round-robin so
    # consecutively assigned double words rotate through types (no 'll' run).
    by_type: dict[str, list[banks.Entry]] = {}
    for e in double_nouns:
        by_type.setdefault(_doubled_types(e.word), []).append(e)
    type_keys = list(by_type)
    gen.shuffle(type_keys)
    for k in type_keys:
        gen.shuffle(by_type[k])
    double_order: list[banks.Entry] = []
    idx = 0
    while len(double_order) < len(double_nouns):
        progressed = False
        for k in type_keys:
            bucket = by_type[k]
            if idx < len(bucket):
                double_order.append(bucket[idx])
                progressed = True
        idx += 1
        if not progressed:
            break

    pairs = [(f, e) for f in _FRAMES for e in double_order]
    gen.shuffle(pairs)

    bases: list[Base] = []
    seen_ids: set[str] = set()
    seen_surfaces: set[str] = set()
    salt_target = int(round(_SALT_BASE_FRAC * _N_BASES))
    n_salted = 0
    for frame, dentry in pairs:
        dword = dentry.word
        bid = f"{frame}||{dword}"
        if bid in seen_ids:
            continue

        # decide salt vs plain for THIS base; honor the running salt target.
        remaining = _N_BASES - len(bases)
        need_salt = (salt_target - n_salted) >= remaining  # must salt to hit target
        salted = need_salt or (n_salted < salt_target)

        # candidate tier-matched, length-matched double-free FALSE fillers; cycle
        # through them so frame+tier-twins do not collapse to the same surface.
        tier = dentry.frequency_tier
        if salted:
            pool = nonadj_by_tier.get(tier) or [e.word for e in nonadj_nouns]
            cands = _matched_candidates(dentry.length, pool, nonadj_len)
            free_bank = _NONADJ_BANK
        else:
            pool = plain_by_tier.get(tier) or [e.word for e in plain_nouns]
            cands = _matched_candidates(dentry.length, pool, plain_len)
            free_bank = _NOUN_BANK

        true_text = to_sentence_case(fill_frame(frame, {"X": dword}))
        wc = word_count(true_text)
        if not (_MIN_WORDS <= wc <= _MAX_WORDS):
            continue
        if true_text in seen_surfaces:  # double-word surfaces must be distinct
            continue

        # choose the first length-matched FALSE filler that yields a still-unused,
        # distinct-from-True surface (deterministic walk over the seeded candidate
        # order).
        chosen: str | None = None
        false_text = ""
        for cand in cands:
            ft = to_sentence_case(fill_frame(frame, {"X": cand}))
            if ft == true_text or ft in seen_surfaces:
                continue
            chosen = cand
            false_text = ft
            break
        if chosen is None:
            continue  # no distinct length-matched filler left for this pair

        seen_ids.add(bid)
        seen_surfaces.add(true_text)
        seen_surfaces.add(false_text)
        bases.append(
            Base(
                base_id=bid,
                frame=frame,
                double_word=dword,
                free_word=chosen,
                free_bank=free_bank,
                double_type=_doubled_types(dword),
            )
        )
        if salted:
            n_salted += 1
        if len(bases) >= _N_BASES:
            break

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"double_letter_word: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  -> {X} filled with the DOUBLE_WORDS noun (one adjacent doubled letter
             -> rule labels True).
    False -> {X} filled with the matched double-FREE noun (no adjacent double
             anywhere; the frame is double-free too -> rule labels False).
    Deterministic; ``gen`` is unused (the fillers are pinned on the base) but kept
    in the signature to match the interface."""
    filler = spec.double_word if label else spec.free_word
    text = to_sentence_case(fill_frame(spec.frame, {"X": filler}))
    meta = {
        "frame": spec.frame,
        "slot": "X",
        "filler": filler,
        "filler_bank": (_DOUBLE_BANK if label else spec.free_bank),
        "double_word": spec.double_word,
        "free_word": spec.free_word,
        "double_type": spec.double_type,
        "salted_false": spec.free_bank == _NONADJ_BANK,
        "transform": "double_substitution" if label else "free_substitution",
    }
    return text, meta
