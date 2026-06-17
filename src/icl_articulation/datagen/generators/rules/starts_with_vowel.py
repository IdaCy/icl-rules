"""Rule 7: starts_with_vowel (category surface).

Canonical articulation: True iff the FIRST LETTER of the first word is a vowel
letter (a, e, i, o, u), regardless of case and regardless of pronunciation;
digit-initial -> False (none in training). The ground-truth predicate is
``groundtruth._r7_starts_with_vowel``.

Construction (rule-specs recipe + distribution_guards, verbatim intent):

  Frame ``'{W} {rest...}'`` where ``{W}`` is the sentence-initial (capitalized)
  first word and ``{rest}`` is a continuation shared VERBATIM across the two
  classes ('... waited near the station', '... appeared on the morning train').
    * True  variant: W from VOWEL_INITIAL.
    * False variant: W from CONSONANT_INITIAL.
  base_id = the continuation (+ the matched word pair / POS that key it); each
  base gets exactly one vowel-initial and one consonant-initial variant whose
  first words are the SAME POS, the SAME first-letter bucket, and the SAME
  alphabetic length. POS mix is 70% plural noun / 30% adverb in BOTH classes
  (length-matched bank-to-bank).

Why every non-exempt battery predicate sits at ~50% (the confound machinery):

  The two variants of a base are CHARACTER-IDENTICAL except for the first word,
  and the matched word pair fixes the only first-word features a generic
  predicate can read:
    * same alphabetic length  -> ``first_word_len>=k`` and ``char_count>=k`` take
      the SAME value on both variants of a base (the first words are pure
      alphabetic, so equal alpha length == equal char length).
    * same first-letter bucket -> all four ``first_letter_bucket_*`` predicates
      take the same value on both variants.
    * same POS                -> the six ``first_word_pos=*`` predicates take the
      same value on both variants.
  Everything past position 1 is the shared continuation, so word_count,
  contains_{the,a,and}, count_the, last_ends_{vowel,consonant}, last_word_len,
  contains_digit/comma, nonfirst_word_capitalized and all_lowercase are also
  identical on the two variants. A predicate that is identical on the True and
  False variant of EVERY base agrees with the label on exactly one of each pair
  and so scores exactly 0.5 over the dataset. Only ``first_starts_vowel`` /
  ``first_starts_consonant`` separate the classes, and those are the rule itself
  (equiv-class exempt). All five vowels are represented (each >= 10% of True
  items) so no single-letter shortcut survives.

Mirrors the reference rule ``all_lowercase``'s ``build_bases`` / ``instantiate``
shape; the shared gated pipeline (``base.emit_rule``) enforces all four gates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, base_id as make_base_id
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

_VOWEL_BANK = "VOWEL_INITIAL"
_CONS_BANK = "CONSONANT_INITIAL"

VOWELS = frozenset("aeiou")

# the global [4, 14] word-count window the schema validator re-checks; we keep a
# comfortable interior window so every (first word + continuation) lands inside.
_MIN_WORDS, _MAX_WORDS = 5, 13

# ~70% noun / 30% adverb first-word POS in BOTH classes (recipe). We weight how
# many continuations each matched pair receives by its POS so the per-base POS
# composition lands near this split. (The battery is robust either way — POS is
# matched per base — but the guard is honored at the data level.)
_ADVERB_TARGET_FRAC = 0.30

# build comfortably more than the 340-base floor.
_N_BASES = 372


def _bucket(letter: str) -> str:
    """First-letter bucket, matching battery._bucket / banks._bucket."""
    if letter in "abcdef":
        return "a-f"
    if letter in "ghijklm":
        return "g-m"
    if letter in "nopqrs":
        return "n-s"
    return "t-z"


# Continuations shared VERBATIM across both classes. Each is a past-tense
# intransitive verb phrase (+ optional place adjunct) that reads naturally after
# BOTH a plural-noun subject ('Engines waited near the station') and a
# sentence-initial adverb ('Always waited near the station'). All lowercase, no
# punctuation, no 'I', no proper nouns, no digits, no commas, no temporal words
# that would proxy another rule. Word counts vary (2..8) so the dataset spans a
# range of total lengths; the count is identical on a base's two variants, so
# every length/count predicate stays balanced regardless.
_CONTINUATIONS: tuple[str, ...] = (
    "waited near the station",
    "appeared on the morning train",
    "gathered by the old harbour wall",
    "arrived before the heavy rain",
    "remained quiet through the long night",
    "returned after the summer break",
    "vanished into the thick fog",
    "lingered beside the garden gate",
    "traveled across the wide plain",
    "departed from the busy port",
    "wandered through the quiet market",
    "rested under the tall pine",
    "settled along the river bank",
    "marched toward the distant hill",
    "drifted past the sleeping town",
    "assembled inside the great hall",
    "scattered across the open field",
    "paused at the narrow bridge",
    "moved through the crowded square",
    "stayed close to the warm fire",
    "spread over the empty road",
    "circled around the stone tower",
    "climbed up the steep path",
    "rushed toward the open door",
    "drifted slowly down the stream",
    "waited patiently for the late bus",
    "appeared suddenly behind the wall",
    "gathered near the wooden fence",
    "returned home through the snow",
    "vanished beyond the far ridge",
)


@dataclass(frozen=True)
class _Pair:
    """A matched first-word pair: same POS, same first-letter bucket, same
    alphabetic length. The vowel word is the True first word, the consonant word
    the False first word."""

    pos: str
    bucket: str
    length: int
    vowel_word: str  # VOWEL_INITIAL surface (lowercase)
    cons_word: str   # CONSONANT_INITIAL surface (lowercase), matched


@dataclass(frozen=True)
class Base:
    """A starts_with_vowel base: a matched word pair + a shared continuation.

    Both variants share ``base_id``; the True variant prepends the vowel word and
    the False variant the consonant word to the SAME ``continuation``."""

    base_id: str
    pair: _Pair
    continuation: str


def _matched_pairs() -> list[_Pair]:
    """Every (POS, bucket, length)-matched (vowel, consonant) first-word pair.

    Grouping both banks by (pos, first-letter bucket, alphabetic length) and
    zipping within each group guarantees each pair shares POS + bucket + length,
    which is exactly what neutralizes the first-word battery predicates. Sorted
    deterministically (no RNG here) so the pair set is stable."""
    vbank = banks.get_bank(_VOWEL_BANK)
    cbank = banks.get_bank(_CONS_BANK)

    def grouped(bank: banks.Bank) -> dict[tuple[str, str, int], list[str]]:
        g: dict[tuple[str, str, int], list[str]] = defaultdict(list)
        for e in bank.entries:
            g[(e.pos, _bucket(e.initial), e.length)].append(e.word)
        for words in g.values():
            words.sort()
        return g

    gv = grouped(vbank)
    gc = grouped(cbank)
    pairs: list[_Pair] = []
    for key in sorted(set(gv) & set(gc)):
        pos, bucket, length = key
        vs, cs = gv[key], gc[key]
        for vw, cw in zip(vs, cs):
            # sanity: the vowel word really starts with a vowel letter, the
            # consonant word really does not (defends against a bank edit).
            if vw[0].lower() not in VOWELS or cw[0].lower() in VOWELS:
                raise ValueError(
                    f"starts_with_vowel: mispaired ({vw!r}, {cw!r}) for {key}"
                )
            pairs.append(_Pair(pos, bucket, length, vw, cw))
    if not pairs:
        raise ValueError("starts_with_vowel: no matched first-word pairs built")
    return pairs


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (the GENERATOR INTERFACE).

    Cross every matched first-word pair with continuations (seeded order),
    weighting continuation budgets so ~30% of bases use an adverb-initial pair
    and ~70% a plural-noun pair (the recipe's POS mix, honored at the data
    level). Each base's surface strings are distinct by construction (distinct
    (first word, continuation) combos); we still dedup defensively. Raises if it
    cannot reach the 340 floor (loud)."""
    pairs = _matched_pairs()
    noun_pairs = [p for p in pairs if p.pos == "noun"]
    adverb_pairs = [p for p in pairs if p.pos == "adverb"]
    if not noun_pairs or not adverb_pairs:
        raise ValueError("starts_with_vowel: need both noun and adverb pairs")

    conts = list(_CONTINUATIONS)
    gen.shuffle(conts)

    # Target counts: ~30% adverb-first bases. Each (pair, continuation) is one
    # base, so #bases = #pairs_used_for_class * #conts_per_pair (roughly). We
    # give every pair the full continuation list, then take a seeded subset to
    # hit the POS split and the base floor.
    n_adverb_target = int(round(_N_BASES * _ADVERB_TARGET_FRAC))
    n_noun_target = _N_BASES - n_adverb_target

    def enumerate_bases(pair_list: list[_Pair], want: int) -> list[Base]:
        """Round-robin pairs x continuations until ``want`` distinct bases."""
        out: list[Base] = []
        seen_ids: set[str] = set()
        # deterministic stream: for each continuation, walk all pairs (the pair
        # order is the stable _matched_pairs order; continuation order is the
        # seeded shuffle), so the cross product is enumerated reproducibly.
        for cont in conts:
            for pair in pair_list:
                bid = make_base_id("starts_with_vowel", pair.pos, pair.bucket,
                                   pair.length, pair.vowel_word, pair.cons_word, cont)
                if bid in seen_ids:
                    continue
                seen_ids.add(bid)
                out.append(Base(base_id=bid, pair=pair, continuation=cont))
                if len(out) >= want:
                    return out
        return out

    noun_bases = enumerate_bases(noun_pairs, n_noun_target)
    adverb_bases = enumerate_bases(adverb_pairs, n_adverb_target)
    bases = noun_bases + adverb_bases

    # defensive surface-uniqueness check across BOTH variants of every base
    # (the pipeline also rejects duplicate surfaces in Gate A, but failing here
    # is a clearer signal of an under-sized continuation pool).
    seen_surface: set[str] = set()
    for b in bases:
        for w in (b.pair.vowel_word, b.pair.cons_word):
            surface = _surface(w, b.continuation)
            if surface in seen_surface:
                raise ValueError(
                    f"starts_with_vowel: duplicate surface {surface!r} — "
                    "enlarge _CONTINUATIONS or the matched-pair set"
                )
            seen_surface.add(surface)

    gen.shuffle(bases)
    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"starts_with_vowel: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN} (enlarge _CONTINUATIONS)"
        )
    return bases


def _surface(first_word: str, continuation: str) -> str:
    """Sentence-case surface for ``'{first_word} {continuation}'``.

    Only the first letter is uppercased (globals.casing.default sentence case);
    the first word and continuation are otherwise lowercase, so the first LETTER
    is the first word's initial — the only thing the rule reads."""
    return (first_word[0].upper() + first_word[1:]) + " " + continuation


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  -> vowel-initial first word  (rule labels True).
    False -> consonant-initial first word (rule labels False).
    The continuation is shared verbatim; ``gen`` is unused (deterministic)."""
    first_word = spec.pair.vowel_word if label else spec.pair.cons_word
    text = _surface(first_word, spec.continuation)
    wc = word_count(text)
    if not (_MIN_WORDS <= wc <= _MAX_WORDS):
        raise ValueError(
            f"starts_with_vowel: word count {wc} out of [{_MIN_WORDS}, "
            f"{_MAX_WORDS}] for {text!r}"
        )
    meta = {
        "transform": "vowel_initial" if label else "consonant_initial",
        "first_word": first_word,
        "first_word_pos": spec.pair.pos,
        "first_letter_bucket": spec.pair.bucket,
        "first_word_len": spec.pair.length,
        "vowel_word": spec.pair.vowel_word,
        "cons_word": spec.pair.cons_word,
        "continuation": spec.continuation,
    }
    return text, meta
