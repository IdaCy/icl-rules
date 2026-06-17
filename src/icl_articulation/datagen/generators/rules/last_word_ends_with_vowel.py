"""Rule 20: last_word_ends_with_vowel (category positional).

Canonical articulation: True iff the LAST LETTER of the input's LAST WORD is a
vowel letter (a, e, i, o, or u); y does NOT count as a vowel. LETTER-based, so
silent e counts ('table' -> True). Punctuation is stripped first (training has no
terminal punctuation anyway).

================================ CONSTRUCTION ================================
The two variants of a base are CHARACTER-LENGTH-IDENTICAL except for the final
word (the substitution mode the spec pins: "Shared frames '... {W}'"). A base is

    (frame, vowel_word, cons_word)

where the frame ends in the slot '{W}', and vowel_word / cons_word are the True /
False fillers:

    True  variant = frame with {W} = vowel_word  (ends in a/e/i/o/u  -> True)
    False variant = frame with {W} = cons_word   (ends in a consonant -> False)

The fillers are drawn so that, per base, BOTH classes are matched on every
confound axis the recipe + distribution_guards pin:

  * POS              both banks are NOUNS-only (TERMINAL_VOWEL is noun-only;
                     False draws are the noun, final_y=False entries of
                     TERMINAL_CONSONANT). So 'last word is a noun' is True for
                     both classes (~50% agreement, well under the gate).
  * NO y-enders      False draws EXCLUDE every final_y=True entry (the canonical
                     y stand is for graders/probes, never trained); 'ends with
                     letter y' labels everything False (~50%).
  * length matched   vowel_word and cons_word share the SAME alphabetic length
                     PER BASE, so the True and False surface strings have an
                     IDENTICAL word count AND an identical character count. Every
                     length / position / first-word battery predicate is then
                     literally identical on the two variants of every base and so
                     sits at EXACTLY 50% over the dataset (the same "the frame is
                     the only shared signal, the final word is the only
                     difference" property the reference rule exploits) -- nothing
                     but the final letter can separate the classes.
  * >= 50% non-e     the vowel fillers are drawn to keep >= 50% non-e endings and
    >= 25% silent-e  >= 25% silent-final-e across the True items (the bank is
                     27/52 non-e, 25/52 silent-e; we sample to preserve that), so
                     'ends with letter e' disagrees on ~25% and 'ends with a vowel
                     SOUND' (silent-e + discordant) disagrees on >= 25%. Neither
                     is a frozen battery predicate, but the recipe pins them and
                     the construction honours them; they are reported in
                     slots_meta / verifiable from the items.

Because the frame (hence the first word, every interior token, the word count and
char count) is shared between a base's two variants, the ONLY frozen battery
predicates that can separate the classes are last_ends_vowel (THE rule) and its
complement last_ends_consonant -- both equivalence-class exempt. Every other
frozen predicate is identical across the two variants of every base => 50%.

This is the GENERATOR INTERFACE (build_bases / instantiate); the pipeline
(base.emit_rule) runs the four gates. The module is wired into the registry via
the single documented one-line ``_REGISTRY`` entry (alphabetical, minimal diff).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, base_id as make_base_id, fill_frame
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

# Shared frames, each ending in the substitution slot '{W}' (the spec's
# "... {W}"). 6-10 words with the slot filled; sentence case; no proper nouns,
# no 'I', no commas, no terminal punctuation (all true here by construction, and
# Gate A re-checks the default style for this rule_id).
_FRAMES: list[str] = [
    "They quietly walked toward the old {W}",
    "She carefully looked behind the wooden {W}",
    "We slowly moved closer to the {W}",
    "He gently placed his hand on the {W}",
    "Everyone gathered around the freshly painted {W}",
    "The children ran straight past the {W}",
    "Someone had carelessly left out the {W}",
    "They spent the whole afternoon near the {W}",
    "She kept staring at the distant {W}",
    "We finally arrived beside the small {W}",
    "He pointed firmly toward the broken {W}",
    "The visitors admired the beautifully carved {W}",
    "They were standing right next to the {W}",
    "She wrote a short note about the {W}",
    "We talked for hours about the strange {W}",
    "He had clearly forgotten about the {W}",
    "Everyone seemed worried about the missing {W}",
    "They searched everywhere for the lost {W}",
    "She reached out and touched the cold {W}",
    "We sat together beside the quiet {W}",
]

_VOWEL_BANK = "TERMINAL_VOWEL"
_CONS_BANK = "TERMINAL_CONSONANT"

# build comfortably over the 340-base floor (100 + 120 + 100 + >= 20 spare); the
# equal-length (vowel, cons) pair pool is ~575 and there are 20 frames, so 380 is
# easily disjoint.
_N_BASES = 380

# the spec's window for the substituted sentence (informational; Gate A also
# re-checks the global [4, 14]).
_MIN_WORDS, _MAX_WORDS = 6, 10

# distribution_guards (checked over the True/vowel fillers actually used):
_NON_E_MIN = 0.50   # >= 50% non-e vowel endings
_SILENT_E_MIN = 0.25  # >= 25% silent-final-e words


@dataclass(frozen=True)
class Base:
    """One base: a frame plus the equal-length vowel / consonant final words.

    ``base_id`` is content-hashed from (frame, vowel_word, cons_word) so the two
    variants of a base share it. ``length`` is the shared alphabetic length of
    the two fillers (recorded for provenance / the silent-e audit)."""

    base_id: str
    frame: str
    vowel_word: str
    cons_word: str
    length: int


def _is_silent_e(word: str) -> bool:
    """A silent-final-e noun ('table', 'house'): ends in 'e' and is not a bare
    'Ce'/short word where the e is voiced -- for this noun bank every 'e'-ender
    is a silent terminal e, so the letter test suffices."""
    return word.endswith("e")


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (the GENERATOR INTERFACE).

    Each base pairs a frame with a (vowel_word, cons_word) of EQUAL alphabetic
    length, so the True and False variants are character-length-identical. Bases
    are enumerated deterministically (seeded shuffle of frame x equal-length word
    pair), deduped on BOTH variant surface strings, and selected so the vowel
    fillers used keep >= 50% non-e and >= 25% silent-e endings. Raises (loud) if
    the floor or a distribution guard cannot be met."""
    vowel_bank = banks.get_bank(_VOWEL_BANK)
    cons_bank = banks.get_bank(_CONS_BANK)

    # False pool: NOUN, final_y=False (clear consonant enders, NO y-enders).
    cons_entries = [
        e for e in cons_bank.entries if e.pos == "noun" and not e.final_y
    ]
    vowel_entries = list(vowel_bank.entries)  # noun-only by bank quota

    # group both pools by alphabetic length so we can pair within a length.
    v_by_len: dict[int, list[str]] = {}
    c_by_len: dict[int, list[str]] = {}
    for e in vowel_entries:
        v_by_len.setdefault(e.length, []).append(e.word)
    for e in cons_entries:
        c_by_len.setdefault(e.length, []).append(e.word)

    # every equal-length (vowel, cons) word pair, then x every frame.
    candidates: list[tuple[str, str, str, int]] = []  # (frame, vowel, cons, len)
    for L in sorted(set(v_by_len) & set(c_by_len)):
        for vw in v_by_len[L]:
            for cw in c_by_len[L]:
                for frame in _FRAMES:
                    candidates.append((frame, vw, cw, L))
    gen.shuffle(candidates)

    bases: list[Base] = []
    seen_surface: set[str] = set()
    n_non_e = 0
    n_silent_e = 0
    for frame, vw, cw, L in candidates:
        true_text = fill_frame(frame, {"W": vw})
        false_text = fill_frame(frame, {"W": cw})
        if true_text in seen_surface or false_text in seen_surface:
            continue
        wc = word_count(true_text)
        if not (_MIN_WORDS <= wc <= _MAX_WORDS):
            continue
        # both variants share the frame's word count (one final word either way)
        seen_surface.add(true_text)
        seen_surface.add(false_text)
        bid = make_base_id("last_word_ends_with_vowel", frame, vw, cw)
        bases.append(Base(base_id=bid, frame=frame, vowel_word=vw, cons_word=cw, length=L))
        if not vw.endswith("e"):
            n_non_e += 1
        if _is_silent_e(vw):
            n_silent_e += 1
        if len(bases) >= _N_BASES:
            break

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"last_word_ends_with_vowel: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )

    # distribution_guards: >= 50% non-e vowel endings, >= 25% silent-final-e,
    # over the vowel fillers actually used (each base contributes one).
    non_e_frac = n_non_e / len(bases)
    silent_e_frac = n_silent_e / len(bases)
    if non_e_frac < _NON_E_MIN:
        raise ValueError(
            f"last_word_ends_with_vowel: non-e vowel-ending share {non_e_frac:.3f} "
            f"< {_NON_E_MIN} (distribution_guards)"
        )
    if silent_e_frac < _SILENT_E_MIN:
        raise ValueError(
            f"last_word_ends_with_vowel: silent-final-e share {silent_e_frac:.3f} "
            f"< {_SILENT_E_MIN} (distribution_guards)"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True variant  = frame with {W} = the vowel-ending noun  (ends in a vowel).
    False variant = frame with {W} = the equal-length consonant-ending noun.
    Deterministic; ``gen`` is unused (the fillers are fixed on the base) but kept
    in the signature to match the interface."""
    final_word = spec.vowel_word if label else spec.cons_word
    text = fill_frame(spec.frame, {"W": final_word})
    meta = {
        "frame": spec.frame,
        "final_word": final_word,
        "vowel_word": spec.vowel_word,
        "cons_word": spec.cons_word,
        "matched_length": spec.length,
        "final_letter": final_word[-1],
        "is_silent_e": (label and _is_silent_e(final_word)),
        "transform": "substitute_vowel" if label else "substitute_consonant",
    }
    return text, meta
