"""Rule 29: first_two_words_alphabetical (category: hard_articulation).

Canonical articulation: True iff the input's first word precedes its second word
in alphabetical (lexicographic, case-insensitive) order; a tie is False. The
ground-truth verifier ``_r29_first_two_words_alphabetical`` recomputes the label
as ``stripped_lower(text)[0] < stripped_lower(text)[1]``.

Construction (per the private rule spec, not in this repository; id: first_two_words_alphabetical)
-------------------------------------------------------------------------------------
Frame ``{Adj} {nouns} {verb-past} {place-adjunct...}`` (5-9 words). The FIRST word
is an adjective from ADJ_BY_LETTER (sentence-cased), the SECOND a plural noun from
NOUN_PLURAL_BY_LETTER; both banks cover the 14 letters {b,c,d,f,g,h,l,m,n,p,r,s,t,w}
with >= 5 entries. The verb (past tense of a VERB_REGULAR base) and the trailing
ADVERB_PLACE adjunct(s) are SHARED VERBATIM between a base's two variants, so word
count, char count, last word, "the"/"a"/"and" content -- every battery feature that
reads past the first two words -- is IDENTICAL across the two variants of a base and
therefore sits at exactly 50% on the few-shot split.

Two base families realise the recipe's confound machinery:

  CROSS bases (~90% of bases) -- the (x,y)/(y,x) MIRROR + PAIR WINDOWING:
    A base fixes a windowed initial pair (lo, hi), lo < hi alphabetically and
    within 4 alphabet positions of each other (gaps 1-4, windows spread b..w).
      True  variant: Adj initial = lo, Noun initial = hi  (lo < hi  -> True).
      False variant: Adj initial = hi, Noun initial = lo  (hi > lo  -> False).
    So every True letter-pair (x,y) has a mirror False (y,x): the per-position
    letter marginals stay symmetric and the four first_letter_bucket predicates
    (and any "first/second letter <= m" proxy) stay near chance. The adjective in
    the True variant and the adjective in the False variant are chosen with the
    SAME length (likewise the two nouns), so first_word_len and char_count are
    identical across the two variants -> 50% by construction.

  SAME-INITIAL bases (<= 10% of items) -- the rule's letter-2 fine print:
    Both first words share an initial L; the label is decided at letter 2 by full
    lexicographic comparison.
      True  variant: Adj_T, Noun_T with Adj_T < Noun_T (e.g. "Big birds ...").
      False variant: Adj_F, Noun_F with Adj_F > Noun_F (e.g. "Brown books ...").
    Lengths are matched across the variants (len Adj_T == len Adj_F, len Noun_T ==
    len Noun_F) so first_word_len / char_count stay at 50%. There is NO cross-letter
    signal here (both words share the initial), so these only train the fine print.

base_id = genutils.base_id(frame-tail content + letter window + the two variants'
adjective/noun words), stable and shared by the base's True and False variant.

This module exposes the GENERATOR INTERFACE (build_bases / instantiate); it is
auto-discovered by ``generators/registry.py`` and run through the shared gated
pipeline (``base.emit_rule``) by the plain CLI
``python -m icl_articulation.datagen.generators first_two_words_alphabetical``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...banks import _regular_verb_forms
from ...genutils import (
    Gen,
    adjunct_word_lengths,
    base_id as make_base_id,
    equalize_word_count,
    to_sentence_case,
)
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

STYLE_RULE_ID = "first_two_words_alphabetical"  # default global style ('', no comma)

# The 14 letters both banks cover with >= 5 entries (rule-specs recipe).
_LETTERS = "bcdfghlmnprstw"

# Word-count window the recipe names (5-9); the global [4, 14] cap is re-checked
# by the schema validator. Targets are spread across the window so word_count>=k
# predicates have variety (every base's two variants share its target).
_MIN_WORDS, _MAX_WORDS = 5, 9
_WC_TARGETS = (5, 6, 7, 8, 9)

# How many bases to build (comfortably over the 340 floor so the by-base split
# 100 + 120 + 100 + >= 20 spare has headroom). ~10% are same-initial bases.
_N_BASES = 380
_SAME_INITIAL_FRAC = 0.09  # <= 10% of items (the recipe cap)

# Window: the two initials are within 4 alphabet positions (gaps 1..4) for the
# CROSS family; same-initial is gap 0. Both are "within 4" -> 100% windowed.
_MAX_GAP = 4


def _pos(letter: str) -> int:
    return ord(letter) - ord("a")


def _windowed_pairs() -> list[tuple[str, str]]:
    """All ordered (lo, hi) initial pairs from the 14-letter set with lo < hi and
    the two within ``_MAX_GAP`` alphabet positions. Windows span b..w."""
    out: list[tuple[str, str]] = []
    for x in _LETTERS:
        for y in _LETTERS:
            if _pos(x) < _pos(y) and 1 <= _pos(y) - _pos(x) <= _MAX_GAP:
                out.append((x, y))
    return out


@dataclass(frozen=True)
class Base:
    """One base spec carrying BOTH variants' word choices (so instantiate is pure).

    family       'cross' (the (x,y)/(y,x) mirror) or 'same' (letter-2 fine print).
    adj_true/noun_true  the True variant's first two words (adj precedes noun).
    adj_false/noun_false the False variant's first two words (adj follows noun).
    verb_past    the shared past-tense verb (3rd word, identical in both variants).
    tail         the shared adjunct material appended to reach the word target.
    lo/hi        the window initials (lo == hi for the same-initial family).
    wc_target    the shared word count both variants hit.
    """

    base_id: str
    family: str
    adj_true: str
    noun_true: str
    adj_false: str
    noun_false: str
    verb_past: str
    tail: str
    lo: str
    hi: str
    wc_target: int


def _by_initial(words: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for w in words:
        out.setdefault(w[0].lower(), []).append(w)
    return out


def _by_len(words: list[str]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for w in words:
        out.setdefault(len(w), []).append(w)
    return out


def _surface_core(adj: str, noun: str, verb_past: str) -> str:
    """The 3-word core 'Adj noun verb' (lowercased words; cased by the caller)."""
    return f"{adj.lower()} {noun.lower()} {verb_past}"


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (the GENERATOR INTERFACE).

    Deterministic given ``gen``. CROSS bases pick a windowed pair (lo, hi) and a
    length-matched adjective/noun choice for each role; SAME-INITIAL bases pick a
    letter and a length-matched True (adj < noun) / False (adj > noun) quadruple.
    The shared verb + adjunct tail equalises every base's two variants to one word
    target. Raises (loud) if the base floor cannot be reached."""
    adj_words = banks.get_bank("ADJ_BY_LETTER").words()
    noun_words = banks.get_bank("NOUN_PLURAL_BY_LETTER").words()
    verb_bases = banks.get_bank("VERB_REGULAR").words()
    adverbs = banks.get_bank("ADVERB_PLACE").words()

    adj_by_initial = _by_initial(adj_words)
    noun_by_initial = _by_initial(noun_words)
    adj_len = {L: _by_len(ws) for L, ws in adj_by_initial.items()}
    noun_len = {L: _by_len(ws) for L, ws in noun_by_initial.items()}

    # past-tense verb forms (3rd word; never reads into the first-two comparison)
    verbs_past = [_regular_verb_forms(v)[2] for v in verb_bases]

    adjuncts_by_len = adjunct_word_lengths(adverbs)  # {1: [...], 2: [...], 3: [...]}

    pairs = _windowed_pairs()

    n_same_target = int(round(_N_BASES * _SAME_INITIAL_FRAC))
    n_cross_target = _N_BASES - n_same_target

    bases: list[Base] = []
    seen_ids: set[str] = set()
    seen_true: set[str] = set()
    seen_false: set[str] = set()

    bgen = gen.derive("bases")

    def _tail(core: str, target: int) -> str | None:
        """Adjunct material that brings ``core`` (3 words) up to ``target`` words.
        Returns just the appended tail (may be empty if target == core wc)."""
        if word_count(core) > target:
            return None
        full = equalize_word_count(core, target, adjuncts_by_len, bgen)
        tail = full[len(core):].lstrip()
        return tail

    def _register(b: Base, true_text: str, false_text: str) -> bool:
        if b.base_id in seen_ids:
            return False
        if true_text in seen_true or false_text in seen_false:
            return False
        if true_text in seen_false or false_text in seen_true:
            return False
        seen_ids.add(b.base_id)
        seen_true.add(true_text)
        seen_false.add(false_text)
        bases.append(b)
        return True

    # ---- CROSS family (windowed (lo,hi) / (hi,lo) mirror) --------------------
    # Enumerate every length-matched (adj_lo, adj_hi, noun_hi, noun_lo) word combo
    # per windowed pair, shuffle deterministically, and take until the target.
    cross_specs: list[tuple[str, str, str, str, str, str]] = []
    for lo, hi in pairs:
        a_lo, a_hi = adj_len.get(lo, {}), adj_len.get(hi, {})
        n_lo, n_hi = noun_len.get(lo, {}), noun_len.get(hi, {})
        common_adj_len = sorted(set(a_lo) & set(a_hi))
        common_noun_len = sorted(set(n_lo) & set(n_hi))
        for la in common_adj_len:
            for adj_true in a_lo[la]:
                for adj_false in a_hi[la]:
                    for ln in common_noun_len:
                        for noun_true in n_hi[ln]:
                            for noun_false in n_lo[ln]:
                                cross_specs.append(
                                    (lo, hi, adj_true, noun_true, adj_false, noun_false)
                                )
    bgen.shuffle(cross_specs)

    for (lo, hi, adj_true, noun_true, adj_false, noun_false) in cross_specs:
        if sum(1 for b in bases if b.family == "cross") >= n_cross_target:
            break
        verb_past = bgen.choice(verbs_past)
        target = bgen.choice(list(_WC_TARGETS))
        # tail computed off the TRUE core; both variants share the same verb and
        # word target, and adj/noun are length-matched, so the FALSE core has the
        # same word count -> the same tail equalises both to ``target``.
        core_true = _surface_core(adj_true, noun_true, verb_past)
        tail = _tail(core_true, target)
        if tail is None:
            continue
        true_text = to_sentence_case((core_true + " " + tail).strip())
        core_false = _surface_core(adj_false, noun_false, verb_past)
        false_text = to_sentence_case((core_false + " " + tail).strip())
        wc = word_count(true_text)
        if not (_MIN_WORDS <= wc <= _MAX_WORDS) or word_count(false_text) != wc:
            continue
        bid = make_base_id(
            "cross", lo, hi, adj_true, noun_true, adj_false, noun_false, verb_past, tail
        )
        b = Base(
            base_id=bid, family="cross",
            adj_true=adj_true, noun_true=noun_true,
            adj_false=adj_false, noun_false=noun_false,
            verb_past=verb_past, tail=tail, lo=lo, hi=hi, wc_target=target,
        )
        _register(b, true_text, false_text)

    # ---- SAME-INITIAL family (letter-2 fine print, <= 10% of items) ----------
    same_specs: list[tuple[str, str, str, str, str]] = []
    for L in _LETTERS:
        A = adj_by_initial.get(L, [])
        N = noun_by_initial.get(L, [])
        trues = [(a, n) for a in A for n in N if a.lower() < n.lower()]
        falses = [(a, n) for a in A for n in N if a.lower() > n.lower()]
        for (aT, nT) in trues:
            for (aF, nF) in falses:
                if len(aT) == len(aF) and len(nT) == len(nF):
                    same_specs.append((L, aT, nT, aF, nF))
    bgen.shuffle(same_specs)

    for (L, aT, nT, aF, nF) in same_specs:
        if sum(1 for b in bases if b.family == "same") >= n_same_target:
            break
        verb_past = bgen.choice(verbs_past)
        target = bgen.choice(list(_WC_TARGETS))
        core_true = _surface_core(aT, nT, verb_past)
        tail = _tail(core_true, target)
        if tail is None:
            continue
        true_text = to_sentence_case((core_true + " " + tail).strip())
        core_false = _surface_core(aF, nF, verb_past)
        false_text = to_sentence_case((core_false + " " + tail).strip())
        wc = word_count(true_text)
        if not (_MIN_WORDS <= wc <= _MAX_WORDS) or word_count(false_text) != wc:
            continue
        bid = make_base_id("same", L, aT, nT, aF, nF, verb_past, tail)
        b = Base(
            base_id=bid, family="same",
            adj_true=aT, noun_true=nT, adj_false=aF, noun_false=nF,
            verb_past=verb_past, tail=tail, lo=L, hi=L, wc_target=target,
        )
        _register(b, true_text, false_text)

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"first_two_words_alphabetical: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN} (enlarge banks / loosen window)"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True  -> first word (adjective) alphabetically precedes the second (noun).
    False -> first word follows the second. The verb + adjunct tail is shared
    verbatim; ``gen`` is unused (every choice was frozen in build_bases, so this
    is pure)."""
    if label:
        adj, noun = spec.adj_true, spec.noun_true
        transform = "adj_precedes_noun"
    else:
        adj, noun = spec.adj_false, spec.noun_false
        transform = "adj_follows_noun"
    core = _surface_core(adj, noun, spec.verb_past)
    text = to_sentence_case((core + " " + spec.tail).strip())
    meta = {
        "family": spec.family,
        "transform": transform,
        "adjective": adj.lower(),
        "noun": noun.lower(),
        "verb_past": spec.verb_past,
        "tail": spec.tail,
        "first_initial": adj[0].lower(),
        "second_initial": noun[0].lower(),
        "window_lo": spec.lo,
        "window_hi": spec.hi,
        "word_count": spec.wc_target,
    }
    return text, meta
