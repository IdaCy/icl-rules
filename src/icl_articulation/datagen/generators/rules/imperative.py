"""Rule 10: imperative (category syntactic).

Canonical articulation: True iff the input is an imperative (a command addressed
to the listener, with NO expressed subject); False iff it is a declarative
statement with a subject.

GENERATOR INTERFACE (mirrors the reference rule all_lowercase): ``build_bases``
returns >= 340 distinct content frames; ``instantiate`` renders ONE variant
(True imperative / False declarative) of a frame. The two variants of a base
share their CONTENT (verb stem, object phrase, tail adjunct, the adverb-start
decision and that adverb); they differ ONLY in the subjectless-command vs
subject-bearing-declarative shape:

    structure        True (imperative)              False (declarative)
    -------------    ---------------------------    -------------------------------
    bare-verb (75%)  Verb OBJ PAD TAIL              Pron Verb3sg OBJ TAIL
    adverb (25%)     Adv Verb OBJ PAD TAIL          Adv Pron Verb3sg OBJ TAIL

where Pron in {she, he, they, we}. The word count is equalized PER BASE without
the equalizer: the imperative carries a 1-word adverb PAD (mid-sentence, after
the object) exactly where the declarative carries the subject Pron; the bare
verb (1 word) matches the 3sg verb (1 word). So |imperative| == |declarative|
EXACTLY for every base (Gate D length-matching is satisfied by construction, not
by averaging).

Confound design (why the four gates pass):
  * Everything except the first word and the {PAD vs Pron} swap is SHARED per
    base (object phrase, tail adjunct, optional sentence-initial adverb, verb
    stem). So every whole-text token predicate (contains the/a/and, count_the,
    all_lowercase, comma/digit) and every LAST-word predicate is IDENTICAL on
    the two variants of a base -> exactly 0.5 agreement on the dataset.
  * The adverb-start mix is SYMMETRIC (25% of BOTH classes draw the SAME
    sentence-initial adverb pool), so the 25% adverb-initial items contribute
    exactly 0.5 to every first-word predicate.
  * FIRST-WORD predicates are the only place the classes can differ (verb-start
    imperative vs pronoun-start declarative on the other 75%). The frozen
    battery EXEMPTS first_word_pos=verb and first_word_pos=pronoun (imperatives
    are verb-initial by definition). The NON-exempt first-word predicates
    (first_letter_bucket_*, first_starts_vowel/consonant, first_word_len>=k) are
    held <= 0.75 by (a) restricting imperative verbs to the letter buckets the
    subject pronouns occupy (g-m via 'he', n-s via 'she', t-z via 'they'/'we')
    so all four bucket predicates sit at 0.5, and (b) choosing the verb-length
    and pronoun-length marginals + the PAD length so first_word_len and
    vowel/consonant stay near 0.5 (worst-case ~0.65, well under the gate).
  * OBJECT MIX (recipe): 40% of objects are an object pronoun 'it' or a bare
    plural noun in BOTH classes; 60% are 'the {N}'. Shared per base, so it does
    not move any battery predicate (those are base-symmetric) but it removes the
    'second word is the' / determiner-position multiple-choice distractor the recipe pins.

'Please' is banned everywhere; no digits, no commas, no terminal punctuation
(global style; imperative is not in RULE_STYLE_POLICY -> strict default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, base_id as make_base_id
from ...schema import (
    PROGRAMMATIC_N_BASES_MIN,
    WORD_COUNT_MAX,
    WORD_COUNT_MIN,
    word_count,
)

# style alias: imperative uses the strict global style (no terminal, no comma),
# which is what style_policy_for returns for an unlisted rule_id, so no alias.

_VERB_BANK = "VERB_REGULAR"
_NOUN_BANK = "NOUN_CONCRETE"
_ADVERB_INIT_BANK = "ADVERB_SENT_INITIAL"
_ADVERB_PLACE_BANK = "ADVERB_PLACE"

# build comfortably above the 340-base floor (100+120+100+>=20 spare).
_N_BASES = 380

# recipe word-count window for this rule (kept inside the global [4, 14]).
_MIN_WORDS, _MAX_WORDS = 5, 9

# fraction of bases that use the adverb-initial shape (symmetric across classes).
_ADVERB_START_RATE = 0.25
# fraction of bases whose object is an object-pronoun 'it' or a bare plural
# (recipe target: 40% non-determiner objects in BOTH classes). Set slightly
# above 0.40 because dedup pressure on the smaller 'it'/bare-plural content
# space rejects more of those candidates, so the REALIZED rate lands ~0.40.
_NONDET_OBJECT_RATE = 0.48

# 'please' is banned everywhere (no verb here is 'please'; assert anyway).
_BANNED_TOKENS = frozenset({"please", "i"})


def _coarse_bucket(letter: str) -> str:
    if letter in "abcdef":
        return "a-f"
    if letter in "ghijklm":
        return "g-m"
    if letter in "nopqrs":
        return "n-s"
    return "t-z"


# Subject pronouns the declarative plants, grouped by the first-letter bucket
# they occupy. Restricting imperative verbs to these SAME buckets makes the four
# first_letter_bucket_* battery predicates sit at exactly 0.5 (a-f never opens
# either class, g-m/n-s/t-z share their per-base frequency between classes).
#   g-m: he   n-s: she   t-z: they / we
_PRON_BY_BUCKET: dict[str, tuple[str, ...]] = {
    "g-m": ("he",),
    "n-s": ("she",),
    "t-z": ("they", "we"),
}
# buckets the imperative verb is allowed to open with (== pronoun buckets).
_VERB_BUCKETS: tuple[str, ...] = ("g-m", "n-s", "t-z")

# per-bucket base frequencies (must be reachable by both a verb and a pronoun).
# Tuned (with the 25% adverb-start dilution) so the worst NON-exempt first-word
# predicate stays well under 0.75: the only first-word asymmetry left is verb
# length (>= 4 chars, vs the short subject pronouns), so the t-z bucket is
# weighted up and 'they' (len 4) lifts the declarative first-word-length
# marginal toward the verbs'. (Bucket predicates are 0.5 for ANY weights because
# verb-bucket == pron-bucket per base; the weights only move first_word_len>=k.)
#
# INCIDENTAL 'they' TOKEN BALANCE (this fix round): the subject pronoun is the
# only declarative-class token, so an over-weighted 'they' became a single-token
# confound (max(agree,1-agree) ~ 0.76, over the 0.75 incidental-token bar). But
# 'they' is ALSO the only len-4 pronoun, so it must carry enough declarative
# first-word length to keep first_word_len>=4 under 0.75 too. The two pull in
# opposite directions; with the spec-pinned 25% adverb-start the feasible window
# is ~ 70-120 'they' declaratives. These weights (t-z 0.48, they 0.78 of t-z)
# land 'they' at ~95 declaratives -> the 'they' single token ~0.70 AND
# first_word_len>=4 ~0.71, both comfortably inside the gate.
_BUCKET_WEIGHTS: dict[str, float] = {"g-m": 0.25, "n-s": 0.27, "t-z": 0.48}
_THEY_SHARE_IN_TZ = 0.78  # of t-z bases use 'they' (len 4), rest 'we'

# PAD: a 1-word mid-sentence adverb the imperative carries where the declarative
# carries its subject pronoun. Drawn so its char length tracks |pron| + 1 (the
# declarative also gains the 3sg '-s'), keeping char_count balanced. These read
# acceptably after an object ('close the window quickly downtown'); confound
# structure is the priority (naturalness is best-effort for this rule).
_PAD_ADVERBS_SHORT = ("here", "soon", "fast", "well", "now")          # ~4 chars
_PAD_ADVERBS_MED = ("today", "twice", "later", "again", "alone")      # ~5 chars
_PAD_ADVERBS_LONG = ("nearby", "calmly", "neatly", "firmly")          # ~6 chars


@dataclass(frozen=True)
class Base:
    """One imperative/declarative content frame.

    Carries everything both variants need; the two variants are rendered purely
    from these fields (instantiate does no extra random draw)."""

    base_id: str
    verb: str          # bare verb (imperative form; declarative form for they/we)
    verb_decl: str     # declarative present verb agreeing with the subject:
                       # 3sg (closes) for she/he, bare plural (close) for they/we
    pron: str          # subject pronoun (she/he/they/we)
    obj: str           # object phrase, shared by both classes
    tail: str          # tail adjunct (ADVERB_PLACE), shared, always the LAST words
    pad: str           # 1-word adverb the imperative carries (declarative omits)
    adverb_start: bool
    adverb: str        # sentence-initial adverb (only if adverb_start)


def _third_person(verb: str) -> str:
    """Regular 3rd-singular present: +es after sibilants/-o, y->ies after a
    consonant, else +s. Exact for every VERB_REGULAR entry."""
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    if verb.endswith(("s", "x", "z", "ch", "sh", "o")):
        return verb + "es"
    return verb + "s"


def _regular_plural_ok(noun: str) -> bool:
    """True iff ``noun`` pluralizes by a bare +s (used for bare-plural objects)."""
    if noun.endswith(("s", "x", "z", "ch", "sh", "o", "y", "f", "fe")):
        return False
    return True


_SINGULAR_PRONS = frozenset({"she", "he"})


def _pick_pad(pron: str, gen: Gen) -> str:
    """A 1-word PAD adverb whose length tracks the extra characters the
    declarative carries where the imperative carries PAD: the subject pronoun
    plus, for a singular subject, the 3sg '-s' on the verb. Keeps char_count
    matched between the two classes."""
    target = len(pron) + (1 if pron in _SINGULAR_PRONS else 0)
    if target <= 4:
        pool = _PAD_ADVERBS_SHORT
    elif target == 5:
        pool = _PAD_ADVERBS_MED
    else:
        pool = _PAD_ADVERBS_LONG
    return gen.choice(pool)


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct content frames (the GENERATOR INTERFACE).

    Deterministic given ``gen``. Each frame fixes a (bucket -> verb, pronoun),
    an object phrase, a tail adjunct, the adverb-start decision and adverb, and
    the imperative's PAD adverb. Frames are filtered so BOTH rendered variants
    land in [5, 9] words, deduped by content tuple, and capped at _N_BASES."""
    verbs_all = banks.get_bank(_VERB_BANK).words()
    nouns_all = banks.get_bank(_NOUN_BANK).words()
    adverbs_init = banks.get_bank(_ADVERB_INIT_BANK).words()
    adverbs_place = banks.get_bank(_ADVERB_PLACE_BANK).words()

    verbs_by_bucket: dict[str, list[str]] = {b: [] for b in _VERB_BUCKETS}
    for v in verbs_all:
        b = _coarse_bucket(v[0])
        if b in verbs_by_bucket:
            verbs_by_bucket[b].append(v)

    plural_nouns = [n for n in nouns_all if _regular_plural_ok(n)]

    # tail adjuncts grouped by word count (place phrases, 1-3 words).
    tails_by_len: dict[int, list[str]] = {}
    for ph in adverbs_place:
        tails_by_len.setdefault(word_count(ph), []).append(ph)

    bucket_keys = list(_BUCKET_WEIGHTS)
    bucket_w = [_BUCKET_WEIGHTS[b] for b in bucket_keys]

    bases: list[Base] = []
    seen_content: set[tuple] = set()
    seen_surface: set[str] = set()

    # generous attempt budget; the construction is dense so this terminates fast.
    attempts = 0
    max_attempts = _N_BASES * 200
    while len(bases) < _N_BASES and attempts < max_attempts:
        attempts += 1

        bucket = gen.rng.choices(bucket_keys, weights=bucket_w, k=1)[0]
        verb = gen.choice(verbs_by_bucket[bucket])

        prons = _PRON_BY_BUCKET[bucket]
        if bucket == "t-z":
            pron = "they" if gen.rng.random() < _THEY_SHARE_IN_TZ else "we"
        else:
            pron = prons[0]

        # declarative verb agrees with the subject: 3sg for she/he, bare plural
        # (== imperative form) for they/we. groundtruth labels by subject
        # position, not verb form, so either keeps the label correct.
        verb_decl = _third_person(verb) if pron in _SINGULAR_PRONS else verb

        # object phrase (40% pronoun/bare-plural, 60% determiner), shared.
        roll = gen.rng.random()
        if roll < _NONDET_OBJECT_RATE / 2:
            obj = "it"
            obj_kind = "pronoun"
        elif roll < _NONDET_OBJECT_RATE:
            noun = gen.choice(plural_nouns)
            obj = noun + "s"
            obj_kind = "bare_plural"
        else:
            noun = gen.choice(nouns_all)
            obj = "the " + noun
            obj_kind = "determiner"

        tail_len = gen.choice([1, 2, 3])
        tail = gen.choice(tails_by_len[tail_len])

        adverb_start = gen.rng.random() < _ADVERB_START_RATE
        adverb = gen.choice(adverbs_init) if adverb_start else ""

        pad = _pick_pad(pron, gen)

        spec = Base(
            base_id="",  # filled below
            verb=verb,
            verb_decl=verb_decl,
            pron=pron,
            obj=obj,
            tail=tail,
            pad=pad,
            adverb_start=adverb_start,
            adverb=adverb,
        )

        # word-count gate (recipe 5-9, inside global [4,14]); both variants are
        # equal length by construction, so checking one suffices, but check both.
        t_text, _ = _render(spec, True)
        f_text, _ = _render(spec, False)
        wc_t, wc_f = word_count(t_text), word_count(f_text)
        if wc_t != wc_f:  # defensive: the design guarantees equality
            continue
        if not (_MIN_WORDS <= wc_t <= _MAX_WORDS):
            continue
        if not (WORD_COUNT_MIN <= wc_t <= WORD_COUNT_MAX):
            continue

        content = (
            verb, verb_decl, pron, obj, tail, pad, adverb_start, adverb, obj_kind,
        )
        if content in seen_content:
            continue
        if t_text in seen_surface or f_text in seen_surface or t_text == f_text:
            continue

        bid = make_base_id("imperative", *content)
        spec = Base(
            base_id=bid,
            verb=verb,
            verb_decl=verb_decl,
            pron=pron,
            obj=obj,
            tail=tail,
            pad=pad,
            adverb_start=adverb_start,
            adverb=adverb,
        )
        seen_content.add(content)
        seen_surface.add(t_text)
        seen_surface.add(f_text)
        bases.append(spec)

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"imperative: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN} (attempts={attempts})"
        )
    return bases


def _capitalize_first(text: str) -> str:
    """Sentence case: uppercase the first alphabetic char, leave the rest."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def _render(spec: Base, label: bool) -> tuple[str, str]:
    """Render the ``label`` variant of ``spec``; returns (text, transform_tag).

    True  (imperative)  = [Adv] Verb OBJ PAD TAIL   (subjectless command)
    False (declarative) = [Adv] Pron Verb3sg OBJ TAIL
    Word counts are equal per base: the imperative's PAD (1 word) replaces the
    declarative's subject Pron (1 word); bare verb (1) == 3sg verb (1)."""
    if label:
        # imperative: verb-initial (or adverb + verb), no subject, carries PAD.
        if spec.adverb_start:
            parts = [spec.adverb, spec.verb, spec.obj, spec.pad, spec.tail]
            tag = "imperative_adverb_start"
        else:
            parts = [spec.verb, spec.obj, spec.pad, spec.tail]
            tag = "imperative_bare_verb"
    else:
        # declarative: subject pronoun + agreeing present verb (or adverb +
        # subject + verb).
        if spec.adverb_start:
            parts = [spec.adverb, spec.pron, spec.verb_decl, spec.obj, spec.tail]
            tag = "declarative_adverb_start"
        else:
            parts = [spec.pron, spec.verb_decl, spec.obj, spec.tail]
            tag = "declarative_subject_initial"
    text = _capitalize_first(" ".join(parts))
    return text, tag


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    Deterministic; ``gen`` is unused (all randomness was resolved in
    build_bases and frozen onto the spec)."""
    text, transform = _render(spec, label)
    # defensive style assertions (the gates re-check, but fail fast & local).
    toks_lower = [t.lower() for t in text.split()]
    if any(t.strip(".,!?;:\"'()[]-") in _BANNED_TOKENS for t in toks_lower):
        raise ValueError(f"imperative: banned token in {text!r}")
    meta = {
        "transform": transform,
        "structure": "adverb_start" if spec.adverb_start else "canonical",
        "verb": spec.verb,
        "verb_decl": spec.verb_decl,
        "pron": spec.pron,
        "object": spec.obj,
        "tail": spec.tail,
        "pad": spec.pad,
        "adverb": spec.adverb,
    }
    return text, meta
