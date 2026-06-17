"""Rule 11: question_word_order (category: syntactic).

Canonical articulation: True iff the input has yes/no-question word order, with
the auxiliary or copular verb BEFORE the subject (terminal punctuation absent in
all inputs); False iff it has declarative word order. The ground-truth checker
(``groundtruth._r11_question_word_order``) recomputes this from text as: the
first stripped/lowercased token is in the aux/copular set
{is, are, was, were, can, will, should, must, could}.

The two variants of a base are the SAME word multiset reordered (no '?' — the
global style strips terminal punctuation):

    base_id        = the DECLARATIVE form (rule-spec: 'base_id = declarative form')
    False variant  = declarative ('The {N} ... ')      -> first word 'The' -> False
    True  variant  = question   ('{Aux} the {N} ... ')  -> first word aux  -> True

Two families, 50% each of all bases (rule-spec generation.recipe):
  (a) MODAL    decl 'The {N1} {aux} {V} the {N2} {adjunct}'
               ques '{Aux} the {N1} {V} the {N2} {adjunct}'  aux in
               {can, will, should, must, could}
  (b) COPULAR  decl 'The {N1} {cop} {adj} {adjunct}'
               ques '{Cop} the {N1} {adj} {adjunct}'  cop in {is, are, was, were}

CONFOUND AVOIDANCE (where the gate-C battery earns its keep). Because every
False (declarative) item begins with 'The', the only first-word signal an
honest first-word predicate can see is 'is the True item's aux vowel-/consonant-
initial / in a given letter bucket / >= length k'. We therefore PIN the aux
distribution of the True class so each such predicate stays <= 0.75:

  * first_starts_vowel / _consonant : False items never start with a vowel
    ('the'), so a True item starting with a vowel (is/are) pushes the vowel
    predicate up. We hold vowel-initial auxes (is/are) to a MINORITY of True
    items (target ~30%), keeping both predicates well under 0.75.
  * first_letter_bucket_t-z         : 'The' -> t-z, so EVERY False item is in
    the t-z bucket. To keep this non-separating we make >= half of True items
    ALSO start with a t-z aux (was/were/will). The binding constraint.
  * first_word_len>=4..8            : 'the' has length 3, so a True item with a
    >=4-length aux pushes these up; we hold the >=4-length aux fraction of the
    True class to ~<= 0.5 by leaning on short auxes (is/are/was/can, len 2-3).
  * first_word_pos=verb / =determiner: True always starts with a verb, False
    always with a determiner -> ~100% each, but BOTH are EXEMPT for this rule
    via equiv_keys (they instantiate the equivalence-class string 'starts with
    an auxiliary or copular verb ...'). Not a confound to fix.

EVERYTHING ELSE is neutralized by the identical-multiset construction:
  * Word count is pinned to a constant 8 for EVERY item via the ADVERB_PLACE
    equalizer, so all word_count>=k predicates are constant (agreement 0.5) and
    the length-matching assert is trivially 0.
  * Both variants of a pair share their word multiset, so char_count, the
    last-word predicates, contains_a/and, contains_digit/comma are pair-constant
    and (with label-independent base construction) sit at ~0.5.
  * count_the>=2 / contains_the: modal cores already contain 'the' twice; the
    copular family is given a 'the'-bearing 3-word adjunct so it ALSO has 'the'
    twice -> both predicates True on every item (agreement 0.5).
  * Only the first word is capitalized in either variant (sentence case), so
    nonfirst_word_capitalized and all_lowercase are False everywhere (0.5).

Aux variety across the True class (will/can/should/must/could and
is/are/was/were all appear) defeats the 'starts with can' / 'starts with is'
distractors, per distribution_guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, adjunct_word_lengths
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

# the spec's banks for this rule
_NOUN_BANK = "NOUN_CONCRETE"
_VERB_BANK = "VERB_REGULAR"
_ADJ_BANK = "ADJ_PLAIN"
_ADVERB_BANK = "ADVERB_PLACE"

# aux/copular families (rule-spec generation.recipe; mirrors the frozen
# ground-truth aux set, partitioned by family).
_MODAL_AUX = ("can", "will", "should", "must", "could")
_COPULAR_AUX = ("is", "are", "was", "were")

# Constant per-item word count (inside the global [4, 14] window). A constant
# count makes every word_count>=k battery predicate non-separating and the
# length-matching assert trivially zero.
_TARGET_WC = 8

# Build comfortably above the 340-base floor (100 few_shot + 120 held_out +
# 100 confirmation + >= 20 spare). 50/50 modal/copular.
_N_BASES = 420

# --- aux assignment design (the confound dials) -------------------------------
# Per the module docstring: hold vowel-initial copulas (is/are) to a minority of
# True items, and make >= half the True class start with a t-z aux (was/were/
# will). We realise this with fixed per-family aux MIXTURES (counts out of a
# small cycle) that are deterministically assigned across bases.
#
# first_letter_bucket_t-z is the BINDING constraint (every False item is 'The'
# -> t-z), so we drive the t-z fraction of the True class to ~0.60 (score
# ~0.70) for margin against the split/variant sampling noise, while keeping the
# len>=4 fraction ~0.40 (score ~0.70). The lever that raises t-z WITHOUT raising
# len>=4 is the short t-z copula 'was' (length 3), so the copular mixture leans
# heavily on 'was'.
#
# COPULAR mixture (out of 10 copular bases): was 6, were 1, is 2, are 1.
#   -> t-z (was+were)      = 7/10 ; vowel-initial (is+are) = 3/10 ;
#      len>=4 (were)       = 1/10.
# MODAL mixture (out of 20 modal bases): will 10, can 6, should 2, must 1,
#   could 1 (all five present for aux variety; defeats 'starts with can').
#   -> t-z (will)          = 10/20 ; vowel-initial = 0 ;
#      len>=4 (will/should/must/could) = 14/20.
#
# Combined over the True class (modal & copular each ~50% of bases):
#   t-z            ~= 0.5*0.70 + 0.5*0.50 = 0.60   (>= 0.5 with margin)
#   vowel-initial  ~= 0.5*0.30 + 0.5*0.00 = 0.15   (<= 0.5 with margin)
#   len>=4         ~= 0.5*0.10 + 0.5*0.70 = 0.40   (<= 0.5 with margin)
# All measured exactly by the gate-C battery; tuned for headroom.
_COPULAR_MIX = (("was", 6), ("were", 1), ("is", 2), ("are", 1))
_MODAL_MIX = (("will", 10), ("can", 6), ("should", 2), ("must", 1), ("could", 1))


def _expand_mix(mix: tuple[tuple[str, int], ...]) -> list[str]:
    """Expand a (token, count) mixture into a flat cycle list."""
    out: list[str] = []
    for tok, n in mix:
        out.extend([tok] * n)
    return out


@dataclass(frozen=True)
class Base:
    """One question_word_order base.

    ``base_id`` is the DECLARATIVE surface string (rule-spec: base_id =
    declarative form). The fields carry everything ``instantiate`` needs to
    rebuild BOTH variants deterministically (no further randomness)."""

    base_id: str          # == the declarative surface string (False variant)
    family: str           # "modal" | "copular"
    aux: str              # the aux/copular verb (lowercase)
    n1: str               # subject noun
    n2: str               # object noun (modal only; "" for copular)
    verb: str             # main verb (modal only; "" for copular)
    adj: str              # adjective (copular only; "" for modal)
    adjunct: str          # the equalized place adjunct string (shared by both variants)


def _capitalize_first(text: str) -> str:
    """Uppercase the first alphabetic character (the question's first word)."""
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1 :]
    return text


def _declarative(base: Base) -> str:
    """The False (declarative) surface string for a base (sentence case)."""
    if base.family == "modal":
        core = f"the {base.n1} {base.aux} {base.verb} the {base.n2}"
    else:
        core = f"the {base.n1} {base.aux} {base.adj}"
    body = f"{core} {base.adjunct}".strip()
    return _capitalize_first(body)


def _question(base: Base) -> str:
    """The True (question) surface string for a base (sentence case): the aux
    moves to the front (capitalized), the former 'The' subject 'the' lowercases.
    SAME word multiset as the declarative, reordered; no terminal '?'."""
    if base.family == "modal":
        core = f"{base.aux} the {base.n1} {base.verb} the {base.n2}"
    else:
        core = f"{base.aux} the {base.n1} {base.adj}"
    body = f"{core} {base.adjunct}".strip()
    return _capitalize_first(body)


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases, 50/50 modal/copular, with the designed aux
    mixtures and each pinned to a constant word count of 8 (GENERATOR INTERFACE).

    Deterministic given ``gen``. The place adjunct is chosen HERE (once per base)
    and stored, so ``instantiate`` appends the IDENTICAL adjunct to both variants
    (identical multiset, identical word count)."""
    nouns = banks.get_bank(_NOUN_BANK).words()
    verbs = banks.get_bank(_VERB_BANK).words()
    adjs = banks.get_bank(_ADJ_BANK).words()
    adverbs = banks.get_bank(_ADVERB_BANK).words()

    by_len = adjunct_word_lengths(adverbs)
    # adjuncts that contain the token 'the' (all 3-word ADVERB_PLACE phrases do),
    # used to give the copular family a SECOND 'the' so count_the>=2 is True on
    # every item (neutralising that battery predicate).
    the_adjuncts_3w = [
        p for p in by_len.get(3, []) if "the" in [w.lower() for w in p.split()]
    ]
    if not the_adjuncts_3w:
        raise ValueError("question_word_order: no 3-word 'the'-bearing adjunct in bank")
    one_word = by_len.get(1, [])
    two_word = by_len.get(2, [])
    if not one_word or not two_word:
        raise ValueError("question_word_order: need 1- and 2-word adjuncts to equalize")

    n_each = _N_BASES // 2  # bases per family
    modal_cycle = _expand_mix(_MODAL_MIX)
    copular_cycle = _expand_mix(_COPULAR_MIX)

    # independent seeded streams so the family loops do not share draw order
    g_modal = gen.derive("modal")
    g_copular = gen.derive("copular")
    g_adj = gen.derive("adjunct")

    bases: list[Base] = []
    seen_decl: set[str] = set()

    # ----- MODAL family -------------------------------------------------------
    # core 'The N1 aux V the N2' = 6 words -> deficit 2 -> one 2-word adjunct
    # (no 'the'); modal cores already contain 'the' twice.
    for i in range(n_each):
        aux = modal_cycle[i % len(modal_cycle)]
        n1 = g_modal.choice(nouns)
        n2 = g_modal.choice(nouns)
        verb = g_modal.choice(verbs)
        # 2-word adjunct with NO 'the' so modal count_the stays exactly 2
        adjunct = g_adj.choice(two_word)
        base = Base(
            base_id="",  # filled after we render the declarative
            family="modal", aux=aux, n1=n1, n2=n2, verb=verb, adj="", adjunct=adjunct,
        )
        decl = _declarative(base)
        if word_count(decl) != _TARGET_WC or decl in seen_decl:
            continue
        seen_decl.add(decl)
        bases.append(Base(
            base_id=decl, family="modal", aux=aux, n1=n1, n2=n2, verb=verb,
            adj="", adjunct=adjunct,
        ))

    # ----- COPULAR family -----------------------------------------------------
    # core 'The N1 cop adj' = 4 words -> deficit 4 -> a 3-word 'the'-adjunct +
    # a 1-word adjunct = 4 words; the 'the' adjunct gives copular its 2nd 'the'.
    for i in range(n_each):
        cop = copular_cycle[i % len(copular_cycle)]
        n1 = g_copular.choice(nouns)
        adj = g_copular.choice(adjs)
        the_adj = g_adj.choice(the_adjuncts_3w)
        tail = g_adj.choice(one_word)
        adjunct = f"{the_adj} {tail}"
        base = Base(
            base_id="", family="copular", aux=cop, n1=n1, n2="", verb="",
            adj=adj, adjunct=adjunct,
        )
        decl = _declarative(base)
        if word_count(decl) != _TARGET_WC or decl in seen_decl:
            continue
        seen_decl.add(decl)
        bases.append(Base(
            base_id=decl, family="copular", aux=cop, n1=n1, n2="", verb="",
            adj=adj, adjunct=adjunct,
        ))

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"question_word_order: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (GENERATOR INTERFACE).

    True  = question form ('{Aux} the {N} ...')   -> first word aux  -> label True
    False = declarative form ('The {N} ...')       -> first word 'The'-> label False
    Both share the exact word multiset (identical word count, no '?'); ``gen`` is
    unused (the transform carries no randomness) but kept to match the interface.
    """
    if label:
        text = _question(spec)
        form = "question"
    else:
        text = spec.base_id  # the declarative surface string
        form = "declarative"
    meta = {
        "family": spec.family,
        "aux": spec.aux,
        "n1": spec.n1,
        "n2": spec.n2,
        "verb": spec.verb,
        "adj": spec.adj,
        "adjunct": spec.adjunct,
        "form": form,
        "target_word_count": _TARGET_WC,
        "declarative": spec.base_id,
    }
    return text, meta
