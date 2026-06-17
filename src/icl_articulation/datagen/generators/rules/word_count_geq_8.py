"""Rule 23: word_count_geq_8 (category: numeric).

Canonical articulation: True iff the input contains at least 8 words (global
tokenizer; hyphenated words count as one, digits count as a word — but no digits
appear in this rule's data).

Construction (rule-specs ``id: word_count_geq_8`` generation.recipe — "Core
clauses (4-5 words) + optional adjunct slots (1-3 words each)"):
  * Each CONCORDANT base is a GRAMMATICAL English sentence. Its FALSE variant is
    a complete past-tense clause of exactly the (sub-8) false target word count:
        <subjDet> <subjAdj?> [adverb?] <pastVerb> <objDet> <ender>
    (e.g. "The old worker slowly cleaned the gate"). Past tense makes the clause
    number-agreement-safe — no "a tree wash" / "a letter arrive" errors. The
    object noun (``ender``) ends the clause.
  * The TRUE variant is the SAME clause with optional ADJUNCT SLOTS (1-3 words
    each: "quietly", "with care", "by hand at home") APPENDED, padding it to the
    (>=8) true target ("The old worker slowly cleaned the gate quietly by hand").
    The recipe's "True/False variants share the core and differ in adjunct count":
    the only difference between the pair is a run of trailing adjuncts, so WORD
    COUNT is the only intended signal.
  * The optional adjuncts are ARTICLE-FREE ('downtown', 'with care', 'by hand',
    ...). The False variant IS the core, so the True variant has the SAME number
    of 'the'/'a' as its False partner — contains_the / contains_a / count_the>=2
    cannot separate the classes.

GRAMMATICALITY (a hand review found ungrammatical items in the prior build): TRUE items are GRAMMATICAL sentences, not a
pile of trailing bare nouns. The old construction (a) appended the ``ender`` noun
as a glued final token after an already-complete clause ('... with care lock',
'... often seed') and (b) used bare plural verb forms with singular subjects ('A
tree wash a box', 'A letter slowly arrive ...'). The new construction uses PAST
TENSE (agreement-safe), makes the ``ender`` the grammatical OBJECT of the clause,
and pads the True variant with real adjunct phrases — so every item reads as
ordinary English.

CHAR-LENGTH DISCORDANCE (the recipe's headline confound guard):
  * >= 25% of items are char-discordant. A DISCORDANT base builds its True
    variant as a GRAMMATICAL short-word sentence (every content word <= 3 chars,
    past tense: 'The red fox sat on a wet log by the dam' = 11 words, ~30 chars
    -> long word-count, LOW char-count) and its False variant as a GRAMMATICAL
    long-word clause from the LONGEST entries of the rule's OWN banks ('The
    ancient gardener collected the photographs' = 6 words, ~46 chars -> short
    word-count, HIGH char-count). This INVERTS the natural "more words => more
    chars" correlation, so char_count>=35/40/45 cannot separate the classes. The
    discordant fraction is ~33% (> the 25% quota).

Why each NON-exempt battery predicate stays <= 0.75:
  * word_count>=8 IS the rule (exempt via equiv_keys).
  * word_count>=7 / >=9 — the near-boundary mass. With p8 = fraction of True
    items exactly 8 words and p7 = fraction of False items exactly 7 words,
    agreement(word_count>=9) = 0.5 + 0.5*(1-p8) and agreement(word_count>=7) =
    0.5 + 0.5*(1-p7); the boundary-heavy schedules keep both under 0.75.
  * char_count>=35/40/45 — defeated by the >= 25% char-discordance quota.
  * contains_the / contains_a / count_the>=2 — the True variant of a concordant
    base is its False core plus article-free adjuncts, so the pair's article
    counts are identical; the discordant pair is article-count-balanced too.
  * last_word_len<=k — both classes draw their final word's length from the same
    distribution (per-base last-word length category), so these sit ~50%.
  * first-word / first-letter / POS predicates — the two variants of a concordant
    base share the SAME opening words; the discordant pair shares a leading
    determiner. So these sit ~50%.

length_match exemption: the spec's length_matching.policy explicitly EXEMPTS
rules 23 and 25 from the |mean_wc(T) - mean_wc(F)| <= 0.2 match (word count IS the
rule, so the class-conditional means must differ by >= 1.0). That single confound
check is the documented per-rule exemption (RuleSpec.length_match_exempt = True);
EVERY other gate (schema, groundtruth, the full 40-predicate battery, and the
confound report's battery half) still passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import (
    Gen,
    GenError,
    base_id as make_base_id,
    to_sentence_case,
)
from ...schema import PROGRAMMATIC_N_BASES_MIN, WORD_COUNT_MAX, WORD_COUNT_MIN, word_count

_NOUN_BANK = "NOUN_CONCRETE"
_VERB_BANK = "VERB_REGULAR"
_ADJ_BANK = "ADJ_PLAIN"
_SHORT_BANK = "SHORT_WORDS_BY_POS"

# Transitive members of VERB_REGULAR — verbs that take a direct-object noun, so
# "<subj> <pastVerb> the <ender>" is grammatical. Past tense is used throughout
# (agreement-safe), so the base form here is inflected by _past_tense.
_TRANSITIVE_VERBS: frozenset[str] = frozenset({
    "open", "close", "clean", "paint", "watch", "help", "start", "finish",
    "climb", "pull", "push", "wash", "cook", "plant", "carry", "study", "copy",
    "move", "use", "order", "offer", "answer", "enter", "visit", "repeat",
    "collect", "protect", "expect", "accept", "prepare", "share", "prove",
    "remove", "solve", "serve", "receive", "imagine", "mention", "explain",
    "provide", "reduce", "approach", "search", "count", "call", "enjoy",
})

# Sentence adverbs that sit between subject and verb ("slowly cleaned"), used to
# build the 7-word core. Article-free.
_PRE_VERB_ADVERBS: tuple[str, ...] = (
    "slowly", "quietly", "quickly", "calmly", "gladly", "carefully", "suddenly",
)

# ARTICLE-FREE optional adjunct slots (1-3 words each), the ONLY thing that
# differs between the True and False variant of a concordant base. None contains
# 'the' or 'a', so padding the True variant longer never changes the item's
# article counts relative to its False partner. Grouped by word count so the
# padder can hit an exact target.
# NOTE: the 1-word pool is kept DISJOINT from _PRE_VERB_ADVERBS (manner adverbs)
# so the 7-word core's pre-verb adverb can never be duplicated by a trailing
# 1-word adjunct ("... carefully removed the painting carefully").
_ADJUNCTS_BY_LEN: dict[int, tuple[str, ...]] = {
    1: (
        "downtown", "outside", "upstairs", "nearby", "overhead", "inside",
        "everywhere", "abroad", "today", "again", "nowadays", "afterwards",
    ),
    2: (
        "on foot", "by hand", "with care", "at home", "at work", "in town",
        "at once", "right away", "for now",
    ),
    3: (
        "without much fuss", "in good time", "with great care", "all day long",
        "for a while",
    ),
}

# Near-boundary count schedules (recipe "near-boundary sampling per PLAN"), with
# the boundary classes a little heavier than the spec's nominal 40% so the two
# threshold-neighbour predicates word_count>=7 / >=9 land at ~0.72 (a margin
# under the 0.75 line) rather than on it.
#   True (>= 8):  8: 55%, 9: 22%, 10: 14%, 11: 9%
#   False (< 8):  7: 55%, 6: 22%,  5: 14%,  4: 9%
_TRUE_COUNT_WEIGHTS: tuple[tuple[int, int], ...] = ((8, 55), (9, 22), (10, 14), (11, 9))
_FALSE_COUNT_WEIGHTS: tuple[tuple[int, int], ...] = ((7, 55), (6, 22), (5, 14), (4, 9))

# Fraction of bases that are CHAR-DISCORDANT (short-word True / long-word False).
# Recipe requires >= 25%; we use ~1/3 for margin.
_DISCORDANT_EVERY = 3  # every 3rd base is discordant -> ~33%

# Build comfortably above the 340-base floor (split needs 100+120+100+>=20=340).
_N_BASES = 372

_MIN_WORDS, _MAX_WORDS = WORD_COUNT_MIN, WORD_COUNT_MAX  # global [4, 14]


@dataclass(frozen=True)
class Base:
    """A word_count_geq_8 base.

    ``base_id`` hashes the base's identity. ALL randomness is resolved at build
    time: the FINAL surface strings of both variants are stored so ``instantiate``
    is a pure lookup and the strings Gate A dedups are exactly the strings
    build_bases validated.

    Concordant base: the False variant is a complete grammatical clause of the
    false target length; the True variant is that SAME clause with article-free
    adjuncts appended to the true target length.

    Discordant base: the True variant is a grammatical SHORT-word sentence (low
    chars), the False variant a grammatical LONG-word clause (high chars) ->
    char-discordant."""

    base_id: str
    kind: str               # "concordant" | "discordant"
    true_target: int        # word count of the True variant (>= 8)
    false_target: int       # word count of the False variant (< 8)
    core: str               # concordant: the shared False clause ("" for discordant)
    true_text: str          # FINAL True-variant surface string
    false_text: str         # FINAL False-variant surface string


def _weighted_schedule(weights: tuple[tuple[int, int], ...], n: int) -> list[int]:
    """A length-``n`` list of counts realising ``weights`` (count -> percent) as
    closely as integer rounding allows. Deterministic (no RNG)."""
    total = sum(w for _, w in weights)
    out: list[int] = []
    for count, w in weights:
        out.extend([count] * round(n * w / total))
    boundary = weights[0][0]
    while len(out) < n:
        out.append(boundary)
    return out[:n]


def _an(word: str) -> str:
    """'a' vs 'an' for the FOLLOWING word (vowel-letter heuristic). The emit
    pipeline re-normalises this label-neutrally; choosing it here keeps the
    strings build_bases validates grammatical on their own."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _past_tense(verb: str) -> str:
    """Regular past tense ("clean"->"cleaned", "carry"->"carried",
    "share"->"shared"). Every VERB_REGULAR / SHORT verb used here inflects
    regularly (the short-word verbs are already past tense and bypass this)."""
    if verb.endswith("e"):
        return verb + "d"
    if verb.endswith("y") and len(verb) >= 2 and verb[-2] not in "aeiou":
        return verb[:-1] + "ied"
    return verb + "ed"


def _plural(noun: str) -> str:
    """Regular plural for the 4-word core's bare-plural object ("gate"->"gates",
    "box"->"boxes", "city"->"cities")."""
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and len(noun) >= 2 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def _concordant_core(
    length: int,
    subj_adj: str,
    subj_noun: str,
    past_verb: str,
    ender: str,
    gen: Gen,
) -> str:
    """A grammatical past-tense transitive clause of EXACTLY ``length`` words
    (4..7), ending on the object ``ender``.

      4: <det> <noun> <verb> <plural-ender>          "The worker cleaned gates"
      5: <det> <noun> <verb> the <ender>             "The worker cleaned the gate"
      6: <det> <adj> <noun> <verb> the <ender>       "The old worker cleaned the gate"
      7: <det> <adj> <noun> <adv> <verb> the <ender> "The old worker slowly cleaned the gate"
    """
    det = "The"
    if length == 4:
        return " ".join([det, subj_noun, past_verb, _plural(ender)])
    if length == 5:
        return " ".join([det, subj_noun, past_verb, "the", ender])
    if length == 6:
        return " ".join([det, subj_adj, subj_noun, past_verb, "the", ender])
    if length == 7:
        adv = gen.choice(_PRE_VERB_ADVERBS)
        return " ".join([det, subj_adj, subj_noun, adv, past_verb, "the", ender])
    raise GenError(f"word_count_geq_8: unsupported core length {length}")


def _solve_adjuncts(deficit: int, gen: Gen) -> list[str]:
    """Pick DISTINCT article-free adjunct phrases whose word counts sum to exactly
    ``deficit`` (0..7). Deterministic given ``gen``; raises if unreachable (it
    never is: 1-word adjuncts are plentiful)."""
    if deficit < 0:
        raise GenError(f"word_count_geq_8: negative adjunct deficit {deficit}")
    chosen: list[str] = []
    used: set[str] = set()
    remaining = deficit
    guard = 0
    while remaining > 0:
        guard += 1
        if guard > 200:
            raise GenError(f"word_count_geq_8: could not solve adjuncts for {deficit}")
        max_len = min(3, remaining)
        lengths = list(range(max_len, 0, -1))
        gen.shuffle(lengths)
        placed = False
        for n in lengths:
            pool = [p for p in _ADJUNCTS_BY_LEN.get(n, ()) if p not in used]
            if not pool:
                continue
            phrase = gen.choice(pool)
            chosen.append(phrase)
            used.add(phrase)
            remaining -= n
            placed = True
            break
        if not placed:
            raise GenError(f"word_count_geq_8: ran out of adjuncts for {deficit}")
    gen.shuffle(chosen)
    return chosen


# body-length recipes for the short-word True sentence: (core_len, chunk_sizes)
# realising each target (after reserving the 3-word final PP "by the <ender>").
# core 3 = "<lead> <noun> <verb>", core 4 = "<lead> <adj> <noun> <verb>";
# chunks: 2 = "<prep> <noun>", 3 = "<prep> a <noun>", 4 = "<prep> a <adj> <noun>".
_SHORT_RECIPES: dict[int, tuple[int, tuple[int, ...]]] = {
    8: (3, (2,)),
    9: (4, (2,)),
    10: (4, (3,)),
    11: (4, (4,)),
}
# place prepositions for the MIDDLE locative PP; "by" is reserved for the final
# PP ("by the <ender>") so the sentence never repeats "by ... by ...".
_SHORT_PLACE_PREPS: tuple[str, ...] = ("on", "in", "near", "past")
# short nouns that are mass/uncountable, so they are NOT pluralised in the
# bare-plural 2-word PP ("on logs" is fine, "on ices" is not).
_SHORT_MASS_NOUNS: frozenset[str] = frozenset({"sky", "sun", "ice", "oil", "tea"})


def _short_plural(noun: str) -> str:
    """Regular plural for the short (<=3 char) countable nouns used in the
    bare-plural 2-word locative PP ("box"->"boxes", "bus"->"buses",
    "fox"->"foxes", "key"->"keys")."""
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def _short_sentence(
    target: int,
    lead: str,
    ender: str,
    short_nouns: list[str],
    short_verbs: list[str],
    short_adjs: list[str],
    gen: Gen,
) -> list[str]:
    """A ``target``-word (8..11) GRAMMATICAL past-tense sentence whose content
    words are all <= 3 chars (low char count), ending on ``ender`` via a final PP.

    Pattern: '<lead> [<adj>] <noun> <pastverb> [<PP> ...] by the <ender>'. Past
    tense (ran/sat/ate/...) avoids any agreement question. The middle PPs use 'a'
    (never a second 'the'), so the short-word True carries the SAME 'the' budget
    as the long-word False: the leading determiner plus the one 'the' in the final
    PP."""
    core_len, chunk_sizes = _SHORT_RECIPES[target]
    if core_len == 4:
        toks: list[str] = [lead, gen.choice(short_adjs), gen.choice(short_nouns), gen.choice(short_verbs)]
    else:
        toks = [lead, gen.choice(short_nouns), gen.choice(short_verbs)]
    # countable short nouns only (for the bare-plural 2-word PP "on logs");
    # mass nouns (sky/sun/ice/oil/tea) are excluded there.
    countable = [n for n in short_nouns if n not in _SHORT_MASS_NOUNS]
    for size in chunk_sizes:
        if size == 2:
            # "<prep> <plural-noun>" — grammatical bare-plural locative ("by cars")
            toks += [gen.choice(_SHORT_PLACE_PREPS), _short_plural(gen.choice(countable))]
        elif size == 3:
            toks += [gen.choice(_SHORT_PLACE_PREPS), "a", gen.choice(countable)]
        else:  # 4
            toks += [gen.choice(_SHORT_PLACE_PREPS), "a", gen.choice(short_adjs), gen.choice(countable)]
    toks += ["by", "the", ender]
    return toks


def _long_sentence(
    target: int,
    lead: str,
    ender: str,
    long_adjs: list[str],
    long_nouns: list[str],
    long_verbs: list[str],
    gen: Gen,
) -> list[str]:
    """A ``target``-word (4..7) GRAMMATICAL past-tense clause built from the
    LONGEST entries of the rule's own banks (high char count, few words), ending
    on ``ender`` as the direct object.

    Pattern: '<lead> [<longadj> ...] <longnoun> <longpastverb> the <ender>'. Only
    TRANSITIVE long verbs are used (so "<noun> <verb> the <ender>" is
    grammatical). One 'the' (the object's) plus the leading determiner — matching
    the short-word True's 'the' budget."""
    verb = _past_tense(gen.choice(long_verbs))
    head = gen.choice(long_nouns)
    if target == 4:
        # no room for "the <ender>" (that needs 5); use a bare-plural object so
        # the clause is grammatical and still ends on the long ender noun.
        return [lead, head, verb, _plural(ender)]
    # target >= 5: lead (1) + head (1) + verb (1) + "the <ender>" (2) = 5;
    # any remaining words are long adjectives stacked before the head noun.
    n_adj = target - 5
    toks = [lead]
    for _ in range(n_adj):
        toks.append(gen.choice(long_adjs))
    toks += [head, verb, "the", ender]
    return toks[:target]


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct word_count_geq_8 bases (the GENERATOR INTERFACE).

    Deterministic given ``gen``. ~2/3 concordant bases (a grammatical False clause
    + article-free adjuncts for the True variant) and ~1/3 char-discordant bases
    (grammatical short-word True vs grammatical long-word False). Per-base targets
    come from the near-boundary schedules; both variants must land in [4, 14]
    words with globally distinct surface strings."""
    nouns = banks.get_bank(_NOUN_BANK).words()
    verbs = banks.get_bank(_VERB_BANK).words()
    adjs = banks.get_bank(_ADJ_BANK).words()
    transitive = [v for v in verbs if v in _TRANSITIVE_VERBS]
    if not transitive:
        raise ValueError("word_count_geq_8: no transitive verbs in VERB_REGULAR")

    short = banks.get_bank(_SHORT_BANK)
    short_nouns = [e.word for e in short.entries if e.pos == "noun"]
    short_verbs = [e.word for e in short.entries if e.pos == "verb"]
    short_adjs = [e.word for e in short.entries if e.pos == "adjective"]
    if not (short_nouns and short_verbs and short_adjs):
        raise ValueError("word_count_geq_8: SHORT_WORDS_BY_POS missing a POS class")

    # longest entries of the rule's OWN banks for the discordant long-word side.
    # The verb pool is restricted to TRANSITIVE verbs so "<noun> <verb> the
    # <ender>" is grammatical.
    long_nouns = sorted(nouns, key=len, reverse=True)[:40]
    long_verbs = sorted(transitive, key=len, reverse=True)[:30]
    long_adjs = sorted(adjs, key=len, reverse=True)[:40]

    # END-WORD pools: one SHORT pool (<= 4 letters) and one LONG pool (>= 7
    # letters) from NOUN_CONCRETE. The per-base ender-length CATEGORY is chosen
    # ~50/50 and applied to BOTH variants, so the last word's length distribution
    # is the same in both classes and the last_word_len<=k predicates sit ~50%.
    end_short = sorted({w for w in nouns if len(w) <= 4})
    end_long = sorted({w for w in nouns if len(w) >= 7})
    if not end_short or not end_long:
        raise ValueError("word_count_geq_8: NOUN_CONCRETE lacks short/long enders")

    true_targets = _weighted_schedule(_TRUE_COUNT_WEIGHTS, _N_BASES)
    false_targets = _weighted_schedule(_FALSE_COUNT_WEIGHTS, _N_BASES)
    gen.shuffle(true_targets)
    gen.shuffle(false_targets)

    g_conc = gen.derive("conc")
    g_short = gen.derive("short")
    g_long = gen.derive("long")

    bases: list[Base] = []
    seen_surface: set[str] = set()
    seen_base: set[str] = set()
    attempts = 0
    i = 0
    max_attempts = _N_BASES * 60
    while len(bases) < _N_BASES and attempts < max_attempts:
        attempts += 1
        tt = true_targets[i % _N_BASES]
        ft = false_targets[i % _N_BASES]
        discordant = (i % _DISCORDANT_EVERY == 0)
        i += 1

        # per-base ender-length category, applied to BOTH variants (concordant
        # pairing; the discordant True caps to a short ender so it stays low-char).
        ender_short = (i % 2 == 0)

        if discordant:
            # shared leading determiner for BOTH variants -> first-word/first-
            # letter battery predicates do not track the class.
            lead = g_short.choice(("The", "A"))
            # BOTH variants end on an ender from the SAME per-base length category
            # (ender_short), so the last word's length is NOT correlated with the
            # class — the char-discordance comes from the BODY words (short-word
            # True vs long-word False), not from the ender. This keeps the
            # last_word_len<=k predicates near 0.5 instead of letting the discordant
            # pair drive them upward.
            ender_pool = end_short if ender_short else end_long
            true_ender = g_short.choice(ender_pool)
            false_ender = g_long.choice(ender_pool)
            true_toks = _short_sentence(tt, lead, true_ender, short_nouns, short_verbs, short_adjs, g_short)
            false_toks = _long_sentence(ft, lead, false_ender, long_adjs, long_nouns, long_verbs, g_long)
            true_text = to_sentence_case(" ".join(true_toks))
            false_text = to_sentence_case(" ".join(false_toks))
            bid = make_base_id("wcg8", "disc", " ".join(true_toks), " ".join(false_toks))
            core_repr = ""
        else:
            ender = g_conc.choice(end_short if ender_short else end_long)
            subj_adj = g_conc.choice(adjs)
            subj_noun = g_conc.choice(nouns)
            past_verb = _past_tense(g_conc.choice(transitive))
            # False variant IS the core, built to exactly the false target.
            core = _concordant_core(ft, subj_adj, subj_noun, past_verb, ender, g_conc)
            false_text = to_sentence_case(core)
            # True variant = core + article-free adjuncts to the true target.
            adjuncts = _solve_adjuncts(tt - ft, g_conc)
            true_text = to_sentence_case(" ".join([core, *adjuncts]))
            bid = make_base_id("wcg8", "conc", core, str(tt))
            core_repr = core

        if bid in seen_base:
            continue
        wt, wf = word_count(true_text), word_count(false_text)
        if not (_MIN_WORDS <= wf <= _MAX_WORDS and _MIN_WORDS <= wt <= _MAX_WORDS):
            continue
        if wt < 8 or wf >= 8:
            continue  # ground-truth guard: True >= 8, False < 8
        if true_text in seen_surface or false_text in seen_surface:
            continue
        seen_surface.add(true_text)
        seen_surface.add(false_text)
        seen_base.add(bid)
        bases.append(
            Base(
                base_id=bid,
                kind="discordant" if discordant else "concordant",
                true_target=wt,
                false_target=wf,
                core=core_repr,
                true_text=true_text,
                false_text=false_text,
            )
        )

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"word_count_geq_8: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (the GENERATOR INTERFACE).

    True variant -> a grammatical sentence with >= 8 words (rule labels True);
    False variant -> a grammatical sentence with < 8 words (rule labels False).
    All randomness was resolved at build time, so this is a pure lookup of the
    stored final surface string; ``gen`` is unused (kept to match the interface)."""
    if label:
        text = spec.true_text
        transform = "short_words" if spec.kind == "discordant" else "core+adjuncts"
        target = spec.true_target
    else:
        text = spec.false_text
        transform = "long_words" if spec.kind == "discordant" else "core+adjuncts"
        target = spec.false_target
    meta = {
        "kind": spec.kind,
        "core": spec.core,
        "target_word_count": target,
        "transform": transform,
    }
    return text, meta
