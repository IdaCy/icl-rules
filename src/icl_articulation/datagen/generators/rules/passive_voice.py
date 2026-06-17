"""Rule 9: passive_voice (plan rule 9, category syntactic).

Canonical articulation: True iff the main clause is in the PASSIVE voice (the
subject is acted upon), False iff it is ACTIVE. On the emitted data the
distinguishing surface shape (and the ground-truth checker, groundtruth._r9)
is ``was/were`` IMMEDIATELY followed by an ``-ed``/``-en`` past participle (an
``-ing`` gerund after ``was`` is a progressive ACTIVE, not passive).

GENERATION (rule-specs ``id: passive_voice`` recipe), built over the pinned
banks ``[VERB_REGULAR, NOUN_CONCRETE, ADVERB_PLACE]``:

    base_id = (agent, V, patient) triple.

    Per base a seeded ``use_by`` flag (50/50) drives the 'by' token in BOTH
    variants in lockstep so 'by' cannot separate the classes (see the 'contains
    by' guard below):

    FALSE variant (active):
        use_by      'The {agent} {V-past} the {patient} by hand'   (carries 'by')
        else simple-past  'The {agent} {V-past} the {patient} {adjunct}'
        else progressive  'The {agent} was {V-ing} the {patient} {adjunct}'  (50%)

    TRUE variant (passive):
        use_by      'The {patient} was {V-ed} by the {agent}'      (carries 'by')
        else agentless    'The {patient} was {V-ed} {adjunct}'     (NON-'by' phrase)

GRAMMATICALITY (the fix this module exists for — a hand review found
ungrammatical items in the prior build):

  (1) TRANSITIVITY. The frames put the verb in a clause that takes a DIRECT
      OBJECT — the passive 'The {patient} was {V-ed} (by the {agent})' and the
      active 'The {agent} {V-ed}/{was V-ing} the {patient}'. Only a TRANSITIVE
      verb (one that takes a direct object and so can passivize) is
      grammatical there. The full VERB_REGULAR bank mixes transitives with
      INTRANSITIVES (smile, arrive, stay, walk, talk, jump, ...) that CANNOT
      take an object or passivize: 'The store was smiled ...' / 'The leaf
      smiled the store' are ungrammatical. This generator therefore draws verbs
      ONLY from the curated ``_TRANSITIVE_VERBS`` subset of VERB_REGULAR below
      (every member takes a direct object and passivizes cleanly), so every
      emitted active and passive is grammatical. The subset is physical
      manipulation / handling / affect verbs, which also read naturally with
      the (inanimate) NOUN_CONCRETE agent and patient the pinned banks force.

  (2) ADJUNCTS — AT MOST ONE PLACE PHRASE. Word-count equalization appends AT
      MOST ONE ADVERB_PLACE adjunct per item (never a stack like
      '... in silence at school abroad'). This is achievable because the
      equalization target is set to _TARGET_WC = 7, the unique word count that
      every core shape reaches with a SINGLE 0-3-word adjunct AND lands at the
      same 'the' count (see _TARGET_WC + the the-count plan below):

          agentless passive  'The {p} was {V-ed}'              4 words, 1 'the'
            -> + one 3-word 'has-the' phrase ('on the table')  -> 7 words, 2 'the'
          by-agent passive   'The {p} was {V-ed} by the {a}'   7 words, 2 'the'
            -> + NO adjunct (already at target)                -> 7 words, 2 'the'
          active simple-past 'The {a} {V-ed} the {p}'          5 words, 2 'the'
            -> + one 2-word 'no-the' phrase ('at home')        -> 7 words, 2 'the'
          active progressive 'The {a} was {V-ing} the {p}'     6 words, 2 'the'
            -> + one 1-word 'no-the' phrase ('downtown')       -> 7 words, 2 'the'

      Because the deficit is filled by exactly one phrase (or zero, for the
      by-agent passive that is already at length), NO sentence ever stacks two
      or three place adjuncts. The single appended phrase is always DISTINCT
      from any other token; there is nothing to repeat.

WORD COUNT + the-count balancing (Gate C / Gate D). EVERY emitted item is
exactly _TARGET_WC (= 7) words and carries exactly TWO 'the' tokens, in BOTH
classes. So mean_wc(T) == mean_wc(F) exactly (Gate D length-match is 0) and
'count_the>=2' / 'contains_the' sit at exactly 100% in both classes
(non-separating, well under the 0.75 Gate C threshold). The single adjunct that
fills a core is chosen from the partition that supplies the needed word count
AND the needed 'the's at once:
  * agentless passive (1 'the', needs +1 'the', +3 words) -> a 3-word HAS-THE
    phrase ('on the table'), the only adjunct that carries a 'the';
  * the other three shapes already have 2 'the', so they take a NO-THE phrase
    (1- or 2-word) or no phrase.

CONFOUND GUARDS the recipe pins, and how this generator satisfies them against
the FROZEN battery (Gate C, max(agree, 1-agree) <= 0.75 on every non-exempt
predicate):

  * 'contains was/were'  — DEFUSED: every progressive active (50% of actives)
    also carries 'was', so 'was' is present in 100% of passives and ~50% of
    actives. The battery has no bare 'contains was/were' predicate; the closest
    scored cues (POS/length/the-count) are all balanced below threshold.
  * 'contains by'        — BALANCED by the per-base ``use_by`` flag (seeded
    50/50). When use_by is set, BOTH variants carry 'by': the TRUE variant is the
    by-agent passive ('... by the {agent}') and the FALSE variant is the active
    SIMPLE-PAST + 'by hand' (a 2-word no-'the' adjunct that keeps the item at 7
    words / 2 'the'). When use_by is unset, NEITHER variant carries 'by' (the
    TRUE variant is the agentless passive with a NON-'by' has-the phrase; the
    FALSE active draws only NON-'by' adjuncts — _partition_adjuncts excludes every
    'by'-prefixed phrase). Because the flag is per-base and drives both variants
    in lockstep, the 'by' rate is identical (~50%) in BOTH classes, so the 'by'
    single-token confound collapses to ~0.50 (a prior build left 'by' at ~0.81 in
    the True class only — passive by-agent — with the active class near 0; that is
    the incidental confound this round removes). There is no 'contains by'
    predicate in the frozen battery, but the stricter incidental single-token
    audit (max(agree,1-agree) <= 0.75 over every token) now passes for 'by'. No
    item ever stacks two place adjuncts: 'by hand' IS the active's single adjunct.
  * count('the') >= 2 / contains_the — DEFUSED by construction: every item in
    BOTH classes carries exactly 2 'the' (the the-count plan), so the predicate
    is constant-True and non-separating (100% in both classes).
  * word counts            — matched exactly (every item is 7 words).
  * 'first noun is a person' / animacy — see ANIMACY NOTE below.
  * last-word cues          — the trailing token of an item is either a NOUN
    (the by-agent passive ends in the {agent} noun; the agentless passive ends
    in the has-the phrase's noun) or a place adverb/noun from the appended
    phrase. The recipe pins no last-word equivalence; the frozen battery's
    last-word predicates ('last_word_len<=k', 'last_ends_vowel/consonant') are
    verified below the 0.75 bar by the emit pipeline's battery on the full
    dataset (the by-agent-passive share is only 0.6*0.5 = 30% of all items, and
    its agent noun is drawn from the same NOUN_CONCRETE pool that supplies the
    patients, so its length/ending distribution overlaps the active class's
    object/adjunct nouns). Recorded in open_concerns.

ANIMACY NOTE (open tension, recorded honestly): the recipe's ANIMACY MIX
(35% animate-agent / 35% animate-patient / 15-15 A-A / I-I, with named examples
'the chef / the meal', 'the storm / the hikers') presupposes a source of
ANIMATE nouns. The pinned bank list for this rule is exactly
``[VERB_REGULAR, NOUN_CONCRETE, ADVERB_PLACE]`` and NOUN_CONCRETE contains only
INANIMATE concrete nouns (table, garden, river, ...): there is no animate noun
to draw from without adding an off-spec bank. The mix's stated PURPOSE —
"put first-noun animacy at exactly 50/50 in BOTH classes" so 'first noun is a
person' cannot separate the classes — is achieved a fortiori here: first-noun
animacy is 0% in BOTH classes (no animate nouns exist on this data), so the
'first noun is a person' cue is constant-False and non-separating. The frozen
battery has no animacy predicate, so this does not threaten Gate C. The literal
animate/inanimate share mix is therefore NOT built (it is unsatisfiable with the
pinned banks); the confound it targets is instead neutralised by uniform
inanimacy. Choosing the TRANSITIVE verb subset above keeps every item
grammatical despite both arguments being inanimate (the verbs take physical
objects, and an inanimate subject of a physical-action verb is syntactically
well-formed — 'The crane lifted the beam' / 'The basket carried the apples').
This is recorded in open_concerns.

This module conforms to the GENERATOR INTERFACE in
``icl_articulation.datagen.generators.base`` (build_bases / instantiate) and is
dispatched via the registry; the shared gated pipeline enforces all four gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, GenError, adjunct_word_lengths
from ...schema import PROGRAMMATIC_N_BASES_MIN, WORD_COUNT_MAX, word_count

# pinned banks (rule-specs passive_voice generation.banks)
_VERB_BANK = "VERB_REGULAR"
_NOUN_BANK = "NOUN_CONCRETE"
_ADJUNCT_BANK = "ADVERB_PLACE"

# --- TRANSITIVITY: curated transitive subset of VERB_REGULAR ------------------
#
# Every verb here takes a DIRECT OBJECT and passivizes cleanly, so all four
# frames ('The {p} was {V-ed} (by the {a})' and 'The {a} {V-ed}/was {V-ing} the
# {p}') are GRAMMATICAL. The set is physical manipulation / handling / affect
# verbs, which also read naturally with the inanimate NOUN_CONCRETE agent and
# patient the pinned banks force (e.g. 'The basket carried the apples', 'The
# wall covered the window' — well-formed even with inanimate subjects).
#
# EXCLUDED from VERB_REGULAR (intransitive or object-less in these frames, so
# they would make the passive/active ungrammatical): walk, talk, climb, reply,
# dry, play, stay, jump, live, wave, smile, arrive, return, decide, plus the
# communication/mental/benefactive verbs that demand an ANIMATE agent and so
# read as ungrammatical with the inanimate agent here (answer, explain,
# imagine, expect, accept, believe, offer, share, serve, provide, receive,
# enjoy, call, visit, search, help, protect-of-people, repeat-speech, solve,
# reduce, approach, enter, order, mention). We keep ONLY the clearly transitive
# physical-action core, which is the conservative grammatical choice.
_TRANSITIVE_VERBS: tuple[str, ...] = (
    "open",
    "close",
    "clean",
    "paint",
    "watch",
    "start",
    "finish",
    "pull",
    "push",
    "wash",
    "cook",
    "plant",
    "carry",
    "study",
    "copy",
    "count",
    "move",
    "use",
    "collect",
    "protect",
    "prepare",
    "remove",
)

# fixed seed-independent target word count for BOTH variants of every base, and
# the UNIQUE value that lets every core shape reach the target with AT MOST ONE
# 0-3-word ADVERB_PLACE adjunct (no stacking) while also landing at exactly 2
# 'the' tokens. Cores: agentless-passive = 4 words / 1 'the', active-simple =
# 5 / 2, active-progressive = 6 / 2, by-agent-passive = 7 / 2. At target 7 the
# single-adjunct deficits are 3 / 2 / 1 / 0 words respectively, and the only
# shape short a 'the' (agentless passive) needs +3 words -> a single 3-word
# 'has-the' phrase supplies BOTH the missing word count and the missing 'the'.
# 7 is well inside the global [4, 14] word window.
_TARGET_WC = 7

# build well past the 340-base floor so the by-base split (100+120+100+>=20) has
# headroom after the distinct-(agent,V,patient) filter.
_N_BASES = 380

# recipe distribution knobs (per-base, seeded at build time so instantiate is
# deterministic).
#
# 'by' TOKEN BALANCE (incidental-confound fix this round): a per-base ``use_by``
# flag drives the 'by' token in BOTH variants in lockstep so 'by' cannot separate
# the classes (single-token agreement ~0.5 for ANY use_by share):
#   use_by  -> TRUE = by-agent passive ('... by the {agent}', carries 'by')
#              FALSE = active SIMPLE-PAST + 'by hand' adjunct      (carries 'by')
#   not     -> TRUE = AGENTLESS passive + a NON-'by' has-the phrase (no 'by')
#              FALSE = active (progressive/simple-past) + non-'by' adjunct (no 'by')
# 'by' True rate == 'by' False rate == _USE_BY_SHARE -> 'by' agreement is exactly
# 0.5 regardless of the share.
#
# 'was' TOKEN BALANCE (coupled to use_by): EVERY passive carries 'was' ('was
# V-ed'), so 'was' is 100% in the True class; to hold the 'was' single token <=
# 0.75 the ACTIVE class must carry 'was' in >= 50% of items (was_score =
# (480 - 240*was_F_rate)/480 <= 0.75  <=>  was_F_rate >= 0.5). Only the
# progressive active ('was V-ing') carries 'was', and a use_by active is forced
# to simple-past (no 'was'). So the non-use_by actives must be almost all
# progressive to reach the 50% floor:
#     was_F_rate = (1 - _USE_BY_SHARE) * _PROGRESSIVE_ACTIVE_SHARE
# With _USE_BY_SHARE = 0.40 and _PROGRESSIVE_ACTIVE_SHARE = 0.95 -> was_F_rate
# ~ 0.57 -> 'was' single token ~ 0.715 (and 'by' ~ 0.50), both inside the gate.
_USE_BY_SHARE = 0.40
_PROGRESSIVE_ACTIVE_SHARE = 0.95   # of NON-use_by actives use past-progressive

# the single 2-word no-'the' agent adjunct that injects 'by' into an active
# without a 'the' (keeps the active's 2-'the' count). 'by hand' reads as a
# grammatical manner adjunct on a transitive active ('The crane lifted the beam
# by hand'). Must exist in ADVERB_PLACE (asserted in build_bases).
_BY_HAND = "by hand"

# every item carries exactly this many 'the' tokens in BOTH classes.
_THE_TARGET = 2


@dataclass(frozen=True)
class Base:
    """A passive_voice base: the (agent, V, patient) triple + the per-base seeded
    shape choices, so BOTH variants instantiate deterministically and equalize
    to the same word count and the-count."""

    base_id: str
    agent: str
    patient: str
    verb_base: str
    verb_past: str   # == past participle for regular verbs (-ed)
    verb_ing: str
    # per-base shape choices (seeded at build time)
    active_progressive: bool   # False variant uses past-progressive (else simple past)
    use_by: bool               # both variants carry 'by' (passive by-agent + active 'by hand')


# --- ADVERB_PLACE adjunct partitions (has 'the' vs no 'the') ------------------


def _partition_adjuncts(gen: Gen) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Split ADVERB_PLACE phrases into (no-'the', has-'the') pools, each grouped
    by word length, EXCLUDING any 'by'-prefixed phrase. The 3-word 'prep the
    noun' phrases ('on the table') carry a 'the'; the 1-/2-word phrases
    ('downtown', 'at home') do not. The single appended adjunct is drawn from the
    pool that supplies the needed word count AND the needed 'the' count at once.

    'by'-prefixed phrases ('by hand', 'by the door') are EXCLUDED here so the
    no_by branches never accidentally inject a 'by'; the use_by branches inject
    'by' explicitly (the active via ``_BY_HAND``, the passive via 'by the
    {agent}'). This keeps the 'by' token fully under the per-base ``use_by``
    control that balances it across classes."""
    entries = banks.get_bank(_ADJUNCT_BANK).words()
    no_the: list[str] = []
    has_the: list[str] = []
    for ph in entries:
        toks = ph.lower().split()
        if toks and toks[0] == "by":
            continue  # 'by'-phrases are injected explicitly by the use_by branches
        (has_the if "the" in toks else no_the).append(ph)
    # deterministic order independent of bank insertion order
    no_the.sort()
    has_the.sort()
    return adjunct_word_lengths(no_the), adjunct_word_lengths(has_the)


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct (agent, V, patient) bases (GENERATOR INTERFACE).

    Enumerate distinct (agent, verb, patient) triples over NOUN_CONCRETE x
    _TRANSITIVE_VERBS x NOUN_CONCRETE (agent != patient), seeded-shuffle, take
    the first _N_BASES, and attach the per-base seeded shape choices.
    Deterministic given ``gen``."""
    nouns = banks.get_bank(_NOUN_BANK).words()
    verb_words = set(banks.get_bank(_VERB_BANK).words())

    # GRAMMATICALITY guard: every curated transitive verb must exist in the bank
    # (a bank edit that drops one must fail loudly here, not silently shrink the
    # verb space or fall back to an intransitive).
    missing = [v for v in _TRANSITIVE_VERBS if v not in verb_words]
    if missing:
        raise GenError(
            f"passive_voice: curated transitive verbs not in {_VERB_BANK}: {missing}"
        )

    # 'by' BALANCE guard: the active-side 'by' injector ('by hand') must exist in
    # ADVERB_PLACE (it is the 2-word no-'the' phrase that adds 'by' to an active
    # without disturbing the 2-'the' count); fail loud if a bank edit removed it.
    adjunct_words = set(banks.get_bank(_ADJUNCT_BANK).words())
    if _BY_HAND not in adjunct_words:
        raise GenError(
            f"passive_voice: {_ADJUNCT_BANK} lacks {_BY_HAND!r} (needed to balance "
            "the 'by' token across the active class)"
        )

    # distinct triples (agent != patient). The space is huge
    # (22 verbs * 131 * 130 ~ 375k), so build the candidate list then shuffle.
    triples: list[tuple[str, str, str]] = []
    for v in _TRANSITIVE_VERBS:
        for a in nouns:
            for p in nouns:
                if a == p:
                    continue
                triples.append((a, v, p))
    gen.shuffle(triples)

    # seeded per-base choice streams (independent of draw order via derive)
    choice_gen = gen.derive("shape_choices")

    bases: list[Base] = []
    seen_ids: set[str] = set()
    for agent, vbase, patient in triples:
        bid = f"pv:{agent}|{vbase}|{patient}"
        if bid in seen_ids:
            continue
        _, _, vpast, ving = banks._regular_verb_forms(vbase)

        # per-base seeded shape choices
        active_progressive = choice_gen.randint(1, 100) <= int(_PROGRESSIVE_ACTIVE_SHARE * 100)
        # use_by drives 'by' in BOTH variants in lockstep (balance, see _USE_BY_SHARE)
        use_by = choice_gen.randint(1, 100) <= int(_USE_BY_SHARE * 100)
        # a use_by base's FALSE variant must be simple-past (only it can carry the
        # 'by hand' 2-word adjunct; progressive's deficit is 1 word). Force it.
        if use_by:
            active_progressive = False

        bases.append(
            Base(
                base_id=bid,
                agent=agent,
                patient=patient,
                verb_base=vbase,
                verb_past=vpast,
                verb_ing=ving,
                active_progressive=active_progressive,
                use_by=use_by,
            )
        )
        seen_ids.add(bid)
        if len(bases) >= _N_BASES:
            break

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise GenError(
            f"passive_voice: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def _the_count(text: str) -> int:
    return sum(1 for t in text.lower().split() if t.strip(".,") == "the")


def _equalize_one_adjunct(
    core: str,
    *,
    no_the: dict[int, list[str]],
    has_the: dict[int, list[str]],
    gen: Gen,
) -> str:
    """Append AT MOST ONE ADVERB_PLACE adjunct so ``core`` reaches EXACTLY
    _TARGET_WC words AND exactly _THE_TARGET 'the' tokens — NEVER a stack.

    The single adjunct must supply the whole word deficit (_TARGET_WC - words)
    AND the whole 'the' deficit (_THE_TARGET - core_the) at once:
      * 'the' deficit 1  (only the agentless passive) -> a HAS-THE phrase, which
        is always 3 words and adds exactly one 'the'. The word deficit must
        therefore be exactly 3 (guaranteed by _TARGET_WC = 7 for that core).
      * 'the' deficit 0  -> a NO-THE phrase of the exact word deficit (1 or 2),
        or NO phrase when the deficit is 0 (the by-agent passive, already at
        length).
    Raises (loud) if a single adjunct cannot satisfy both deficits exactly —
    that would mean a core/target combination that needs stacking, which the
    _TARGET_WC choice is designed to make impossible."""
    have = word_count(core)
    word_deficit = _TARGET_WC - have
    the_deficit = _THE_TARGET - _the_count(core)

    if word_deficit < 0:
        raise GenError(f"core already over target {_TARGET_WC}: {core!r}")
    if the_deficit < 0:
        raise GenError(f"core already over the-target {_THE_TARGET}: {core!r}")

    if the_deficit == 0:
        if word_deficit == 0:
            text = core  # by-agent passive: at length and at the-count, no adjunct
        else:
            pool = no_the.get(word_deficit)
            if not pool:
                raise GenError(
                    f"no {word_deficit}-word 'no-the' adjunct for core {core!r}"
                )
            text = f"{core} {gen.choice(pool)}"
    elif the_deficit == 1:
        # only a 3-word HAS-THE phrase carries a 'the'; the word deficit must
        # match (3) so a SINGLE phrase fixes both — no stacking.
        if word_deficit != 3:
            raise GenError(
                f"the-deficit 1 needs a 3-word has-the phrase but word deficit is "
                f"{word_deficit} for core {core!r} (would force stacking)"
            )
        pool = has_the.get(3)
        if not pool:
            raise GenError("no 3-word 'has-the' adjunct available")
        text = f"{core} {gen.choice(pool)}"
    else:
        raise GenError(f"unexpected the-deficit {the_deficit} for core {core!r}")

    if word_count(text) != _TARGET_WC:
        raise GenError(
            f"equalizer produced {word_count(text)} words, expected {_TARGET_WC}: "
            f"{text!r}"
        )
    if _the_count(text) != _THE_TARGET:
        raise GenError(
            f"equalizer produced {_the_count(text)} 'the', expected {_THE_TARGET}: "
            f"{text!r}"
        )
    return text


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (GENERATOR INTERFACE).

    label True  -> PASSIVE (by-agent or agentless 'was {V-ed}'), ground-truth True.
    label False -> ACTIVE  (simple-past or past-progressive),    ground-truth False.

    Both variants are padded to _TARGET_WC words and _THE_TARGET 'the' tokens via
    AT MOST ONE ADVERB_PLACE adjunct, so the two variants are word-count- and
    the-count-identical and NO sentence stacks place adjuncts."""
    no_the, has_the = _partition_adjuncts(gen)

    agent, patient = spec.agent, spec.patient
    vpast, ving = spec.verb_past, spec.verb_ing

    if label:
        # PASSIVE (transitive verb -> grammatical passive)
        if spec.use_by:
            # by-agent passive already at 7 words / 2 'the' and carries 'by'.
            core = f"The {patient} was {vpast} by the {agent}"
            text = core
            shape = "passive_by_agent"
        else:
            # agentless passive (4w/1 'the') + a NON-'by' 3-word has-the phrase.
            core = f"The {patient} was {vpast}"
            text = _equalize_one_adjunct(core, no_the=no_the, has_the=has_the, gen=gen)
            shape = "passive_agentless"
    else:
        # ACTIVE (transitive verb -> grammatical active with a direct object)
        if spec.use_by:
            # active SIMPLE-PAST (5w/2 'the') + 'by hand' (2w no-'the') -> 7w/2
            # 'the', injecting 'by' so the active class matches the passive class's
            # 'by' rate. build_bases forces simple-past whenever use_by is set.
            core = f"The {agent} {vpast} the {patient}"
            text = f"{core} {_BY_HAND}"
            shape = "active_simple_past_by"
        elif spec.active_progressive:
            core = f"The {agent} was {ving} the {patient}"
            text = _equalize_one_adjunct(core, no_the=no_the, has_the=has_the, gen=gen)
            shape = "active_progressive"
        else:
            core = f"The {agent} {vpast} the {patient}"
            text = _equalize_one_adjunct(core, no_the=no_the, has_the=has_the, gen=gen)
            shape = "active_simple_past"

    if word_count(text) != _TARGET_WC:
        raise GenError(f"passive_voice item not {_TARGET_WC} words: {text!r}")
    if _the_count(text) != _THE_TARGET:
        raise GenError(f"passive_voice item not {_THE_TARGET} 'the': {text!r}")
    if word_count(text) > WORD_COUNT_MAX:
        raise GenError(f"passive_voice item over word cap: {text!r}")

    meta = {
        "shape": shape,
        "agent": agent,
        "patient": patient,
        "verb_base": spec.verb_base,
        "verb_form": vpast if label else (ving if spec.active_progressive else vpast),
        "use_by": spec.use_by,
        "target_wc": _TARGET_WC,
        "the_target": _THE_TARGET,
        "core": core,
    }
    return text, meta
