"""Per-rule configuration for the shared llm_validated pipeline (rules 15, 16).

One ``LLMRuleConfig`` per rule carries everything the rule-agnostic pipeline needs:

  - rule_id and the two ground-truth label classes (True/False) with the human
    names the validators use (e.g. "positive"/"negative", "food"/"not food").
  - the GENERATION prompt builder: gpt-4.1-mini is asked for matched pairs of
    sentences per topic, under the shared style constraints
    (globals.sentence_style: 4-12 words, sentence case, NO exclamation / emoji /
    contraction / quote / terminal punctuation).
  - the TOPIC structure: rule 15 = >= 8 polarity topics; rule 16 = >= 6 food
    subdomains (True) and >= 6 non-food topics (False, each 10-25% of False).
  - the two VALIDATOR prompts: pass A (gpt-4.1-mini) and pass B (gpt-4.1), each
    DIFFERENTLY worded, each returning a 3-way label (intended-True /
    intended-False / neutral-or-unclear). The pipeline keeps an item only when
    BOTH passes return the intended NON-neutral label.
  - rule-specific generation knobs (rule 16's >= 55% no-eat/drink/taste/cook-verb
    quota for True items, so the keyword predicate sits ~72.5%).

The style/audit constants are transcribed from the spec globals + each rule's
distribution_guards.audit_thresholds; the pipeline reads them so the audit is
parameterised, not hand-coded per rule. The private rule-specs.yaml is the source
of these strings (rules 15/16 recipe + audit_thresholds); they are copied here
verbatim where they are load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
# Shared style constraints (rule-specs globals.sentence_style; identical for 15
# and 16). The generation prompt embeds these; the pipeline ALSO enforces them
# programmatically post-hoc (strip terminal '.', drop any item with '!'/'?'/
# emoji/quote/contraction/hyphen), so a non-compliant model output is dropped
# rather than trusted.
# ---------------------------------------------------------------------------
STYLE_RULES: tuple[str, ...] = (
    "between 4 and 12 words long",
    "written in sentence case (only the first word capitalized; no ALL-CAPS)",
    "no exclamation marks and no question marks",
    "no emoji and no non-ASCII characters",
    "no contractions (write 'do not', never 'don't')",
    "no quotation marks of any kind",
    "no terminal punctuation (do not end the sentence with a period)",
    "no hyphenated words and no semicolons or colons",
    "never use the word 'I'",
)

# words whose presence drops a candidate during the programmatic style filter
# (cheap pre-validation gate; mirrors globals.sentence_style bans). Apostrophes
# and quotes are caught by char scan; this set catches stray emoji is handled by
# the ascii check.
_BANNED_CHARS = "!?\"';:()[]{}…–—-"


@dataclass(frozen=True)
class LLMRuleConfig:
    """Everything the rule-agnostic pipeline needs for one llm_validated rule."""

    rule_id: str
    # human label names used in prompts / provenance (true_name, false_name)
    true_name: str
    false_name: str
    # the third, "drop" option each validator offers (neutral / unclear)
    neutral_name: str
    # topic structure
    true_topics: tuple[str, ...]
    false_topics: tuple[str, ...]
    # generation: how many matched pairs to request per generation call per topic
    pairs_per_call: int
    # a one-line task description embedded in the generation prompt
    generation_task: str
    # builders for the two validator prompts (pass A = mini, pass B = 4.1)
    validator_a_instruction: str
    validator_b_instruction: str
    # extra per-rule generation directive (e.g. rule 16's no-verb quota text)
    extra_generation_directive: str = ""
    # rule-specific candidate-level constraint checker hooks (applied as advisory
    # tags in slots_meta; the keyword-quota audit reads them). Maps a tag name to
    # a predicate over text. Used by rule 16's no-eat-verb quota audit.
    candidate_tags: dict[str, Callable[[str], bool]] = field(default_factory=dict)

    # ----- generation prompt -------------------------------------------------
    def generation_messages(self, label: bool, topic: str, n_pairs: int) -> list[dict[str, str]]:
        """Chat messages asking gpt-4.1-mini for ``n_pairs`` sentences of the
        intended class on ``topic``. Returns the (system, user) message list the
        client.complete signature expects. The model is asked for ONE sentence
        per line so the pipeline can split deterministically."""
        intended = self.true_name if label else self.false_name
        style = "\n".join(f"- {s}" for s in STYLE_RULES)
        directive = (
            f"\n{self.extra_generation_directive}" if self.extra_generation_directive else ""
        )
        user = (
            f"{self.generation_task}\n\n"
            f"Write exactly {n_pairs} different sentences that are clearly "
            f"{intended.upper()} about the topic: {topic}.\n\n"
            f"Every sentence MUST follow ALL of these constraints:\n{style}{directive}\n\n"
            f"Output ONLY the sentences, one per line, with no numbering, no "
            f"bullet points, and no extra commentary."
        )
        return [
            {"role": "system", "content": "You write short, plain sentences to exact constraints."},
            {"role": "user", "content": user},
        ]

    # ----- validator prompts -------------------------------------------------
    def validator_messages(self, which: str, text: str) -> list[dict[str, str]]:
        """Chat messages for one validation pass over ``text``.

        ``which`` is 'A' (gpt-4.1-mini) or 'B' (gpt-4.1). The two instructions
        are DIFFERENTLY worded (rule-specs: 'a differently worded prompt') but
        ask for the SAME 3-way classification: the True name, the False name, or
        the neutral/unclear name. The model is told to answer with exactly one of
        the three words so the pipeline can parse a single token."""
        if which == "A":
            instruction = self.validator_a_instruction
        elif which == "B":
            instruction = self.validator_b_instruction
        else:
            raise ValueError(f"validator pass must be 'A' or 'B', got {which!r}")
        options = f"{self.true_name}, {self.false_name}, or {self.neutral_name}"
        user = (
            f"{instruction}\n\n"
            f"Sentence: {text}\n\n"
            f"Answer with exactly one word: {options}."
        )
        return [
            {"role": "system", "content": "You are a careful, literal annotator."},
            {"role": "user", "content": user},
        ]

    def parse_validator_label(self, text: str) -> bool | None:
        """Map a validator completion to True (intended-True), False
        (intended-False), or None (neutral / unclear / unparseable). Case- and
        punctuation-insensitive prefix match against the three option names. A
        completion that matches NEITHER non-neutral name parses as None (drop)."""
        t = text.strip().lower().strip(".,!?:;\"'")
        for name, value in (
            (self.true_name.lower(), True),
            (self.false_name.lower(), False),
            (self.neutral_name.lower(), None),
        ):
            if t.startswith(name):
                return value
        return None


# ---------------------------------------------------------------------------
# Rule 16 generation tag: does the True sentence contain an eat/drink/taste/cook
# verb? The >= 55% no-verb quota (recipe MAJOR-2) keeps 'mentions eating' a
# distractor at ~72.5% agreement, not an equivalent. This is a GENERATION-time
# tag (stamped in slots_meta and used by the quota audit), NOT the ground-truth
# label — the label is the validators' food/not-food call.
# ---------------------------------------------------------------------------
_EAT_VERB_STEMS: tuple[str, ...] = (
    "eat", "eats", "eaten", "ate", "eating",
    "drink", "drinks", "drank", "drunk", "drinking",
    "taste", "tastes", "tasted", "tasting",
    "cook", "cooks", "cooked", "cooking",
    "chew", "chews", "chewed", "chewing",
    "swallow", "swallows", "swallowed", "swallowing",
    "sip", "sips", "sipped", "sipping",
    "dine", "dines", "dined", "dining",
)


def contains_eat_verb(text: str) -> bool:
    """True iff a whitespace token (lowercased, stripped of trailing punctuation)
    is one of the eat/drink/taste/cook verb forms. Word-level so 'cookbook' and
    'cookie' do NOT count (they are not the verb)."""
    toks = [t.strip(".,!?:;\"'()[]").lower() for t in text.split()]
    stems = set(_EAT_VERB_STEMS)
    return any(t in stems for t in toks)


# ===========================================================================
# RULE 15 — positive_sentiment
# ===========================================================================
_R15 = LLMRuleConfig(
    rule_id="positive_sentiment",
    true_name="positive",
    false_name="negative",
    neutral_name="neutral",
    # >= 8 topics (rule-specs recipe lists exactly these)
    true_topics=(
        "restaurants", "movies", "weather", "work",
        "travel", "products", "sports", "music",
    ),
    false_topics=(
        "restaurants", "movies", "weather", "work",
        "travel", "products", "sports", "music",
    ),
    pairs_per_call=8,
    generation_task=(
        "Generate short opinion sentences that express a clear evaluative "
        "stance. Each sentence must be CLEARLY positive or CLEARLY negative — "
        "never neutral, factual, mixed, or sarcastic."
    ),
    validator_a_instruction=(
        "Classify the sentiment expressed by the speaker of this sentence. "
        "Decide whether the speaker is expressing a favourable (positive) "
        "opinion, an unfavourable (negative) opinion, or neither/unclear."
    ),
    validator_b_instruction=(
        "Read the sentence and judge the writer's attitude. Is the overall "
        "evaluation approving and pleased, disapproving and displeased, or is it "
        "purely factual, mixed, or impossible to tell?"
    ),
)

# ===========================================================================
# RULE 16 — food_topic
# ===========================================================================
_R16 = LLMRuleConfig(
    rule_id="food_topic",
    true_name="food",
    false_name="not food",
    neutral_name="unclear",
    # >= 6 food subdomains (True)
    true_topics=(
        "cooking at home", "restaurants", "ingredients",
        "baking", "meals", "tasting and flavour",
    ),
    # >= 6 non-food topics (False); gardening WITHOUT edible plants
    false_topics=(
        "sports", "weather", "transport",
        "work", "music", "gardening flowers",
    ),
    pairs_per_call=8,
    generation_task=(
        "Generate short, sentiment-neutral sentences (no opinion adjectives, no "
        "praise or complaint). Each sentence must be plainly about its assigned "
        "topic."
    ),
    extra_generation_directive=(
        "- do NOT use any eat/drink/taste/cook verb (no 'eat', 'drink', 'taste', "
        "'cook', 'dine', 'sip') in MOST of the food sentences; describe the food, "
        "dish, ingredient, recipe, or kitchen instead"
    ),
    validator_a_instruction=(
        "Decide what this sentence is ABOUT. Is its topic food or cooking "
        "(meals, dishes, ingredients, cooking, eating, restaurants as places to "
        "eat), some unrelated topic, or is it unclear?"
    ),
    validator_b_instruction=(
        "Judge the subject matter of the sentence. Does it concern food, "
        "cuisine, cooking, or eating; does it concern something with nothing to "
        "do with food; or can you not tell?"
    ),
    candidate_tags={"eat_verb": contains_eat_verb},
)


LLM_RULE_CONFIGS: dict[str, LLMRuleConfig] = {
    _R15.rule_id: _R15,
    _R16.rule_id: _R16,
}


def get_rule_config(rule_id: str) -> LLMRuleConfig:
    """Look up the config for an llm_validated rule (LOUD if unknown)."""
    if rule_id not in LLM_RULE_CONFIGS:
        raise KeyError(
            f"no llm_validated config for rule_id {rule_id!r}; "
            f"known: {sorted(LLM_RULE_CONFIGS)}"
        )
    return LLM_RULE_CONFIGS[rule_id]
