"""Prompt templates.

ONE generic step-1 classification template for ALL rules (PLAN: no rule names,
no hints, identical across rules). The template text lives in constants below;
``step1_template_hash()`` is logged into every run config so any edit is
visible in the results.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

STEP1_SYSTEM = "You are a precise classifier."

STEP1_EXAMPLE_TEMPLATE = "Input: {text}\nLabel: {label}"

STEP1_USER_TEMPLATE = (
    "Here are labeled examples:\n"
    "\n"
    "{examples}\n"
    "\n"
    "Classify the next input. Answer with exactly True or False.\n"
    "\n"
    "Input: {query}\n"
    "Label:"
)


def step1_template_hash() -> str:
    """sha256 over all step-1 template constants (logged in run configs)."""
    blob = "\n---\n".join([STEP1_SYSTEM, STEP1_EXAMPLE_TEMPLATE, STEP1_USER_TEMPLATE])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_step1(examples: Sequence[tuple[str, bool]], query: str) -> list[dict[str, str]]:
    """Messages for one step-1 classification call.

    ``examples``: (sentence, label) pairs, already shuffled by the caller with
    a logged seed. ``query``: the held-out sentence to classify.
    """
    block = "\n\n".join(
        STEP1_EXAMPLE_TEMPLATE.format(text=text, label="True" if label else "False")
        for text, label in examples
    )
    return [
        {"role": "system", "content": STEP1_SYSTEM},
        {"role": "user", "content": STEP1_USER_TEMPLATE.format(examples=block, query=query)},
    ]


# --- rule-given zero-shot baseline (P3) ---------------------------------------

RULE_GIVEN_USER_TEMPLATE = (
    "Classify the next input according to this rule:\n"
    "\n"
    "{rule}\n"
    "\n"
    "Answer with exactly True or False.\n"
    "\n"
    "Input: {query}\n"
    "Label:"
)


def rule_given_template_hash() -> str:
    """sha256 over the rule-given baseline template constants."""
    blob = "\n---\n".join([STEP1_SYSTEM, RULE_GIVEN_USER_TEMPLATE])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_rule_given(rule_text: str, query: str) -> list[dict[str, str]]:
    """Rule-given zero-shot application baseline (PLAN step-1 baselines).

    The model is told the canonical rule, sees NO examples, and gets the same
    "Answer with exactly True or False." line and trailing "Label:" as step 1
    (so the answer-token/logprob handling is identical).
    """
    return [
        {"role": "system", "content": STEP1_SYSTEM},
        {
            "role": "user",
            "content": RULE_GIVEN_USER_TEMPLATE.format(rule=rule_text.strip(), query=query),
        },
    ]


# --- step-2 free-form articulation (PLAN step-2 free-form) ---------------------

# The same neutral classifier system prompt as step 1, so the only thing that
# changes between classifying and articulating is the instruction.
FREEFORM_SYSTEM = STEP1_SYSTEM

# Two request PHRASINGS x two VARIANTS. 'direct' asks for the one-sentence rule
# only (no reasoning). 'think-then-state' permits chain-of-thought BEFORE a
# final one-sentence rule on its own line — CoT is allowed in step-2
# articulation (unlike the no-CoT step-1 classification). The final answer is
# fenced with FINAL_RULE_MARKER so the runner can extract just the sentence.
FINAL_RULE_MARKER = "RULE:"

FREEFORM_DIRECT_PHRASINGS = (
    (
        "Above are inputs that were each labeled True or False by a single "
        "fixed rule.\n\n"
        "State that labeling rule in ONE sentence. Do not explain or show your "
        "reasoning; give only the rule, prefixed with '{marker}'."
    ),
    (
        "All of the examples were labeled by the same hidden rule.\n\n"
        "In a SINGLE sentence, what is the rule that decides the True/False "
        "label? Answer with only the rule on one line, beginning with "
        "'{marker}'."
    ),
)

FREEFORM_THINK_PHRASINGS = (
    (
        "Above are inputs that were each labeled True or False by a single "
        "fixed rule.\n\n"
        "First think step by step about what distinguishes the True inputs "
        "from the False inputs. Then, on a final line by itself, state the "
        "labeling rule in ONE sentence prefixed with '{marker}'."
    ),
    (
        "All of the examples were labeled by the same hidden rule.\n\n"
        "Work out what the rule is, reasoning carefully about the True vs "
        "False inputs. When you are done, write the rule as a single sentence "
        "on its own final line, beginning with '{marker}'."
    ),
)

# (variant, phrasing_index) -> instruction template; locked order so the run
# config + the 2x2 grid stay reproducible.
FREEFORM_VARIANTS: dict[str, tuple[str, ...]] = {
    "direct": FREEFORM_DIRECT_PHRASINGS,
    "think-then-state": FREEFORM_THINK_PHRASINGS,
}

FREEFORM_USER_TEMPLATE = "Here are labeled examples:\n\n{examples}\n\n{instruction}"

# No-examples control: the SAME articulation request with the few-shot block
# removed (measures a-priori guessability, quality bar #7).
FREEFORM_NOEX_TEMPLATE = (
    "I labeled some short text inputs True or False using a single fixed "
    "rule.\n\n{instruction}"
)


def freeform_template_hash() -> str:
    """sha256 over all free-form template constants (logged in run configs)."""
    blob = "\n---\n".join(
        [
            FREEFORM_SYSTEM,
            FINAL_RULE_MARKER,
            *FREEFORM_DIRECT_PHRASINGS,
            *FREEFORM_THINK_PHRASINGS,
            FREEFORM_USER_TEMPLATE,
            FREEFORM_NOEX_TEMPLATE,
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _freeform_instruction(variant: str, phrasing: int) -> str:
    if variant not in FREEFORM_VARIANTS:
        raise ValueError(f"unknown free-form variant {variant!r}")
    phrasings = FREEFORM_VARIANTS[variant]
    if not 0 <= phrasing < len(phrasings):
        raise ValueError(f"phrasing index {phrasing} out of range for variant {variant!r}")
    return phrasings[phrasing].format(marker=FINAL_RULE_MARKER)


def render_freeform_articulation(
    examples: Sequence[tuple[str, bool]], variant: str = "direct", phrasing: int = 0
) -> list[dict[str, str]]:
    """Free-form articulation (PLAN step-2 free-form).

    After the SAME few-shot block step 1 used, ask the model to state the
    labeling rule in one sentence. ``variant`` is 'direct' (one sentence, no
    reasoning) or 'think-then-state' (CoT permitted, then a final sentence);
    ``phrasing`` selects one of the two locked request phrasings.
    """
    instruction = _freeform_instruction(variant, phrasing)
    block = "\n\n".join(
        STEP1_EXAMPLE_TEMPLATE.format(text=text, label="True" if label else "False")
        for text, label in examples
    )
    return [
        {"role": "system", "content": FREEFORM_SYSTEM},
        {
            "role": "user",
            "content": FREEFORM_USER_TEMPLATE.format(examples=block, instruction=instruction),
        },
    ]


def render_freeform_no_examples(variant: str = "direct", phrasing: int = 0) -> list[dict[str, str]]:
    """No-examples free-form control: same articulation request, no few-shot block.

    Measures a-priori guessability (quality bar #7): the model is asked for the
    rule with NO examples to read it off, so any grade reflects prior bias only.
    """
    instruction = _freeform_instruction(variant, phrasing)
    return [
        {"role": "system", "content": FREEFORM_SYSTEM},
        {"role": "user", "content": FREEFORM_NOEX_TEMPLATE.format(instruction=instruction)},
    ]


def extract_rule(text: str) -> str:
    """Pull the one-sentence rule out of a free-form completion.

    Prefers the text after the LAST 'RULE:' marker (think-then-state ends with
    it; direct begins with it). Falls back to the last non-empty line, then to
    the whole stripped text — a model that ignored the marker still yields a
    gradeable candidate rather than an empty string.
    """
    stripped = text.strip()
    if FINAL_RULE_MARKER in stripped:
        candidate = stripped.rsplit(FINAL_RULE_MARKER, 1)[1].strip()
        if candidate:
            return candidate.splitlines()[0].strip() if "\n" in candidate else candidate
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return stripped


# --- CoT same-session classify-then-articulate (CoT same-session diagnostic) --------------
#
# A DELIBERATE departure from the no-CoT brief: one multi-turn conversation that
# (turn 1) classifies a batch of held-out items WITH chain-of-thought, then
# (turn 2, same session) asks the model to state the rule it just used, again
# with CoT. Turn 1 reuses the STEP1 example block verbatim; turn 2 reuses the
# free-form RULE: marker tail so ``extract_rule`` works unchanged. The runner
# builds turn 2 by appending the model's turn-1 assistant reply between the two
# user turns (see scripts/run_cot_same_session.py).

# Turn 1: classify N numbered items with CoT, ending in a parseable answer block.
# The "Answer K: True/False" form is load-bearing for parse_cot_labels; keep exact.
COT_TURN1_TEMPLATE = (
    "Here are labeled examples:\n"
    "\n"
    "{examples}\n"
    "\n"
    "Each input above was labeled True or False by a single fixed rule. Apply "
    "that same rule to classify each of the {n} inputs below. Think step by step, "
    "then on the LAST lines write one verdict per input in the exact form\n"
    "Answer 1: True\n"
    "Answer 2: False\n"
    "... (one line per numbered input, only the word True or False after the "
    "colon, and nothing else on those lines).\n"
    "\n"
    "{numbered_items}"
)

# Turn 2: ask for the rule the model just used, CoT permitted, RULE: tail reused
# from the free-form think-then-state phrasing so extract_rule recovers it.
COT_TURN2_TEMPLATE = (
    "What rule did you use to assign those True/False labels? Reason about it if "
    "that helps, then on a final line by itself state the labeling rule in ONE "
    "sentence prefixed with '{marker}'."
)


def cot_same_session_template_hash() -> str:
    """sha256 over the CoT same-session template constants (logged in configs)."""
    blob = "\n---\n".join([STEP1_SYSTEM, STEP1_EXAMPLE_TEMPLATE,
                           COT_TURN1_TEMPLATE, COT_TURN2_TEMPLATE, FINAL_RULE_MARKER])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_cot_turn1(query_texts: Sequence[str], examples: Sequence[tuple[str, bool]]) -> list[dict[str, str]]:
    """Turn-1 messages: the SAME few-shot block as step 1 + a CoT classification
    instruction over ``query_texts`` (numbered 1..N), ending in the answer block.
    """
    block = "\n\n".join(
        STEP1_EXAMPLE_TEMPLATE.format(text=text, label="True" if label else "False")
        for text, label in examples
    )
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(query_texts, 1))
    user = COT_TURN1_TEMPLATE.format(examples=block, n=len(query_texts), numbered_items=numbered)
    return [
        {"role": "system", "content": STEP1_SYSTEM},
        {"role": "user", "content": user},
    ]


def cot_turn2_user() -> str:
    """The turn-2 user content (ends with the RULE: marker clause)."""
    return COT_TURN2_TEMPLATE.format(marker=FINAL_RULE_MARKER)


# --- in-session per-item classify-then-articulate (in-session) ----------------------
#
# The CORRECTED redo of CoT same-session: classify each held-out item ONE AT A TIME (the
# normal Step-1 per-item completion format, NOT a batch) inside ONE preserved
# conversation, then ask the rule. Two experiments share this structure and
# differ only by CoT: Exp 1 = no reasoning anywhere; Exp 2 = reasoning while
# classifying each item AND while stating the rule. Turn 1 (no-CoT) reuses
# render_step1 verbatim, so it is a byte-for-byte normal Step-1 prompt. The
# runner (scripts/run_insession_articulation.py) drives the sequential turns.

# Exp 2 turn-1: few-shot block + a CoT classification instruction (one item now;
# more come as follow-ups). The "Answer: True/False" line is the parse anchor.
INSESSION_COT_TURN1_TEMPLATE = (
    "Here are labeled examples:\n"
    "\n"
    "{examples}\n"
    "\n"
    "Each input above was labeled True or False by a single fixed rule. I will give "
    "you inputs one at a time; apply that same rule to each. Think step by step, then "
    "end your reply with a line in the exact form 'Answer: True' or 'Answer: False' "
    "(only that word after the colon).\n"
    "\n"
    "Input: {query}"
)

# per-item follow-up user turns (item 2..N), no few-shot re-sent (it is in context).
INSESSION_FOLLOWUP_TEMPLATE = "Input: {query}\nLabel:"                 # Exp 1 (no-CoT)
INSESSION_COT_FOLLOWUP_TEMPLATE = (                                    # Exp 2 (with-CoT)
    "Input: {query}\n(Reason, then end with 'Answer: True' or 'Answer: False'.)"
)

# Exp 1 articulation: direct, no reasoning (ends in the RULE: marker; Exp 2 reuses
# cot_turn2_user, which also ends in the marker, so extract_rule works for both).
INSESSION_ARTICULATE_DIRECT = (
    "Those are all the inputs. What single rule decided the True/False labels you just "
    "assigned? State it in ONE sentence. Do not explain or show your reasoning; give "
    "only the rule, prefixed with '{marker}'."
)


def insession_template_hash() -> str:
    """sha256 over the in-session template constants (logged in run configs)."""
    blob = "\n---\n".join([
        STEP1_SYSTEM, STEP1_EXAMPLE_TEMPLATE, STEP1_USER_TEMPLATE,
        INSESSION_COT_TURN1_TEMPLATE, INSESSION_FOLLOWUP_TEMPLATE,
        INSESSION_COT_FOLLOWUP_TEMPLATE, INSESSION_ARTICULATE_DIRECT,
        COT_TURN2_TEMPLATE, FINAL_RULE_MARKER,
    ])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_insession_turn1(
    examples: Sequence[tuple[str, bool]], query: str, *, cot: bool
) -> list[dict[str, str]]:
    """Turn-1 messages. No-CoT: the EXACT Step-1 prompt (render_step1). CoT: the
    same few-shot block + a reason-then-'Answer:' instruction."""
    if not cot:
        return render_step1(examples, query)
    block = "\n\n".join(
        STEP1_EXAMPLE_TEMPLATE.format(text=text, label="True" if label else "False")
        for text, label in examples
    )
    return [
        {"role": "system", "content": STEP1_SYSTEM},
        {"role": "user", "content": INSESSION_COT_TURN1_TEMPLATE.format(examples=block, query=query)},
    ]


def insession_followup_user(query: str, *, cot: bool) -> str:
    """The per-item follow-up user turn for item 2..N."""
    template = INSESSION_COT_FOLLOWUP_TEMPLATE if cot else INSESSION_FOLLOWUP_TEMPLATE
    return template.format(query=query)


def insession_articulation_user(*, cot: bool) -> str:
    """The final 'what rule did you use?' user turn (ends in the RULE: marker)."""
    if cot:
        return cot_turn2_user()
    return INSESSION_ARTICULATE_DIRECT.format(marker=FINAL_RULE_MARKER)


# --- step-2 multiple-choice articulation (PLAN step-2 multiple-choice) ---------------------

# Generic across ALL rules (no rule name, no hint): the same few-shot block as
# step 1, then a fixed question + lettered options + a single-letter answer.
MC_SYSTEM = STEP1_SYSTEM

MC_USER_TEMPLATE = (
    "Here are labeled examples:\n"
    "\n"
    "{examples}\n"
    "\n"
    "Which rule best describes how the labels were assigned?\n"
    "\n"
    "{options}\n"
    "\n"
    "Answer with the single letter of the best option."
)

# No-examples control: the SAME question/options with the few-shot block
# removed (a-priori guessability, chance = 1/n_options; quality bar #7).
MC_NOEX_TEMPLATE = (
    "I labeled some short text inputs True or False using a single fixed rule.\n"
    "\n"
    "Which rule best describes how the labels were assigned?\n"
    "\n"
    "{options}\n"
    "\n"
    "Answer with the single letter of the best option."
)

MC_OPTION_TEMPLATE = "{letter}) {text}"


def mc_template_hash() -> str:
    """sha256 over all step-2 multiple-choice template constants (logged in run configs)."""
    blob = "\n---\n".join([MC_SYSTEM, MC_USER_TEMPLATE, MC_NOEX_TEMPLATE, MC_OPTION_TEMPLATE])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _render_mc_options(letters: Sequence[str], options: Sequence[str]) -> str:
    if len(letters) != len(options):
        raise ValueError(f"{len(letters)} letters but {len(options)} options")
    return "\n".join(
        MC_OPTION_TEMPLATE.format(letter=letter, text=text.strip())
        for letter, text in zip(letters, options)
    )


def _default_letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def render_mc_articulation(
    examples: Sequence[tuple[str, bool]],
    options: Sequence[str],
    letters: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Multiple-choice articulation (PLAN step-2 multiple-choice).

    Same few-shot block as step 1, then 'Which rule best describes how the
    labels were assigned?', the lettered options (already SHUFFLED by the caller
    with a logged seed), and a single-letter answer instruction.
    """
    if letters is None:
        letters = _default_letters(len(options))
    block = "\n\n".join(
        STEP1_EXAMPLE_TEMPLATE.format(text=text, label="True" if label else "False")
        for text, label in examples
    )
    user = MC_USER_TEMPLATE.format(examples=block, options=_render_mc_options(letters, options))
    return [{"role": "system", "content": MC_SYSTEM}, {"role": "user", "content": user}]


def render_mc_no_examples(
    options: Sequence[str], letters: Sequence[str] | None = None
) -> list[dict[str, str]]:
    """No-examples multiple-choice control: SAME question + options, few-shot block removed."""
    if letters is None:
        letters = _default_letters(len(options))
    user = MC_NOEX_TEMPLATE.format(options=_render_mc_options(letters, options))
    return [{"role": "system", "content": MC_SYSTEM}, {"role": "user", "content": user}]
