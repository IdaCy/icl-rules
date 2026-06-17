"""Step-2 free-form articulation grading: LLM-judge (primary) + extensional check.

Two graded paths, for "Step-2 free-form":

LLM-JUDGE (primary). A gpt-4.1 judge with the WRITTEN RUBRIC below grades each
candidate articulation 2/1/0 given the rule's ground-truth canonical_articulation
+ equivalence_class (both from the COMMITTED data/spec_extract.json) + the
candidate + a few example items. Grade 2 = correct OR
extensionally equivalent on the training distribution; 1 = partially correct /
related but wrong scope; 0 = wrong/vague. Polarity- and complement-phrased
articulations are accepted (the rubric says so explicitly). The rubric is a
module constant so it is auditable; the judge prompt FEEDS the canonical
articulation + equivalence class so "grade 2 for an equivalent" is principled,
not the judge's private opinion.

EXTENSIONAL CHECK (corroboration, surface/positional/numeric/structural rules).
There is no automatic way to compile an arbitrary natural-language candidate
into a python predicate, so we do NOT try. Instead, for a rule whose own
ground truth is recomputable (groundtruth.RULE_PREDICATES[rule_id].label_of is
not None), we run that label_of on N probe items to record the rule's behavior,
and we ALSO ask the judge to state whether the candidate is extensionally
equivalent to the rule on the training distribution. The judge's equivalence
verdict is the load-bearing signal; the recomputed labels are logged so a human
can audit the probe distribution the judge reasoned about. This is best-effort
and documented as such: the LLM-judge grade is PRIMARY, the extensional check is
CORROBORATION for surface rules (PLAN).

Pre-registered metrics (PLAN):
- primary  = MEDIAN grade of the DIRECT variant across the 3 contexts.
- secondary ("elicitable") = the BEST grade across all variants.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

from .client import OpenAIClient, response_text
from .rule_ids import canonical_rule_id

# --- the written rubric (auditable module constant) ---------------------------

JUDGE_MODEL = "gpt-4.1"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 400  # rubric asks for a short rationale + a strict JSON verdict
JUDGE_SEED = 0

GRADING_RUBRIC = """\
You are grading how well a CANDIDATE articulation describes a hidden labeling \
rule. You are given the rule's ground-truth canonical statement, a list of \
statements that are EXTENSIONALLY EQUIVALENT to it on the training \
distribution, and a few labeled example sentences. Assign exactly one integer \
grade.

GRADE 2 (correct or extensionally equivalent):
- The candidate states the canonical rule, OR
- The candidate is logically equivalent to it (a paraphrase), OR
- The candidate is EXTENSIONALLY EQUIVALENT on the training distribution: it \
labels every shown example the way the true rule does, even if phrased \
differently (e.g. "no capital letters anywhere" for an all-lowercase rule, or \
"the sentence uses was/were + past participle" for a passive-voice rule). The \
equivalence-class list enumerates accepted equivalents; a candidate that \
matches the spirit of one of them is a 2.
- POLARITY/COMPLEMENT phrasing is accepted: stating the rule for the False \
class ("labeled False when it contains a digit") or the contrapositive is a 2 \
if it picks out the same partition.

GRADE 1 (partially correct / related but wrong scope):
- The candidate names the right FEATURE but gets the threshold, direction, or \
scope wrong (e.g. "contains a number greater than fifty" for a plain \
contains-a-digit rule; "at least three words are capitalized" for a \
second-word-capitalized rule).
- The candidate is a near-miss that agrees with the true rule on many but not \
all examples.

GRADE 0 (wrong or vague):
- The candidate names an unrelated feature, contradicts the examples, or is so \
vague it does not commit to a checkable rule ("the sentences share a pattern", \
"some are positive and some negative").

Rules:
- Judge the candidate ONLY against the ground truth and the examples provided. \
Do not invent facts about the rule beyond what is given.
- Casing, hedging, and wordiness do not change the grade; the labeling \
PARTITION it implies is what matters.
- Output STRICT JSON on a single line and nothing else:
  {"grade": <0|1|2>, "extensionally_equivalent": <true|false>, "rationale": "<one sentence>"}
- "extensionally_equivalent" is true iff the candidate would label every shown \
example exactly as the true rule does (this is what a grade of 2 via the \
equivalence route means).
"""


def rubric_hash() -> str:
    """sha256 of the rubric text (logged with every grading run so an edit shows)."""
    import hashlib

    return hashlib.sha256(GRADING_RUBRIC.encode("utf-8")).hexdigest()


# --- spec extract access (PUBLIC committed extract only) ----------------------


def load_spec_extract(path: str | Path) -> dict[str, Any]:
    """Load the committed data/spec_extract.json (the public machine extract).

    NEVER read the private rule-specs.yaml at runtime; this is the only ground
    truth a grading run is allowed to consult.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = data.get("rules")
    if not isinstance(rules, dict) or not rules:
        raise ValueError(f"{path}: spec extract has no 'rules' object")
    return rules


def gold_for(spec_rules: dict[str, Any], rule_id: str) -> dict[str, Any]:
    """Canonical articulation + equivalence class for one rule (loud on miss)."""
    base_rule_id = canonical_rule_id(rule_id)
    if base_rule_id not in spec_rules:
        raise KeyError(f"rule {rule_id!r} not in spec extract")
    entry = spec_rules[base_rule_id]
    canonical = entry.get("canonical_articulation")
    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError(f"rule {rule_id!r}: no canonical_articulation in spec extract")
    equivalence = entry.get("equivalence_class") or []
    if not isinstance(equivalence, list):
        raise ValueError(f"rule {rule_id!r}: equivalence_class is not a list")
    return {
        "canonical_articulation": canonical.strip(),
        "equivalence_class": [str(s) for s in equivalence],
    }


# --- the judge prompt ----------------------------------------------------------


def render_judge(
    candidate: str,
    canonical_articulation: str,
    equivalence_class: Sequence[str],
    examples: Sequence[tuple[str, bool]],
) -> list[dict[str, str]]:
    """Messages for one grading call: rubric + gold + candidate + examples.

    ``examples``: (sentence, label) pairs the judge sees so it can reason about
    extensional equivalence on the actual training distribution.
    """
    equiv_block = (
        "\n".join(f"- {s}" for s in equivalence_class)
        if equivalence_class
        else "(none provided)"
    )
    example_block = "\n".join(
        f"{'True ' if label else 'False'}: {text}" for text, label in examples
    )
    user = (
        f"GROUND-TRUTH CANONICAL RULE:\n{canonical_articulation}\n\n"
        f"EXTENSIONALLY-EQUIVALENT STATEMENTS (each grades 2):\n{equiv_block}\n\n"
        f"LABELED EXAMPLE SENTENCES:\n{example_block}\n\n"
        f"CANDIDATE ARTICULATION TO GRADE:\n{candidate.strip()}\n\n"
        "Grade the candidate per the rubric. Output the strict JSON verdict only."
    )
    return [
        {"role": "system", "content": GRADING_RUBRIC},
        {"role": "user", "content": user},
    ]


def parse_judge(text: str) -> dict[str, Any]:
    """Parse the judge's strict-JSON verdict.

    Tolerates fencing / leading prose by extracting the first {...} block, but a
    missing or out-of-range grade raises LOUDLY (a silent default would corrupt
    a metric). Returns {grade:int in {0,1,2}, extensionally_equivalent:bool,
    rationale:str}.
    """
    raw = text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"judge produced no JSON object: {text!r}")
    obj = json.loads(raw[start : end + 1])
    if "grade" not in obj:
        raise ValueError(f"judge verdict missing 'grade': {obj!r}")
    grade = obj["grade"]
    if grade not in (0, 1, 2):
        raise ValueError(f"judge grade out of range (expected 0/1/2): {grade!r}")
    return {
        "grade": int(grade),
        "extensionally_equivalent": bool(obj.get("extensionally_equivalent", False)),
        "rationale": str(obj.get("rationale", "")).strip(),
    }


async def grade_one(
    client: OpenAIClient,
    candidate: str,
    canonical_articulation: str,
    equivalence_class: Sequence[str],
    examples: Sequence[tuple[str, bool]],
    *,
    model: str = JUDGE_MODEL,
) -> dict[str, Any]:
    """One judge call -> {grade, extensionally_equivalent, rationale, record}."""
    messages = render_judge(candidate, canonical_articulation, equivalence_class, examples)
    record = await client.complete(
        model,
        messages,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
        seed=JUDGE_SEED,
    )
    verdict = parse_judge(response_text(record))
    return {**verdict, "record": record}


# --- extensional check (corroboration for recomputable rules) ------------------


def extensional_probe(
    label_of: Callable[[str], bool] | None,
    probes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Run the RULE'S OWN ground-truth label_of on probe items (corroboration).

    This does NOT test the candidate articulation directly (no general way to
    compile NL into a predicate); it records the true rule's behavior on the
    probe set so a human auditor can see the distribution the judge's
    extensional-equivalence verdict was made against. ``applicable`` is False for
    validator-derived (LLM-judged) rules that carry no recomputable predicate.
    """
    if label_of is None:
        return {"applicable": False, "reason": "rule has no recomputable predicate"}
    n_true = sum(1 for p in probes if label_of(p["text"]))
    return {
        "applicable": True,
        "n_probes": len(probes),
        "true_rule_n_true": n_true,
        "true_rule_n_false": len(probes) - n_true,
        # stored-vs-recomputed agreement, where the probe carries a stored label
        "stored_label_agreement": _stored_agreement(label_of, probes),
    }


def _stored_agreement(
    label_of: Callable[[str], bool], probes: Sequence[dict[str, Any]]
) -> float | None:
    labeled = [p for p in probes if "label" in p]
    if not labeled:
        return None
    agree = sum(1 for p in labeled if label_of(p["text"]) == p["label"])
    return agree / len(labeled)


# --- pre-specified metrics -----------------------------------------------------


def median_of_direct(grades: Sequence[int]) -> float:
    """PRIMARY metric: median grade of the DIRECT variant across contexts.

    ``grades`` are the direct-variant grades (one per context x phrasing). Uses
    the statistics median (interpolates between 1 and 2 -> 1.5 for an even count
    straddling, which is the pre-specified behavior — we do NOT round)."""
    if not grades:
        raise ValueError("median_of_direct: no direct-variant grades")
    return float(statistics.median(grades))


def best_grade(grades: Sequence[int]) -> int:
    """SECONDARY 'elicitable' metric: best grade across ALL variants."""
    if not grades:
        raise ValueError("best_grade: no grades")
    return int(max(grades))


def summarize_rule(graded: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-rule metric block from graded generation rows.

    Each row carries: variant ('direct'|'think-then-state'), phrasing index,
    context_index, has_examples (False == the no-examples control), grade.
    Primary = median of the DIRECT variant's WITH-EXAMPLES grades; secondary =
    best across all with-examples variants; the no-examples control is
    summarized separately (a-priori guessability)."""
    with_ex = [g for g in graded if g["has_examples"]]
    control = [g for g in graded if not g["has_examples"]]
    direct = [g["grade"] for g in with_ex if g["variant"] == "direct"]
    out: dict[str, Any] = {
        "n_generations": len(with_ex),
        "primary_median_direct": median_of_direct(direct) if direct else None,
        "secondary_best_variant": best_grade([g["grade"] for g in with_ex]) if with_ex else None,
        "grade_counts": _grade_counts(g["grade"] for g in with_ex),
        "by_variant": {},
    }
    for variant in sorted({g["variant"] for g in with_ex}):
        vg = [g["grade"] for g in with_ex if g["variant"] == variant]
        out["by_variant"][variant] = {
            "median": float(statistics.median(vg)) if vg else None,
            "max": max(vg) if vg else None,
            "grades": vg,
        }
    if control:
        cg = [g["grade"] for g in control]
        out["no_examples_control"] = {
            "n": len(cg),
            "median": float(statistics.median(cg)),
            "max": max(cg),
            "grade_counts": _grade_counts(cg),
        }
    return out


def _grade_counts(grades: Any) -> dict[str, int]:
    counts = {"0": 0, "1": 0, "2": 0}
    for g in grades:
        counts[str(g)] += 1
    return counts
