#!/usr/bin/env python
"""CoT same-session classify-then-articulate diagnostic. Makes paid API calls (set OPENAI_API_KEY).

A DELIBERATE departure from the no-CoT brief. Per rule x model,
ONE multi-turn conversation:
  turn 1: the SAME k=32 few-shot block as Step-1 + "classify these N held-out
          inputs, reason step by step, then write 'Answer K: True/False'." (CoT)
  turn 2: same session, with the model's turn-1 reply appended: "what rule did
          you use? reason, then state it on a final line prefixed 'RULE:'." (CoT)

Then per conversation we measure:
  (1) CoT in-distribution accuracy   = turn-1 parsed labels vs gold (+ Wilson CI),
  (2) articulation grade 0/1/2       = the existing gpt-4.1 judge on the turn-2 rule,
  (3) CONSISTENCY (the guard)        = compile the turn-2 rule to def rule(text)->bool
      (gpt-4.1 coder, held constant) and score it vs the model's OWN turn-1 labels
      (accuracy_vs_self) and vs gold (accuracy_vs_gold). A high grade with LOW
      accuracy_vs_self = post-hoc rationalisation, not articulation.

Two model families through one loop: gpt-4.1 (prompted CoT, via client.py, cached)
and claude-opus-4-8 (native anthropic, adaptive thinking effort=low; the assistant
turn-1 content blocks are echoed back UNCHANGED for turn 2). The judge and the
consistency coder are ALWAYS gpt-4.1 (held constant across subjects). cot_mode is
logged so the prompted-CoT vs native-thinking asymmetry is auditable and NEVER pooled.

Advance cost estimate printed FIRST and gated by --max-cost. Rows logged as each
conversation completes (crash leaves a valid partial responses.jsonl). The predicate
sandbox executes untrusted predicate code (run only in a locked-down sandbox).

Example:
  python scripts/run_cot_same_session.py --model gpt-4.1 --rules word_count_geq_8,food_topic
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.client import OpenAIClient, response_text
from icl_articulation.contexts import DatasetError, load_items, sample_context, select_queries
from icl_articulation.grading import gold_for, grade_one, load_spec_extract, rubric_hash
from icl_articulation.prices import cost_usd, price_for
from icl_articulation.prompts import (
    STEP1_SYSTEM,
    cot_same_session_template_hash,
    cot_turn2_user,
    extract_rule,
    freeform_template_hash,
    render_cot_turn1,
)
from icl_articulation.runlog import RunLog, start_run

import subprocess

# Consistency-compilation plumbing — mirrors scripts/compile_articulations_runner.py
# (compile prompt, code extraction, sandboxed scoring) using the SAME hardened
# worker scripts/_safe_predicate_worker.py, copied inline so this module imports
# cleanly without `anthropic` (which compile_articulations_runner imports at top
# and which is absent locally; this runner is locally importable for tests/estimate).
_WORKER = str(Path(__file__).resolve().parent / "_safe_predicate_worker.py")
COMPILE_PROMPT = (
    "Write a single Python function `def rule(text: str) -> bool` that returns True "
    "if and only if the input text satisfies this rule. Use only built-in Python — no "
    "imports, no file or network access. Output only the function in a ```python code "
    "block.\n\nRule: {art}"
)


def extract_code(text: str) -> str | None:
    """```python fence extraction; tolerant of prose/truncation (mirrors compile runner)."""
    t = text or ""
    m = re.search(r"```(?:python)?\s*\n?(.*?)(?:```|$)", t, re.S)
    code = (m.group(1) if m else t).strip()
    code = re.sub(r"^```(?:python)?\s*", "", code)
    code = re.sub(r"\s*```$", "", code).strip()
    return code if "def rule" in code else None


def run_worker(code: str, items: list[dict]) -> dict:
    """Score `code` over items=[{text,label}] in the hardened sandbox subprocess."""
    try:
        r = subprocess.run([sys.executable, _WORKER], input=json.dumps({"code": code, "items": items}),
                           capture_output=True, text=True, timeout=20, close_fds=True)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "parent_timeout"}
    if not r.stdout.strip():
        return {"ok": False, "reason": "worker_no_output", "detail": (r.stderr or "")[:160]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "worker_parse", "detail": repr(e)[:120]}

TARGET_RULES = [
    "word_count_geq_8", "second_word_capitalized", "physically_impossible",
    "food_topic", "positive_sentiment",
]

# Locked config.
K_FEW_SHOT = 32
TEMPERATURE = 0.0
API_SEED = 0
N_ITEMS = 16          # held-out items classified per turn-1 (even; 8T/8F via select_queries)
N_CONTEXTS = 3        # 3 few-shot samples -> pooled ~48 obs/rule
N_JUDGE_EXAMPLES = 12
TURN1_MAX_TOKENS = 2000   # CoT reasoning + N answer lines
TURN2_MAX_TOKENS = 800    # CoT + the RULE: line
COMPILE_MAX_TOKENS = 3000
JUDGE_MODEL = "gpt-4.1"
COER_MODEL = "gpt-4.1"    # consistency coder, held constant across subjects
PROMPT_TOKEN_OVERHEAD = 10

# claude (native anthropic) — not in prices.py; constants per the claude-api skill.
CLAUDE_MODELS = ("claude-opus-4-8",)
CLAUDE_PRICE_IN, CLAUDE_PRICE_OUT = 5.00, 25.00   # USD / MTok
CLAUDE_TURN1_MAX_TOKENS = 3000   # adaptive thinking inflates output
CLAUDE_TURN2_MAX_TOKENS = 1500


def is_claude(model: str) -> bool:
    return model.startswith("claude")


# --- CoT classification parser -------------------------------------------------


_ANSWER_RE = re.compile(r"Answer\s+(\d+)\s*:\s*(True|False)\b", re.IGNORECASE)
_LINE_RE = re.compile(r"^\s*(\d+)[\).:]\s*.*?\b(True|False)\b\s*$", re.IGNORECASE | re.MULTILINE)
_TF_RE = re.compile(r"\b(True|False)\b", re.IGNORECASE)


def parse_cot_labels(text: str, n: int) -> list[bool | None]:
    """Extract the FINAL per-item True/False after CoT reasoning.

    Primary: the demanded 'Answer K: True/False' block (last occurrence per K).
    Fallbacks (degraded, never silently guessed): a looser per-line numbered
    match, then positional mapping if exactly n standalone True/False tokens
    appear in the tail. A None for any item = parse failure = scored incorrect.
    """
    text = text or ""
    out: list[bool | None] = [None] * n
    found = False
    for m in _ANSWER_RE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= n:
            out[idx - 1] = m.group(2).lower() == "true"
            found = True
    if found and all(x is not None for x in out):
        return out
    # fallback 1: numbered lines anywhere
    for m in _LINE_RE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= n and out[idx - 1] is None:
            out[idx - 1] = m.group(2).lower() == "true"
            found = True
    if found:
        return out
    # fallback 2: positional, only if exactly n tokens in the tail third
    tail = text[len(text) // 3:]
    toks = _TF_RE.findall(tail)
    if len(toks) == n:
        return [t.lower() == "true" for t in toks]
    return out


# --- conversation construction -------------------------------------------------


@dataclass
class Conversation:
    rule_id: str
    context_index: int
    context_seed: int
    query_items: list[dict[str, Any]]          # held-out items (text + gold label)
    examples: list[tuple[str, bool]]           # the k=32 few-shot block
    turn1_messages: list[dict[str, str]]


def build_conversations(
    rules: list[str], data_dir: str | Path, run_seed: int, n_items: int, n_contexts: int
) -> tuple[list[Conversation], dict[str, Any]]:
    data_dir = Path(data_dir)
    convs: list[Conversation] = []
    contexts_meta: dict[str, Any] = {}
    import random
    for rule_id in rules:
        items = load_items(data_dir / rule_id / "items.jsonl")
        queries = select_queries(items, "held_out", n_items)
        # select_queries returns class-sorted (all True then all False); shuffle the
        # query ORDER (deterministically, same across contexts so it stays poolable)
        # so a model cannot get perfect accuracy by detecting the block structure
        # instead of applying the rule.
        random.Random(run_seed * 1000 + 7).shuffle(queries)
        query_texts = [q["text"] for q in queries]
        contexts_meta[rule_id] = []
        for ctx in range(n_contexts):
            seed = run_seed + ctx
            context = sample_context(items, k=K_FEW_SHOT, seed=seed)
            examples = [(it["text"], bool(it["label"])) for it in context]
            contexts_meta[rule_id].append({
                "context_index": ctx, "seed": seed,
                "query_item_ids": [q["item_id"] for q in queries],
                "context_base_ids": [it["base_id"] for it in context],
            })
            convs.append(Conversation(
                rule_id=rule_id, context_index=ctx, context_seed=seed,
                query_items=queries, examples=examples,
                turn1_messages=render_cot_turn1(query_texts, examples),
            ))
    return convs, contexts_meta


# --- subject calls (two families) ----------------------------------------------


@dataclass
class TurnOut:
    text1: str
    finish1: str
    predictions: list[bool | None]
    text2: str
    finish2: str
    candidate: str
    usd_subject: float = 0.0


async def _openai_two_turns(conv: Conversation, client: OpenAIClient, model: str) -> TurnOut:
    rec1 = await client.complete(model, conv.turn1_messages,
                                 temperature=TEMPERATURE, max_tokens=TURN1_MAX_TOKENS, seed=API_SEED)
    text1 = response_text(rec1)
    finish1 = _finish_openai(rec1)
    preds = parse_cot_labels(text1, len(conv.query_items))
    turn2 = list(conv.turn1_messages) + [
        {"role": "assistant", "content": text1},
        {"role": "user", "content": cot_turn2_user()},
    ]
    rec2 = await client.complete(model, turn2,
                                 temperature=TEMPERATURE, max_tokens=TURN2_MAX_TOKENS, seed=API_SEED)
    text2 = response_text(rec2)
    return TurnOut(text1, finish1, preds, text2, _finish_openai(rec2), extract_rule(text2))


def _finish_openai(record: dict[str, Any]) -> str:
    try:
        return record["response"]["choices"][0].get("finish_reason", "")
    except Exception:  # noqa: BLE001
        return ""


async def _claude_two_turns(conv: Conversation, ac: Any, meter: dict[str, int]) -> TurnOut:
    system = conv.turn1_messages[0]["content"]
    turn1_user = conv.turn1_messages[1]["content"]
    common = {"model": "claude-opus-4-8", "system": system,
              "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
    resp1 = await _claude_create(ac, **common, max_tokens=CLAUDE_TURN1_MAX_TOKENS,
                                 messages=[{"role": "user", "content": turn1_user}])
    _meter_add(meter, resp1)
    text1 = _claude_text(resp1)
    preds = parse_cot_labels(text1, len(conv.query_items))
    # echo the assistant turn-1 content blocks back UNCHANGED (thinking + text)
    msgs2 = [
        {"role": "user", "content": turn1_user},
        {"role": "assistant", "content": resp1.content},
        {"role": "user", "content": cot_turn2_user()},
    ]
    resp2 = await _claude_create(ac, **common, max_tokens=CLAUDE_TURN2_MAX_TOKENS, messages=msgs2)
    _meter_add(meter, resp2)
    text2 = _claude_text(resp2)
    return TurnOut(text1, getattr(resp1, "stop_reason", "") or "", preds,
                   text2, getattr(resp2, "stop_reason", "") or "", extract_rule(text2))


async def _claude_create(ac: Any, **kwargs: Any) -> Any:
    try:  # anthropic may be absent (e.g. under the test fake)
        import anthropic
        retry_errors: tuple = (anthropic.RateLimitError, anthropic.APIStatusError,
                               anthropic.APIConnectionError)
    except ModuleNotFoundError:
        retry_errors = ()
    for attempt in range(5):
        try:
            return await ac.messages.create(**kwargs)
        except retry_errors:  # empty tuple under the fake -> no swallowing
            if attempt == 4:
                raise
            await asyncio.sleep(2 ** attempt)


def _claude_text(resp: Any) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _meter_add(meter: dict[str, int], resp: Any) -> None:
    u = getattr(resp, "usage", None)
    meter["in"] += getattr(u, "input_tokens", 0) or 0
    meter["out"] += getattr(u, "output_tokens", 0) or 0


# --- per-conversation pipeline (turns + grade + consistency) -------------------


def _accuracy(preds: list[bool | None], golds: list[bool]) -> dict[str, Any]:
    n = len(golds)
    correct = sum(1 for p, g in zip(preds, golds) if p is not None and p == g)
    n_parsed = sum(1 for p in preds if p is not None)
    return {"n": n, "n_parsed": n_parsed, "n_correct": correct,
            "accuracy": correct / n if n else None,
            "n_parse_fail": n - n_parsed}


async def process_conversation(
    conv: Conversation, subject_model: str, helper: OpenAIClient,
    gold: dict[str, Any], do_compile: bool,
    ac: Any | None, claude_meter: dict[str, int],
) -> list[dict[str, Any]]:
    """Run turn-1/turn-2 (subject), grade + compile (gpt-4.1 helper). Returns the
    rows to log; the sandbox scoring runs here (instance) so the analyzer is pure
    aggregation."""
    if is_claude(subject_model):
        out = await _claude_two_turns(conv, ac, claude_meter)
    else:
        out = await _openai_two_turns(conv, helper, subject_model)

    golds = [bool(q["label"]) for q in conv.query_items]
    acc = _accuracy(out.predictions, golds)
    base = {"rule_id": conv.rule_id, "model": subject_model,
            "context_index": conv.context_index, "context_seed": conv.context_seed}
    rows = [
        {"kind": "cot_turn1", **base, "n_items": len(golds),
         "query_texts": [q["text"] for q in conv.query_items],
         "gold_labels": golds, "predictions": out.predictions,
         "accuracy": acc, "finish_reason": out.finish1, "completion_text": out.text1},
        {"kind": "cot_turn2", **base, "candidate": out.candidate,
         "finish_reason": out.finish2, "completion_text": out.text2},
    ]

    # grade the turn-2 articulation (gpt-4.1 judge, held constant)
    g = await grade_one(helper, out.candidate, gold["canonical_articulation"],
                        gold["equivalence_class"], conv.examples[:N_JUDGE_EXAMPLES], model=JUDGE_MODEL)
    rows.append({"kind": "grade", **base, "candidate": out.candidate,
                 "grade": g["grade"], "extensionally_equivalent": g["extensionally_equivalent"],
                 "rationale": g["rationale"]})

    # consistency: compile the articulation (gpt-4.1 coder), score vs self + gold
    if do_compile:
        rows.append(await _consistency_row(conv, out, base, helper))
    return rows


async def _consistency_row(conv: Conversation, out: TurnOut, base: dict[str, Any],
                           helper: OpenAIClient) -> dict[str, Any]:
    rec = await helper.complete(COER_MODEL,
                                [{"role": "user", "content": COMPILE_PROMPT.format(art=out.candidate)}],
                                temperature=0.0, max_tokens=COMPILE_MAX_TOKENS, seed=API_SEED)
    code = extract_code(response_text(rec))
    row: dict[str, Any] = {"kind": "compile", **base, "coder": COER_MODEL,
                           "candidate": out.candidate, "code": code}
    if code is None:
        row["result"] = {"ok": False, "reason": "no_code_in_response"}
        return row
    texts = [q["text"] for q in conv.query_items]
    golds = [bool(q["label"]) for q in conv.query_items]
    items_gold = [{"text": t, "label": g} for t, g in zip(texts, golds)]
    # self labels: only items the model actually classified (drop None)
    items_self = [{"text": t, "label": p} for t, p in zip(texts, out.predictions) if p is not None]
    row["result_vs_gold"] = run_worker(code, items_gold)
    row["result_vs_self"] = run_worker(code, items_self) if items_self else {"ok": False, "reason": "no_self_labels"}
    return row


# --- estimate ------------------------------------------------------------------


def estimate_cost(model: str, convs: list[Conversation], do_compile: bool) -> float:
    total = 0.0
    claude = is_claude(model)
    for conv in convs:
        prompt_chars = sum(len(m["content"]) for m in conv.turn1_messages)
        t1_in = int(prompt_chars / 4) + PROMPT_TOKEN_OVERHEAD
        t1_out = CLAUDE_TURN1_MAX_TOKENS if claude else TURN1_MAX_TOKENS
        t2_in = t1_in + t1_out  # turn-1 prompt + echoed turn-1 output
        t2_out = CLAUDE_TURN2_MAX_TOKENS if claude else TURN2_MAX_TOKENS
        if claude:
            total += ((t1_in + t2_in) * CLAUDE_PRICE_IN + (t1_out + t2_out) * CLAUDE_PRICE_OUT) / 1e6
        else:
            total += cost_usd(model, t1_in, t1_out) + cost_usd(model, t2_in, t2_out)
        total += cost_usd(JUDGE_MODEL, 1300, 400)              # judge
        if do_compile:
            total += cost_usd(COER_MODEL, 200, COMPILE_MAX_TOKENS)  # compile worst case
    return total


# --- execution -----------------------------------------------------------------


async def execute(convs: list[Conversation], run: RunLog, model: str, spec_rules: dict[str, Any],
                  do_compile: bool, concurrency: int, cache_dir: str | Path,
                  api: Any | None, anthropic_api: Any | None) -> tuple[dict[str, Any], dict[str, Any]]:
    helper = OpenAIClient(concurrency=concurrency, cache_dir=cache_dir, api=api)
    ac = anthropic_api
    if is_claude(model) and ac is None:
        import anthropic
        ac = anthropic.AsyncAnthropic(max_retries=0)
    claude_meter = {"in": 0, "out": 0}
    sem = asyncio.Semaphore(concurrency)
    golds = {r: gold_for(spec_rules, r) for r in {c.rule_id for c in convs}}
    try:
        async def one(conv: Conversation) -> list[dict[str, Any]]:
            async with sem:
                return await process_conversation(conv, model, helper, golds[conv.rule_id],
                                                  do_compile, ac, claude_meter)
        futures = [asyncio.ensure_future(one(c)) for c in convs]
        try:
            for fut in asyncio.as_completed(futures):
                for row in await fut:
                    run.log_response(row)
        except BaseException:
            for f in futures:
                f.cancel()
            await asyncio.gather(*futures, return_exceptions=True)
            raise
        claude_usd = (claude_meter["in"] * CLAUDE_PRICE_IN + claude_meter["out"] * CLAUDE_PRICE_OUT) / 1e6
        cost = helper.cost.summary()
        cost["claude_usd"] = claude_usd
        cost["total_usd"] = cost.get("total_usd", 0.0) + claude_usd
        return helper.stats(), cost
    finally:
        await helper.aclose()
        if is_claude(model) and anthropic_api is None and ac is not None:
            await ac.close()


# --- CLI -----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="gpt-4.1 or claude-opus-4-8")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--rules", help="comma-separated rule_ids (default: the 5 targets)")
    sel.add_argument("--all-targets", action="store_true")
    p.add_argument("--run-seed", type=int, default=0)
    p.add_argument("--n-items", type=int, default=N_ITEMS)
    p.add_argument("--n-contexts", type=int, default=N_CONTEXTS)
    p.add_argument("--no-compile", action="store_true", help="skip the consistency compile+sandbox")
    p.add_argument("--max-cost", type=float, default=200.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--spec-extract", default="data/spec_extract.json")
    return p.parse_args(argv)


def resolve_rules(args: argparse.Namespace) -> list[str]:
    if args.rules:
        rules = [r.strip() for r in args.rules.split(",") if r.strip()]
        if not rules:
            raise DatasetError("--rules: empty")
        return rules
    return list(TARGET_RULES)


def main(argv: list[str] | None = None, api: Any | None = None, anthropic_api: Any | None = None) -> int:
    args = parse_args(argv)
    claude = is_claude(args.model)
    # price sanity: gpt models must be priced; claude uses hardcoded constants.
    if not claude:
        price_for(args.model)
    price_for(JUDGE_MODEL)
    if args.n_items % 2 != 0:
        print("ERROR: --n-items must be even (balanced select_queries)", file=sys.stderr)
        return 2

    rules = resolve_rules(args)
    spec_rules = load_spec_extract(args.spec_extract)
    for r in rules:
        gold_for(spec_rules, r)  # fail loud now if any target lacks gold
    do_compile = not args.no_compile

    convs, contexts_meta = build_conversations(rules, args.data_dir, args.run_seed,
                                              args.n_items, args.n_contexts)
    estimate = estimate_cost(args.model, convs, do_compile)
    cot_mode = "adaptive_thinking_effort_low" if claude else "prompted_cot"
    print(f"advance cost estimate: ${estimate:.4f} for {len(convs)} conversations "
          f"({len(rules)} rules x {args.n_contexts} contexts, {args.n_items} items each) "
          f"to {args.model} [cot_mode={cot_mode}] + judge/compile; --max-cost ${args.max_cost:.2f}")
    if estimate > args.max_cost:
        print(f"ABORT: estimate ${estimate:.4f} exceeds --max-cost ${args.max_cost:.2f}", file=sys.stderr)
        return 1

    config = {
        "task": "cot-same-session", "model": args.model, "cot_mode": cot_mode,
        "judge_model": JUDGE_MODEL, "coder_model": COER_MODEL if do_compile else None,
        "rules": rules, "run_seed": args.run_seed,
        "context_seeds": [args.run_seed + i for i in range(args.n_contexts)],
        "n_contexts": args.n_contexts, "n_items": args.n_items, "k_few_shot": K_FEW_SHOT,
        "temperature": TEMPERATURE, "api_seed": API_SEED,
        "turn1_max_tokens": CLAUDE_TURN1_MAX_TOKENS if claude else TURN1_MAX_TOKENS,
        "turn2_max_tokens": CLAUDE_TURN2_MAX_TOKENS if claude else TURN2_MAX_TOKENS,
        "compile_consistency": do_compile,
        "claude_price_per_mtok": {"in": CLAUDE_PRICE_IN, "out": CLAUDE_PRICE_OUT} if claude else None,
        "cot_same_session_template_hash": cot_same_session_template_hash(),
        "freeform_template_hash": freeform_template_hash(),
        "rubric_hash": rubric_hash(),
        "expected_conversations": len(convs), "data_dir": str(args.data_dir),
        "contexts": contexts_meta,
    }
    run = start_run(name=f"cot-same-session-{args.model}", config=config,
                    cost_estimate_usd=estimate, results_dir=args.results_dir)
    print(f"run dir: {run.run_dir}  ({len(convs)} conversations)")

    t0 = time.monotonic()
    client_stats, cost = asyncio.run(execute(convs, run, args.model, spec_rules, do_compile,
                                             args.concurrency, args.cache_dir, api, anthropic_api))
    wall = time.monotonic() - t0
    run.write_metrics({"task": "cot-same-session", "model": args.model, "cot_mode": cot_mode,
                       "wall_seconds": wall, "client_stats": client_stats, "cost": cost})
    run.finish(cost_actual_usd=cost["total_usd"], extra={"client_stats": client_stats, "cost": cost})
    print(f"done: {len(convs)} conversations, {wall:.1f}s, actual cost ${cost['total_usd']:.4f} "
          f"(estimate ${estimate:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
