#!/usr/bin/env python
"""In-session per-item classify-then-articulate. Makes paid API calls (set OPENAI_API_KEY).

The corrected redo of the earlier batched variant (which batched all items into one prompt). Two
experiments, run SEPARATELY, both using the NORMAL per-item completion format
(NOT a batch) inside ONE preserved conversation:

  Exp 1 (--no-cot): classify each held-out item ONE AT A TIME the normal way
        (bare True/False, no reasoning — turn 1 is a byte-for-byte Step-1 prompt),
        keeping the session open, then ask "what rule did you use?" directly.
  Exp 2 (--cot):    the SAME structure, but reasoning is allowed BOTH while
        classifying each item AND while stating the rule.

The conversation is GROWING: turn k re-sends the full prior conversation (the
model sees its own earlier answers — this is the "preserve state" the user asked
for). Within a conversation the turns are strictly sequential; across the 15
conversations (5 rules x 3 contexts) they run concurrently.

CoT is in Exp 2 ONLY. Subjects: gpt-4.1 (prompted CoT) and claude-opus-4-8
(thinking OFF for Exp 1; adaptive thinking effort=low for Exp 2; assistant
content blocks echoed back unchanged each turn). The judge is always gpt-4.1.
Advance estimate printed first + --max-cost gate; rows logged per turn (crash-safe).

Example:
  python scripts/run_insession_articulation.py --no-cot --model gpt-4.1 --rules food_topic
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "src"))

from icl_articulation.client import OpenAIClient, response_text
from icl_articulation.contexts import DatasetError, load_items, sample_context, select_queries
from icl_articulation.grading import gold_for, grade_one, load_spec_extract, rubric_hash
from icl_articulation.prices import cost_usd, price_for
from icl_articulation.prompts import (
    STEP1_SYSTEM,
    extract_rule,
    freeform_template_hash,
    insession_articulation_user,
    insession_followup_user,
    insession_template_hash,
    render_insession_turn1,
)
from icl_articulation.runlog import RunLog, start_run
from run_step1 import parse_label  # reuse the exact Step-1 no-CoT prefix parser

TARGET_RULES = [
    "word_count_geq_8", "second_word_capitalized", "physically_impossible",
    "food_topic", "positive_sentiment",
]

K_FEW_SHOT = 32
TEMPERATURE = 0.0
API_SEED = 0
N_ITEMS = 16
N_CONTEXTS = 3
N_JUDGE_EXAMPLES = 12
JUDGE_MODEL = "gpt-4.1"
# per-turn max_tokens
CLASSIFY_MAX_NOCOT = 2
CLASSIFY_MAX_COT = 300
ARTICULATE_MAX_NOCOT = 90
ARTICULATE_MAX_COT = 600
PROMPT_TOKEN_OVERHEAD = 10

CLAUDE_PRICE_IN, CLAUDE_PRICE_OUT = 5.00, 25.00
CLAUDE_CLASSIFY_MAX_NOCOT = 16
CLAUDE_CLASSIFY_MAX_COT = 1200
CLAUDE_ARTICULATE_MAX = 1500


def is_claude(model: str) -> bool:
    return model.startswith("claude")


# --- CoT single-item parser ----------------------------------------------------
_FINAL_ANS_RE = re.compile(r"Answer\s*:?\s*\**\s*(True|False)\b", re.IGNORECASE)
_TF_RE = re.compile(r"\b(True|False)\b", re.IGNORECASE)


def parse_cot_label(text: str) -> bool | None:
    """Final True/False after CoT for ONE item: prefer the demanded 'Answer: X'
    (last occurrence), else the last bare True/False token, else None (=incorrect)."""
    text = text or ""
    m = _FINAL_ANS_RE.findall(text)
    if m:
        return m[-1].lower() == "true"
    toks = _TF_RE.findall(text)
    return toks[-1].lower() == "true" if toks else None


# --- conversations -------------------------------------------------------------
@dataclass
class Conversation:
    rule_id: str
    context_index: int
    context_seed: int
    query_items: list[dict[str, Any]]
    examples: list[tuple[str, bool]]


def build_conversations(rules, data_dir, run_seed, n_items, n_contexts):
    import random
    data_dir = Path(data_dir)
    convs: list[Conversation] = []
    meta: dict[str, Any] = {}
    for rule_id in rules:
        items = load_items(data_dir / rule_id / "items.jsonl")
        queries = select_queries(items, "held_out", n_items)
        random.Random(run_seed * 1000 + 7).shuffle(queries)  # break class-block order
        meta[rule_id] = []
        for ctx in range(n_contexts):
            seed = run_seed + ctx
            context = sample_context(items, k=K_FEW_SHOT, seed=seed)
            examples = [(it["text"], bool(it["label"])) for it in context]
            meta[rule_id].append({"context_index": ctx, "seed": seed,
                                  "query_item_ids": [q["item_id"] for q in queries]})
            convs.append(Conversation(rule_id, ctx, seed, queries, examples))
    return convs, meta


# --- per-conversation execution (sequential turns) -----------------------------
async def _classify_openai(client, model, messages, cot):
    rec = await client.complete(model, messages, temperature=TEMPERATURE,
                                max_tokens=(CLASSIFY_MAX_COT if cot else CLASSIFY_MAX_NOCOT), seed=API_SEED)
    text = response_text(rec)
    fin = _finish_openai(rec)
    return text, fin, (parse_cot_label(text) if cot else parse_label(text))


async def _articulate_openai(client, model, messages, cot):
    rec = await client.complete(model, messages, temperature=TEMPERATURE,
                                max_tokens=(ARTICULATE_MAX_COT if cot else ARTICULATE_MAX_NOCOT), seed=API_SEED)
    return response_text(rec), _finish_openai(rec)


def _finish_openai(rec):
    try:
        return rec["response"]["choices"][0].get("finish_reason", "")
    except Exception:  # noqa: BLE001
        return ""


async def _claude_create(ac, **kwargs):
    try:
        import anthropic
        retry = (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError)
    except ModuleNotFoundError:
        retry = ()
    for attempt in range(5):
        try:
            return await ac.messages.create(**kwargs)
        except retry:
            if attempt == 4:
                raise
            await asyncio.sleep(2 ** attempt)


def _claude_text(resp):
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _claude_kwargs(cot, *, max_tokens):
    kw = {"model": "claude-opus-4-8", "system": STEP1_SYSTEM, "max_tokens": max_tokens}
    if cot:
        kw["thinking"] = {"type": "adaptive"}
        kw["output_config"] = {"effort": "low"}
    return kw


async def process_conversation(conv, model, helper, gold, cot, ac, claude_meter, experiment):
    base = {"rule_id": conv.rule_id, "model": model, "experiment": experiment, "cot": cot,
            "context_index": conv.context_index, "context_seed": conv.context_seed}
    rows: list[dict[str, Any]] = []
    claude = is_claude(model)

    if claude:
        first = render_insession_turn1(conv.examples, conv.query_items[0]["text"], cot=cot)
        messages = [{"role": "user", "content": first[1]["content"]}]
        prev_content = None
        for k, item in enumerate(conv.query_items):
            if k > 0:
                messages = messages + [{"role": "assistant", "content": prev_content},
                                       {"role": "user", "content": insession_followup_user(item["text"], cot=cot)}]
            resp = await _claude_create(ac, messages=messages,
                                        **_claude_kwargs(cot, max_tokens=(CLAUDE_CLASSIFY_MAX_COT if cot else CLAUDE_CLASSIFY_MAX_NOCOT)))
            _meter(claude_meter, resp)
            text = _claude_text(resp)
            prev_content = resp.content
            rows.append(_classify_row(base, k, item, (parse_cot_label(text) if cot else parse_label(text)),
                                      getattr(resp, "stop_reason", "") or "", text))
        messages = messages + [{"role": "assistant", "content": prev_content},
                               {"role": "user", "content": insession_articulation_user(cot=cot)}]
        resp = await _claude_create(ac, messages=messages, **_claude_kwargs(cot, max_tokens=CLAUDE_ARTICULATE_MAX))
        _meter(claude_meter, resp)
        art_text, art_fin = _claude_text(resp), getattr(resp, "stop_reason", "") or ""
    else:
        messages = render_insession_turn1(conv.examples, conv.query_items[0]["text"], cot=cot)
        prev_text = None
        for k, item in enumerate(conv.query_items):
            if k > 0:
                messages = messages + [{"role": "assistant", "content": prev_text},
                                       {"role": "user", "content": insession_followup_user(item["text"], cot=cot)}]
            text, fin, pred = await _classify_openai(helper, model, messages, cot)
            prev_text = text
            rows.append(_classify_row(base, k, item, pred, fin, text))
        messages = messages + [{"role": "assistant", "content": prev_text},
                               {"role": "user", "content": insession_articulation_user(cot=cot)}]
        art_text, art_fin = await _articulate_openai(helper, model, messages, cot)

    candidate = extract_rule(art_text)
    rows.append({"kind": "articulate", **base, "candidate": candidate,
                 "finish_reason": art_fin, "completion_text": art_text})
    g = await grade_one(helper, candidate, gold["canonical_articulation"],
                        gold["equivalence_class"], conv.examples[:N_JUDGE_EXAMPLES], model=JUDGE_MODEL)
    rows.append({"kind": "grade", **base, "candidate": candidate, "grade": g["grade"],
                 "extensionally_equivalent": g["extensionally_equivalent"], "rationale": g["rationale"]})
    return rows


def _classify_row(base, turn_index, item, pred, finish, text):
    gold = bool(item["label"])
    return {"kind": "classify", **base, "turn_index": turn_index,
            "item_id": item["item_id"], "base_id": item["base_id"], "text": item["text"],
            "true_label": gold, "predicted": pred, "parse_ok": pred is not None,
            "correct": pred is not None and pred == gold,
            "finish_reason": finish, "completion_text": text}


def _meter(meter, resp):
    u = getattr(resp, "usage", None)
    meter["in"] += getattr(u, "input_tokens", 0) or 0
    meter["out"] += getattr(u, "output_tokens", 0) or 0


# --- cost estimate (simulate the growing conversation) -------------------------
def estimate_cost(model, convs, cot):
    claude = is_claude(model)
    ans_chars = 6 if not cot else (700 if not claude else 1400)   # representative completion size
    out_tok = 2 if not cot else (180 if not claude else 350)
    art_out = 60 if not cot else 350
    total = 0.0
    for conv in convs:
        msgs = render_insession_turn1(conv.examples, conv.query_items[0]["text"], cot=cot)
        running = sum(len(m["content"]) for m in msgs)
        in_tok = out_tok_sum = 0.0
        for k, item in enumerate(conv.query_items):
            if k > 0:
                running += ans_chars + len(insession_followup_user(item["text"], cot=cot))
            in_tok += running / 4 + PROMPT_TOKEN_OVERHEAD
            out_tok_sum += out_tok
        running += ans_chars + len(insession_articulation_user(cot=cot))
        in_tok += running / 4 + PROMPT_TOKEN_OVERHEAD
        out_tok_sum += art_out
        if claude:
            total += (in_tok * CLAUDE_PRICE_IN + out_tok_sum * CLAUDE_PRICE_OUT) / 1e6
        else:
            total += cost_usd(model, int(in_tok), int(out_tok_sum))
        total += cost_usd(JUDGE_MODEL, 1300, 400)  # judge
    return total


# --- execution -----------------------------------------------------------------
async def execute(convs, run, model, spec_rules, cot, experiment, concurrency, cache_dir, api, anthropic_api):
    helper = OpenAIClient(concurrency=concurrency, cache_dir=cache_dir, api=api)
    ac = anthropic_api
    if is_claude(model) and ac is None:
        import anthropic
        ac = anthropic.AsyncAnthropic(max_retries=0)
    claude_meter = {"in": 0, "out": 0}
    sem = asyncio.Semaphore(concurrency)
    golds = {r: gold_for(spec_rules, r) for r in {c.rule_id for c in convs}}
    try:
        async def one(conv):
            async with sem:
                return await process_conversation(conv, model, helper, golds[conv.rule_id],
                                                  cot, ac, claude_meter, experiment)
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
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="gpt-4.1 or claude-opus-4-8")
    cot = p.add_mutually_exclusive_group(required=True)
    cot.add_argument("--cot", dest="cot", action="store_true", help="Exp 2: reasoning on (classify + articulate)")
    cot.add_argument("--no-cot", dest="cot", action="store_false", help="Exp 1: no reasoning anywhere")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--rules")
    sel.add_argument("--all-targets", action="store_true")
    p.add_argument("--run-seed", type=int, default=0)
    p.add_argument("--n-items", type=int, default=N_ITEMS)
    p.add_argument("--n-contexts", type=int, default=N_CONTEXTS)
    p.add_argument("--max-cost", type=float, default=200.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--spec-extract", default="data/spec_extract.json")
    return p.parse_args(argv)


def resolve_rules(args):
    if args.rules:
        rules = [r.strip() for r in args.rules.split(",") if r.strip()]
        if not rules:
            raise DatasetError("--rules: empty")
        return rules
    return list(TARGET_RULES)


def main(argv=None, api=None, anthropic_api=None):
    args = parse_args(argv)
    claude = is_claude(args.model)
    if not claude:
        price_for(args.model)
    price_for(JUDGE_MODEL)
    if args.n_items % 2 != 0:
        print("ERROR: --n-items must be even", file=sys.stderr)
        return 2
    rules = resolve_rules(args)
    spec_rules = load_spec_extract(args.spec_extract)
    for r in rules:
        gold_for(spec_rules, r)

    convs, meta = build_conversations(rules, args.data_dir, args.run_seed, args.n_items, args.n_contexts)
    cot = args.cot
    experiment = "exp2-cot" if cot else "exp1-no-cot"
    cot_mode = ("adaptive_thinking_effort_low" if claude else "prompted_cot") if cot else "none"
    estimate = estimate_cost(args.model, convs, cot)
    print(f"advance cost estimate: ${estimate:.4f} for {len(convs)} conversations "
          f"({len(rules)} rules x {args.n_contexts} contexts x {args.n_items} items, sequential turns) "
          f"to {args.model} [{experiment}, cot_mode={cot_mode}] + judge; --max-cost ${args.max_cost:.2f}")
    if estimate > args.max_cost:
        print(f"ABORT: estimate ${estimate:.4f} exceeds --max-cost ${args.max_cost:.2f}", file=sys.stderr)
        return 1

    config = {
        "task": "insession-articulation", "experiment": experiment, "cot": cot, "cot_mode": cot_mode,
        "model": args.model, "judge_model": JUDGE_MODEL, "rules": rules, "run_seed": args.run_seed,
        "context_seeds": [args.run_seed + i for i in range(args.n_contexts)],
        "n_contexts": args.n_contexts, "n_items": args.n_items, "k_few_shot": K_FEW_SHOT,
        "temperature": TEMPERATURE, "api_seed": API_SEED,
        "classify_max_tokens": (CLAUDE_CLASSIFY_MAX_COT if claude and cot else
                                CLAUDE_CLASSIFY_MAX_NOCOT if claude else
                                CLASSIFY_MAX_COT if cot else CLASSIFY_MAX_NOCOT),
        "claude_price_per_mtok": {"in": CLAUDE_PRICE_IN, "out": CLAUDE_PRICE_OUT} if claude else None,
        "insession_template_hash": insession_template_hash(),
        "freeform_template_hash": freeform_template_hash(), "rubric_hash": rubric_hash(),
        "expected_conversations": len(convs), "data_dir": str(args.data_dir), "contexts": meta,
    }
    run = start_run(name=f"insession-{experiment}-{args.model}", config=config,
                    cost_estimate_usd=estimate, results_dir=args.results_dir)
    print(f"run dir: {run.run_dir}  ({len(convs)} conversations)")

    t0 = time.monotonic()
    client_stats, cost = asyncio.run(execute(convs, run, args.model, spec_rules, cot, experiment,
                                             args.concurrency, args.cache_dir, api, anthropic_api))
    wall = time.monotonic() - t0
    run.write_metrics({"task": "insession-articulation", "experiment": experiment, "model": args.model,
                       "wall_seconds": wall, "client_stats": client_stats, "cost": cost})
    run.finish(cost_actual_usd=cost["total_usd"], extra={"client_stats": client_stats, "cost": cost})
    print(f"done: {len(convs)} conversations, {wall:.1f}s, actual cost ${cost['total_usd']:.4f} "
          f"(estimate ${estimate:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
