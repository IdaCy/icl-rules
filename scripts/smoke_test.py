#!/usr/bin/env python
"""P1 smoke test. Makes paid API calls (set OPENAI_API_KEY).

What it does:
1. prints the cost estimate BEFORE any call (must be < $0.05);
2. lists available models, checks gpt-4.1 / gpt-4.1-mini / gpt-4o;
3. sends ~20 trivial classification calls (contains-a-digit toy rule) to
   gpt-4.1 with max_tokens=2, T=0, top_logprobs=5; verifies single-token
   True/False answers + logprobs, prints the exact label tokenization
   ("True" vs " True");
4. measures wall-clock throughput at concurrency 16 and reports the 429 count;
5. writes a proper results/<run_id>/ via runlog with cost actuals.

Usage:  python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.client import OpenAIClient, first_token_logprobs, response_text
from icl_articulation.prices import cost_usd
from icl_articulation.prompts import render_step1, step1_template_hash
from icl_articulation.runlog import start_run
from icl_articulation.stats import wilson_ci

MODEL = "gpt-4.1"
WANTED_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4o"]
CONCURRENCY = 16
SEED = 0

# Hardcoded toy task: True iff the sentence contains a digit.
FEW_SHOT: list[tuple[str, bool]] = [
    ("The train leaves at 7 sharp.", True),
    ("She bought fresh bread this morning.", False),
    ("There are 12 chairs in the room.", True),
    ("The cat slept on the warm sofa.", False),
    ("He scored 99 points last night.", True),
    ("Rain fell quietly over the hills.", False),
    ("My address has a 4 in it.", True),
    ("They walked home after the show.", False),
]

QUERIES: list[tuple[str, bool]] = [
    ("The recipe needs 3 eggs.", True),
    ("Birds sang in the garden.", False),
    ("Call me at 5 tomorrow.", True),
    ("The river froze last winter.", False),
    ("Page 42 has the answer.", True),
    ("She painted the fence green.", False),
    ("He ran 10 miles today.", True),
    ("The soup smelled wonderful.", False),
    ("Gate 8 is now boarding.", True),
    ("The dog chased its tail.", False),
    ("I owe you 20 dollars.", True),
    ("Clouds drifted across the sky.", False),
    ("Room 6 is down the hall.", True),
    ("The children built a sandcastle.", False),
    ("Add 2 cups of flour.", True),
    ("The lamp flickered at night.", False),
    ("Bus 14 stops here daily.", True),
    ("Her voice echoed in the hall.", False),
    ("Take exit 9 on the left.", True),
    ("The garden bloomed in spring.", False),
]


def estimate_cost_usd() -> float:
    """Advance estimate: ~4 chars/token heuristic on rendered prompts + 2 completion tokens."""
    total = 0.0
    for query, _ in QUERIES:
        messages = render_step1(FEW_SHOT, query)
        prompt_chars = sum(len(m["content"]) for m in messages)
        prompt_tokens = prompt_chars / 4 + 10  # + chat formatting overhead
        total += cost_usd(MODEL, int(prompt_tokens), 2)
    return total


async def main() -> int:
    estimate = estimate_cost_usd()
    print(f"estimated cost: ${estimate:.4f} for {len(QUERIES)} calls to {MODEL}")
    if estimate >= 0.05:
        print("ERROR: estimate >= $0.05 — smoke test should be trivial; aborting.")
        return 1

    client = OpenAIClient(concurrency=CONCURRENCY)

    print("\n--- model availability ---")
    available = await client.list_models()
    for wanted in WANTED_MODELS:
        ok = wanted in available
        snapshots = [m for m in available if m.startswith(wanted + "-")]
        print(f"  {wanted}: {'PRESENT' if ok else 'MISSING'}"
              + (f" (snapshots: {', '.join(snapshots)})" if snapshots else ""))

    run = start_run(
        name="smoke",
        config={
            "model": MODEL,
            "seed": SEED,
            "temperature": 0.0,
            "max_tokens": 2,
            "logprobs": True,
            "top_logprobs": 5,
            "concurrency": CONCURRENCY,
            "n_calls": len(QUERIES),
            "task": "toy contains-a-digit, 8-shot",
            "template_hash": step1_template_hash(),
        },
        cost_estimate_usd=estimate,
    )
    print(f"\nrun dir: {run.run_dir}")

    print(f"\n--- {len(QUERIES)} classification calls at concurrency {CONCURRENCY} ---")
    t0 = time.monotonic()
    records = await asyncio.gather(
        *(
            client.complete(
                MODEL,
                render_step1(FEW_SHOT, query),
                temperature=0.0,
                max_tokens=2,
                logprobs=True,
                top_logprobs=5,
                seed=SEED,
            )
            for query, _ in QUERIES
        )
    )
    wall = time.monotonic() - t0

    n_correct = 0
    n_parse_ok = 0
    n_logprobs_ok = 0
    label_tokens: set[str] = set()
    for (query, truth), record in zip(QUERIES, records):
        run.log_response({"query": query, "true_label": truth, **record})
        text = response_text(record).strip()
        parsed = text in ("True", "False")
        n_parse_ok += parsed
        if parsed and (text == "True") == truth:
            n_correct += 1
        entry = first_token_logprobs(record)
        if entry is not None:
            n_logprobs_ok += 1
            label_tokens.add(entry["token"])

    print(f"  parseable True/False answers: {n_parse_ok}/{len(QUERIES)}")
    print(f"  correct: {n_correct}/{len(QUERIES)}"
          f"  (wilson 95% CI: {tuple(round(x, 3) for x in wilson_ci(n_correct, len(QUERIES)))})")
    print(f"  responses with logprobs: {n_logprobs_ok}/{len(QUERIES)}")
    print(f"  exact label tokenizations seen: {sorted(repr(t) for t in label_tokens)}")
    if records:
        entry = first_token_logprobs(records[0])
        if entry is not None:
            tops = [(t["token"], round(t["logprob"], 4)) for t in entry.get("top_logprobs", [])]
            print(f"  top-5 at answer position (call 0): {[(repr(t), lp) for t, lp in tops]}")

    stats = client.stats()
    cost_actual = client.cost.total_usd
    throughput = len(QUERIES) / wall if wall > 0 else float("inf")
    print(f"\n  wall clock: {wall:.2f}s -> {throughput:.2f} calls/s at concurrency {CONCURRENCY}")
    print(f"  429s: {stats['n_429']}  retryable errors: {stats['retryable_errors']}"
          f"  cache hits: {stats['cache_hits']}")
    print(f"  actual cost: ${cost_actual:.4f} (estimate was ${estimate:.4f})")
    if stats["cache_hits"]:
        print("  NOTE: cache hits present — throughput/cost numbers are not a clean API measurement.")

    run.write_metrics(
        {
            "n_calls": len(QUERIES),
            "n_parse_ok": n_parse_ok,
            "n_correct": n_correct,
            "n_logprobs_ok": n_logprobs_ok,
            "label_tokens_seen": sorted(label_tokens),
            "wall_seconds": wall,
            "throughput_calls_per_s": throughput,
            "client_stats": stats,
            "cost": client.cost.summary(),
            "models_wanted": WANTED_MODELS,
            "models_present": [m for m in WANTED_MODELS if m in available],
        }
    )
    run.finish(
        cost_actual_usd=cost_actual,
        extra={"client_stats": stats, "cost": client.cost.summary()},
    )
    await client.aclose()

    ok = n_parse_ok == len(QUERIES) and n_logprobs_ok == len(QUERIES)
    print(f"\nsmoke test {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
