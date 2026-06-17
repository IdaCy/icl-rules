#!/usr/bin/env python
"""Elicit the free-form articulation for word_count_geq_8_v2 (deconfounded), both
models, to see whether the model now articulates LENGTH instead of the original
confabulated 'an adverb / prepositional phrase after the verb' — i.e. whether the
original confabulation was driven by the lexical confound (external-review P2b).

Makes paid API calls (set OPENAI_API_KEY). Small (3 contexts x 2 phrasings x 2 models).

Run:  python scripts/probe_wc8v2_articulation.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.client import OpenAIClient, response_text
from icl_articulation.contexts import load_items, sample_context
from icl_articulation.prices import cost_usd, price_for
from icl_articulation.prompts import FINAL_RULE_MARKER, render_freeform_articulation
from icl_articulation.runlog import start_run

REPO = Path(__file__).resolve().parent.parent
RULE = "word_count_geq_8_v2"
MODELS = ["gpt-4.1", "gpt-4.1-mini"]


def _extract(text: str) -> str:
    s = text.strip()
    if FINAL_RULE_MARKER in s:
        return s.rsplit(FINAL_RULE_MARKER, 1)[1].strip()
    return s


async def main() -> int:
    items = load_items(REPO / "data" / RULE / "items.jsonl")
    contexts = [sample_context(items, k=32, seed=s) for s in (0, 1, 2)]
    jobs = []
    for m in MODELS:
        for ci, ctx in enumerate(contexts):
            ex = [(it["text"], it["label"]) for it in ctx]
            for phr in (0, 1):
                jobs.append((m, ci, phr, render_freeform_articulation(ex, "direct", phr)))

    est = sum(cost_usd(m, sum(len(x["content"]) for x in msg) // 4 + 10, 120)
              for m, _, _, msg in jobs)
    print(f"advance cost estimate: ${est:.4f} for {len(jobs)} calls")
    run = start_run("step2-freeform-wc8v2-probe",
                    {"task": "wc8v2-articulation-probe", "rule": RULE, "models": MODELS,
                     "price_per_mtok": {m: price_for(m) for m in MODELS}},
                    cost_estimate_usd=est, results_dir=str(REPO / "results"))

    client = OpenAIClient(concurrency=4, cache_dir=str(REPO / "cache"))
    rows = []
    try:
        async def one(m, ci, phr, msg):
            rec = await client.complete(m, msg, temperature=0.0, max_tokens=120, seed=0)
            art = _extract(response_text(rec))
            return {"model": m, "context_index": ci, "phrasing": phr,
                    "articulation": art, **rec}
        futs = [asyncio.ensure_future(one(*j)) for j in jobs]
        for f in asyncio.as_completed(futs):
            r = await f
            run.log_response(r); rows.append(r)
        cost = client.cost.summary()
    finally:
        await client.aclose()
    run.finish(cost_actual_usd=cost["total_usd"], extra={"cost": cost})

    print(f"\n=== word_count_geq_8_v2 free-form articulations (direct) ${cost['total_usd']:.4f} ===")
    for m in MODELS:
        print(f"\n[{m}]")
        for r in sorted([x for x in rows if x["model"] == m], key=lambda x: (x["context_index"], x["phrasing"])):
            print(f"  ctx{r['context_index']} phr{r['phrasing']}: {r['articulation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
