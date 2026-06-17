#!/usr/bin/env python
"""Cross-family (Claude) runner for the cross-family extension — native `anthropic` SDK only.

Two pre-registered experiments (frozen design):

  B1  deconfound generality: replay the EXACT logged gpt-4.1 Step-1 prompts
      (request.messages) for word_count_geq_8 (v1/v2) and second_word_capitalized
      (v1/v2) to a strong cross-family model, so the model sees identical few-shot
      blocks and items. Two regimes: thinking OFF (the no-CoT comparison arm, all
      4 datasets) and thinking ON (effort=low confirmation arm, the two v1 sets).

  B2  third cross-family validator: for each item in the 3 LLM-rule pools, ask the
      rule's yes/no question and compare to the stored (OpenAI-consensus) label.

Subject/validator model references (`claude-…`, `gpt-…`) are scientific. Makes
paid API calls (set OPENAI_API_KEY and ANTHROPIC_API_KEY).

Resume-safe: results append to JSONL and already-done (rule,item,context,mode)
keys are skipped, so a re-run never double-spends.

Run:  python scripts/cross_family_runner.py --exp b1
                         python scripts/cross_family_runner.py --exp b1-think
                         python scripts/cross_family_runner.py --exp b2
                         python scripts/cross_family_runner.py --exp deconfounded-step1 --source-run results/deconfounded-...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "results"
OUT = RESULTS / "cross_family"

MODEL = "claude-opus-4-8"
PRICE_IN, PRICE_OUT = 5.00, 25.00  # USD / MTok (verified via claude-api skill 2026-06-04)
CONCURRENCY = 8

# B1: rule -> the gpt-4.1 Step-1 run whose exact prompts we replay.
B1_RUNS = {
    "word_count_geq_8": "results/step1-full-gpt-4.1-20260611T000748Z",
    "second_word_capitalized": "results/step1-full-gpt-4.1-20260611T000748Z",
    "word_count_geq_8_v2": "results/step1-full-gpt-4.1-20260612T011559Z",
    "second_word_capitalized_v2": "results/step1-full-gpt-4.1-20260612T213439Z",
}
B1_V1 = ["word_count_geq_8", "second_word_capitalized"]  # thinking-on confirmation arm

# B2: rule -> (yes/no question, mapping of yes->label-True)
B2_QUESTIONS = {
    "physically_impossible": "Is the following sentence describing something that is physically impossible — something that could not happen in the real world? Answer with exactly Yes or No.",
    "food_topic": "Is the following sentence about food, ingredients, cooking, or meal preparation? Answer with exactly Yes or No.",
    "positive_sentiment": "Does the following sentence express a positive sentiment or feeling? Answer with exactly Yes or No.",
}

_TF = re.compile(r"\b(true|false)\b", re.IGNORECASE)
_YN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_label(text: str, pat: re.Pattern, pos: str) -> bool | None:
    """First standalone token match -> bool; None if unparseable. pos is the True word."""
    if text is None:
        return None
    m = pat.search(text.strip())
    if not m:
        return None
    return m.group(1).lower() == pos


def done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        keys.add((r["rule"], r.get("item_id"), r.get("context_index"), r["mode"]))
    return keys


class Meter:
    def __init__(self) -> None:
        self.pin = self.pout = 0

    def add(self, u: Any) -> None:
        self.pin += getattr(u, "input_tokens", 0) or 0
        self.pout += getattr(u, "output_tokens", 0) or 0

    @property
    def usd(self) -> float:
        return self.pin / 1e6 * PRICE_IN + self.pout / 1e6 * PRICE_OUT


async def call_claude(client, system: str, user: str, *, thinking: bool, meter: Meter) -> tuple[str | None, str]:
    """Return (parsed_text, raw_text). thinking=False omits the param (off by
    default on opus-4-8); thinking=True uses adaptive + effort low."""
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if thinking:
        kwargs["max_tokens"] = 256
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": "low"}
    else:
        kwargs["max_tokens"] = 16
    resp = await client.messages.create(**kwargs)
    meter.add(resp.usage)
    text = ""
    for block in resp.content:
        if block.type == "text":
            text = block.text
    return text, text


async def run_b1(client, meter: Meter, *, thinking: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mode = "think_on" if thinking else "think_off"
    out_path = OUT / f"claude-step1-{mode}.jsonl"
    done = done_keys(out_path)
    rules = B1_V1 if thinking else list(B1_RUNS)
    sem = asyncio.Semaphore(CONCURRENCY)
    fh = out_path.open("a")

    async def one(rule: str, row: dict) -> None:
        msgs = row["request"]["messages"]
        system = msgs[0]["content"]
        user = msgs[-1]["content"]
        key = (rule, row.get("item_id"), row.get("context_index"), mode)
        if key in done:
            return
        async with sem:
            for attempt in range(5):
                try:
                    parsed_text, raw = await call_claude(client, system, user, thinking=thinking, meter=meter)
                    break
                except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(2 ** attempt)
        pred = parse_label(parsed_text, _TF, "true")
        rec = {
            "rule": rule, "mode": mode, "item_id": row.get("item_id"),
            "context_index": row.get("context_index"), "true_label": row.get("true_label"),
            "predicted": pred, "parse_ok": pred is not None,
            "raw": (raw or "")[:200],
        }
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    tasks = []
    for rule in rules:
        run_dir = REPO / B1_RUNS[rule]
        rows = [json.loads(l) for l in (run_dir / "responses.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r.get("rule_id") == rule]
        for row in rows:
            tasks.append(one(rule, row))
    print(f"[B1 {mode}] {len(tasks)} calls queued ({len(done)} already done)")
    await asyncio.gather(*tasks)
    fh.close()
    print(f"[B1 {mode}] wrote {out_path} | running cost ${meter.usd:.2f}")


async def run_b2(client, meter: Meter) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "claude-validate.jsonl"
    done = done_keys(out_path)
    sem = asyncio.Semaphore(CONCURRENCY)
    fh = out_path.open("a")

    async def one(rule: str, item: dict, q: str) -> None:
        key = (rule, item.get("id") or item.get("item_id"), None, "validate")
        if key in done:
            return
        user = f"{q}\n\nSentence: {item['text']}"
        async with sem:
            for attempt in range(5):
                try:
                    parsed_text, raw = await call_claude(client, "You are a careful annotator.", user, thinking=False, meter=meter)
                    break
                except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(2 ** attempt)
        yes = parse_label(parsed_text, _YN, "yes")  # yes -> label True
        rec = {
            "rule": rule, "mode": "validate",
            "item_id": item.get("id") or item.get("item_id"),
            "true_label": bool(item["label"]),
            "claude_says_true": yes, "parse_ok": yes is not None,
            "impossibility_type": item.get("slots_meta", {}).get("impossibility_type"),
            "base_id": item.get("base_id"),
            "raw": (raw or "")[:200],
        }
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    tasks = []
    for rule, q in B2_QUESTIONS.items():
        items = [json.loads(l) for l in (DATA / rule / "items.jsonl").read_text().splitlines() if l.strip()]
        for item in items:
            tasks.append(one(rule, item, q))
    print(f"[B2] {len(tasks)} calls queued ({len(done)} already done)")
    await asyncio.gather(*tasks)
    fh.close()
    print(f"[B2] wrote {out_path} | running cost ${meter.usd:.2f}")


async def run_deconfounded_step1(
    client,
    meter: Meter,
    *,
    source_run: Path,
    rules: list[str] | None,
    out_name: str,
) -> None:
    """Replay exact Step-1 prompt messages from a fresh Deconfounded OpenAI-compatible run."""
    if not (source_run / "responses.jsonl").is_file():
        raise SystemExit(f"--source-run has no responses.jsonl: {source_run}")
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / out_name
    done = done_keys(out_path)
    wanted = set(rules or [])
    rows = [
        json.loads(l)
        for l in (source_run / "responses.jsonl").read_text().splitlines()
        if l.strip()
    ]
    rows = [r for r in rows if not wanted or r.get("rule_id") in wanted]
    sem = asyncio.Semaphore(CONCURRENCY)
    fh = out_path.open("a")

    async def one(row: dict) -> None:
        rule = row["rule_id"]
        key = (rule, row.get("item_id"), row.get("context_index"), "deconfounded_step1")
        if key in done:
            return
        msgs = row["request"]["messages"]
        system = msgs[0]["content"]
        user = msgs[-1]["content"]
        async with sem:
            for attempt in range(5):
                try:
                    parsed_text, raw = await call_claude(
                        client, system, user, thinking=False, meter=meter
                    )
                    break
                except (
                    anthropic.RateLimitError,
                    anthropic.APIStatusError,
                    anthropic.APIConnectionError,
                ):
                    if attempt == 4:
                        raise
                    await asyncio.sleep(2 ** attempt)
        pred = parse_label(parsed_text, _TF, "true")
        rec = {
            "rule": rule,
            "mode": "deconfounded_step1",
            "source_run": str(source_run),
            "item_id": row.get("item_id"),
            "context_index": row.get("context_index"),
            "context_seed": row.get("context_seed"),
            "true_label": row.get("true_label"),
            "predicted": pred,
            "parse_ok": pred is not None,
            "raw": (raw or "")[:200],
        }
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    print(f"[Deconfounded step1] {len(rows)} calls queued from {source_run} ({len(done)} already done)")
    await asyncio.gather(*(one(row) for row in rows))
    fh.close()
    print(f"[Deconfounded step1] wrote {out_path} | running cost ${meter.usd:.2f}")


async def main() -> None:
    load_dotenv(REPO / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set (expected in env or .env)")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp",
        required=True,
        choices=["b1", "b1-think", "b2", "smoke", "deconfounded-step1"],
    )
    ap.add_argument("--source-run", help="Step-1 run directory to replay for --exp deconfounded-step1")
    ap.add_argument("--rules", help="optional comma-separated rule ids for --exp deconfounded-step1")
    ap.add_argument(
        "--out-name",
        default="claude-deconfounded-step1.jsonl",
        help="output filename under results/cross_family for --exp deconfounded-step1",
    )
    args = ap.parse_args()
    client = anthropic.AsyncAnthropic(max_retries=0)
    meter = Meter()
    try:
        if args.exp == "smoke":
            t, _ = await call_claude(client, "You are a precise classifier.",
                                     "Answer with exactly True or False.\n\nInput: the cat sat on the mat\nLabel:",
                                     thinking=False, meter=meter)
            print("SMOKE thinking-off ->", repr(t))
            t2, _ = await call_claude(client, "You are a precise classifier.",
                                      "Answer with exactly True or False.\n\nInput: THE DOG RAN\nLabel:",
                                      thinking=True, meter=meter)
            print("SMOKE thinking-on  ->", repr(t2))
            print(f"smoke cost ${meter.usd:.4f}")
        elif args.exp == "b1":
            await run_b1(client, meter, thinking=False)
        elif args.exp == "b1-think":
            await run_b1(client, meter, thinking=True)
        elif args.exp == "b2":
            await run_b2(client, meter)
        elif args.exp == "deconfounded-step1":
            if not args.source_run:
                raise SystemExit("--exp deconfounded-step1 requires --source-run")
            rules = [r.strip() for r in args.rules.split(",") if r.strip()] if args.rules else None
            await run_deconfounded_step1(
                client,
                meter,
                source_run=Path(args.source_run),
                rules=rules,
                out_name=args.out_name,
            )
    finally:
        await client.close()
        (OUT).mkdir(parents=True, exist_ok=True)
        (OUT / f"_cost_{args.exp}.json").write_text(json.dumps(
            {"model": MODEL, "input_tokens": meter.pin, "output_tokens": meter.pout, "usd": meter.usd}, indent=2))
        print(f"[{args.exp}] FINAL cost ${meter.usd:.2f}  (in {meter.pin}, out {meter.pout})")


if __name__ == "__main__":
    asyncio.run(main())
