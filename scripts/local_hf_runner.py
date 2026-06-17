#!/usr/bin/env python
"""Local open-weights step-1 classification runner (SECOND-FAMILY robustness).

Replicates the OpenAI step-1 in-context CLASSIFICATION run with an open model
(default Qwen/Qwen2.5-7B-Instruct), REUSING the exact same prompt rendering,
context sampling, and answer parsing as scripts/run_step1.py so the format
matches the OpenAI runs bit-for-bit:

  - data loaded via icl_articulation.contexts.load_items
  - query items via select_queries(items, "held_out", 120)  (60 True / 60 False)
  - 3 contexts, each k=32 (16/16) sampled with sample_context(seed=run_seed+i)
  - messages via icl_articulation.prompts.render_step1(examples, query)
  - chat template applied with the model's tokenizer
  - GREEDY decode, max_new_tokens=2, NO chain-of-thought
  - answer parsed with the SAME parse_label() as run_step1.py (case-insensitive
    True/False prefix; parse failure -> counts as incorrect)

GPU efficiency: prompts are generated in batches (default 24) with left
padding, all on cuda in bf16.

Raw output: results/local-qwen-step1-<rule>.jsonl, one line per
(rule, context_index, item):
  {rule_id, context_index, item_id, text, true_label, predicted, completion_text}

A per-rule + overall metrics json is written to
results/local-qwen-step1-metrics.json. Pooled accuracy = mean of the 3
per-context accuracies (same definition as run_step1.py compute_metrics).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.contexts import load_items, sample_context, select_queries
from icl_articulation.prompts import render_step1

# parse_label lives in scripts/run_step1.py; import it from there so the answer
# parsing is byte-identical to the OpenAI runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_step1 import parse_label  # type: ignore  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

K_FEW_SHOT = 32
N_CONTEXTS = 3
N_ITEMS = 120
SPLIT = "held_out"

RULES = [
    "physically_impossible",
    "food_topic",
    "all_lowercase",
    "second_word_capitalized",
    "second_word_capitalized_v2",
    "word_count_geq_8",
    "word_count_geq_8_v2",
    "mentions_animal",
    "positive_sentiment",
]


def build_prompt_text(tokenizer, messages: list[dict[str, str]]) -> str:
    """Apply the model chat template; add the generation prompt so the model
    emits the assistant turn. The step-1 user message already ends in 'Label:'
    so the model continues with True/False."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Greedy decode a batch of fully-rendered prompt strings; return the
    NEW text (completion only) per prompt."""
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,  # chat template already added them
    ).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def run_rule(
    rule_id: str,
    model,
    tokenizer,
    data_dir: Path,
    results_dir: Path,
    run_seed: int,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    items = load_items(data_dir / rule_id / "items.jsonl")
    queries = select_queries(items, SPLIT, N_ITEMS)

    out_path = results_dir / f"local-qwen-step1-{rule_id}.jsonl"
    rows: list[dict[str, Any]] = []
    per_ctx_acc: list[float] = []

    with out_path.open("w", encoding="utf-8") as fh:
        for ctx_index in range(N_CONTEXTS):
            seed = run_seed + ctx_index
            context = sample_context(items, k=K_FEW_SHOT, seed=seed)
            examples = [(it["text"], it["label"]) for it in context]

            prompts = [
                build_prompt_text(tokenizer, render_step1(examples, q["text"]))
                for q in queries
            ]

            completions: list[str] = []
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                completions.extend(generate_batch(model, tokenizer, batch, max_new_tokens))

            n_correct = 0
            for q, comp in zip(queries, completions):
                predicted = parse_label(comp)
                correct = predicted is not None and predicted == q["label"]
                n_correct += int(correct)
                row = {
                    "rule_id": rule_id,
                    "context_index": ctx_index,
                    "context_seed": seed,
                    "item_id": q["item_id"],
                    "text": q["text"],
                    "true_label": q["label"],
                    "predicted": predicted,
                    "completion_text": comp,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            fh.flush()
            acc = n_correct / len(queries)
            per_ctx_acc.append(acc)
            print(
                f"    [{rule_id}] ctx{ctx_index} seed={seed}: acc={acc:.3f} "
                f"({n_correct}/{len(queries)})",
                flush=True,
            )

    pooled = sum(per_ctx_acc) / len(per_ctx_acc)
    n_parse_fail = sum(1 for r in rows if r["predicted"] is None)
    n_pred_true = sum(1 for r in rows if r["predicted"] is True)
    n_pred_false = sum(1 for r in rows if r["predicted"] is False)
    print(
        f"  {rule_id}: per-context {['%.3f' % a for a in per_ctx_acc]} "
        f"pooled={pooled:.3f}  parse_fail={n_parse_fail} "
        f"pred(T/F/none)={n_pred_true}/{n_pred_false}/{n_parse_fail}",
        flush=True,
    )
    return {
        "rule_id": rule_id,
        "per_context_accuracy": per_ctx_acc,
        "pooled_accuracy": pooled,
        "n_items_per_context": len(queries),
        "n_contexts": N_CONTEXTS,
        "context_seeds": [run_seed + i for i in range(N_CONTEXTS)],
        "n_parse_failures": n_parse_fail,
        "predictions": {"true": n_pred_true, "false": n_pred_false, "parse_failure": n_parse_fail},
        "raw_file": str(out_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--rules", default=",".join(RULES), help="comma-separated rule_ids")
    p.add_argument("--run-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--max-new-tokens", type=int, default=2)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model {args.model} ...", flush=True)
    t_load = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # decoder-only batched generation needs LEFT padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print(
        f"model loaded in {time.monotonic() - t_load:.1f}s; "
        f"device={model.device} dtype={next(model.parameters()).dtype}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(
            f"cuda mem after load: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated",
            flush=True,
        )

    config = {
        "model": args.model,
        "decode": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "greedy": True,
            "no_cot": True,
        },
        "k_few_shot": K_FEW_SHOT,
        "n_contexts": N_CONTEXTS,
        "n_items_per_context": N_ITEMS,
        "query_split": SPLIT,
        "run_seed": args.run_seed,
        "context_seeds": [args.run_seed + i for i in range(N_CONTEXTS)],
        "batch_size": args.batch_size,
        "rules": rules,
        "transformers": __import__("transformers").__version__,
        "torch": torch.__version__,
    }

    results: dict[str, Any] = {}
    t0 = time.monotonic()
    for rule_id in rules:
        print(f"--- running rule {rule_id} ---", flush=True)
        tr = time.monotonic()
        results[rule_id] = run_rule(
            rule_id, model, tokenizer, data_dir, results_dir,
            args.run_seed, args.batch_size, args.max_new_tokens,
        )
        results[rule_id]["wall_seconds"] = time.monotonic() - tr
    wall = time.monotonic() - t0

    metrics = {
        "config": config,
        "wall_seconds": wall,
        "rules": results,
        "pooled_accuracy_table": {r: results[r]["pooled_accuracy"] for r in rules},
    }
    metrics_path = results_dir / "local-qwen-step1-metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n=== DONE in {wall:.1f}s ===", flush=True)
    print(f"metrics: {metrics_path}", flush=True)
    print("pooled accuracy per rule:", flush=True)
    for r in rules:
        print(f"  {r}: {results[r]['pooled_accuracy']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
