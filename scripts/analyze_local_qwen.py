#!/usr/bin/env python
"""Recompute local Qwen Step-1 metrics from frozen JSONL outputs.

This is local/no-API and does not require transformers or a GPU. It is the
reproduction companion to scripts/local_hf_runner.py, which produced the raw
files on a GPU instance.

Run:
  .venv/bin/python scripts/analyze_local_qwen.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = RESULTS / "local-qwen-step1-metrics.json"


def _load_existing_config() -> dict[str, Any]:
    if not OUT.is_file():
        return {}
    try:
        return json.loads(OUT.read_text()).get("config", {})
    except json.JSONDecodeError:
        return {}


def _rule_from_path(path: Path) -> str:
    stem = path.stem
    prefix = "local-qwen-step1-"
    if not stem.startswith(prefix):
        raise ValueError(f"unexpected local Qwen path: {path}")
    return stem[len(prefix):]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    raw_paths = sorted(
        p
        for p in RESULTS.glob("local-qwen-step1-*.jsonl")
        if p.name != "local-qwen-step1-metrics.json"
    )
    if not raw_paths:
        raise SystemExit("no results/local-qwen-step1-*.jsonl files found")

    rules: dict[str, Any] = {}
    for path in raw_paths:
        rule = _rule_from_path(path)
        rows = _read_rows(path)
        by_ctx: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_ctx[int(row["context_index"])].append(row)

        per_context_accuracy: list[float] = []
        seeds: list[int] = []
        for ctx in sorted(by_ctx):
            ctx_rows = by_ctx[ctx]
            n_correct = sum(
                1 for row in ctx_rows
                if row.get("predicted") is not None and row.get("predicted") == row.get("true_label")
            )
            per_context_accuracy.append(n_correct / len(ctx_rows))
            seeds.append(int(ctx_rows[0].get("context_seed", ctx)))

        preds = Counter(
            "parse_failure" if row.get("predicted") is None
            else "true" if row.get("predicted") is True
            else "false"
            for row in rows
        )
        rules[rule] = {
            "rule_id": rule,
            "per_context_accuracy": per_context_accuracy,
            "pooled_accuracy": sum(per_context_accuracy) / len(per_context_accuracy),
            "n_items_per_context": len(next(iter(by_ctx.values()))),
            "n_contexts": len(by_ctx),
            "context_seeds": seeds,
            "n_parse_failures": preds["parse_failure"],
            "predictions": {
                "true": preds["true"],
                "false": preds["false"],
                "parse_failure": preds["parse_failure"],
            },
            "raw_file": str(path.relative_to(REPO)),
        }

    existing_config = _load_existing_config()
    metrics = {
        "config": existing_config,
        "rules": rules,
        "pooled_accuracy_table": {
            rule: block["pooled_accuracy"] for rule, block in sorted(rules.items())
        },
        "recomputed_from_raw_jsonl": True,
    }
    OUT.write_text(json.dumps(metrics, indent=2) + "\n")

    print("=== local Qwen Step-1 metrics (recomputed from JSONL) ===")
    for rule, block in sorted(rules.items()):
        print(f"  {rule:30s} {block['pooled_accuracy']:.3f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
