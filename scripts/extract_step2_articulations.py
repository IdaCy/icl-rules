#!/usr/bin/env python
"""Extract representative Step-2 direct articulations for Step-3.

The Step-3 runner needs a model's own stated rule for ARM 2. For Deconfounded reruns this
must come from the fresh Step-2 free-form run, not from legacy hard-coded
articulation constants. This script reads a `responses.jsonl` from
`scripts/run_step2_freeform.py` and writes a deterministic JSON sidecar:

  {"articulations": {"rule_id": "candidate text", ...}, "selection": {...}}

Selection policy: use direct, with-examples generation rows; take the modal
normalized candidate per rule; break ties by lowest context index, then phrasing,
then original row order.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def extract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in enumerate(rows):
        if row.get("kind") != "generation":
            continue
        if row.get("variant") != "direct" or not row.get("has_examples"):
            continue
        candidate = str(row.get("candidate") or "").strip()
        if not candidate:
            continue
        groups.setdefault(str(row["rule_id"]), []).append((idx, row))

    articulations: dict[str, str] = {}
    selection: dict[str, Any] = {}
    for rule_id, entries in sorted(groups.items()):
        counts = Counter(_norm(str(row.get("candidate") or "")) for _idx, row in entries)
        best_norm, best_count = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        tied = [
            (idx, row)
            for idx, row in entries
            if _norm(str(row.get("candidate") or "")) == best_norm
        ]
        idx, chosen = sorted(
            tied,
            key=lambda it: (
                int(it[1].get("context_index", 999)),
                int(it[1].get("phrasing", 999)),
                it[0],
            ),
        )[0]
        candidate = str(chosen.get("candidate") or "").strip()
        articulations[rule_id] = candidate
        selection[rule_id] = {
            "row_index": idx,
            "context_index": chosen.get("context_index"),
            "phrasing": chosen.get("phrasing"),
            "modal_count": best_count,
            "n_direct_with_examples": len(entries),
            "candidate": candidate,
        }

    return {
        "articulations": articulations,
        "selection": selection,
        "selection_policy": "modal normalized direct with-examples candidate; tie by context_index, phrasing, row order",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("responses_jsonl", help="Path to a Step-2 free-form responses.jsonl")
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = extract(_read_jsonl(Path(args.responses_jsonl)))
    if not payload["articulations"]:
        raise SystemExit("no direct with-examples generation rows found")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(payload['articulations'])} rules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
