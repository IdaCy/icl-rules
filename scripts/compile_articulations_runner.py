#!/usr/bin/env python
"""Compile-articulations instrument: each stated rule articulation is compiled into an
executable Python predicate, which is then run (sandboxed) on the data — a
judge-independent, author-DOF-free upgrade of the hand-written compiled-predicate check.

For each of the 66 gpt-4.1 `direct` articulations (11 probed rules x 6 samples), two
coders compile it to `def rule(text)->bool`: `gpt-4.1` and `claude-opus-4-8` (a fixed
cross-family coder, to separate articulation precision from coding skill). Each
predicate is executed by the hardened sandbox `_safe_predicate_worker.py` (AST allowlist
+ restricted builtins + non-root + rlimits + per-process timeout).

Pre-registered design (frozen, design + security reviewed).
Compilation makes paid API calls (set OPENAI_API_KEY); execution runs the resulting
predicate code in a locked-down sandbox.

Run:  python scripts/compile_articulations_runner.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import argparse
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "results" / "compiled_predicates"
FREEFORM = REPO / "results" / "step2-freeform-gpt-4.1-20260611T011216Z" / "responses.jsonl"
WORKER = str(REPO / "scripts" / "_safe_predicate_worker.py")

PROMPT = (
    "Write a single Python function `def rule(text: str) -> bool` that returns True "
    "if and only if the input text satisfies this rule. Use only built-in Python — no "
    "imports, no file or network access. Output only the function in a ```python code "
    "block.\n\nRule: {art}"
)
PRICES = {"gpt-4.1": (2.0, 8.0), "claude-opus-4-8": (5.0, 25.0)}  # USD/MTok in,out


def extract_code(text: str) -> str | None:
    """Robust to leading prose, a missing closing fence (truncation), and stray
    fences. Captures from the opening ```python fence to the closing fence OR to
    end-of-string (so a truncated-but-valid function can still parse)."""
    t = text or ""
    m = re.search(r"```(?:python)?\s*\n?(.*?)(?:```|$)", t, re.S)
    code = (m.group(1) if m else t).strip()
    code = re.sub(r"^```(?:python)?\s*", "", code)
    code = re.sub(r"\s*```$", "", code).strip()
    return code if "def rule" in code else None


def run_worker(code: str, items: list[dict]) -> dict:
    try:
        r = subprocess.run(
            [sys.executable, WORKER],
            input=json.dumps({"code": code, "items": items}),
            capture_output=True, text=True, timeout=20, close_fds=True,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "parent_timeout"}
    if not r.stdout.strip():
        return {"ok": False, "reason": "worker_no_output", "detail": (r.stderr or "")[:160]}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": "worker_parse", "detail": repr(e)[:120]}


def done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    return {(r["coder"], r["rule"], r["art_idx"]) for r in
            (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeform-responses", default=str(FREEFORM))
    p.add_argument("--data-dir", default=str(DATA))
    p.add_argument("--out-dir", default=str(OUT))
    p.add_argument("--rules", help="optional comma-separated rule ids")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(REPO / ".env")
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    freeform = Path(args.freeform_responses)
    rule_filter = {r.strip() for r in args.rules.split(",") if r.strip()} if args.rules else None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compiled.jsonl"
    done = done_keys(out_path)

    # 66 direct gpt-4.1 articulations, grouped by rule (stable order)
    arts: dict[str, list[str]] = {}
    for line in freeform.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "generation" and r.get("variant") == "direct" and r.get("has_examples"):
            if rule_filter is not None and r["rule_id"] not in rule_filter:
                continue
            arts.setdefault(r["rule_id"], []).append((r.get("candidate") or "").strip())
    items_by_rule = {rule: [json.loads(l) for l in (data_dir / rule / "items.jsonl").read_text().splitlines() if l.strip()]
                     for rule in arts}

    oai = openai.OpenAI()
    ant = anthropic.Anthropic()
    cost = {"gpt-4.1": [0, 0], "claude-opus-4-8": [0, 0]}
    fh = out_path.open("a")
    n_calls = 0

    def gen(coder: str, art: str) -> tuple[str, tuple[int, int]]:
        for attempt in range(4):
            try:
                if coder == "gpt-4.1":
                    rr = oai.chat.completions.create(
                        model="gpt-4.1", temperature=0, max_tokens=3000,
                        messages=[{"role": "user", "content": PROMPT.format(art=art)}])
                    return rr.choices[0].message.content or "", (rr.usage.prompt_tokens, rr.usage.completion_tokens)
                rr = ant.messages.create(
                    model="claude-opus-4-8", max_tokens=3000,
                    system="You write small, correct Python functions.",
                    messages=[{"role": "user", "content": PROMPT.format(art=art)}])
                txt = "".join(b.text for b in rr.content if b.type == "text")
                return txt, (rr.usage.input_tokens, rr.usage.output_tokens)
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise
                import time
                time.sleep(2 ** attempt)

    for coder in ("gpt-4.1", "claude-opus-4-8"):
        for rule, art_list in sorted(arts.items()):
            items = items_by_rule[rule]
            for idx, art in enumerate(art_list):
                if (coder, rule, idx) in done:
                    continue
                raw, (pin, pout) = gen(coder, art)
                cost[coder][0] += pin
                cost[coder][1] += pout
                n_calls += 1
                code = extract_code(raw)
                if code is None:
                    rec = {"coder": coder, "rule": rule, "art_idx": idx, "articulation": art[:300],
                           "code": None, "result": {"ok": False, "reason": "no_code_in_response"}}
                else:
                    rec = {"coder": coder, "rule": rule, "art_idx": idx, "articulation": art[:300],
                           "code": code, "result": run_worker(code, items)}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        usd = sum(cost[c][0] / 1e6 * PRICES[c][0] + cost[c][1] / 1e6 * PRICES[c][1] for c in cost)
        print(f"[{coder}] done; running cost ${usd:.3f}")
    fh.close()
    total = round(sum(cost[c][0] / 1e6 * PRICES[c][0] + cost[c][1] / 1e6 * PRICES[c][1] for c in cost), 4)
    (out_dir / "_cost.json").write_text(json.dumps({
        "freeform_responses": str(freeform),
        "data_dir": str(data_dir),
        "rules": sorted(arts),
        "by_coder_tokens": cost,
        "total_usd": total,
        "n_calls": n_calls,
    }, indent=2))
    print(f"[compiled] wrote {out_path}  | FINAL cost ${total}  ({n_calls} compile calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
