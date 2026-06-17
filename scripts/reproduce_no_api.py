#!/usr/bin/env python
"""Regenerate report-critical no-API derived artifacts from frozen outputs.

This script intentionally does not run paid API calls and does not run the local
HF model. It rebuilds analysis sidecars, the frozen local-Qwen metric summary
from raw JSONL, and all public report figures that can be regenerated from
existing data/results.

Run:
  .venv/bin/python scripts/reproduce_no_api.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

COMMANDS = [
    ["scripts/analyze_step3.py"],
    ["scripts/make_figures.py", "--tables-only"],
    ["scripts/analyze_confounds.py"],
    ["scripts/analyze_deconfound.py"],
    ["scripts/analyze_swc_deconfound.py"],
    ["scripts/analyze_confound_grade.py"],
    ["scripts/analyze_local_qwen.py"],
    ["scripts/make_figures.py"],
    # divergence (divergence faithfulness) + CoT same-session (CoT same-session) — all no-API; the
    # CoT same-session analyzer/figure no-op cleanly when their run dirs are absent.
    ["scripts/analyze_behavioral_faithfulness.py"],
    ["scripts/analyze_divergence.py"],
    ["scripts/make_divergence_figures.py"],
    ["scripts/analyze_cot_same_session.py"],
    ["scripts/make_cot_same_session_figures.py"],
    ["scripts/analyze_insession_articulation.py"],
    ["scripts/make_insession_articulation_figures.py"],
]

TEST_COMMAND = ["-m", "pytest"]


def _run(args: list[str]) -> None:
    cmd = [sys.executable, *args]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-tests", action="store_true", help="also run the test suite")
    args = parser.parse_args()

    if args.include_tests:
        _run(TEST_COMMAND)
    for command in COMMANDS:
        _run(command)

    print("\nno-API reproduction completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
