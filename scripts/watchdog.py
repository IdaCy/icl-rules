#!/usr/bin/env python
"""Per-run watchdog: watches a step-1 run dir + its nohup log.

Every --interval seconds (default 120) it checks:
  - stall: responses.jsonl unchanged for 10 minutes while the run is unfinished
  - parse-failure rate > 5% (over all logged records, once >= 20 exist)
  - error/429 burst in the log file (>= 10 new error-ish lines since last check)
  - metered cost (recomputed from responses.jsonl usage, fresh calls only)
    > 1.5x the advance estimate
  - all-one-class drift: > 80% single-class predictions over the last 200 records

One timestamped status line per check is appended to <rundir>/WATCHDOG_STATUS;
any alert is ALSO appended to <rundir>/ALERT (and the watchdog keeps running).
Exits when responses.jsonl reaches the expected total or config.json gains
finished_utc.

Expected total and cost estimate are read from the run dir's config.json
(expected_total_calls / cost_estimate_usd); --expected-total/--cost-estimate
are overrides. If started before the run dir exists, the watchdog waits for
it (up to 15 minutes) instead of crashing — it never creates the dir itself.

Usage:
  python scripts/watchdog.py --run-dir results/step1-full-gpt-4.1-<ts> \\
      --log-file step1-full.log \\
      [--expected-total 10800] [--cost-estimate 12.50]
  (--log-file is wherever you redirected the run's stdout/stderr.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.prices import cost_usd

STALL_SECONDS = 600
PARSE_FAIL_RATE_ALERT = 0.05
PARSE_FAIL_MIN_RECORDS = 20
ERROR_BURST_ALERT = 10  # new error-ish log lines between two checks
COST_RATIO_ALERT = 1.5
DRIFT_WINDOW = 200
DRIFT_FRACTION = 0.80
RUN_DIR_WAIT_SECONDS = 900  # how long main() waits for the run dir to appear
# word-boundary matching so benign substrings never count as error lines:
# the runner's own done line contains '429s=0', and a '429' inside an id or
# token count must not inflate err_lines either.
ERROR_LINE_RE = re.compile(
    r"\b(429|RateLimitError|Traceback|ERROR|APIError|APIConnectionError)\b"
)


def _utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_records(path: Path) -> list[dict[str, Any]]:
    """responses.jsonl records; a partially-written last line is skipped."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # mid-write tail
    return records


def metered_cost_usd(records: list[dict[str, Any]]) -> float:
    """Actual spend so far, recomputed from logged usage (fresh calls only)."""
    total = 0.0
    for r in records:
        if r.get("cached"):
            continue
        response = r.get("response") or {}
        usage = response.get("usage") or {}
        model = (r.get("request") or {}).get("model") or response.get("model")
        if not model or not usage:
            continue
        try:
            total += cost_usd(model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        except KeyError:
            continue  # unpriced model: skip rather than crash the watchdog
    return total


def count_error_lines(log_file: Path) -> int:
    if not log_file.is_file():
        return 0
    n = 0
    with log_file.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if ERROR_LINE_RE.search(line):
                n += 1
    return n


def read_config(run_dir: Path) -> dict[str, Any]:
    """config.json of the watched run; {} when absent or mid-write."""
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def check_once(
    run_dir: Path,
    log_file: Path,
    expected_total: int,
    cost_estimate: float,
    state: dict[str, Any],
    now: float | None = None,
) -> tuple[str, list[str], bool]:
    """One watchdog pass. Returns (status line, alerts, run finished).

    ``state`` persists between passes (last record count + its change time,
    last error-line count). ``now`` is injectable for tests.
    """
    now = time.time() if now is None else now
    alerts: list[str] = []
    records = read_records(run_dir / "responses.jsonl")
    n = len(records)

    # stall: no new records for STALL_SECONDS while unfinished
    if n != state.get("last_n"):
        state["last_n"] = n
        state["last_change"] = now
    stalled_for = now - state.setdefault("last_change", now)
    finished = "finished_utc" in read_config(run_dir)
    if not finished and n < expected_total and stalled_for >= STALL_SECONDS:
        alerts.append(
            f"STALL: responses.jsonl unchanged for {int(stalled_for)}s at {n}/{expected_total}"
        )

    # parse-failure rate
    if n >= PARSE_FAIL_MIN_RECORDS:
        n_fail = sum(1 for r in records if not r.get("parse_ok", True))
        rate = n_fail / n
        if rate > PARSE_FAIL_RATE_ALERT:
            alerts.append(f"PARSE-FAILURES: {n_fail}/{n} = {rate:.1%} (> 5%)")

    # error/429 burst in the log
    err_lines = count_error_lines(log_file)
    new_errors = err_lines - state.get("err_lines", 0)
    state["err_lines"] = err_lines
    if new_errors >= ERROR_BURST_ALERT:
        alerts.append(f"ERROR-BURST: {new_errors} new error/429 lines in {log_file}")

    # metered cost vs estimate
    cost = metered_cost_usd(records)
    if cost_estimate > 0 and cost > COST_RATIO_ALERT * cost_estimate:
        alerts.append(
            f"COST: metered ${cost:.4f} > {COST_RATIO_ALERT}x estimate ${cost_estimate:.4f}"
        )

    # all-one-class drift over the last DRIFT_WINDOW records
    if n >= DRIFT_WINDOW:
        window = records[-DRIFT_WINDOW:]
        for label in (True, False):
            frac = sum(1 for r in window if r.get("predicted") is label) / len(window)
            if frac > DRIFT_FRACTION:
                alerts.append(
                    f"DRIFT: {frac:.0%} of the last {len(window)} predictions are {label}"
                )

    done = finished or n >= expected_total
    status = (
        f"{_utc_stamp()} n={n}/{expected_total} cost=${cost:.4f} est=${cost_estimate:.4f} "
        f"err_lines={err_lines} stalled_for={int(stalled_for)}s "
        f"finished={finished} alerts={len(alerts)}"
    )
    return status, alerts, done


def _append(path: Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--log-file", required=True)
    p.add_argument("--expected-total", type=int, default=None,
                   help="override of config.json expected_total_calls")
    p.add_argument("--cost-estimate", type=float, default=None,
                   help="override of config.json cost_estimate_usd")
    p.add_argument("--interval", type=float, default=120.0)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    log_file = Path(args.log_file)

    # If started before the runner has created the run dir, wait for it.
    # Deliberately NOT mkdir'ing it: creating the dir here would mask a runner
    # that failed to start. After RUN_DIR_WAIT_SECONDS of absence, alert and
    # exit nonzero — a watchdog silently dead at t=0 is the worst failure mode.
    poll = min(args.interval, 10.0)
    waited = 0.0
    while not run_dir.is_dir():
        if waited >= RUN_DIR_WAIT_SECONDS:
            print(
                f"ALERT: run dir {run_dir} still absent after {int(waited)}s — "
                "did the runner fail to start?",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(f"watchdog: waiting for run dir {run_dir} ({int(waited)}s)", flush=True)
        time.sleep(poll)
        waited += poll

    state: dict[str, Any] = {}
    while True:
        # config.json carries the authoritative totals; CLI flags override
        config = read_config(run_dir)
        expected_total = args.expected_total
        if expected_total is None:
            expected_total = config.get("expected_total_calls")
        cost_estimate = args.cost_estimate
        if cost_estimate is None:
            cost_estimate = config.get("cost_estimate_usd")
        if expected_total is None or cost_estimate is None:
            if waited >= RUN_DIR_WAIT_SECONDS:
                print(
                    f"ALERT: {run_dir / 'config.json'} never provided "
                    "expected_total_calls/cost_estimate_usd and no CLI override given",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            print(
                f"watchdog: waiting for totals from {run_dir / 'config.json'} "
                f"({int(waited)}s)",
                flush=True,
            )
            time.sleep(poll)
            waited += poll
            continue
        status, alerts, done = check_once(run_dir, log_file, expected_total, cost_estimate, state)
        _append(run_dir / "WATCHDOG_STATUS", [status])
        print(status, flush=True)
        if alerts:
            stamped = [f"{_utc_stamp()} {a}" for a in alerts]
            _append(run_dir / "ALERT", stamped)
            for a in stamped:
                print(f"ALERT: {a}", file=sys.stderr, flush=True)
        if done:
            print("watchdog: run finished, exiting", flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
