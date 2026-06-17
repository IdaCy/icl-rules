"""Watchdog check logic tests (pure check_once + file outputs; no sleeping)."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_step1
import watchdog
from tests.conftest import write_rule_dataset
from tests.test_run_step1 import RuleFollowingAPI, _argv, _single_run_dir


def _write_records(run_dir: Path, records: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "responses.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


def _record(predicted: bool | None, prompt_tokens: int = 1000, cached: bool = False) -> dict[str, Any]:
    return {
        "predicted": predicted,
        "parse_ok": predicted is not None,
        "cached": cached,
        "request": {"model": "gpt-4.1"},
        "response": {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2}},
    }


def test_metered_cost_skips_cached_calls() -> None:
    records = [_record(True), _record(False), _record(True, cached=True)]
    # gpt-4.1: $2/Mtok prompt, $8/Mtok completion; 2 fresh calls
    expected = 2 * (1000 * 2.00 + 2 * 8.00) / 1_000_000
    assert watchdog.metered_cost_usd(records) == expected


def test_partial_last_line_is_skipped(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_records(run_dir, [_record(True)])
    with (run_dir / "responses.jsonl").open("a") as fh:
        fh.write('{"predicted": tr')  # mid-write tail
    assert len(watchdog.read_records(run_dir / "responses.jsonl")) == 1


def test_no_alerts_on_healthy_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_records(run_dir, [_record(i % 2 == 0) for i in range(100)])
    log = tmp_path / "run.log"
    log.write_text("all fine\n")
    status, alerts, done = watchdog.check_once(run_dir, log, 400, 1.0, state={})
    assert alerts == []
    assert done is False
    assert "n=100/400" in status


def test_stall_alert_after_ten_minutes(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_records(run_dir, [_record(i % 2 == 0) for i in range(50)])
    log = tmp_path / "run.log"
    log.write_text("")
    state: dict[str, Any] = {}
    t0 = time.time()
    _, alerts, _ = watchdog.check_once(run_dir, log, 400, 1.0, state, now=t0)
    assert alerts == []
    _, alerts, _ = watchdog.check_once(run_dir, log, 400, 1.0, state, now=t0 + 601)
    assert any("STALL" in a for a in alerts)


def test_parse_failure_cost_and_drift_alerts(tmp_path) -> None:
    run_dir = tmp_path / "run"
    # 200 records: 90% predicted True (drift), 10% parse failures, expensive
    records = [_record(True, prompt_tokens=200_000) for _ in range(180)]
    records += [_record(None, prompt_tokens=200_000) for _ in range(20)]
    _write_records(run_dir, records)
    log = tmp_path / "run.log"
    log.write_text("")
    _, alerts, _ = watchdog.check_once(run_dir, log, 400, 0.5, state={})
    kinds = {a.split(":")[0] for a in alerts}
    assert "PARSE-FAILURES" in kinds  # 10% > 5%
    assert "COST" in kinds  # 200 * $0.4 = $80 > 1.5 * $0.5
    assert "DRIFT" in kinds  # 90% single-class over the last 200


def test_error_burst_alert(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_records(run_dir, [_record(i % 2 == 0) for i in range(40)])
    log = tmp_path / "run.log"
    log.write_text("ok\n")
    state: dict[str, Any] = {}
    _, alerts, _ = watchdog.check_once(run_dir, log, 400, 1.0, state)
    assert alerts == []
    log.write_text("ok\n" + "openai.RateLimitError: 429 too many requests\n" * 12)
    _, alerts, _ = watchdog.check_once(run_dir, log, 400, 1.0, state)
    assert any("ERROR-BURST" in a for a in alerts)


def test_error_lines_match_word_boundaries_only(tmp_path) -> None:
    # the runner's own done line ('429s=0') and ids containing 429 must NOT
    # count as error lines; real 429/RateLimitError lines must
    log = tmp_path / "run.log"
    log.write_text(
        "done: 360/360 correct, 0 parse failures, 12.3s, actual cost $0.0500 "
        "(estimate $0.0600), 429s=0 cache_hits=100\n"
        "logging item i4290 and key abc429def\n"
        "timestamp 20260429T101010Z\n"
    )
    assert watchdog.count_error_lines(log) == 0
    with log.open("a") as fh:
        fh.write("retry: 429 RateLimitError attempt 1/6 sleeping 1.0s\n")
        fh.write("Traceback (most recent call last):\n")
    assert watchdog.count_error_lines(log) == 2


def test_done_on_expected_total_or_finished_utc(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_records(run_dir, [_record(i % 2 == 0) for i in range(40)])
    log = tmp_path / "run.log"
    log.write_text("")
    *_, done = watchdog.check_once(run_dir, log, 40, 1.0, state={})
    assert done is True  # reached expected total
    *_, done = watchdog.check_once(run_dir, log, 400, 1.0, state={})
    assert done is False
    (run_dir / "config.json").write_text(json.dumps({"finished_utc": "20260610T000000Z"}))
    *_, done = watchdog.check_once(run_dir, log, 400, 1.0, state={})
    assert done is True  # config gained finished_utc


def test_main_writes_status_and_alert_files(tmp_path) -> None:
    run_dir = tmp_path / "run"
    records = [_record(True) for _ in range(180)] + [_record(None) for _ in range(20)]
    _write_records(run_dir, records)
    log = tmp_path / "run.log"
    log.write_text("")
    rc = watchdog.main(
        [
            "--run-dir", str(run_dir),
            "--log-file", str(log),
            "--expected-total", "200",  # already complete -> exits after one pass
            "--cost-estimate", "1.0",
            "--interval", "0.01",
        ]
    )
    assert rc == 0
    status_lines = (run_dir / "WATCHDOG_STATUS").read_text().splitlines()
    assert len(status_lines) == 1 and "n=200/200" in status_lines[0]
    alert_lines = (run_dir / "ALERT").read_text().splitlines()
    assert any("DRIFT" in a for a in alert_lines)  # alerts also land in ALERT


def test_main_reads_totals_from_config_and_flags_override(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_records(run_dir, [_record(i % 2 == 0) for i in range(40)])
    (run_dir / "config.json").write_text(
        json.dumps({"expected_total_calls": 40, "cost_estimate_usd": 1.0})
    )
    log = tmp_path / "run.log"
    log.write_text("")
    # no --expected-total/--cost-estimate: read from config.json
    rc = watchdog.main(
        ["--run-dir", str(run_dir), "--log-file", str(log), "--interval", "0.01"]
    )
    assert rc == 0
    status = (run_dir / "WATCHDOG_STATUS").read_text().splitlines()[-1]
    assert "n=40/40" in status and "est=$1.0000" in status
    # CLI flag overrides the (deliberately wrong) config value
    (run_dir / "config.json").write_text(
        json.dumps({"expected_total_calls": 999, "cost_estimate_usd": 1.0})
    )
    rc = watchdog.main(
        ["--run-dir", str(run_dir), "--log-file", str(log),
         "--expected-total", "40", "--interval", "0.01"]
    )
    assert rc == 0
    status = (run_dir / "WATCHDOG_STATUS").read_text().splitlines()[-1]
    assert "n=40/40" in status


def test_main_waits_for_run_dir_instead_of_crashing(tmp_path) -> None:
    # started before the runner created the run dir: poll, never mkdir, then
    # proceed once the dir appears (with totals taken from its config.json)
    run_dir = tmp_path / "results" / "run"
    staging = tmp_path / "staging"
    _write_records(staging, [_record(i % 2 == 0) for i in range(40)])
    (staging / "config.json").write_text(
        json.dumps({"expected_total_calls": 40, "cost_estimate_usd": 1.0})
    )
    log = tmp_path / "run.log"
    log.write_text("")

    def appear() -> None:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(run_dir)

    timer = threading.Timer(0.05, appear)
    timer.start()
    try:
        rc = watchdog.main(
            ["--run-dir", str(run_dir), "--log-file", str(log), "--interval", "0.01"]
        )
    finally:
        timer.cancel()
    assert rc == 0
    assert "n=40/40" in (run_dir / "WATCHDOG_STATUS").read_text()


def test_main_alerts_and_exits_nonzero_if_run_dir_never_appears(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(watchdog, "RUN_DIR_WAIT_SECONDS", 0.0)
    rc = watchdog.main(
        ["--run-dir", str(tmp_path / "never"), "--log-file", str(tmp_path / "run.log"),
         "--interval", "0.01"]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "ALERT" in err and "absent" in err
    assert not (tmp_path / "never").exists()  # the watchdog never creates the dir


# --- integration: watchdog on REAL runner output (review N9) -----------------------


def test_check_once_on_real_runner_output(tmp_path) -> None:
    # run a small pilot end-to-end (FakeAPI, zero network), then point the
    # watchdog at the actual run dir: a field rename in run_step1.analyze()
    # must break THIS test rather than silently blinding the watchdog
    write_rule_dataset(tmp_path / "data")
    assert run_step1.main(_argv(tmp_path, "pilot"), api=RuleFollowingAPI()) == 0
    run_dir = _single_run_dir(tmp_path)
    config = json.loads((run_dir / "config.json").read_text())
    log = tmp_path / "run.log"
    log.write_text("")

    records = watchdog.read_records(run_dir / "responses.jsonl")
    assert len(records) == 40 == config["expected_total_calls"]
    # the watchdog's metered cost sees the runner's usage fields (>0 with the
    # FakeAPI's nonzero usage) and matches the runner's own actuals
    cost = watchdog.metered_cost_usd(records)
    assert cost > 0
    assert cost == pytest.approx(config["cost_actual_usd"])
    # parse/drift checks see the runner's parse_ok/predicted fields
    assert all("parse_ok" in r and "predicted" in r for r in records)

    status, alerts, done = watchdog.check_once(
        run_dir, log, config["expected_total_calls"], config["cost_estimate_usd"], state={}
    )
    assert done is True  # n reached expected total AND config has finished_utc
    assert "n=40/40" in status and "finished=True" in status
    assert alerts == []  # clean run: parse rate 0, cost within estimate, no drift
