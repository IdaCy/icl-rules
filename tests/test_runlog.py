"""Run logging tests: directory layout, config contents, jsonl, metrics,
git provenance (rev-parse + GIT_COMMIT fallback)."""

from __future__ import annotations

import json
import re
import subprocess

import icl_articulation.runlog as runlog
from icl_articulation.runlog import _git_provenance, start_run


def test_run_dir_layout_and_config(tmp_path) -> None:
    run = start_run(
        name="unittest",
        config={"model": "gpt-4.1", "seed": 0, "template_hash": "abc"},
        cost_estimate_usd=0.01,
        results_dir=tmp_path / "results",
    )
    assert re.fullmatch(r"unittest-\d{8}T\d{6}Z", run.run_id)
    config = json.loads((run.run_dir / "config.json").read_text())
    assert config["model"] == "gpt-4.1"
    assert config["seed"] == 0
    assert config["template_hash"] == "abc"
    assert config["run_id"] == run.run_id
    assert config["cost_estimate_usd"] == 0.01
    assert "started_utc" in config
    assert "git_commit" in config and "git_dirty" in config
    run.finish(cost_actual_usd=0.009)


def test_responses_jsonl_and_metrics(tmp_path) -> None:
    run = start_run("unittest", {"model": "m"}, 0.0, results_dir=tmp_path / "results")
    run.log_response({"query": "a", "response": {"choices": []}})
    run.log_response({"query": "b", "response": {"choices": []}})
    run.write_metrics({"accuracy": 0.95, "n": 2})
    run.finish(cost_actual_usd=0.001, extra={"n_429": 0})

    lines = (run.run_dir / "responses.jsonl").read_text().splitlines()
    assert [json.loads(l)["query"] for l in lines] == ["a", "b"]
    metrics = json.loads((run.run_dir / "metrics.json").read_text())
    assert metrics == {"accuracy": 0.95, "n": 2}
    config = json.loads((run.run_dir / "config.json").read_text())
    assert config["cost_actual_usd"] == 0.001
    assert config["n_429"] == 0
    assert "finished_utc" in config


def test_run_dir_collision_gets_suffix(tmp_path, monkeypatch) -> None:
    # the run-dir stamp has 1-second resolution: a second run within the same
    # second must get a -2/-3 suffix instead of dying with FileExistsError
    monkeypatch.setattr(runlog, "_utc_stamp", lambda: "20260610T000000Z")
    runs = [
        start_run("unittest", {}, 0.0, results_dir=tmp_path / "results") for _ in range(3)
    ]
    names = [r.run_id for r in runs]
    assert names == [
        "unittest-20260610T000000Z",
        "unittest-20260610T000000Z-2",
        "unittest-20260610T000000Z-3",
    ]
    for r in runs:
        assert r.run_dir.is_dir()
        assert json.loads((r.run_dir / "config.json").read_text())["run_id"] == r.run_id
        r.finish(0.0)


def _git(repo, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo, check=True, capture_output=True,
    )


def test_git_provenance_from_real_repo(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "f.txt").write_text("hello\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "init")
    commit, dirty = _git_provenance(start=repo)
    assert commit is not None and re.fullmatch(r"[0-9a-f]{40}", commit)
    assert dirty is False
    (repo / "f.txt").write_text("changed\n")  # dirty tree now
    commit2, dirty2 = _git_provenance(start=repo)
    assert commit2 == commit
    assert dirty2 is True


def test_git_provenance_git_commit_file_fallback(tmp_path) -> None:
    # No .git anywhere (synced to a host without .git): falls back to GIT_COMMIT
    # at the repo root, found by walking up; dirty is unknown -> None.
    root = tmp_path / "synced"
    pkg = root / "src" / "icl_articulation"
    pkg.mkdir(parents=True)
    (root / "GIT_COMMIT").write_text("a" * 40 + "\n")
    commit, dirty = _git_provenance(start=pkg)
    assert commit == "a" * 40
    assert dirty is None


def test_git_provenance_dirty_marker_fallback(tmp_path) -> None:
    # A sibling GIT_DIRTY marker (written by the sync step) makes the fallback
    # report observed dirtiness honestly instead of None.
    root = tmp_path / "synced"
    pkg = root / "src" / "icl_articulation"
    pkg.mkdir(parents=True)
    (root / "GIT_COMMIT").write_text("b" * 40 + "\n")
    (root / "GIT_DIRTY").write_text("true\n")
    commit, dirty = _git_provenance(start=pkg)
    assert commit == "b" * 40
    assert dirty is True
    (root / "GIT_DIRTY").write_text("clean\n")
    assert _git_provenance(start=pkg)[1] is False
    (root / "GIT_DIRTY").write_text("garbage\n")  # unparseable -> unknown, not a lie
    assert _git_provenance(start=pkg)[1] is None


def test_git_provenance_nothing_available(tmp_path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    commit, dirty = _git_provenance(start=bare)
    assert commit is None and dirty is None


def test_no_markdown_emitted(tmp_path) -> None:
    # .gitignore drops *.md — results must be json/jsonl/csv/png only
    run = start_run("unittest", {}, 0.0, results_dir=tmp_path / "results")
    run.write_metrics({})
    run.finish(0.0)
    suffixes = {p.suffix for p in run.run_dir.iterdir()}
    assert suffixes <= {".json", ".jsonl", ".csv", ".png"}
