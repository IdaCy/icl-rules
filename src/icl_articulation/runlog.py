"""Per-run logging into results/<run_id>/.

Every run writes:
- config.json    model, seeds, all params, prompt template hash, git commit +
                 dirty flag, timestamp, advance cost estimate; actual cost
                 added by finish().
- responses.jsonl  one record per call (full request messages, response text,
                   top logprobs, usage — the record dicts from client.complete).
- metrics.json   whatever the run computes.

NOTE: .gitignore drops *.md (except README.md) — emit ONLY .json/.jsonl/.csv/
.png into results/, never markdown.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, TextIO


def _git_provenance(start: Path | None = None) -> tuple[str | None, bool | None]:
    """(commit, dirty) for the code being run.

    Tries ``git rev-parse HEAD`` first (cwd pinned to this module's directory,
    immune to the caller's cwd). If .git is unavailable — e.g. the repo was
    synced to a host where .git is absent — falls back to a ``GIT_COMMIT`` file
    found by walking up from this module to the repo root (written at sync
    time). ``dirty`` is True/False from
    ``git status --porcelain`` and None when unknown.

    Fallback-path honesty: a sibling ``GIT_DIRTY`` file next to ``GIT_COMMIT``
    (containing ``true``/``false``, case-insensitive) lets the sync step record
    the working-tree dirtiness it observed locally; absent, ``dirty`` stays None
    (genuinely unknown) rather than silently claiming a clean tree.
    """
    here = (start or Path(__file__)).resolve()
    cwd = here if here.is_dir() else here.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            commit = out.stdout.strip() or None
            dirty: bool | None = None
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if status.returncode == 0:
                dirty = bool(status.stdout.strip())
            return commit, dirty
    except (OSError, subprocess.SubprocessError):
        pass
    for parent in [cwd, *cwd.parents]:
        marker = parent / "GIT_COMMIT"
        if marker.is_file():
            try:
                commit = marker.read_text(encoding="utf-8").strip() or None
            except OSError:
                return None, None
            dirty = _read_dirty_marker(parent / "GIT_DIRTY")
            return commit, dirty
    return None, None


def _read_dirty_marker(path: Path) -> bool | None:
    """Optional sibling of GIT_COMMIT recording observed dirtiness; None if
    absent/unparseable so we never falsely claim a clean tree."""
    try:
        token = path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    if token in ("true", "1", "dirty"):
        return True
    if token in ("false", "0", "clean"):
        return False
    return None


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


class RunLog:
    """Create with start_run(); then log_response() per call, write_metrics(),
    and finish(cost_actual_usd) at the end."""

    def __init__(self, run_dir: Path, config: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.run_id = run_dir.name
        self._config = config
        self._responses: TextIO = (run_dir / "responses.jsonl").open("a", encoding="utf-8")

    def log_response(self, record: dict[str, Any]) -> None:
        self._responses.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._responses.flush()

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        _write_json(self.run_dir / "metrics.json", metrics)

    def finish(self, cost_actual_usd: float, extra: dict[str, Any] | None = None) -> None:
        self._config["cost_actual_usd"] = cost_actual_usd
        self._config["finished_utc"] = _utc_stamp()
        if extra:
            self._config.update(extra)
        _write_json(self.run_dir / "config.json", self._config)
        self._responses.close()


def start_run(
    name: str,
    config: dict[str, Any],
    cost_estimate_usd: float,
    results_dir: str | Path = "results",
) -> RunLog:
    """Create results/<name>-<UTCtimestamp>/ and write the initial config.json.

    ``config`` must carry model, seeds, all run params, and the prompt template
    hash; run_id, timestamp, git commit, and the advance cost estimate are
    added here.
    """
    base_id = f"{name}-{_utc_stamp()}"
    # the stamp has 1-second resolution: on collision (e.g. a crash-then-retry
    # loop), append -2, -3, ... instead of dying with FileExistsError
    run_id = base_id
    suffix = 2
    while True:
        run_dir = Path(results_dir) / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run_id = f"{base_id}-{suffix}"
            suffix += 1
    git_commit, git_dirty = _git_provenance()
    full = dict(config)
    full.update(
        {
            "run_id": run_id,
            "started_utc": _utc_stamp(),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "cost_estimate_usd": cost_estimate_usd,
        }
    )
    _write_json(run_dir / "config.json", full)
    return RunLog(run_dir, full)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
