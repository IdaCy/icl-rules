"""CLI: python -m icl_articulation.datagen.generators.llm <rule_id> [options]

Runs the shared llm_validated pipeline (generate -> 2-pass validate -> rebalance
-> emit) for rule 15 (positive_sentiment) or 16 (food_topic). On success it
writes data/<rule_id>/items.jsonl + data/<rule_id>/confound_report.json and
prints the PipelineResult as JSON.

MODES
  (default)  REAL run: every model call goes through icl_articulation.client
             (disk cache + cost meter + retries). Makes paid API calls (set
             OPENAI_API_KEY). Prints an ADVANCE cost estimate FIRST and aborts
             if it exceeds --max-cost (default $200). The run is logged into
             results/<run_id>/.
  --mock     OFFLINE run: wires the deterministic, network-free mock seam in
             pipeline (mock_generator / mock_labeler). No API key, no cost, no
             network — exercises the whole flow on a tiny fake corpus.

EXAMPLES
  # offline smoke (no network):
  python -m icl_articulation.datagen.generators.llm positive_sentiment --mock --max-candidates 240
  # real run:
  python -m icl_articulation.datagen.generators.llm food_topic --max-candidates 800
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .cost import estimate_cost
from .pipeline import mock_generator, mock_labeler, run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m icl_articulation.datagen.generators.llm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "rule_id",
        choices=["positive_sentiment", "food_topic", "physically_impossible"],
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=800,
        help="target number of candidates to generate (>= 600 for the real run)",
    )
    p.add_argument("--seed", type=int, default=0, help="generation/validation/split seed")
    p.add_argument(
        "--mock",
        action="store_true",
        help="OFFLINE: use the network-free mock generator/validator (no API key, no cost)",
    )
    p.add_argument(
        "--max-cost",
        type=float,
        default=200.0,
        help="abort the real run if the advance estimate exceeds this (USD)",
    )
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--data-dir", default=None, help="output root (defaults to repo data/)")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--no-write", action="store_true", help="run gates but do not write items.jsonl")
    p.add_argument("--no-pos", action="store_true", help="skip the nltk POS battery predicates")
    return p.parse_args(argv)


def _run_mock(args: argparse.Namespace) -> int:
    result = run_pipeline(
        args.rule_id,
        mock_generator,
        mock_labeler,
        seed=args.seed,
        max_candidates=args.max_candidates,
        data_dir=args.data_dir,
        write=not args.no_write,
        run_pos=not args.no_pos,
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 0


def _run_real(args: argparse.Namespace) -> int:
    # advance cost estimate first, then the gate.
    est = estimate_cost(args.rule_id, args.max_candidates)
    print(
        f"advance cost estimate: ${est['total_usd']:.4f} for rule {args.rule_id!r} "
        f"(~{args.max_candidates} candidates; gen={est['gen_calls']} calls, "
        f"validate={est['validation_calls']} calls across passes A+B); "
        f"--max-cost gate ${args.max_cost:.2f}"
    )
    print(json.dumps(est, indent=2))
    if est["total_usd"] > args.max_cost:
        print(
            f"ABORT: estimate ${est['total_usd']:.4f} exceeds --max-cost ${args.max_cost:.2f}",
            file=sys.stderr,
        )
        return 1

    # import the real seam + run logging only on the real path (no client import
    # is forced for --mock / the offline test).
    from ....runlog import start_run
    from .api import ClientSeam
    from .pipeline import run_pipeline_async

    config = {
        "task": "llm-datagen",
        "rule_id": args.rule_id,
        "max_candidates": args.max_candidates,
        "seed": args.seed,
        "gen_model": est["gen_model"],
        "validator_a_model": est["validator_a_model"],
        "validator_b_model": est["validator_b_model"],
        "concurrency": args.concurrency,
        "cost_estimate": est,
    }
    run = start_run(
        name=f"llm-datagen-{args.rule_id}",
        config=config,
        cost_estimate_usd=est["total_usd"],
        results_dir=args.results_dir,
    )
    print(f"run dir: {run.run_dir}")

    async def _drive() -> tuple[Any, dict[str, Any]]:
        """The ENTIRE real run under ONE event loop: build the seam, run the whole
        generate->validate->regen->emit flow through the concurrent client, log
        metrics, AND ``await seam.aclose()`` — all on this single loop.

        Sequential ``asyncio.run(...)`` calls (one per phase, plus a final
        ``asyncio.run(seam.aclose())``) bound the client's httpx pool to a loop the
        next phase had already closed -> 'RuntimeError: Event loop is closed', so
        validation/aclose never completed and items.jsonl was never written. Now a
        single loop wraps every phase and the close, so the pool stays valid the
        whole run. The seam still fans calls out via generate_many/label_many under
        the client's Semaphore(16); request content + seed mapping are unchanged."""
        seam = ClientSeam(concurrency=args.concurrency, cache_dir=args.cache_dir, run_log=run)
        try:
            # pass the seam itself (not its bound methods) so the pipeline drives
            # the CONCURRENT batch dispatch (generate_many / label_many) rather
            # than one request at a time — the per-call seeds/content are unchanged.
            result = await run_pipeline_async(
                args.rule_id,
                seam,
                seed=args.seed,
                max_candidates=args.max_candidates,
                data_dir=args.data_dir,
                write=not args.no_write,
                run_pos=not args.no_pos,
            )
            cost = seam.cost_summary()
            stats = seam.stats()
            run.write_metrics({**result.as_dict(), "client_stats": stats})
            run.finish(cost_actual_usd=cost["total_usd"], extra={"cost": cost, "client_stats": stats})
            return result, cost
        finally:
            # close the client on the SAME loop the calls ran on (single-loop fix).
            await seam.aclose()

    result, cost = asyncio.run(_drive())

    print(json.dumps(result.as_dict(), indent=2))
    print(
        f"done: rule {args.rule_id!r} emitted {result.n_emitted} items, "
        f"drop_rate={result.drop_rate:.3f}, actual cost ${cost['total_usd']:.4f} "
        f"(estimate ${est['total_usd']:.4f})"
    )
    return 0 if result.confound_overall_pass else 1


def _run_physically_impossible(args: argparse.Namespace) -> int:
    """Rule 18 has a DIFFERENT recipe (minimal-pair frames + by-base survival +
    programmatic split), so it dispatches to its own module rather than the
    generate->rebalance pipeline shared by rules 15/16. ``--max-candidates`` here
    caps the number of BASES validated (None = all)."""
    from . import physically_impossible as r18

    # --max-candidates defaults to 800 for the 15/16 generation target; for rule
    # 18 'all bases' is the natural default, so treat the parser default as None.
    max_candidates = args.max_candidates if args.max_candidates != 800 else None

    if args.mock:
        summary = r18.run_build(
            r18.make_mock_pair_validator(),
            seed=args.seed,
            data_dir=args.data_dir,
            write=not args.no_write,
            run_pos=not args.no_pos,
        )
        print(json.dumps(summary.as_dict(), indent=2))
        return 0

    # REAL path: advance cost estimate first, then the --max-cost gate.
    n_bases = len(r18.build_bases()) if max_candidates is None else max_candidates
    est = r18.estimate_cost(2 * n_bases)
    print(
        f"advance cost estimate: ${est['total_usd']:.4f} for rule "
        f"'physically_impossible' ({n_bases} bases x 2 variants x 2 passes = "
        f"{est['n_calls']} calls); --max-cost gate ${args.max_cost:.2f}"
    )
    print(json.dumps(est, indent=2))
    if est["total_usd"] > args.max_cost:
        print(
            f"ABORT: estimate ${est['total_usd']:.4f} exceeds --max-cost "
            f"${args.max_cost:.2f}",
            file=sys.stderr,
        )
        return 1

    summary = r18.run_api_build(
        seed=args.seed,
        max_candidates=max_candidates,
        data_dir=args.data_dir,
        write=not args.no_write,
        run_pos=not args.no_pos,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        concurrency=args.concurrency,
    )
    print(json.dumps(summary.as_dict(), indent=2))
    all_pass = summary.gate_schema and summary.gate_groundtruth and summary.gate_confound
    return 0 if all_pass else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rule_id == "physically_impossible":
        return _run_physically_impossible(args)
    if args.mock:
        return _run_mock(args)
    return _run_real(args)


if __name__ == "__main__":
    sys.exit(main())
