"""The REAL OpenAI seam for the llm_validated pipeline. Makes paid API calls (set OPENAI_API_KEY).

Wires the pipeline's Generator / Labeler callables to
``icl_articulation.client.OpenAIClient`` so every model call goes through the
disk cache, the cost meter, retries/backoff, and per-run logging.

This module is NEVER imported by the offline test (which uses the mock seam in
``pipeline``); it is the production path the CLI uses without ``--mock``. It is
the only place in the llm package that imports the OpenAI client.

Calls are issued CONCURRENTLY through the async client: the pipeline collects a
whole phase's worth of requests (all the generation calls for a round, all the
validation calls for the candidate set) and hands them to the batch methods
below, which ``asyncio.gather`` them in ONE event loop. The client's
``asyncio.Semaphore(concurrency)`` bounds the in-flight calls, so gathering
everything saturates the concurrency budget without overshooting it. The cost
meter on the shared client accumulates across the whole run;
``ClientSeam.cost_summary()`` reports it for the run log.

The dispatch is the ONLY thing that changed from the old sequential bridge: each
request below carries the IDENTICAL (model, messages, params, seed) the
sequential path used, so the disk cache key is byte-identical and re-runs HIT
the existing cache. Parallelising never alters which seed maps to which call.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ....client import OpenAIClient, response_text
from .config import get_rule_config
from .pipeline import (
    GEN_MODEL,
    VALIDATOR_A_MODEL,
    VALIDATOR_B_MODEL,
    GenRequest,
    LabelRequest,
)

# generation sampling: matched-pair opinion sentences benefit from temperature;
# validation is a judgement call run at temperature 0 for reproducibility.
GEN_TEMPERATURE = 0.8
GEN_MAX_TOKENS = 400
VAL_TEMPERATURE = 0.0
VAL_MAX_TOKENS = 4


class ClientSeam:
    """Holds one OpenAIClient and exposes the pipeline's BATCHED dispatch bound to
    it (so caching + cost metering are shared across the run).

    The pipeline drives the real run through ``generate_many`` / ``label_many``,
    which fan out a whole phase's requests CONCURRENTLY via the async client
    (bounded by the client's ``Semaphore(concurrency)``). The single-call
    ``generator`` / ``labeler`` callables are kept for back-compat / direct use,
    but the pipeline prefers the batch path when the seam exposes it.

    Use as (SINGLE event loop wraps the whole run, incl. aclose)::

        async def drive():
            seam = ClientSeam(concurrency=16, cache_dir="cache")
            try:
                result = await run_pipeline_async(rule_id, seam, ...)
                cost = seam.cost_summary()
                return result, cost
            finally:
                await seam.aclose()       # SAME loop the calls ran on

        result, cost = asyncio.run(drive())

    A fresh ``asyncio.run`` per phase (generation, validation, aclose) would bind
    the underlying httpx pool to a loop the next phase had already closed
    ('RuntimeError: Event loop is closed'); awaiting every phase + aclose on the
    one loop above is the fix.
    """

    # marks this object as a concurrent dispatcher so run_pipeline fans out
    # generation + validation rather than calling one request at a time.
    is_concurrent: bool = True

    def __init__(
        self,
        *,
        concurrency: int = 16,
        cache_dir: str = "cache",
        api: Any | None = None,
        run_log: Any | None = None,
    ) -> None:
        self._client = OpenAIClient(concurrency=concurrency, cache_dir=cache_dir, api=api)
        self._run_log = run_log

    # ---- single-call seam (back-compat; one event loop per call) -----------
    # NB: these open a fresh ``asyncio.run`` per call, so they MUST NOT be used to
    # drive a multi-phase real run over a shared loop-bound client (that is the
    # 'Event loop is closed' bug). The production path passes the seam ITSELF to
    # ``run_pipeline_async`` (concurrent batch dispatch under one loop). These stay
    # only for direct, one-off single-call use (e.g. a one-shot probe / tests).
    def generator(self, rule_id: str, label: bool, topic: str, n: int, *, seed: int) -> list[str]:
        (out,) = asyncio.run(
            self.generate_many([GenRequest(rule_id, label, topic, n, seed)])
        )
        return out

    def labeler(self, rule_id: str, which: str, text: str, *, seed: int) -> bool | None:
        (out,) = asyncio.run(
            self.label_many([LabelRequest(rule_id, which, text, seed)])
        )
        return out

    # ---- async batch dispatch (CONCURRENT, cache-identical) ----------------
    async def _generate_one(self, req: "GenRequest") -> list[str]:
        cfg = get_rule_config(req.rule_id)
        messages = cfg.generation_messages(req.label, req.topic, req.n)
        record = await self._client.complete(
            GEN_MODEL,
            messages,
            temperature=GEN_TEMPERATURE,
            max_tokens=GEN_MAX_TOKENS,
            seed=req.seed,
        )
        if self._run_log is not None:
            self._run_log.log_response(
                {
                    "phase": "generate",
                    "rule_id": req.rule_id,
                    "label": req.label,
                    "topic": req.topic,
                    **record,
                }
            )
        text = response_text(record)
        return [line for line in text.splitlines() if line.strip()]

    async def _label_one(self, req: "LabelRequest") -> bool | None:
        cfg = get_rule_config(req.rule_id)
        model = VALIDATOR_A_MODEL if req.which == "A" else VALIDATOR_B_MODEL
        messages = cfg.validator_messages(req.which, req.text)
        record = await self._client.complete(
            model,
            messages,
            temperature=VAL_TEMPERATURE,
            max_tokens=VAL_MAX_TOKENS,
            seed=req.seed,
        )
        if self._run_log is not None:
            self._run_log.log_response(
                {
                    "phase": f"validate_{req.which}",
                    "rule_id": req.rule_id,
                    "text": req.text,
                    **record,
                }
            )
        return cfg.parse_validator_label(response_text(record))

    async def generate_many(self, requests: list["GenRequest"]) -> list[list[str]]:
        """Issue every generation request concurrently; results stay in input
        order (so the caller's seed->candidate mapping is unchanged)."""
        return list(await asyncio.gather(*(self._generate_one(r) for r in requests)))

    async def label_many(self, requests: list["LabelRequest"]) -> list[bool | None]:
        """Issue every validation request (pass A and pass B alike) concurrently;
        results stay in input order."""
        return list(await asyncio.gather(*(self._label_one(r) for r in requests)))

    def cost_summary(self) -> dict[str, Any]:
        return self._client.cost.summary()

    def stats(self) -> dict[str, Any]:
        return self._client.stats()

    async def aclose(self) -> None:
        await self._client.aclose()
