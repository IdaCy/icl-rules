"""Offline tests for rule 18 (physically_impossible): minimal-pair frame bank +
2-pass validation + by-base survival + programmatic split + emit.

ZERO network: every test drives the build through an INJECTED validator (the
deterministic mock, or a FakeAPI behind the real OpenAIClient). No
OPENAI_API_KEY, no HTTP.

The headline test (`test_full_offline_flow`) asserts the WHOLE
author -> validate -> by-base-survive -> split -> emit flow on the REAL frame
bank produces:
  * the programmatic split sizes (few_shot 200 = 100 bases x 2, held_out 120,
    confirmation 100, spare >= 40) with exact 50/50 balance,
  * a base in exactly one split, both variants sharing the base_id,
  * two-validator-agreement provenance (validated_agreement) on every item,
  * groundtruth.assert_labels_correct passing (validator-derived path),
  * schema.validate_full passing and the contexts loader round-trip,
  * a confound report that is computed and passes (is_llm_rule path).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icl_articulation import contexts
from icl_articulation.datagen import groundtruth, schema
from icl_articulation.datagen.confound import build_confound_report
from icl_articulation.datagen.generators.llm import frames_physically_impossible as bank
from icl_articulation.datagen.generators.llm import physically_impossible as r18


# =============================================================================
# frame bank: the authored invariants
# =============================================================================


def test_frame_bank_audit_passes_at_import() -> None:
    s = bank.AUDIT_SUMMARY
    assert s["n_frames"] >= bank.MIN_FRAMES  # >= 440 authored frames
    # each impossibility type within [20%, 35%]
    for t, share in s["type_shares"].items():
        assert bank.TYPE_SHARE_MIN <= share <= bank.TYPE_SHARE_MAX, (t, share)
    # cross-class word reuse >= 80% (the key 'no lexical proxy' guard)
    assert s["cross_class_reuse_fraction"] >= bank.CROSS_CLASS_REUSE_MIN


def test_frame_bank_single_word_length_matched_mundane() -> None:
    for fr in bank.FRAMES:
        assert fr.template.count("{S}") == 1
        for filler in (fr.plausible, fr.impossible):
            assert len(filler.split()) == 1
            assert filler.isascii() and filler.isalpha()
        # length-matched +/- 2 alphabetic chars
        assert abs(bank._alpha_len(fr.plausible) - bank._alpha_len(fr.impossible)) <= 2
        # mundane only (no fantasy lexicon)
        toks = set(fr.template.lower().replace("{s}", "").split())
        toks |= {fr.plausible.lower(), fr.impossible.lower()}
        assert not (toks & bank.BANNED_LEXICON)


def test_frame_bank_audit_is_loud_on_a_bad_bank() -> None:
    bad = list(bank.FRAMES[:10]) + [
        bank.Frame("the {S} ate the magic potion at dawn", "boy", "wizard",
                   bank.TYPE_INANIMATE_AGENT)
    ]
    with pytest.raises(bank.FrameBankError):
        bank.audit(bad)  # too few frames AND banned lexicon -> raises


def test_all_frames_style_conformant_after_sentence_case() -> None:
    # every filled, sentence-cased variant obeys the global style + word window.
    items = []
    for i, fr in enumerate(bank.FRAMES):
        for w, lab in ((fr.impossible, True), (fr.plausible, False)):
            text = r18._frame_text(fr.template, w)
            items.append({"text": text, "label": lab})
    schema.assert_sentence_style(items, rule_id=r18.RULE_ID)
    schema.assert_word_count_window(items)


# =============================================================================
# bases + verdict parsing
# =============================================================================


def test_build_bases_distinct_and_above_floor() -> None:
    bases = r18.build_bases()
    assert len(bases) >= schema.PROGRAMMATIC_N_BASES_MIN
    ids = [b.base_id for b in bases]
    assert len(set(ids)) == len(ids)  # distinct base_ids
    # every base has exactly two variants, sharing the base_id, differing by one
    # slot word (the minimal pair).
    for b in bases:
        assert b.impossible_variant.base_id == b.base_id == b.plausible_variant.base_id
        assert b.impossible_variant.label is True
        assert b.plausible_variant.label is False
        assert b.impossible_variant.text != b.plausible_variant.text


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("impossible", r18.IMPOSSIBLE),
        ("Possible.", r18.POSSIBLE),
        ("unclear", r18.UNCLEAR),
        ("  IMPOSSIBLE  ", r18.IMPOSSIBLE),
        ("The event is impossible", r18.IMPOSSIBLE),
        ("possible, though strange", r18.POSSIBLE),
    ],
)
def test_parse_verdict(raw: str, expected: str) -> None:
    assert r18.parse_verdict(raw) == expected


def test_parse_verdict_loud_on_garbage() -> None:
    with pytest.raises(ValueError):
        r18.parse_verdict("banana")


# =============================================================================
# survival: a base lives only if BOTH variants pass
# =============================================================================


def test_survival_requires_both_variants() -> None:
    bases = r18.build_bases()[:3]
    verdicts: dict[str, r18.VariantVerdict] = {}
    # base 0: both pass; base 1: impossible variant disputed; base 2: plausible disputed
    for idx, b in enumerate(bases):
        iv, pv = b.impossible_variant, b.plausible_variant
        if idx == 0:
            verdicts[iv.text] = r18.VariantVerdict(iv, r18.IMPOSSIBLE, r18.IMPOSSIBLE)
            verdicts[pv.text] = r18.VariantVerdict(pv, r18.POSSIBLE, r18.POSSIBLE)
        elif idx == 1:
            verdicts[iv.text] = r18.VariantVerdict(iv, r18.IMPOSSIBLE, r18.UNCLEAR)
            verdicts[pv.text] = r18.VariantVerdict(pv, r18.POSSIBLE, r18.POSSIBLE)
        else:
            verdicts[iv.text] = r18.VariantVerdict(iv, r18.IMPOSSIBLE, r18.IMPOSSIBLE)
            verdicts[pv.text] = r18.VariantVerdict(pv, r18.POSSIBLE, r18.IMPOSSIBLE)
    survivors = r18.survive_bases(bases, verdicts)
    assert [b.base_id for b in survivors] == [bases[0].base_id]


def test_survive_bases_loud_on_missing_verdict() -> None:
    bases = r18.build_bases()[:1]
    with pytest.raises(ValueError):
        r18.survive_bases(bases, {})  # no verdict for the base's variants


# =============================================================================
# the full offline flow (perfect mock)
# =============================================================================


def test_full_offline_flow(tmp_path: Path) -> None:
    summary = r18.run_build(
        r18.make_mock_pair_validator(),
        seed=11,
        data_dir=tmp_path,
        write=True,
    )
    # gates
    assert summary.gate_schema and summary.gate_groundtruth and summary.gate_confound
    # programmatic split sizes
    sc = summary.split_counts
    assert sc["few_shot_pool"] == {"true": 100, "false": 100, "total": 200}
    assert sc["held_out"] == {"true": 60, "false": 60, "total": 120}
    assert sc["confirmation"] == {"true": 50, "false": 50, "total": 100}
    assert sc["spare"]["true"] == sc["spare"]["false"]  # balanced spare
    assert sc["spare"]["total"] >= 2 * schema.PROGRAMMATIC_SPARE_MIN

    items_path = tmp_path / r18.RULE_ID / "items.jsonl"
    report_path = tmp_path / r18.RULE_ID / "confound_report.json"
    assert items_path.is_file() and report_path.is_file()

    items = schema.read_items(items_path)
    # every item carries the two-validator agreement provenance == its stored label
    for it in items:
        meta = it["slots_meta"]
        assert meta[groundtruth.VALIDATED_FLAG] == it["label"]
        assert meta["validator_pass_a"]["model"] == r18.PASS_A_MODEL
        assert meta["validator_pass_b"]["model"] == r18.PASS_B_MODEL

    # groundtruth (validator-derived: requires the provenance, never recomputes)
    groundtruth.assert_labels_correct(r18.RULE_ID, items)
    assert groundtruth.verify_dataset(r18.RULE_ID, items_path) == len(items)

    # schema + contexts loader round-trip
    schema.validate_full(items, rule_id=r18.RULE_ID)
    assert len(contexts.load_items(items_path)) == len(items)

    # a base lives in exactly one split; few_shot bases keep both variants
    schema.assert_split_base_disjoint(items)
    fs_bases = {it["base_id"] for it in items if it["split"] == "few_shot_pool"}
    fs_items = [it for it in items if it["split"] == "few_shot_pool"]
    assert len(fs_items) == 2 * len(fs_bases)

    # confound report is the is_llm_rule path and passes; word count identical by
    # construction so the |mean_T - mean_F| diff is ~0.
    report = json.loads(report_path.read_text())
    assert report["is_llm_rule"] is True
    assert report["overall_pass"] is True
    # the two variants of a base share an EXACT word count (single-word swap), so
    # the class-conditional mean word count is closely matched and well inside the
    # llm tolerance (1.0). (It is not exactly 0: held_out/confirmation/spare keep
    # ONE variant per base, so the True and False items there come from different
    # frames; the few_shot pool keeps both and is exactly matched.)
    assert report["word_count_mean_abs_diff"] <= report["length_match_tolerance"]
    assert report["word_count_mean_abs_diff"] < 0.5


def test_offline_flow_with_validator_drops_still_survives(tmp_path: Path) -> None:
    # a 15%-per-variant drop exercises by-base survival; the bank has enough
    # headroom (>= 440 frames) to still fill the split (recipe tolerates ~23%).
    summary = r18.run_build(
        r18.make_mock_pair_validator(drop_fraction=0.15, drop_salt=2),
        seed=5,
        data_dir=tmp_path,
        write=False,
    )
    assert summary.drop_rate_base > 0.0  # bases really were dropped
    assert summary.n_bases_survived >= 340
    assert summary.gate_schema and summary.gate_groundtruth and summary.gate_confound


def test_too_many_drops_is_loud(tmp_path: Path) -> None:
    # an 80% drop cannot fill the programmatic split -> loud, nothing written.
    with pytest.raises(ValueError):
        r18.run_build(
            r18.make_mock_pair_validator(drop_fraction=0.80, drop_salt=9),
            seed=1,
            data_dir=tmp_path,
            write=False,
        )


# =============================================================================
# the FakeAPI path through the REAL OpenAIClient (still zero network)
# =============================================================================


def test_api_validator_through_fake_client(tmp_path: Path) -> None:
    """make_api_validator drives the real OpenAIClient with a FakeAPI that always
    answers 'impossible' -> both passes agree, so impossible variants pass and
    plausible ones are disputed; exercises the async two-pass fan-out + parsing."""
    import asyncio

    from icl_articulation.client import OpenAIClient
    from tests.conftest import FakeAPI, fake_response_data

    api = FakeAPI(fake_response_data(text="impossible"))
    cli = OpenAIClient(api=api, cache_dir=str(tmp_path / "cache"))
    try:
        validate_all = r18.make_api_validator(cli, seed=0)
        bases = r18.build_bases()[:2]
        texts = []
        for b in bases:
            texts.append(b.impossible_variant.text)
            texts.append(b.plausible_variant.text)
        pairs = asyncio.run(validate_all(texts))
    finally:
        asyncio.run(cli.aclose())
    assert len(pairs) == len(texts)
    # every pair is (impossible, impossible) because the FakeAPI always says so
    # (asyncio.gather returns lists, so compare as lists)
    assert all(list(p) == [r18.IMPOSSIBLE, r18.IMPOSSIBLE] for p in pairs)
    # two passes (A+B) per text -> the client made 2 * len(texts) calls
    assert cli.n_api_calls == 2 * len(texts)


# =============================================================================
# REGRESSION: the WHOLE rule-18 real run must complete under ONE event loop
# =============================================================================
#
# run_api_build validated every variant under one asyncio.run, but then closed the
# client under a SECOND asyncio.run(cli.aclose()) in the finally. The client's
# httpx pool was bound to the first (now-closed) loop, so the close raised
# 'RuntimeError: Event loop is closed' — the run never finished and items.jsonl was
# never written. The fix awaits validation AND aclose on ONE loop.


class _LoopBoundR18API:
    """A FAKE AsyncOpenAI mimicking openai.AsyncOpenAI + httpx loop-binding: the
    pool binds to the loop active on first use; using OR closing it from a
    different loop raises 'RuntimeError: Event loop is closed' (the production
    crash). It answers the AUTHOR-intended verdict for every known variant surface
    (impossible-filled -> 'impossible', plausible-filled -> 'possible'), parsed out
    of the validation prompt, so the survive/split/emit flow reaches a passing
    emit with zero network."""

    def __init__(self) -> None:
        self._bound_loop = None
        self.closed = False
        self.chat = self
        self.completions = self
        # surface -> author-intended verdict, from the real bank
        self._intended: dict[str, str] = {}
        for base in r18.build_bases():
            self._intended[base.impossible_variant.text] = r18.IMPOSSIBLE
            self._intended[base.plausible_variant.text] = r18.POSSIBLE

    def _bind(self) -> None:
        import asyncio

        loop = asyncio.get_running_loop()
        if self._bound_loop is None:
            self._bound_loop = loop
        elif loop is not self._bound_loop:
            raise RuntimeError("Event loop is closed")

    @staticmethod
    def _extract_text(user: str) -> str:
        # prompts embed the sentence as 'Sentence: {text}\n\n' (A) or
        # 'Statement: {text}\n\n' (B); pull it back out for an O(1) verdict lookup.
        for marker in ("Sentence: ", "Statement: "):
            if marker in user:
                return user.split(marker, 1)[1].split("\n", 1)[0].strip()
        return ""

    async def create(self, **kwargs):
        self._bind()
        user = kwargs["messages"][-1]["content"]
        text = self._extract_text(user)
        verdict = self._intended.get(text, r18.UNCLEAR)
        data = {
            "model": kwargs["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": verdict}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 1, "total_tokens": 31},
        }

        class _Dump:
            def model_dump(self_inner):
                return data

        return _Dump()

    async def close(self):
        self._bind()  # must close on the bound loop
        self.closed = True


def test_run_api_build_single_loop_emits_and_closes_clean(tmp_path: Path, monkeypatch) -> None:
    """The PRODUCTION rule-18 entrypoint run_api_build (the same path the CLI's
    real run uses) must EMIT items.jsonl and close the client cleanly under ONE
    event loop. We inject a loop-bound FakeAPI into the OpenAIClient it builds;
    if the run used a separate asyncio.run for validation vs aclose, the close
    would raise 'Event loop is closed'."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-used")
    from icl_articulation.client import OpenAIClient

    api = _LoopBoundR18API()

    # inject the loop-bound fake into the client run_api_build constructs
    def _client_factory(*args, **kwargs):
        kwargs.setdefault("api", api)
        return OpenAIClient(*args, **kwargs)

    monkeypatch.setattr(r18.client_mod, "OpenAIClient", _client_factory)

    summary = r18.run_api_build(
        seed=7,
        data_dir=tmp_path,
        write=True,
        cache_dir=str(tmp_path / "cache"),
        results_dir=str(tmp_path / "results"),
    )

    # it EMITTED a passing dataset (the symptom that was missing on the broken run)
    items_path = tmp_path / r18.RULE_ID / "items.jsonl"
    assert items_path.is_file()
    items = schema.read_items(items_path)
    assert len(items) == summary.n_items > 0
    assert summary.gate_schema and summary.gate_groundtruth and summary.gate_confound
    # the client closed cleanly on the bound loop (no 'Event loop is closed')
    assert api.closed is True


def test_r18_old_multi_asyncio_run_pattern_would_crash(tmp_path: Path) -> None:
    """Proof the rule-18 regression is caught: with the SAME loop-bound FakeAPI,
    validating under one asyncio.run and then closing under a SECOND asyncio.run
    (the OLD run_api_build shape) raises 'Event loop is closed' on the close."""
    import asyncio

    from icl_articulation.client import OpenAIClient

    api = _LoopBoundR18API()
    cli = OpenAIClient(api=api, cache_dir=str(tmp_path / "cache"))
    validate_all = r18.make_api_validator(cli, seed=0)
    bases = r18.build_bases()[:2]
    texts = []
    for b in bases:
        texts.append(b.impossible_variant.text)
        texts.append(b.plausible_variant.text)
    # phase: validation in loop #1 (binds the pool)
    asyncio.run(validate_all(texts))
    # the OLD separate aclose under a FRESH loop #2 -> pool bound to closed loop #1
    with pytest.raises(RuntimeError, match="Event loop is closed"):
        asyncio.run(cli.aclose())


def test_api_validator_runs_concurrently(tmp_path: Path) -> None:
    """Rule 18's make_api_validator fans every pass-A/pass-B call out via
    asyncio.gather: a sleeping fake API records max-in-flight > 1 (a serial
    validator would peak at 1)."""
    import asyncio

    from icl_articulation.client import OpenAIClient

    class _TrackingAPI:
        def __init__(self, delay: float = 0.02) -> None:
            self.delay = delay
            self.in_flight = 0
            self.max_in_flight = 0
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                await asyncio.sleep(self.delay)
            finally:
                self.in_flight -= 1
            data = {
                "model": kwargs["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "impossible"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }

            class _Dump:
                def model_dump(self_inner):
                    return data

            return _Dump()

        async def close(self):
            pass

    api = _TrackingAPI()
    cli = OpenAIClient(api=api, cache_dir=str(tmp_path / "cache"), concurrency=16)
    try:
        validate_all = r18.make_api_validator(cli, seed=0)
        bases = r18.build_bases()[:20]
        texts = []
        for b in bases:
            texts.append(b.impossible_variant.text)
            texts.append(b.plausible_variant.text)
        asyncio.run(validate_all(texts))
    finally:
        asyncio.run(cli.aclose())
    # 20 bases x 2 variants x 2 passes = 80 calls, fanned out concurrently.
    assert api.max_in_flight > 1
    assert api.max_in_flight == 16


# =============================================================================
# cost estimate
# =============================================================================


def test_cost_estimate_shape() -> None:
    n_variants = 2 * len(r18.build_bases())
    est = r18.estimate_cost(n_variants)
    assert est["n_variants"] == n_variants
    assert est["n_calls"] == 2 * n_variants  # pass A + pass B per variant
    assert est["pass_a_model"] == r18.PASS_A_MODEL
    assert est["pass_b_model"] == r18.PASS_B_MODEL
    assert est["total_usd"] == pytest.approx(est["pass_a_usd"] + est["pass_b_usd"])
    assert est["total_usd"] > 0.0
