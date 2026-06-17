"""Offline tests for the llm_validated generation pipeline (rules 15, 16).

ZERO network: every test drives the shared pipeline through the injected
mock / fake seam (pipeline.mock_generator / mock_labeler, or a FakeAPI behind
the real ClientSeam). No OPENAI_API_KEY, no HTTP, no nltk download.

The headline test (`test_full_offline_flow_*`) asserts the WHOLE
generate -> validate -> rebalance -> emit flow on a tiny fake corpus produces:
  * the exact llm split sizes (120 / 120 / 100) and a spare remainder,
  * exact 50/50 balance in every balanced split,
  * two-validator-agreement provenance on every item,
  * groundtruth.assert_labels_correct passing (validator-derived path),
  * a confound report that is computed and passes its audit thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from icl_articulation.datagen.confound import build_confound_report
from icl_articulation.datagen.generators.llm import (
    get_rule_config,
    mock_generator,
    mock_labeler,
    run_pipeline,
)
from icl_articulation.datagen.generators.llm.config import contains_eat_verb
from icl_articulation.datagen.generators.llm.cost import estimate_cost
from icl_articulation.datagen.generators.llm.pipeline import (
    Candidate,
    LLMPipelineError,
    normalize_candidate,
    style_ok,
)
from icl_articulation.datagen.groundtruth import (
    VALIDATED_FLAG,
    GroundTruthError,
    assert_labels_correct,
    verify_dataset,
)
from icl_articulation.datagen.schema import LLM_SPLIT_ITEMS, read_items, validate_full

RULES = ["positive_sentiment", "food_topic"]


# --- the headline full-flow test ----------------------------------------------


@pytest.mark.parametrize("rule_id", RULES)
def test_full_offline_flow(tmp_path: Path, rule_id: str) -> None:
    """generate -> validate -> rebalance -> emit on the mock seam, all gates."""
    result = run_pipeline(
        rule_id,
        mock_generator,
        mock_labeler,
        seed=0,
        max_candidates=600,
        data_dir=tmp_path,
        run_pos=False,  # no nltk in CI
    )

    # items + report were written
    items_path = tmp_path / rule_id / "items.jsonl"
    report_path = tmp_path / rule_id / "confound_report.json"
    assert items_path.is_file()
    assert report_path.is_file()
    assert result.items_path == str(items_path)

    items = read_items(items_path)

    # exact split sizes (the llm 120/120/100 scheme) + a spare remainder
    by_split: dict[str, list] = {}
    for it in items:
        by_split.setdefault(it["split"], []).append(it)
    for split, n in LLM_SPLIT_ITEMS.items():
        assert len(by_split[split]) == n, (split, len(by_split[split]))
    assert "spare" in by_split and len(by_split["spare"]) > 0

    # exact 50/50 in every balanced split
    for split in ("few_shot_pool", "held_out", "confirmation"):
        labels = [bool(it["label"]) for it in by_split[split]]
        assert sum(labels) == len(labels) // 2 == labels.count(False)

    # provenance present on EVERY item and equal to the stored label
    for it in items:
        meta = it["slots_meta"]
        assert VALIDATED_FLAG in meta
        assert bool(meta[VALIDATED_FLAG]) == bool(it["label"])
        assert meta["validator_a_model"]
        assert meta["validator_b_model"]
        assert meta["gen_model"]

    # groundtruth verifier passes on the validator-derived rule
    assert verify_dataset(rule_id, items_path) == len(items)

    # confound report present + passed its audit
    report = json.loads(report_path.read_text())
    assert report["is_llm_rule"] is True
    assert report["overall_pass"] is True
    assert result.confound_overall_pass is True

    # base_id == item_id for llm rules
    for it in items:
        assert it["base_id"] == it["item_id"]


@pytest.mark.parametrize("rule_id", RULES)
def test_topic_balance_after_rebalance(tmp_path: Path, rule_id: str) -> None:
    """Every topic appears in the emitted set and no single False topic exceeds
    25% of the False class (rule-specs audit_thresholds for rule 16; rule 15 is
    even tighter — exact 50/50 per topic by construction)."""
    cfg = get_rule_config(rule_id)
    run_pipeline(rule_id, mock_generator, mock_labeler, seed=0, max_candidates=600,
                 data_dir=tmp_path, run_pos=False)
    items = read_items(tmp_path / rule_id / "items.jsonl")
    falses = [it for it in items if not bool(it["label"])]
    by_topic: dict[str, int] = {}
    for it in falses:
        by_topic[it["slots_meta"]["topic"]] = by_topic.get(it["slots_meta"]["topic"], 0) + 1
    n_false = len(falses)
    for topic, n in by_topic.items():
        assert n / n_false <= 0.25 + 1e-9, (topic, n, n_false)


def test_keyword_audit_food_under_threshold(tmp_path: Path) -> None:
    """Rule 16's eat/drink/taste/cook keyword predicate must agree <= 75% with
    the labels (the no-verb quota keeps 'mentions eating' a distractor, not an
    equivalent)."""
    result = run_pipeline("food_topic", mock_generator, mock_labeler, seed=0,
                          max_candidates=600, data_dir=tmp_path, run_pos=False)
    kw = result.keyword_audit
    assert kw["predicate"] == "mentions_eat_drink_taste_cook_verb"
    assert kw["max_agreement"] <= 0.75
    assert kw["passes"] is True


# =============================================================================
# the STRUCTURAL hard gates fire through the EMIT pipeline (rules 15/16)
# =============================================================================
#
# Each test wraps mock_generator so the emitted (compliant-elsewhere) corpus
# DELIBERATELY violates ONE structural audit_threshold (a rate gate or topic
# balance), then asserts run_pipeline RAISES (loud, nothing written). The
# wrappers append a confound to the generated text AFTER the class marker, so
# mock_labeler still recovers the class (the items survive validation and reach
# the emit gate). The injected token sits well over the gap so the gate fires
# deterministically. EXCEPTION: a class-exclusive high-frequency token is NOT a
# structural gate — it is the rule's own signal for sentiment/food and is routed
# to the dataset judge; that case (test_emit_reports_..._for_the_judge) asserts
# the pipeline EMITS and reports the token rather than raising.


def _wrap_gen(transform):
    """Wrap mock_generator, applying ``transform(label, line)`` to each line."""
    def _g(rule_id, label, topic, n, *, seed):
        return [transform(label, line) for line in mock_generator(rule_id, label, topic, n, seed=seed)]
    return _g


def test_emit_reports_token_in_one_class_only_for_the_judge(tmp_path: Path) -> None:
    """A high-frequency token confined to the True class is NOT a hard gate: for
    sentiment/food a class-exclusive >= 3% token is the rule's own signal, routed
    to the dataset JUDGE. The emit pipeline must NOT raise on it — it emits and
    reports the token in the confound JSON's judge_tokens_missing_from_class."""
    # 'spaghetti' inserted into EVERY True line -> 100% of True, 0% of False.
    # Inserted BEFORE the unique "item <uid>" tail (not at the very end) so the
    # last-word battery predicates stay matched across classes — the ONLY skew is
    # the class-exclusive token, exactly the judge-reported case under test.
    def _inject(label, line):
        if not label:
            return line
        toks = line.split()
        # tail is "... item <uid>"; splice 'spaghetti' just before it
        toks.insert(len(toks) - 2, "spaghetti")
        return " ".join(toks)

    gen = _wrap_gen(_inject)
    result = run_pipeline("positive_sentiment", gen, mock_labeler, seed=0,
                          max_candidates=600, data_dir=tmp_path, run_pos=False)
    # it emitted (nothing raised) and the report passed the STRUCTURAL audit
    assert result.confound_overall_pass is True
    items_path = tmp_path / "positive_sentiment" / "items.jsonl"
    report_path = tmp_path / "positive_sentiment" / "confound_report.json"
    assert items_path.exists()
    report = json.loads(report_path.read_text())
    at = report["audit_thresholds"]
    # the class-exclusive token is REPORTED for the judge, NOT in the hard checks
    assert "tokens_missing_from_a_class" not in at["checks"]
    missing = {d["token"]: d for d in at["judge_tokens_missing_from_class"]}
    assert "spaghetti" in missing
    assert missing["spaghetti"]["false_count"] == 0
    assert missing["spaghetti"]["true_count"] > 0
    assert report["overall_pass"] is True


def test_emit_rejects_negator_rate_gap(tmp_path: Path) -> None:
    """A negator confined to the True class trips the negator-rate gate. For
    rule 15 the negator-rate audit_threshold is enforced at the keyword_audit
    pre-check (its predicate IS negator_rate) AND again in the confound
    audit_thresholds; either way the emit pipeline RAISES with a negator message."""
    # 'no' on every True item, none on False -> negator-rate gap ~1.0 > 0.05.
    # ('no' replaces a content word so word count stays matched.)
    gen = _wrap_gen(lambda label, line: line.replace(" felt ", " felt no ").replace(" was ", " was no ") if label else line)
    with pytest.raises(LLMPipelineError) as ei:
        run_pipeline("positive_sentiment", gen, mock_labeler, seed=0,
                     max_candidates=600, data_dir=tmp_path, run_pos=False)
    assert "negator" in str(ei.value).lower()
    assert not (tmp_path / "positive_sentiment" / "items.jsonl").exists()


def test_emit_rejects_word_count_gap(tmp_path: Path) -> None:
    """A word-count gap > 1.0 between classes trips the word-count gate. We
    SHORTEN the False class by ~2 words (drop the literal word 'item' and the
    first 'the' from its tail) while KEEPING the unique numeric id, so False
    items stay distinct (no duplicate-surface) and inside the style window
    [4,12] — the True class is then ~2 words longer and the emit pipeline RAISES
    on the mean-word-count diff."""
    def shorten_false(label, line):
        if label:
            return line
        toks = line.split()
        # drop the literal "item" token (keep the unique number -> still distinct)
        if "item" in toks:
            toks.remove("item")
        # drop the first lowercase "the" to lose a second word
        for k, t in enumerate(toks):
            if t == "the":
                del toks[k]
                break
        return " ".join(toks)

    gen = _wrap_gen(shorten_false)
    with pytest.raises(LLMPipelineError) as ei:
        run_pipeline("positive_sentiment", gen, mock_labeler, seed=0,
                     max_candidates=600, data_dir=tmp_path, run_pos=False)
    msg = str(ei.value)
    # the word-count gate fires (reported as the length match and/or the
    # audit_thresholds word-count check, both keyed on the same mean-wc diff)
    assert "word_count_mean_abs_diff" in msg or "mean_T-mean_F" in msg
    assert not (tmp_path / "positive_sentiment" / "items.jsonl").exists()


def test_emit_rejects_false_topic_over_25pct(tmp_path: Path, monkeypatch) -> None:
    """Collapsing the False class onto a SINGLE topic (> 25% of the False class)
    trips the topic-balance HARD gate through the emit pipeline. We monkeypatch
    the rule config so there is only ONE false_topic: every False item then
    carries that one topic (slots_meta['topic'] = the generation cell topic), so
    the emitted False class is 100% one topic -> > 25% -> the pipeline RAISES."""
    import dataclasses

    import icl_articulation.datagen.generators.llm.pipeline as pipe

    base_cfg = get_rule_config("positive_sentiment")
    # single false topic; keep the 8 true topics so True stays balanced (<=25%)
    one_false_cfg = dataclasses.replace(base_cfg, false_topics=("restaurants",))
    monkeypatch.setattr(pipe, "get_rule_config", lambda rid: one_false_cfg)

    with pytest.raises(LLMPipelineError) as ei:
        run_pipeline("positive_sentiment", mock_generator, mock_labeler, seed=0,
                     max_candidates=600, data_dir=tmp_path, run_pos=False)
    assert "topic_balance" in str(ei.value)
    assert not (tmp_path / "positive_sentiment" / "items.jsonl").exists()


def test_emit_rejects_food_no_verb_quota_short(tmp_path: Path) -> None:
    """Rule 16 MINOR fix: if FEWER than 55% of True (food) items contain NO
    eat/drink/taste/cook verb, the keyword-quota HARD gate rejects the dataset
    (post-validation, not just a prompt request)."""
    # append a 'cooked' verb to EVERY True (food) line -> no-verb fraction 0% < 55%
    # (still food-labelled by the marker, so it survives validation).
    gen = _wrap_gen(lambda label, line: (line + " cooked") if label else line)
    with pytest.raises(LLMPipelineError) as ei:
        run_pipeline("food_topic", gen, mock_labeler, seed=0,
                     max_candidates=600, data_dir=tmp_path, run_pos=False)
    msg = str(ei.value)
    assert "keyword" in msg.lower() or "no_verb_quota" in msg or "audit" in msg.lower()
    assert not (tmp_path / "food_topic" / "items.jsonl").exists()


def test_food_no_verb_quota_value_enforced() -> None:
    """The keyword_audit exposes the realized no-verb fraction and gates on it:
    a corpus where every True item has an eat-verb has fraction 0 and FAILS;
    a verb-free True class has fraction 1 and PASSES (with agreement in range)."""
    from icl_articulation.datagen.generators.llm.config import get_rule_config
    from icl_articulation.datagen.generators.llm.pipeline import keyword_audit

    cfg = get_rule_config("food_topic")

    def _mk(texts_labels):
        return [
            {"item_id": f"i{i}", "label": l, "text": t, "rule_id": "food_topic",
             "base_id": f"i{i}", "slots_meta": {}, "split": "few_shot_pool"}
            for i, (t, l) in enumerate(texts_labels)
        ]

    # True items all contain 'cooked' -> no-verb fraction 0 -> fails the quota
    bad = _mk(
        [("the soup cooked slowly today here", True)] * 10
        + [("the racket sat near the door today", False)] * 10
    )
    kwa = keyword_audit(cfg, bad)
    assert kwa["true_no_verb_fraction"] == 0.0
    assert kwa["no_verb_quota_ok"] is False
    assert kwa["passes"] is False

    # True items verb-free -> fraction 1, agreement balanced -> passes
    good = _mk(
        [("the soup sat in the bowl today", True)] * 10
        + [("the racket sat near the door today", False)] * 10
    )
    kwg = keyword_audit(cfg, good)
    assert kwg["true_no_verb_fraction"] == 1.0
    assert kwg["no_verb_quota_ok"] is True
    assert kwg["agreement_ok"] is True
    assert kwg["passes"] is True


def test_determinism(tmp_path: Path) -> None:
    """Same seed + mock seam -> identical emitted texts (reproducibility)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    run_pipeline("positive_sentiment", mock_generator, mock_labeler, seed=7,
                 max_candidates=600, data_dir=a, run_pos=False)
    run_pipeline("positive_sentiment", mock_generator, mock_labeler, seed=7,
                 max_candidates=600, data_dir=b, run_pos=False)
    ta = [it["text"] for it in read_items(a / "positive_sentiment" / "items.jsonl")]
    tb = [it["text"] for it in read_items(b / "positive_sentiment" / "items.jsonl")]
    assert ta == tb


# --- groundtruth provenance is genuinely enforced -----------------------------


def test_missing_provenance_raises(tmp_path: Path) -> None:
    """Stripping the validated_agreement flag makes the validator-derived
    groundtruth check fail (provenance is load-bearing, not cosmetic)."""
    run_pipeline("food_topic", mock_generator, mock_labeler, seed=0,
                 max_candidates=600, data_dir=tmp_path, run_pos=False)
    items = read_items(tmp_path / "food_topic" / "items.jsonl")
    for it in items:
        it["slots_meta"].pop(VALIDATED_FLAG, None)
    with pytest.raises(GroundTruthError):
        assert_labels_correct("food_topic", items)


def test_provenance_disagreement_raises() -> None:
    """If the stamped agreement disagrees with the stored label, it raises."""
    items = [
        {
            "item_id": "food_topic-0000",
            "base_id": "food_topic-0000",
            "rule_id": "food_topic",
            "label": True,
            "text": "warm soup simmered gently with the garden herbs item 11",
            "slots_meta": {VALIDATED_FLAG: False},  # disagrees with stored True
            "split": "few_shot_pool",
        }
    ]
    with pytest.raises(GroundTruthError):
        assert_labels_correct("food_topic", items)


# --- the style filter ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,ok",
    [
        ("warm soup simmered gently with the garden herbs", True),
        ("the meal was great!", False),            # exclamation
        ("was it good", False),                    # too short (3 words)
        ("the dog ran", False),                    # too short
        ("do not like it the food was bland here", True),
        ("it was lovely don't you think it tasted nice", False),  # contraction apostrophe
        ("they served a well-known dish at the table tonight", False),  # hyphen
        ("café food was pleasant and warm tonight here friends", False),  # non-ascii
        ("I loved the meal it was a wonderful evening out", False),  # banned 'I'
    ],
)
def test_style_ok(text: str, ok: bool) -> None:
    assert style_ok(text) is ok


def test_normalize_strips_terminal_period_and_markers() -> None:
    assert normalize_candidate("1. The soup was warm and very fresh today.") == \
        "The soup was warm and very fresh today"
    assert normalize_candidate("- bright tulips opened along the narrow path") == \
        "bright tulips opened along the narrow path"
    # an item with a question mark is dropped (not repaired)
    assert normalize_candidate("was the soup good and warm today?") is None


# --- the eat-verb tag ---------------------------------------------------------


@pytest.mark.parametrize(
    "text,has",
    [
        ("she cooked the soup slowly", True),
        ("they ate the warm bread", True),
        ("the cookbook sat on the shelf", False),   # cookbook is not the verb
        ("the cookie jar was empty today", False),  # cookie is not 'cook'
        ("ripe tomatoes filled the wooden bowl", False),
    ],
)
def test_contains_eat_verb(text: str, has: bool) -> None:
    assert contains_eat_verb(text) is has


# --- cost estimate ------------------------------------------------------------


@pytest.mark.parametrize("rule_id", RULES)
def test_cost_estimate_shape(rule_id: str) -> None:
    e = estimate_cost(rule_id, 600)
    assert e["total_usd"] > 0
    # the three streams sum to the total
    assert abs(
        e["gen_cost_usd"] + e["validate_pass_a_cost_usd"] + e["validate_pass_b_cost_usd"]
        - e["total_usd"]
    ) < 1e-9
    # pass B (gpt-4.1) is the dominant cost
    assert e["validate_pass_b_cost_usd"] > e["validate_pass_a_cost_usd"]
    assert e["gen_model"] == "gpt-4.1-mini"
    assert e["validator_b_model"] == "gpt-4.1"
    # well under the $200 default max-cost
    assert e["total_usd"] < 200.0


# --- candidate.kept logic -----------------------------------------------------


def test_candidate_kept_requires_both_passes_agree() -> None:
    c = Candidate(text="x", intended_label=True, topic="t")
    c.pass_a, c.pass_b = True, True
    assert c.kept is True
    c.pass_a, c.pass_b = True, False
    assert c.kept is False        # passes disagree
    c.pass_a, c.pass_b = False, False
    assert c.kept is False        # agree but on the WRONG label
    c.pass_a, c.pass_b = None, None
    assert c.kept is False        # both neutral
    c.pass_a, c.pass_b = True, None
    assert c.kept is False        # one neutral


# --- the FakeAPI real-seam wiring (no network) --------------------------------


def test_real_seam_with_fake_api(tmp_path: Path, monkeypatch) -> None:
    """Drive the REAL ClientSeam (the production generate/validate seam, with
    caching + cost metering) against a FakeAPI — proving the api.py wiring is
    correct without any network. The fake returns class-appropriate generation
    lines and validator answers so the pipeline reaches a passing emit."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-used")
    from icl_articulation.datagen.generators.llm.api import ClientSeam

    cfg = get_rule_config("positive_sentiment")

    class FakeModelDump:
        def __init__(self, data):
            self._d = data

        def model_dump(self):
            return self._d

    # Large, decorrelated marker pools + a SHARED neutral scaffold, so the
    # FakeAPI's emitted corpus PASSES the new is_llm_rule audit_thresholds
    # (the token-in-both-classes content gate): the only class-skewed token is
    # the single rotating marker, and each marker stays under the 3% floor.
    POS_MARKERS = [
        "wonderful", "delightful", "brilliant", "lovely", "pleasant", "superb",
        "excellent", "charming", "splendid", "gorgeous", "radiant", "cheerful",
        "elegant", "soothing", "uplifting", "glorious", "vibrant", "refreshing",
        "amazing", "marvellous", "satisfying", "joyful", "serene", "stellar",
        "blissful", "dazzling", "heartening", "wholesome", "agreeable", "sunny",
        "graceful", "fabulous", "magical", "delicious", "rewarding", "pleasing",
        "charismatic", "spirited", "lively", "luminous",
    ]
    NEG_MARKERS = [
        "bland", "tiring", "dull", "clumsy", "miserable", "rude", "stressful",
        "frustrating", "careless", "dreary", "tedious", "awful", "dismal",
        "gloomy", "irritating", "shabby", "tiresome", "dreadful", "unpleasant",
        "lousy", "grim", "annoying", "horrible", "draining", "tasteless",
        "sloppy", "depressing", "harsh", "forgettable", "abysmal", "bleak",
        "joyless", "disagreeable", "exhausting", "wretched", "drab", "clumsiest",
        "tense", "weary", "sour",
    ]
    POS_SET = set(POS_MARKERS)

    class FakeCompletions:
        def __init__(self):
            self.n = 0  # per-call counter so generation lines stay distinct
            # per-class running marker index so markers cycle EVENLY (no marker
            # over-used past the 3% token-in-both-classes floor).
            self.pos_i = 0
            self.neg_i = 0

        async def create(self, **kwargs):
            model = kwargs["model"]
            messages = kwargs["messages"]
            user = messages[-1]["content"]
            # GENERATION request -> emit pairs_per_call class-appropriate lines
            if "Output ONLY the sentences" in user:
                label_pos = "POSITIVE" in user
                markers = POS_MARKERS if label_pos else NEG_MARKERS
                self.n += 1
                # shared scaffold; one rotating marker substituted per line,
                # cycled evenly via a per-class running counter.
                out_lines = []
                for i in range(cfg.pairs_per_call):
                    if label_pos:
                        m = markers[self.pos_i % len(markers)]
                        self.pos_i += 1
                    else:
                        m = markers[self.neg_i % len(markers)]
                        self.neg_i += 1
                    out_lines.append(
                        f"the visit there felt {m} to everyone number {self.n} {i}"
                    )
                content = "\n".join(out_lines)
            else:
                # VALIDATION request -> answer with the implied class word; the
                # sentence under test is in the prompt, so key off its marker.
                tokens = set(user.lower().split())
                content = "positive" if (tokens & POS_SET) else "negative"
            data = {
                "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            }
            return FakeModelDump(data)

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeAPI:
        def __init__(self):
            self.chat = FakeChat()

        async def close(self):
            pass

    # Drive through the CONCURRENT seam directly (the production dispatch), not the
    # single-call back-compat methods — so this exercises generate_many/label_many.
    # Drive through the CONCURRENT seam directly (the production dispatch), not the
    # single-call back-compat methods — so this exercises generate_many/label_many.
    seam = ClientSeam(cache_dir=str(tmp_path / "cache"), api=FakeAPI())
    result = run_pipeline(
        "positive_sentiment", seam,
        seed=0, max_candidates=600, data_dir=tmp_path, run_pos=False,
    )
    # the fake produced enough validated items to fill the splits
    assert result.n_emitted >= sum(LLM_SPLIT_ITEMS.values())
    assert result.confound_overall_pass is True
    # the cost meter saw fresh calls
    cost = seam.cost_summary()
    assert cost["total_usd"] > 0
    import asyncio

    asyncio.run(seam.aclose())


# =============================================================================
# REGRESSION: the WHOLE real run must complete under ONE event loop
# =============================================================================
#
# The bug: __main__._run_real drove the real (async) client with SEVERAL
# asyncio.run(...) calls — one per phase (generation, validation, targeted regen)
# PLUS a final asyncio.run(seam.aclose()). Each asyncio.run opens then CLOSES a
# fresh loop, but the OpenAIClient's httpx pool binds to the FIRST loop it ran in,
# so the second phase + aclose operated on a connection bound to an already-closed
# loop -> 'RuntimeError: Event loop is closed'. items.jsonl was never written.
#
# These tests reproduce that loop-binding with a FAKE async API (no network) and
# prove the fix: the production single-loop entrypoint (run_pipeline_async awaited
# alongside seam.aclose() in ONE asyncio.run) EMITS items.jsonl and closes clean,
# while the OLD per-phase multi-asyncio.run pattern raises 'Event loop is closed'.


class _LoopBoundSentimentAPI:
    """A FAKE AsyncOpenAI that faithfully mimics openai.AsyncOpenAI + httpx: its
    connection pool BINDS to the event loop active on first use, and using OR
    closing it from a DIFFERENT loop raises 'RuntimeError: Event loop is closed'
    (exactly the production crash). Pure-function responses (no shared mutable
    counters) so the concurrent dispatch is deterministic; generation embeds the
    request seed so lines stay distinct, validation answers from the marker."""

    _POS = [
        "wonderful", "delightful", "brilliant", "lovely", "pleasant", "superb",
        "excellent", "charming", "splendid", "gorgeous", "radiant", "cheerful",
        "elegant", "soothing", "uplifting", "glorious", "vibrant", "refreshing",
        "amazing", "marvellous", "satisfying", "joyful", "serene", "stellar",
        "blissful", "dazzling", "heartening", "wholesome", "agreeable", "sunny",
    ]
    _NEG = [
        "bland", "tiring", "dull", "clumsy", "miserable", "rude", "stressful",
        "frustrating", "careless", "dreary", "tedious", "awful", "dismal",
        "gloomy", "irritating", "shabby", "tiresome", "dreadful", "unpleasant",
        "lousy", "grim", "annoying", "horrible", "draining", "tasteless",
        "sloppy", "depressing", "harsh", "forgettable", "abysmal",
    ]

    def __init__(self) -> None:
        self._bound_loop = None
        self.closed = False
        self.chat = self
        self.completions = self
        self._pos = set(self._POS)

    def _bind(self) -> None:
        import asyncio

        loop = asyncio.get_running_loop()
        if self._bound_loop is None:
            self._bound_loop = loop
        elif loop is not self._bound_loop:
            # the exact production failure: pool bound to an already-closed loop
            raise RuntimeError("Event loop is closed")

    async def create(self, **kwargs):
        self._bind()
        user = kwargs["messages"][-1]["content"]
        seed = kwargs.get("seed", 0)
        if "Output ONLY the sentences" in user:
            label_pos = "POSITIVE" in user
            markers = self._POS if label_pos else self._NEG
            lines = [
                f"the visit there felt {markers[(seed * 8 + i) % len(markers)]} "
                f"to everyone today num {seed} {i}"
                for i in range(8)  # pairs_per_call
            ]
            content = "\n".join(lines)
        else:
            tokens = set(user.lower().split())
            content = "positive" if (tokens & self._pos) else "negative"
        data = {
            "model": kwargs["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
        }

        class _Dump:
            def model_dump(self_inner):
                return data

        return _Dump()

    async def close(self):
        # aclose must run on the SAME loop the pool bound to (else it raises).
        self._bind()
        self.closed = True


def test_real_run_single_loop_emits_and_closes_clean(tmp_path: Path, monkeypatch) -> None:
    """The PRODUCTION single-loop structure (the one __main__._run_real now uses):
    build the seam, AWAIT run_pipeline_async, AND await seam.aclose() — all inside
    ONE asyncio.run. With a loop-bound FakeAPI this must EMIT items.jsonl and close
    cleanly, with NO 'Event loop is closed' / RuntimeError."""
    import asyncio

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-used")
    from icl_articulation.datagen.generators.llm.api import ClientSeam
    from icl_articulation.datagen.generators.llm.pipeline import run_pipeline_async

    api = _LoopBoundSentimentAPI()

    async def drive():
        # mirrors __main__._run_real._drive: seam + whole flow + aclose, one loop
        seam = ClientSeam(concurrency=16, cache_dir=str(tmp_path / "cache"), api=api)
        try:
            result = await run_pipeline_async(
                "positive_sentiment", seam, seed=0, max_candidates=600,
                data_dir=tmp_path, run_pos=False,
            )
            return result, seam.cost_summary()
        finally:
            await seam.aclose()

    result, cost = asyncio.run(drive())  # must NOT raise 'Event loop is closed'

    # it EMITTED a dataset (the symptom that was missing on the broken run)
    items_path = tmp_path / "positive_sentiment" / "items.jsonl"
    assert items_path.is_file()
    assert result.items_path == str(items_path)
    items = read_items(items_path)
    assert len(items) == result.n_emitted >= sum(LLM_SPLIT_ITEMS.values())
    assert result.confound_overall_pass is True
    # aclose ran on the bound loop (clean close), and cost was metered
    assert api.closed is True
    assert cost["total_usd"] > 0


def test_old_multi_asyncio_run_pattern_would_crash(tmp_path: Path, monkeypatch) -> None:
    """Proof the new test would have CAUGHT the bug: with the SAME loop-bound
    FakeAPI, the OLD dispatch (a fresh asyncio.run per phase, plus a separate
    asyncio.run(seam.aclose())) raises 'Event loop is closed' the moment the second
    phase / the close touches the pool bound to the first, now-closed loop."""
    import asyncio

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-used")
    from icl_articulation.datagen.generators.llm.api import ClientSeam
    from icl_articulation.datagen.generators.llm.pipeline import GenRequest, LabelRequest

    api = _LoopBoundSentimentAPI()
    seam = ClientSeam(concurrency=16, cache_dir=str(tmp_path / "cache"), api=api)

    # phase 1 (generation) in loop #1 — binds the pool to that loop
    asyncio.run(seam.generate_many([GenRequest("positive_sentiment", True, "movies", 8, 0)]))
    # phase 2 (validation) in a FRESH loop #2 — the OLD per-phase asyncio.run shape;
    # the pool is now bound to the already-closed loop #1 -> 'Event loop is closed'
    with pytest.raises(RuntimeError, match="Event loop is closed"):
        asyncio.run(
            seam.label_many([LabelRequest("positive_sentiment", "A", "the visit felt lovely today here now", 0)])
        )
    # and the separate aclose under yet another loop also raises
    with pytest.raises(RuntimeError, match="Event loop is closed"):
        asyncio.run(seam.aclose())


# =============================================================================
# concurrency: the real seam fans validation + generation calls out in parallel
# =============================================================================


class _ConcurrencyTrackingAPI:
    """Fake AsyncOpenAI whose ``create`` sleeps briefly and records the maximum
    number of in-flight calls. A SEQUENTIAL pipeline (one await at a time) shows
    max_in_flight == 1; the concurrent dispatch should drive it well above 1
    (up to the client's concurrency bound)."""

    def __init__(self, *, delay: float = 0.02) -> None:
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.n_calls = 0
        self.chat = self  # so api.chat.completions.create resolves to us
        self.completions = self

    async def create(self, **kwargs):
        import asyncio

        self.in_flight += 1
        self.n_calls += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        # answer every validation request 'positive' and every generation request
        # with class-appropriate lines so the corpus survives (content is not
        # under test here — only the dispatch concurrency is).
        user = kwargs["messages"][-1]["content"]
        if "Output ONLY the sentences" in user:
            content = "the visit there felt fine to everyone today\n" * 0 + "ok line"
        else:
            content = "positive"
        data = {
            "model": kwargs["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

        class _Dump:
            def model_dump(self_inner):
                return data

        return _Dump()

    async def close(self):
        pass


def test_validation_runs_concurrently(tmp_path: Path, monkeypatch) -> None:
    """The concurrent seam drives MANY validation calls in flight at once.

    We validate a fixed candidate set directly through validate_candidates with a
    real ClientSeam over a sleeping fake API. A sequential dispatch would peak at
    1 in-flight; the gather peaks near the concurrency bound."""
    import asyncio

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-used")
    from icl_articulation.datagen.generators.llm.api import ClientSeam
    from icl_articulation.datagen.generators.llm.pipeline import (
        Candidate,
        validate_candidates,
    )

    cfg = get_rule_config("positive_sentiment")
    api = _ConcurrencyTrackingAPI(delay=0.02)
    seam = ClientSeam(concurrency=16, cache_dir=str(tmp_path / "cache"), api=api)
    # 40 distinct candidates -> 80 validation calls (pass A + B each); distinct
    # texts so none collide on the cache (every call hits the sleeping API).
    cands = [
        Candidate(text=f"the visit there felt good to everyone today n{i}",
                  intended_label=True, topic="t")
        for i in range(40)
    ]
    try:
        validate_candidates(cfg, cands, seam, seed=0)
    finally:
        asyncio.run(seam.aclose())

    assert api.n_calls == 80  # 2 passes x 40 candidates
    # the headline assertion: the dispatch parallelized (a serial loop -> 1).
    assert api.max_in_flight > 1
    # and it saturated the concurrency budget (16) given 80 calls queued.
    assert api.max_in_flight == 16


def test_generation_runs_concurrently(tmp_path: Path, monkeypatch) -> None:
    """The generation phase also fans out: each round issues a call for every
    still-short cell at once, so max-in-flight exceeds 1 (a serial build -> 1)."""
    import asyncio

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-used")
    from icl_articulation.datagen.generators.llm.api import ClientSeam
    from icl_articulation.datagen.generators.llm.pipeline import generate_candidates

    cfg = get_rule_config("positive_sentiment")  # 16 cells
    api = _ConcurrencyTrackingAPI(delay=0.02)
    seam = ClientSeam(concurrency=16, cache_dir=str(tmp_path / "cache"), api=api)
    try:
        # per_cell_target=1: every cell needs >=1 round, so round 0 issues 16
        # generation calls concurrently.
        generate_candidates(cfg, 1, seam, seed=0)
    finally:
        asyncio.run(seam.aclose())

    assert api.max_in_flight > 1


class _DeterministicSentimentAPI:
    """Content-from-request fake (NO shared mutable counters) so the response is a
    pure function of (messages, seed) — exactly like the real cached API. This
    lets the CONCURRENT end-to-end path be checked for determinism: parallel
    dispatch must not change the emitted corpus.

    Generation embeds the call's seed + line index (carried in the request seed,
    threaded by the pipeline) so lines stay distinct and reproducible regardless
    of completion order; validation answers from the sentence's marker."""

    _POS = [
        "wonderful", "delightful", "brilliant", "lovely", "pleasant", "superb",
        "excellent", "charming", "splendid", "gorgeous", "radiant", "cheerful",
        "elegant", "soothing", "uplifting", "glorious", "vibrant", "refreshing",
        "amazing", "marvellous", "satisfying", "joyful", "serene", "stellar",
        "blissful", "dazzling", "heartening", "wholesome", "agreeable", "sunny",
    ]
    _NEG = [
        "bland", "tiring", "dull", "clumsy", "miserable", "rude", "stressful",
        "frustrating", "careless", "dreary", "tedious", "awful", "dismal",
        "gloomy", "irritating", "shabby", "tiresome", "dreadful", "unpleasant",
        "lousy", "grim", "annoying", "horrible", "draining", "tasteless",
        "sloppy", "depressing", "harsh", "forgettable", "abysmal",
    ]

    def __init__(self, *, delay: float = 0.005) -> None:
        self.delay = delay
        self.chat = self
        self.completions = self
        self._pos = set(self._POS)

    async def create(self, **kwargs):
        import asyncio

        await asyncio.sleep(self.delay)  # force overlap under gather
        user = kwargs["messages"][-1]["content"]
        seed = kwargs.get("seed", 0)
        if "Output ONLY the sentences" in user:
            label_pos = "POSITIVE" in user
            markers = self._POS if label_pos else self._NEG
            lines = []
            for i in range(8):  # pairs_per_call
                m = markers[(seed * 8 + i) % len(markers)]
                lines.append(f"the visit there felt {m} to everyone today num {seed} {i}")
            content = "\n".join(lines)
        else:
            tokens = set(user.lower().split())
            content = "positive" if (tokens & self._pos) else "negative"
        data = {
            "model": kwargs["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
        }

        class _Dump:
            def model_dump(self_inner):
                return data

        return _Dump()

    async def close(self):
        pass


def test_concurrent_seam_end_to_end_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    """Running the WHOLE pipeline twice through the concurrent seam (passed as the
    single dispatcher arg) yields byte-identical emitted texts — proving the
    parallel dispatch does not perturb the seed->candidate mapping or output."""
    import asyncio

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-used")
    from icl_articulation.datagen.generators.llm.api import ClientSeam

    def _run(dest: Path) -> list[str]:
        seam = ClientSeam(concurrency=16, cache_dir=str(dest / "cache"),
                          api=_DeterministicSentimentAPI())
        try:
            run_pipeline("positive_sentiment", seam, seed=3, max_candidates=600,
                         data_dir=dest, run_pos=False)
        finally:
            asyncio.run(seam.aclose())
        return [it["text"] for it in read_items(dest / "positive_sentiment" / "items.jsonl")]

    ta = _run(tmp_path / "a")
    tb = _run(tmp_path / "b")
    assert ta == tb
    assert len(ta) >= sum(LLM_SPLIT_ITEMS.values())


def test_concurrent_dispatch_is_cache_key_identical(tmp_path: Path, monkeypatch) -> None:
    """The concurrent batch path issues the EXACT same (model, messages, params,
    seed) tuples the old sequential path would — so the disk-cache keys match and
    a re-run HITS the existing cache. We capture every request through a recording
    seam and assert each request's cache key is reproduced when computed directly
    from the seam's prompt builders + the threaded seeds."""
    import asyncio

    monkeypatch.setenv("OPENAI_API_KEY", "test-not-used")
    from icl_articulation.client import cache_key
    from icl_articulation.datagen.generators.llm.api import (
        GEN_MAX_TOKENS,
        GEN_TEMPERATURE,
        VAL_MAX_TOKENS,
        VAL_TEMPERATURE,
        ClientSeam,
    )
    from icl_articulation.datagen.generators.llm.pipeline import (
        Candidate,
        GenRequest,
        LabelRequest,
        validate_candidates,
    )

    cfg = get_rule_config("food_topic")
    seam = ClientSeam(concurrency=8, cache_dir=str(tmp_path / "cache"), api=_ConcurrencyTrackingAPI())

    # validation: drive a known candidate set, then recompute the keys we expect.
    cands = [Candidate(text=f"the soup sat in the bowl today n{i}", intended_label=True, topic="meals")
             for i in range(5)]
    validate_candidates(cfg, cands, seam, seed=100)

    # the keys the sequential path WOULD have used (seed+i per candidate, A then B)
    expected_keys: set[str] = set()
    for i, c in enumerate(cands):
        ka = cache_key("gpt-4.1-mini", cfg.validator_messages("A", c.text),
                       VAL_TEMPERATURE, VAL_MAX_TOKENS, False, None, 100 + i)
        kb = cache_key("gpt-4.1", cfg.validator_messages("B", c.text),
                       VAL_TEMPERATURE, VAL_MAX_TOKENS, False, None, 100 + i)
        expected_keys.add(ka)
        expected_keys.add(kb)
    # every expected key is now in the disk cache (the concurrent run wrote them),
    # proving the concurrent dispatch reproduces the sequential request content.
    try:
        for k in expected_keys:
            assert seam._client.cache.get(k) is not None, k
    finally:
        asyncio.run(seam.aclose())
