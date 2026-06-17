#!/usr/bin/env python
"""Build controlled Deconfounded semantic datasets with fresh two-pass validation.

Makes paid API calls (set OPENAI_API_KEY). This script avoids LLM text generation: it constructs
highly controlled candidate sentences with train/eval-disjoint semantic lexicons,
then uses the existing two validator prompts for `food_topic` and
`positive_sentiment`. Only candidates where both validators return the intended
label are emitted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.datagen import confound, groundtruth, schema
from icl_articulation.datagen.generators.llm.api import ClientSeam
from icl_articulation.datagen.generators.llm.config import get_rule_config
from icl_articulation.datagen.generators.llm.pipeline import LabelRequest, VALIDATOR_A_MODEL, VALIDATOR_B_MODEL
from icl_articulation.datagen.schema import make_item, write_items
from icl_articulation.prices import cost_usd
from icl_articulation.runlog import start_run

SEED = 20260614
NEEDS = {
    "few_shot_pool": {True: 60, False: 60},
    "held_out": {True: 60, False: 60},
    "confirmation": {True: 50, False: 50},
    "spare": {True: 20, False: 20},
}

TOPICS = ("restaurants", "movies", "weather", "work", "travel", "products", "sports", "music")
FOOD_TRUE_TOPICS = (
    "cooking at home",
    "restaurants",
    "ingredients",
    "baking",
    "meals",
    "tasting and flavour",
)
FOOD_FALSE_TOPICS = (
    "sports",
    "weather",
    "transport",
    "work",
    "music",
    "gardening flowers",
)

FOOD_TRAIN = (
    "bread", "rice", "soup", "cheese", "pasta", "salad", "cake", "coffee",
    "beans", "noodles", "pizza", "sauce", "cookie", "sandwich", "stew", "curry",
)
FOOD_EVAL = (
    "omelet", "muffin", "yogurt", "cereal", "taco", "dumpling", "spinach", "salmon",
    "pancake", "lentils", "risotto", "burrito", "biscuit", "pudding", "smoothie", "ravioli",
)
NONFOOD_TRAIN = (
    "lamp", "ticket", "brush", "jacket", "mirror", "ladder", "pencil", "basket",
    "guitar", "flower", "window", "bicycle", "notebook", "curtain", "blanket", "hammer",
)
NONFOOD_EVAL = (
    "camera", "wallet", "helmet", "statue", "carpet", "printer", "suitcase", "violin",
    "lantern", "cabinet", "postcard", "trophy", "compass", "keyboard", "scarf", "vase",
)

POS_TRAIN = (
    "pleasant", "wonderful", "helpful", "delightful", "excellent", "lovely",
    "reliable", "charming", "enjoyable", "impressive", "smooth", "bright",
)
POS_EVAL = (
    "superb", "satisfying", "refreshing", "admirable", "cheerful", "rewarding",
    "graceful", "effective", "welcome", "brilliant", "uplifting", "polished",
)
NEG_TRAIN = (
    "awful", "poor", "dull", "unpleasant", "frustrating", "disappointing",
    "weak", "clumsy", "tiresome", "broken", "dreary", "messy",
)
NEG_EVAL = (
    "terrible", "annoying", "unhelpful", "flawed", "boring", "wasteful",
    "rough", "stressful", "gloomy", "careless", "confusing", "inferior",
)

FOOD_TEMPLATES = (
    "The {word} sat beside the window",
    "Someone moved the {word} near the shelf",
    "The {word} remained on the table",
    "A small {word} waited in the basket",
    "They placed the {word} near the doorway",
    "The plain {word} stayed inside the room",
    "The quiet {word} rested near the cabinet",
    "People stored the {word} beside the lamp",
    "The {word} stood inside the open box",
    "Someone carried the {word} past the counter",
    "The simple {word} stayed near the hallway",
    "They found the {word} beside the chair",
    "The {word} waited inside the basket",
    "A clean {word} rested near the shelf",
    "The group moved the {word} indoors",
    "Someone noticed the {word} near the corner",
    "The {word} sat inside the cabinet",
    "They kept the {word} beside the doorway",
    "The small {word} remained near the lamp",
    "People carried the {word} through the room",
)

SENT_TEMPLATES = (
    "The {topic} felt {word} today",
    "The {topic} seemed {word} this morning",
    "Everyone found the {topic} {word}",
    "The recent {topic} looked {word}",
    "People called the {topic} {word}",
    "The whole {topic} became {word}",
)


def _pool(rule_id: str, label: bool, split: str) -> tuple[str, ...]:
    eval_split = split != "few_shot_pool"
    if rule_id == "food_topic":
        if label:
            return FOOD_EVAL if eval_split else FOOD_TRAIN
        return NONFOOD_EVAL if eval_split else NONFOOD_TRAIN
    if label:
        return POS_EVAL if eval_split else POS_TRAIN
    return NEG_EVAL if eval_split else NEG_TRAIN


def build_candidates(rule_id: str, multiplier: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    templates = FOOD_TEMPLATES if rule_id == "food_topic" else SENT_TEMPLATES
    idx = 0
    cursors: dict[tuple[bool, str], int] = defaultdict(int)
    for split, by_label in NEEDS.items():
        for label, need in by_label.items():
            pool = _pool(rule_id, label, split)
            total = need * multiplier
            partition = "eval" if split != "few_shot_pool" else "train"
            combos: list[tuple[str, str, str]] = []
            if rule_id == "food_topic":
                topic_pool = FOOD_TRUE_TOPICS if label else FOOD_FALSE_TOPICS
                for word in pool:
                    for template in templates:
                        topic = topic_pool[(len(combos) + len(word)) % len(topic_pool)]
                        combos.append((word, template, topic))
            else:
                for word in pool:
                    for topic in TOPICS:
                        for template in templates:
                            combos.append((word, template, topic))
            start = cursors[(label, partition)]
            stop = start + total
            if stop > len(combos):
                raise RuntimeError(
                    f"{rule_id} {split}/{label} asks for combo slice [{start}:{stop}] "
                    f"but only {len(combos)} unique template-word combinations exist"
                )
            cursors[(label, partition)] = stop
            for word, template, topic in combos[start:stop]:
                if rule_id == "food_topic":
                    text = template.format(word=word)
                else:
                    text = template.format(topic=topic, word=word)
                if text in seen:
                    raise RuntimeError(f"duplicate controlled candidate text: {text!r}")
                seen.add(text)
                out.append(
                    {
                        "candidate_id": f"{rule_id}-cand-{idx:05d}",
                        "rule_id": rule_id,
                        "label": label,
                        "split": split,
                        "topic": topic,
                        "text": text,
                        "word": word,
                    }
                )
                idx += 1
    return out


def estimate_validation_cost(n_candidates: int) -> dict[str, Any]:
    # Validator prompts are short and one-token completions; this conservative
    # estimate mirrors the existing LLM cost helper's scale.
    prompt_tokens = 110
    completion_tokens = 3
    pass_a = n_candidates * cost_usd(VALIDATOR_A_MODEL, prompt_tokens, completion_tokens)
    pass_b = n_candidates * cost_usd(VALIDATOR_B_MODEL, prompt_tokens, completion_tokens)
    return {
        "n_candidates": n_candidates,
        "validation_calls": n_candidates * 2,
        "validator_a_model": VALIDATOR_A_MODEL,
        "validator_b_model": VALIDATOR_B_MODEL,
        "validate_pass_a_cost_usd": pass_a,
        "validate_pass_b_cost_usd": pass_b,
        "total_usd": pass_a + pass_b,
        "assumptions": "validation-only controlled semantic Deconfounded build; ~110 prompt tokens and 3 completion tokens per validator call",
    }


async def validate(candidates: list[dict[str, Any]], *, concurrency: int, cache_dir: str, run_log: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    seam = ClientSeam(concurrency=concurrency, cache_dir=cache_dir, run_log=run_log)
    try:
        requests: list[LabelRequest] = []
        for i, cand in enumerate(candidates):
            requests.append(LabelRequest(cand["rule_id"], "A", cand["text"], SEED + i))
            requests.append(LabelRequest(cand["rule_id"], "B", cand["text"], SEED + i))
        results = await seam.label_many(requests)
        validated = []
        for i, cand in enumerate(candidates):
            a = results[2 * i]
            b = results[2 * i + 1]
            row = dict(cand)
            row["validator_pass_a"] = a
            row["validator_pass_b"] = b
            row["kept"] = a == b == bool(cand["label"])
            validated.append(row)
        return validated, seam.cost_summary(), seam.stats()
    finally:
        await seam.aclose()


def select_items(rule_id: str, validated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        if row["kept"]:
            by_cell[(row["split"], bool(row["label"]))].append(row)
    items: list[dict[str, Any]] = []
    idx = 0
    for split, by_label in NEEDS.items():
        for label, need in by_label.items():
            cell = by_cell[(split, label)]
            if len(cell) < need:
                raise RuntimeError(f"{rule_id} short cell {split}/{label}: need {need}, kept {len(cell)}")
            for row in cell[:need]:
                item_id = f"{rule_id}_deconfounded-{idx:04d}"
                meta = {
                    "seed": SEED,
                    "topic": row["topic"],
                    "controlled_word": row["word"],
                    "intended_label": bool(row["label"]),
                    "validated_agreement": bool(row["label"]),
                    "validator_pass_a": row["validator_pass_a"],
                    "validator_pass_b": row["validator_pass_b"],
                    "validator_a_model": VALIDATOR_A_MODEL,
                    "validator_b_model": VALIDATOR_B_MODEL,
                    "deconfounded_controlled_semantic": True,
                    "source_candidate_id": row["candidate_id"],
                }
                items.append(
                    make_item(
                        item_id=item_id,
                        base_id=item_id,
                        rule_id=f"{rule_id}_deconfounded",
                        label=bool(row["label"]),
                        text=row["text"],
                        slots_meta=meta,
                        split=split,
                    )
                )
                idx += 1
    return items


def write_dataset(rule_id: str, items: list[dict[str, Any]], data_dir: Path) -> dict[str, Any]:
    out_dir = data_dir / f"{rule_id}_deconfounded"
    schema.validate_full(items, rule_id=rule_id)
    groundtruth.assert_labels_correct(f"{rule_id}_deconfounded", items)
    topics = [str(it["slots_meta"]["topic"]) for it in items]
    report = confound.build_confound_report(
        items,
        is_llm_rule=True,
        run_pos=False,
        topics=topics,
    )
    if not report["overall_pass"]:
        raise RuntimeError(f"{rule_id}_deconfounded confound report failed: {report}")
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = write_items(items, out_dir / "items.jsonl")
    report_path = confound.write_confound_report(report, out_dir / "confound_report.json")
    return {"items_path": str(items_path), "confound_report_path": str(report_path), "n_items": len(items)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rule_id", choices=["food_topic", "positive_sentiment"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--candidate-multiplier", type=int, default=2)
    p.add_argument("--max-cost", type=float, default=200.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = build_candidates(args.rule_id, args.candidate_multiplier)
    estimate = estimate_validation_cost(len(candidates))
    print(json.dumps({"advance_cost_estimate": estimate}, indent=2))
    if estimate["total_usd"] > args.max_cost:
        raise SystemExit(f"estimate ${estimate['total_usd']:.4f} exceeds max ${args.max_cost:.2f}")
    run = start_run(
        name=f"deconfounded-controlled-semantic-{args.rule_id}",
        config={
            "task": "deconfounded-controlled-semantic",
            "rule_id": args.rule_id,
            "seed": SEED,
            "candidate_multiplier": args.candidate_multiplier,
            "n_candidates": len(candidates),
            "cost_estimate": estimate,
        },
        cost_estimate_usd=estimate["total_usd"],
        results_dir=args.results_dir,
    )
    validated, cost, stats = asyncio.run(
        validate(candidates, concurrency=args.concurrency, cache_dir=args.cache_dir, run_log=run)
    )
    items = select_items(args.rule_id, validated)
    summary = write_dataset(args.rule_id, items, Path(args.data_dir))
    kept = sum(1 for row in validated if row["kept"])
    metrics = {
        "n_candidates": len(candidates),
        "n_kept": kept,
        "drop_rate": 1 - kept / len(candidates),
        "summary": summary,
        "client_stats": stats,
    }
    run.write_metrics(metrics)
    run.finish(cost_actual_usd=cost["total_usd"], extra={"cost": cost, "client_stats": stats})
    print(json.dumps({**metrics, "cost": cost}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
