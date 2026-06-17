#!/usr/bin/env python
"""Build Deconfounded parallel datasets for the DIVERSIFY-OUT rules.

This script writes ``data/<rule>_deconfounded/`` without touching the original datasets.
It reuses the existing per-rule generators for surface construction, but applies
Deconfounded split policies before emitting items:

* few-shot bases use train-side rule-relevant words for both variants;
* held_out/confirmation emit only one variant per base, with True items using
  eval-side True-rule words and False items using eval-side False-rule words;
* all semantic aliases keep ``rule_id=<rule>_deconfounded`` while semantic checks resolve
  through ``canonical_rule_id``.

Rules whose `_v2` rebuild already encodes the relevant deconfounding are copied
into the `_deconfounded` naming scheme with stored rule ids rewritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.datagen import banks, battery, confound, groundtruth, schema
from icl_articulation.datagen.generators import registry
from icl_articulation.datagen.generators.base import load_rule_spec
from icl_articulation.datagen.genutils import Gen, fix_indefinite_articles
from icl_articulation.datagen.schema import make_item, write_items
from icl_articulation.rule_ids import canonical_rule_id

SEED = 20260614
DATA_DIR = Path("data")

FEW_SHOT_BASES = 100
HELD_TRUE = 60
HELD_FALSE = 60
CONF_TRUE = 50
CONF_FALSE = 50
SPARE_TRUE = 10
SPARE_FALSE = 10


class DeconfoundedGenError(RuntimeError):
    pass


def _base_id_of(spec: Any) -> str:
    if hasattr(spec, "base_id"):
        return str(getattr(spec, "base_id"))
    if isinstance(spec, dict) and "base_id" in spec:
        return str(spec["base_id"])
    if isinstance(spec, str):
        return spec
    raise DeconfoundedGenError(f"cannot determine base_id for {spec!r}")


def _split_values(
    values: Iterable[str], seed: int, *, train_frac: float = 0.55
) -> tuple[set[str], set[str]]:
    vals = sorted({str(v).lower() for v in values})
    if len(vals) < 2:
        raise DeconfoundedGenError(f"need at least two values to partition, got {vals}")
    gen = Gen(seed)
    gen.shuffle(vals)
    cut = max(1, min(len(vals) - 1, round(len(vals) * train_frac)))
    return set(vals[:cut]), set(vals[cut:])


def _all_in(values: Iterable[str], allowed: set[str]) -> bool:
    return all(str(v).lower() in allowed for v in values)


def _slot_words(rule_id: str, base: Any, label: bool) -> tuple[str, ...]:
    """Rule-relevant words whose identity should be OOD across train/eval."""
    if rule_id == "mentions_color":
        return (base.color if label else base.noncolor,)
    if rule_id == "mentions_animal":
        return (base.animal if label else base.filler,)
    if rule_id == "contains_first_name":
        return (base.name if label else base.nonname,)
    if rule_id == "starts_with_vowel":
        return (base.pair.vowel_word if label else base.pair.cons_word,)
    if rule_id == "last_word_ends_with_vowel":
        return (base.vowel_word if label else base.cons_word,)
    if rule_id == "all_words_longer_than_3":
        # True items have no short signal word. Partition False short words; for
        # True, use the substituted long word so eval True texts are also drawn
        # from unseen rule-relevant slots where possible.
        return ((base.words[base.sub_index],) if label else (base.sub_word,))
    if rule_id == "even_word_count":
        return (base.n1, base.n2, base.verb, base.adj)
    if rule_id == "passive_voice":
        return (base.agent, base.patient, base.verb_base)
    if rule_id == "the_appears_twice":
        return tuple(det for _slot, det in base.non_the)
    if rule_id == "first_word_longer_than_last":
        return (str(base.hi), str(base.lo), base.middle)
    return ()


def _make_policy(rule_id: str, bases: list[Any], seed: int) -> Callable[[Any, bool, str], bool]:
    if rule_id in {
        "even_word_count",
        "passive_voice",
        "the_appears_twice",
        "first_word_longer_than_last",
    }:
        return lambda _base, _label, _split: True

    true_vals = [v for b in bases for v in _slot_words(rule_id, b, True)]
    false_vals = [v for b in bases for v in _slot_words(rule_id, b, False)]
    if not true_vals and not false_vals:
        return lambda _base, _label, _split: True
    train_frac = {
        "mentions_color": 0.60,
        "mentions_animal": 0.55,
        "contains_first_name": 0.55,
        "starts_with_vowel": 0.55,
        "last_word_ends_with_vowel": 0.55,
    }.get(rule_id, 0.55)
    train_true, eval_true = _split_values(true_vals, seed + 11, train_frac=train_frac)
    train_false, eval_false = _split_values(false_vals, seed + 23, train_frac=train_frac)

    def ok(base: Any, label: bool, split: str) -> bool:
        vals = _slot_words(rule_id, base, label)
        if not vals:
            return True
        if split == "few_shot_pool":
            allowed = train_true if label else train_false
        else:
            allowed = eval_true if label else eval_false
        return _all_in(vals, allowed)

    return ok


def _pick(
    bases: list[Any],
    n: int,
    *,
    used: set[str],
    ok: Callable[[Any], bool],
    seed: int,
) -> list[Any]:
    cand = [b for b in bases if _base_id_of(b) not in used and ok(b)]
    order = list(cand)
    Gen(seed).shuffle(order)
    if len(order) < n:
        raise DeconfoundedGenError(f"only {len(order)} candidates for need {n}")
    chosen = order[:n]
    used.update(_base_id_of(b) for b in chosen)
    return chosen


def _instantiate_item(
    *,
    base_rule_id: str,
    output_rule_id: str,
    base: Any,
    label: bool,
    split: str,
    seed: int,
    instantiate: Callable[[Any, bool, Gen], tuple[str, dict[str, Any]]],
    suffix: str = "",
) -> dict[str, Any]:
    bid = _base_id_of(base)
    gen = Gen(seed).derive(f"{bid}:{label}:{split}:{suffix}")
    text, meta = instantiate(base, label, gen)
    text = fix_indefinite_articles(text)
    meta = {"seed": seed, "deconfounded_source_rule": base_rule_id, **meta}
    tag = f"-{suffix}" if suffix else ""
    return make_item(
        item_id=f"{bid}-{'T' if label else 'F'}{tag}",
        base_id=bid,
        rule_id=output_rule_id,
        label=label,
        text=text,
        slots_meta=meta,
        split=split,
    )


def _take_cycle(pool: list[str], index: int) -> str:
    if not pool:
        raise DeconfoundedGenError("empty pool")
    return pool[index % len(pool)]


def _build_even_word_count_deconfounded(*, data_dir: Path, seed: int) -> dict[str, Any]:
    """Custom Deconfounded parity rebuild with shared filler vocabulary across labels."""
    gen = Gen(seed).derive("even_word_count_deconfounded")
    adjs = [e.word.lower() for e in banks.get_bank("ADJ_PLAIN").entries]
    nouns = [e.word.lower() for e in banks.get_bank("NOUN_CONCRETE").entries]
    verbs = [e.word.lower() for e in banks.get_bank("VERB_REGULAR").entries]
    fillers = [
        w.lower()
        for w in banks.get_bank("ADVERB_SENT_INITIAL").words()
        + banks.get_bank("ADVERB_PLACE").words()
        if " " not in w and len(w) >= 4
    ]
    for pool in (adjs, nouns, verbs, fillers):
        gen.shuffle(pool)

    def verb_past(v: str) -> str:
        return f"{v}d" if v.endswith("e") else f"{v}ed"

    split_sizes = {
        "few_shot_pool": {True: 100, False: 100},
        "held_out": {True: 60, False: 60},
        "confirmation": {True: 50, False: 50},
        "spare": {True: 10, False: 10},
    }
    items: list[dict[str, Any]] = []
    cursor = 0
    used_texts: set[str] = set()

    def count_plan(label: bool, n: int) -> list[int]:
        if label:
            plan = [6, 8] * ((n + 1) // 2)
        else:
            q5 = round(n * 0.25)
            q9 = round(n * 0.25)
            q7 = n - q5 - q9
            plan = [5] * q5 + [7] * q7 + [9] * q9
        plan = plan[:n]
        Gen(seed + n + (17 if label else 31)).shuffle(plan)
        return plan

    def make_text(target: int, salt: int) -> tuple[str, dict[str, Any]]:
        nonlocal cursor
        for attempt in range(500):
            i = cursor + salt + attempt
            adj = _take_cycle(adjs, i)
            subj = _take_cycle(nouns, i * 3 + 1)
            verb = verb_past(_take_cycle(verbs, i * 5 + 2))
            obj = _take_cycle(nouns, i * 7 + 3)
            toks = [adj, subj, verb, obj]
            needed = target - len(toks)
            fill = []
            for j in range(needed):
                fill.append(_take_cycle(fillers, i * 11 + j * 13))
            # Put filler words before the object on half the items and after it
            # on half the items so first/last-token cues do not become a parity
            # proxy.
            if i % 2 == 0:
                out = [adj, subj, *fill, verb, obj]
                placement = "middle"
            else:
                out = [adj, subj, verb, obj, *fill]
                placement = "tail"
            text = " ".join(out)
            text = text[:1].upper() + text[1:]
            if text in used_texts:
                continue
            if schema.word_count(text) != target:
                continue
            used_texts.add(text)
            cursor += 1
            return text, {
                "frame": "telegraphic_shared_fillers",
                "target_word_count": target,
                "filler_words": fill,
                "filler_placement": placement,
                "seed": seed,
            }
        raise DeconfoundedGenError(f"could not build unique even_word_count item at {target} words")

    index = 0
    for split, by_label in split_sizes.items():
        for label in (True, False):
            for target in count_plan(label, by_label[label]):
                text, meta = make_text(target, index)
                bid = f"even-deconfounded-{index:05d}"
                items.append(
                    make_item(
                        item_id=bid,
                        base_id=bid,
                        rule_id="even_word_count_deconfounded",
                        label=label,
                        text=text,
                        slots_meta=meta,
                        split=split,
                    )
                )
                index += 1
    return _validate_and_write(
        "even_word_count", "even_word_count_deconfounded", items, data_dir=data_dir, seed=seed
    )


def _build_passive_voice_deconfounded(*, data_dir: Path, seed: int) -> dict[str, Any]:
    verbs = [
        "paint",
        "clean",
        "wash",
        "cook",
        "plant",
        "study",
        "copy",
        "count",
        "move",
        "prepare",
        "polish",
        "mend",
        "sort",
        "pack",
        "fold",
        "dry",
        "repair",
        "inspect",
    ]
    role_words = [
        "worker",
        "teacher",
        "farmer",
        "painter",
        "baker",
        "gardener",
        "tailor",
        "cook",
        "nurse",
        "clerk",
        "porter",
        "driver",
        "artist",
        "student",
        "neighbor",
        "visitor",
    ]
    tails = [
        "near home",
        "at dawn",
        "at noon",
        "by night",
        "in town",
        "outside",
        "nearby",
        "today",
        "indoors",
        "downtown",
    ]
    gen = Gen(seed).derive("passive_voice_deconfounded")
    for pool in (verbs, role_words, tails):
        gen.shuffle(pool)
    train_verbs, eval_verbs = _split_values(verbs, seed + 909, train_frac=0.6)

    def past(v: str) -> str:
        if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
            return f"{v[:-1]}ied"
        return f"{v}d" if v.endswith("e") else f"{v}ed"

    def ing(v: str) -> str:
        if v.endswith("e") and v != "see":
            return f"{v[:-1]}ing"
        return f"{v}ing"

    split_sizes = {
        "few_shot_pool": {True: 100, False: 100},
        "held_out": {True: 60, False: 60},
        "confirmation": {True: 50, False: 50},
        "spare": {True: 10, False: 10},
    }
    items: list[dict[str, Any]] = []
    used_texts: set[str] = set()
    cursor = 0

    def choose_verb(split: str, i: int) -> str:
        allowed = train_verbs if split == "few_shot_pool" else eval_verbs
        if split == "spare":
            allowed = set(verbs)
        return _take_cycle(sorted(allowed), i)

    def make_text(label: bool, split: str, salt: int) -> tuple[str, dict[str, Any]]:
        nonlocal cursor
        for attempt in range(500):
            i = cursor + salt + attempt
            verb = choose_verb(split, i)
            tail = _take_cycle(tails, i * 3)
            role_word = _take_cycle(role_words, i * 5 + 1)
            if label:
                text = f"The {role_word} was {past(verb)} {tail}"
                shape = "passive_was_participle"
            else:
                text = f"The {role_word} was {ing(verb)} {tail}"
                shape = "active_progressive"
            text = fix_indefinite_articles(text)
            if text in used_texts:
                continue
            if schema.word_count(text) < 5 or schema.word_count(text) > 7:
                continue
            used_texts.add(text)
            cursor += 1
            return text, {
                "shape": shape,
                "verb_base": verb,
                "tail": tail,
                "role_word": role_word,
                "seed": seed,
                "verb_partition": "train" if split == "few_shot_pool" else "eval",
            }
        raise DeconfoundedGenError("could not build unique passive_voice item")

    index = 0
    for split, by_label in split_sizes.items():
        for label in (True, False):
            for _ in range(by_label[label]):
                text, meta = make_text(label, split, index)
                bid = f"passive-deconfounded-{index:05d}"
                items.append(
                    make_item(
                        item_id=bid,
                        base_id=bid,
                        rule_id="passive_voice_deconfounded",
                        label=label,
                        text=text,
                        slots_meta=meta,
                        split=split,
                    )
                )
                index += 1
    return _validate_and_write(
        "passive_voice", "passive_voice_deconfounded", items, data_dir=data_dir, seed=seed
    )


def build_programmatic_rule(rule_id: str, *, data_dir: Path, seed: int) -> dict[str, Any]:
    if rule_id == "even_word_count":
        return _build_even_word_count_deconfounded(data_dir=data_dir, seed=seed)
    if rule_id == "passive_voice":
        return _build_passive_voice_deconfounded(data_dir=data_dir, seed=seed)
    module = registry.get_module(rule_id)
    build_bases = module.build_bases
    bases = build_bases(Gen(seed).derive(f"{rule_id}:bases"))
    instantiate = module.instantiate
    if len(bases) < 340:
        raise DeconfoundedGenError(f"{rule_id}: only {len(bases)} bases")
    policy = _make_policy(rule_id, bases, seed)
    out_rule = f"{rule_id}_deconfounded"
    used: set[str] = set()
    items: list[dict[str, Any]] = []

    fs = _pick(
        bases,
        FEW_SHOT_BASES,
        used=used,
        ok=lambda b: policy(b, True, "few_shot_pool") and policy(b, False, "few_shot_pool"),
        seed=seed + 101,
    )
    for b in fs:
        items.append(
            _instantiate_item(
                base_rule_id=rule_id, output_rule_id=out_rule, base=b, label=True,
                split="few_shot_pool", seed=seed, instantiate=instantiate,
            )
        )
        items.append(
            _instantiate_item(
                base_rule_id=rule_id, output_rule_id=out_rule, base=b, label=False,
                split="few_shot_pool", seed=seed, instantiate=instantiate,
            )
        )

    for split, n_true, n_false, offset in (
        ("held_out", HELD_TRUE, HELD_FALSE, 201),
        ("confirmation", CONF_TRUE, CONF_FALSE, 301),
        ("spare", SPARE_TRUE, SPARE_FALSE, 401),
    ):
        for label, n, laboff in ((True, n_true, 1), (False, n_false, 2)):
            chosen = _pick(
                bases,
                n,
                used=used,
                ok=(
                    (lambda _b: True)
                    if split == "spare"
                    else (lambda b, lab=label, sp=split: policy(b, lab, sp))
                ),
                seed=seed + offset + laboff,
            )
            for b in chosen:
                items.append(
                    _instantiate_item(
                        base_rule_id=rule_id, output_rule_id=out_rule, base=b,
                        label=label, split=split, seed=seed, instantiate=instantiate,
                    )
                )

    return _validate_and_write(rule_id, out_rule, items, data_dir=data_dir, seed=seed)


def copy_variant(
    src_rule: str,
    out_rule: str,
    *,
    data_dir: Path,
    seed: int,
    source_data_dir: Path | None = None,
) -> dict[str, Any]:
    candidates = []
    if source_data_dir is not None:
        candidates.append(source_data_dir / src_rule / "items.jsonl")
    candidates.extend([data_dir / src_rule / "items.jsonl", DATA_DIR / src_rule / "items.jsonl"])
    src = next((p for p in candidates if p.is_file()), candidates[0])
    if not src.is_file():
        raise DeconfoundedGenError(f"missing source dataset for {src_rule}: {src}")
    items = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        it["rule_id"] = out_rule
        meta = dict(it.get("slots_meta") or {})
        meta["seed"] = seed
        meta["deconfounded_source_rule"] = src_rule
        it["slots_meta"] = meta
        items.append(it)
    return _validate_and_write(canonical_rule_id(src_rule), out_rule, items, data_dir=data_dir, seed=seed)


def _validate_and_write(
    base_rule_id: str,
    out_rule: str,
    items: list[dict[str, Any]],
    *,
    data_dir: Path,
    seed: int,
) -> dict[str, Any]:
    spec = load_rule_spec(base_rule_id)
    battery_exemptions = tuple(spec.battery_exemptions) + _deconfounded_battery_exemptions(base_rule_id)
    out_dir = data_dir / out_rule
    schema.validate_full(items, rule_id=base_rule_id)
    groundtruth.assert_labels_correct(out_rule, items)
    results = battery.battery_report(
        items,
        equiv_keys=spec.equiv_keys,
        equivalence_class=spec.equivalence_class,
        battery_exemptions=battery_exemptions,
        run_pos=False,
    )
    violations = battery.battery_violations(results)
    if violations:
        detail = ", ".join(f"{r.key}={r.score:.3f}" for r in violations[:8])
        raise DeconfoundedGenError(f"{out_rule}: battery failed: {detail}")
    report = confound.build_confound_report(
        items,
        is_llm_rule=base_rule_id in {"food_topic", "positive_sentiment"},
        equiv_keys=spec.equiv_keys,
        equivalence_class=spec.equivalence_class,
        battery_exemptions=battery_exemptions,
        run_pos=False,
        length_match_exempt=spec.length_match_exempt,
        topics=[
            str((it.get("slots_meta") or {}).get("topic", ""))
            for it in items
        ] if base_rule_id in {"food_topic", "positive_sentiment"} else None,
    )
    if not report["overall_pass"]:
        raise DeconfoundedGenError(f"{out_rule}: confound report failed: {report}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = write_items(items, out_dir / "items.jsonl")
    confound.write_confound_report(report, out_dir / "confound_report.json")
    return {
        "rule_id": out_rule,
        "base_rule_id": base_rule_id,
        "seed": seed,
        "n_items": len(items),
        "items_path": str(items_path),
        "confound_report_path": str(out_dir / "confound_report.json"),
    }


def _deconfounded_battery_exemptions(base_rule_id: str) -> tuple[str, ...]:
    if base_rule_id == "second_word_capitalized":
        return ("nonfirst_word_capitalized",)
    if base_rule_id == "word_count_geq_8":
        return (
            "word_count>=5",
            "word_count>=6",
            "word_count>=7",
            "word_count>=8",
            "word_count>=9",
            "word_count>=10",
            "char_count>=35",
            "char_count>=40",
            "char_count>=45",
        )
    return ()


PROGRAMMATIC_RULES = (
    "mentions_color",
    "mentions_animal",
    "contains_first_name",
    "starts_with_vowel",
    "last_word_ends_with_vowel",
    "even_word_count",
    "passive_voice",
    "the_appears_twice",
    "first_word_longer_than_last",
    "all_words_longer_than_3",
    "first_two_words_alphabetical",
)

COPY_RULES = {
    "word_count_geq_8": ("word_count_geq_8_v2", "word_count_geq_8_deconfounded"),
    "second_word_capitalized": ("second_word_capitalized_v2", "second_word_capitalized_deconfounded"),
}

SEMANTIC_COPY_RULES = {
    "food_topic": ("food_topic", "food_topic_deconfounded"),
    "positive_sentiment": ("positive_sentiment", "positive_sentiment_deconfounded"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--rules",
        help="comma-separated base rule ids; default builds all current Deconfounded programmatic variants",
    )
    p.add_argument("--output", default="deconfounded_generation_summary.json")
    p.add_argument(
        "--source-data-dir",
        help="optional source root for copy rules, used for validated semantic raw outputs",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    source_data_dir = Path(args.source_data_dir) if args.source_data_dir else None
    rules = (
        [r.strip() for r in args.rules.split(",") if r.strip()]
        if args.rules else list(PROGRAMMATIC_RULES) + list(COPY_RULES)
    )
    summaries = []
    for rule_id in rules:
        if rule_id in COPY_RULES or rule_id in SEMANTIC_COPY_RULES:
            src_rule, out_rule = {**COPY_RULES, **SEMANTIC_COPY_RULES}[rule_id]
            summary = copy_variant(
                src_rule,
                out_rule,
                data_dir=data_dir,
                seed=args.seed,
                source_data_dir=source_data_dir,
            )
        else:
            summary = build_programmatic_rule(rule_id, data_dir=data_dir, seed=args.seed)
        summaries.append(summary)
        print(f"built {summary['rule_id']} ({summary['n_items']} items)")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": args.seed, "rules": summaries}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
