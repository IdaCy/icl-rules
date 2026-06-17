#!/usr/bin/env python
"""No-API dataset shortcut audit.

Trains simple shortcut baselines on ``few_shot_pool`` and evaluates them on
``held_out`` plus ``confirmation``. The script is intentionally dependency-light:
word and character n-gram baselines use a small built-in multinomial naive Bayes
implementation rather than scikit-learn.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from icl_articulation.contexts import load_items
from icl_articulation.datagen import groundtruth
from icl_articulation.datagen.confound import tokens_missing_from_a_class
from icl_articulation.datagen.groundtruth import RULE_PREDICATES
from icl_articulation.datagen.schema import word_count, words
from icl_articulation.rule_ids import canonical_rule_id

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO / "data"
TRAIN_SPLIT = "few_shot_pool"
EVAL_SPLITS = ("held_out", "confirmation")

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_PUNCT = ".,!?;:\"'()[]-–—…"

STOPWORDS = frozenset(
    """a about after all also am an and any are as at be been before being between
    both but by can could did do does during each few for from had has have he her
    here him his how i if in into is it its just may me might more most must my near
    no nor not of off on only onto or our out over own same shall she should so some
    such than that the their them then there these they this those to too under until
    up us very was we were what when where which who why will with would yet you your""".split()
)

def word_tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(word_tokens(text))


def char_ngrams(text: str, max_n: int = 3) -> list[str]:
    s = f" {text.lower()} "
    feats: list[str] = []
    for n in range(1, max_n + 1):
        feats.extend(s[i : i + n] for i in range(max(0, len(s) - n + 1)))
    return feats


def _alpha_len(tok: str) -> int:
    return sum(1 for ch in tok if ch.isalpha())


def scalar_features(text: str) -> dict[str, float]:
    toks = words(text)
    alpha_lens = [_alpha_len(t) for t in toks]
    return {
        "n_chars": float(len(text)),
        "n_words": float(word_count(text)),
        "avg_alpha_len": float(sum(alpha_lens) / len(alpha_lens)) if alpha_lens else 0.0,
        "first_alpha_len": float(alpha_lens[0]) if alpha_lens else 0.0,
        "last_alpha_len": float(alpha_lens[-1]) if alpha_lens else 0.0,
        "n_upper": float(sum(1 for ch in text if ch.isupper())),
        "n_digits": float(sum(1 for ch in text if ch.isdigit())),
        "n_commas": float(text.count(",")),
        "n_exclamation": float(text.count("!")),
        "capitalized_word_rate": (
            sum(1 for t in toks if t[:1].isupper()) / len(toks) if toks else 0.0
        ),
    }


def _split_indices(items: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    train = [i for i, it in enumerate(items) if it["split"] == TRAIN_SPLIT]
    eval_ = [i for i, it in enumerate(items) if it["split"] in EVAL_SPLITS]
    if not train:
        raise ValueError("no few_shot_pool items")
    if not eval_:
        raise ValueError("no held_out/confirmation items")
    return train, eval_


def _acc(pred: Iterable[bool], gold: Iterable[bool]) -> float:
    pairs = list(zip(pred, gold))
    return sum(p == g for p, g in pairs) / len(pairs) if pairs else 0.0


def multinomial_nb(
    items: list[dict[str, Any]],
    feature_fn: Callable[[str], Iterable[str]],
) -> dict[str, Any]:
    """Train a Laplace-smoothed multinomial NB on few-shot examples."""
    train, eval_ = _split_indices(items)
    counts = {True: Counter(), False: Counter()}
    class_n = {True: 0, False: 0}
    vocab: set[str] = set()
    labels = [bool(it["label"]) for it in items]
    features = [list(feature_fn(it["text"])) for it in items]
    for i in train:
        lab = labels[i]
        class_n[lab] += 1
        counts[lab].update(features[i])
        vocab.update(features[i])
    if not vocab or not all(class_n.values()):
        return {"train_acc": None, "eval_acc": None, "n_train": len(train), "n_eval": len(eval_)}
    v = len(vocab)
    totals = {lab: sum(counts[lab].values()) for lab in (True, False)}
    n_train = sum(class_n.values())

    def logp(i: int, lab: bool) -> float:
        lp = math.log(class_n[lab] / n_train)
        denom = totals[lab] + v
        for feat in features[i]:
            lp += math.log((counts[lab][feat] + 1) / denom)
        return lp

    def pred(i: int) -> bool:
        return logp(i, True) >= logp(i, False)

    return {
        "train_acc": _acc((pred(i) for i in train), (labels[i] for i in train)),
        "eval_acc": _acc((pred(i) for i in eval_), (labels[i] for i in eval_)),
        "n_train": len(train),
        "n_eval": len(eval_),
        "vocab_size": v,
    }


def best_single_token(items: list[dict[str, Any]]) -> dict[str, Any]:
    train, eval_ = _split_indices(items)
    labels = [bool(it["label"]) for it in items]
    toks = [token_set(it["text"]) for it in items]
    vocab = sorted(set().union(*(toks[i] for i in train)))
    best: dict[str, Any] | None = None
    for tok in vocab:
        present = [tok in ts for ts in toks]
        for presence_means_true in (True, False):
            pred = [
                (hit if presence_means_true else not hit)
                for hit in present
            ]
            train_acc = _acc((pred[i] for i in train), (labels[i] for i in train))
            if best is None or train_acc > best["train_acc"]:
                best = {
                    "token": tok,
                    "presence_means_true": presence_means_true,
                    "train_acc": train_acc,
                    "eval_acc": _acc((pred[i] for i in eval_), (labels[i] for i in eval_)),
                    "true_count": sum(1 for i, it in enumerate(items) if labels[i] and tok in toks[i]),
                    "false_count": sum(1 for i, it in enumerate(items) if (not labels[i]) and tok in toks[i]),
                }
    return best or {"token": None, "train_acc": None, "eval_acc": None}


def best_scalar_threshold(items: list[dict[str, Any]]) -> dict[str, Any]:
    train, eval_ = _split_indices(items)
    labels = [bool(it["label"]) for it in items]
    feature_rows = [scalar_features(it["text"]) for it in items]
    best: dict[str, Any] | None = None
    for feat in sorted(feature_rows[0]):
        values = [row[feat] for row in feature_rows]
        for threshold in sorted(set(values[i] for i in train)):
            for ge_means_true in (True, False):
                pred = [
                    ((v >= threshold) if ge_means_true else (v < threshold))
                    for v in values
                ]
                train_acc = _acc((pred[i] for i in train), (labels[i] for i in train))
                if best is None or train_acc > best["train_acc"]:
                    best = {
                        "feature": feat,
                        "threshold": threshold,
                        "ge_means_true": ge_means_true,
                        "train_acc": train_acc,
                        "eval_acc": _acc((pred[i] for i in eval_), (labels[i] for i in eval_)),
                    }
    return best or {"feature": None, "train_acc": None, "eval_acc": None}


def groundtruth_check(rule_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    base = canonical_rule_id(rule_id)
    entry = RULE_PREDICATES.get(base)
    if entry is None:
        return {"canonical_rule_id": base, "recomputable": False, "n_mismatch": None, "reason": "unknown rule"}
    if not entry.recomputable:
        missing = [
            str(it.get("item_id"))
            for it in items
            if not isinstance(it.get("slots_meta"), dict)
            or "validated_agreement" not in it["slots_meta"]
        ]
        return {
            "canonical_rule_id": base,
            "backing": entry.backing.value,
            "recomputable": False,
            "n_mismatch": None,
            "missing_validation_provenance": len(missing),
        }
    bad = [
        str(it.get("item_id"))
        for it in items
        if groundtruth.label_of(base, it["text"]) != bool(it["label"])
    ]
    return {
        "canonical_rule_id": base,
        "backing": entry.backing.value,
        "recomputable": True,
        "n_mismatch": len(bad),
        "mismatch_examples": bad[:10],
    }


def split_separation(items: list[dict[str, Any]]) -> dict[str, Any]:
    train = [it for it in items if it["split"] == TRAIN_SPLIT]
    held = [it for it in items if it["split"] == "held_out"]
    conf = [it for it in items if it["split"] == "confirmation"]
    train_bases = {it["base_id"] for it in train}
    eval_bases = {it["base_id"] for it in held + conf}
    train_toks = set().union(*(token_set(it["text"]) for it in train)) if train else set()
    content_train = train_toks - STOPWORDS

    def _share(group: list[dict[str, Any]], *, content: bool = False) -> float | None:
        toks = set().union(*(token_set(it["text"]) for it in group)) if group else set()
        if content:
            toks = toks - STOPWORDS
            base = content_train
        else:
            base = train_toks
        return (len(toks & base) / len(toks)) if toks else None

    def _frames(group: list[dict[str, Any]]) -> set[str]:
        out = set()
        for it in group:
            meta = it.get("slots_meta") or {}
            frame = meta.get("frame")
            if isinstance(frame, str):
                out.add(frame)
        return out

    train_frames = _frames(train)
    held_frames = _frames(held)
    conf_frames = _frames(conf)
    return {
        "base_overlap_train_eval": len(train_bases & eval_bases),
        "held_out_word_vocab_seen_frac": _share(held),
        "confirmation_word_vocab_seen_frac": _share(conf),
        "held_out_content_vocab_seen_frac": _share(held, content=True),
        "confirmation_content_vocab_seen_frac": _share(conf, content=True),
        "held_out_frame_overlap_frac": (
            len(held_frames & train_frames) / len(held_frames) if held_frames else None
        ),
        "confirmation_frame_overlap_frac": (
            len(conf_frames & train_frames) / len(conf_frames) if conf_frames else None
        ),
        "n_train_frames": len(train_frames),
        "n_held_out_frames": len(held_frames),
        "n_confirmation_frames": len(conf_frames),
    }


def label_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("few_shot_pool", "held_out", "confirmation", "spare"):
        group = [it for it in items if it["split"] == split]
        if not group:
            continue
        n_true = sum(1 for it in group if bool(it["label"]))
        out[split] = {"n": len(group), "n_true": n_true, "n_false": len(group) - n_true}
    return out


def audit_rule(data_dir: Path, rule_id: str) -> dict[str, Any]:
    items = load_items(data_dir / rule_id / "items.jsonl")
    bow = multinomial_nb(items, lambda text: token_set(text))
    char = multinomial_nb(items, char_ngrams)
    single = best_single_token(items)
    scalar = best_scalar_threshold(items)
    shortcut_values = [
        v
        for v in (
            bow["eval_acc"],
            char["eval_acc"],
            single["eval_acc"],
            scalar["eval_acc"],
        )
        if isinstance(v, (int, float))
    ]
    return {
        "rule_id": rule_id,
        "canonical_rule_id": canonical_rule_id(rule_id),
        "n_items": len(items),
        "split_label_counts": label_stats(items),
        "groundtruth": groundtruth_check(rule_id, items),
        "shortcut_baselines": {
            "word_bow_nb": bow,
            "char_1_3gram_nb": char,
            "best_single_token": single,
            "best_scalar_threshold": scalar,
            "max_eval_acc": max(shortcut_values) if shortcut_values else None,
        },
        "one_sided_high_freq_tokens": tokens_missing_from_a_class(items),
        "split_separation": split_separation(items),
    }


def discover_rules(data_dir: Path) -> list[str]:
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    return sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and (d / "items.jsonl").is_file()
    )


def parse_rules(value: str | None, data_dir: Path) -> list[str]:
    if value:
        return [r.strip() for r in value.split(",") if r.strip()]
    return discover_rules(data_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--rules", help="comma-separated rule ids; default audits all datasets under --data-dir")
    p.add_argument("--output", help="optional JSON output path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    rules = parse_rules(args.rules, data_dir)
    audited = {rule: audit_rule(data_dir, rule) for rule in rules}
    out = {
        "data_dir": str(data_dir),
        "train_split": TRAIN_SPLIT,
        "eval_splits": list(EVAL_SPLITS),
        "rules": audited,
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for rule, row in audited.items():
        s = row["shortcut_baselines"]
        print(
            f"{rule:32s} max={s['max_eval_acc']:.3f} "
            f"bow={s['word_bow_nb']['eval_acc']:.3f} "
            f"char={s['char_1_3gram_nb']['eval_acc']:.3f} "
            f"tok={s['best_single_token']['eval_acc']:.3f} "
            f"scalar={s['best_scalar_threshold']['eval_acc']:.3f}"
        )
    if args.output:
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
