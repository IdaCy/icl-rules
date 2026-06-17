#!/usr/bin/env python
"""Gate 2C — blind NATURALNESS rating harness.

NOT a bare "old or new" call. We mix a sample of OLD items with the NEW divergence
items, strip all source/label markers, shuffle, and have fresh agents rate each
item 1-5 on naturalness / "reads like a real, varied, well-formed sentence of this
rule's dataset". PASS if the NEW-set rating distribution is not materially below
the OLD-set's. The source mapping is held here (not shown to the rater), so the
rating is genuinely blind.

  prepare : emit a numbered, source-stripped, shuffled sheet (stdout) + a
            mapping json. n_each OLD + n_each NEW.
  score   : read a {index: rating} json, split by the saved mapping, compare the
            OLD vs NEW rating distributions (means, medians, Mann-Whitney U).

Naturalness here = grammatical/stylistic fluency and "looks like a real item of
this dataset", NOT semantic plausibility — both OLD and NEW deliberately contain
implausible/odd sentences (impossible events, word-count word-salad), so plausibility
must NOT drive the rating.

Usage:
  python scripts/gate2c_naturalness.py prepare --rule physically_impossible \
      --new data/physically_impossible_divergence/items.jsonl \
      --old data/physically_impossible/items.jsonl --n-each 24
  # ... raters return ratings as a {index: rating} json ...
  python scripts/gate2c_naturalness.py score --rule physically_impossible \
      --ratings ratings_physically_impossible_raterA.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parent.parent
SCRATCH = REPO / "out"


def load_texts(path: Path, splits: list[str]) -> list[str]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["split"] in splits:
            out.append(r["text"])
    return out


def prepare(args) -> int:
    new = load_texts(Path(args.new), ["held_out"])
    old = load_texts(Path(args.old), ["held_out", "confirmation"])
    rng = np.random.default_rng(args.seed)
    new_s = list(rng.permutation(new)[: args.n_each])
    old_s = list(rng.permutation(old)[: args.n_each])
    items = [{"text": t, "source": "new"} for t in new_s] + \
            [{"text": t, "source": "old"} for t in old_s]
    order = rng.permutation(len(items))
    sheet = [items[i] for i in order]
    mapping = {str(i): sheet[i]["source"] for i in range(len(sheet))}
    map_path = SCRATCH / f"divergence_2c_{args.rule}_mapping.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    print(f"# Gate 2C naturalness sheet — rule {args.rule}  ({len(sheet)} items)")
    print("# Rate each 1-5 (5 = perfectly natural/fluent and reads like a real, varied")
    print("# item of this dataset; 1 = stilted/templated/broken). Judge FLUENCY &")
    print("# VARIETY, NOT whether the event is plausible. Return {\"index\": rating}.")
    for i, it in enumerate(sheet):
        print(f"{i}\t{it['text']}")
    print(f"\n# (private mapping saved to {map_path.relative_to(REPO)})")
    return 0


def score(args) -> int:
    mapping = json.loads((SCRATCH / f"divergence_2c_{args.rule}_mapping.json").read_text())
    ratings = json.loads(Path(args.ratings).read_text())
    old_r, new_r = [], []
    for idx, src in mapping.items():
        if idx not in ratings:
            continue
        (old_r if src == "old" else new_r).append(float(ratings[idx]))
    old_a, new_a = np.array(old_r), np.array(new_r)
    if len(old_a) == 0 or len(new_a) == 0:
        print("ERROR: missing ratings for one source"); return 1
    # one-sided test: is NEW materially BELOW OLD?
    u, p_less = mannwhitneyu(new_a, old_a, alternative="less")
    out = {
        "rule": args.rule, "ratings_file": str(args.ratings),
        "n_old": len(old_a), "n_new": len(new_a),
        "old_mean": round(float(old_a.mean()), 3), "new_mean": round(float(new_a.mean()), 3),
        "old_median": float(np.median(old_a)), "new_median": float(np.median(new_a)),
        "mean_gap_new_minus_old": round(float(new_a.mean() - old_a.mean()), 3),
        "mannwhitney_u": float(u), "p_new_below_old": float(p_less),
        # PASS = NEW not MATERIALLY below OLD (effect-size reading of the addendum):
        # mean gap >= -0.5 AND median not lower. p_new_below_old is reported for
        # transparency but is NOT the gate (low-variance high-naturalness data makes
        # a small, immaterial gap statistically significant).
        "significantly_lower_p_lt_0.05": bool(p_less < 0.05),
        "pass": bool((new_a.mean() - old_a.mean()) >= -0.5
                     and np.median(new_a) >= np.median(old_a)),
    }
    dest = SCRATCH / f"divergence_2c_{args.rule}_score_{Path(args.ratings).stem}.json"
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {dest.relative_to(REPO)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--rule", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--old", required=True)
    p.add_argument("--n-each", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260614)
    s = sub.add_parser("score")
    s.add_argument("--rule", required=True)
    s.add_argument("--ratings", required=True)
    args = ap.parse_args()
    return prepare(args) if args.cmd == "prepare" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
