#!/usr/bin/env python
"""Build the public rule_given rules-file from the committed spec extract.

`scripts/run_step1.py --mode rule_given` needs a JSON `{rule_id: rule text}`
mapping (the canonical articulation told to the model in the zero-shot baseline).
The original runs read a private articulations file (not in this repository),
which a public clone does not have. The same 30 canonical strings already live,
verbatim, in the committed `data/spec_extract.json` under each rule's
`canonical_articulation` (verified identical to the texts logged in the
rule_given run configs). This script projects them into a flat, public
`data/rule_texts.json` that the README command points at.

Run:  python scripts/make_rules_file.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPEC_EXTRACT = REPO / "data" / "spec_extract.json"
OUT_PATH = REPO / "data" / "rule_texts.json"


def build_rule_texts(spec_extract_path: Path = SPEC_EXTRACT) -> dict[str, str]:
    """{rule_id: canonical_articulation} for every rule in the spec extract."""
    data = json.loads(spec_extract_path.read_text(encoding="utf-8"))
    rules = data.get("rules")
    if not isinstance(rules, dict) or not rules:
        raise SystemExit(f"{spec_extract_path}: no 'rules' object — wrong file?")
    texts: dict[str, str] = {}
    for rule_id, entry in sorted(rules.items()):
        canon = (entry or {}).get("canonical_articulation")
        if not isinstance(canon, str) or not canon.strip():
            raise SystemExit(f"{spec_extract_path}: rule {rule_id!r} has no canonical_articulation")
        texts[rule_id] = canon.strip()
    return texts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec-extract", default=str(SPEC_EXTRACT))
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args(argv)

    texts = build_rule_texts(Path(args.spec_extract))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(texts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(texts)} rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
