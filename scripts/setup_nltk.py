#!/usr/bin/env python
"""Download the nltk data the POS battery needs. Run manually once per machine.

The battery's first-word-POS predicates use nltk's averaged_perceptron_tagger
(rule-specs implementation_pins.pos_tagger). That tagger ships its model as
downloadable data, NOT in the pip package. Downloading must never happen at
import time (it would turn an offline test into a network call), so this is a
standalone script the operator runs once per machine.

Run:  .venv/bin/python scripts/setup_nltk.py
On a remote/compute instance, use that instance's own Python interpreter.
"""

from __future__ import annotations

import sys

# both the modern and legacy resource names (nltk renamed it across versions);
# downloading both is harmless and keeps the tagger working on either nltk.
RESOURCES = ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "punkt")


def main() -> int:
    import nltk

    ok = True
    for res in RESOURCES:
        try:
            got = nltk.download(res)
            print(f"{res}: {'ok' if got else 'FAILED'}")
            ok = ok and got
        except Exception as exc:  # network / resource errors are reported, not swallowed
            print(f"{res}: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
            ok = False
    if not ok:
        print("some nltk resources failed to download", file=sys.stderr)
        return 1
    # smoke test
    from nltk import pos_tag

    print("smoke:", pos_tag(["close", "the", "window"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
