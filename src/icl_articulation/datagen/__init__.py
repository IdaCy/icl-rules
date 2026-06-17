"""Dataset-generation framework for the ICL rule-articulation study (B0 infra).

This package is the *framework* the per-rule dataset generators (phases B1-B4)
build on. It deliberately emits NO dataset here: it provides the bank
infrastructure, the item record + split machinery, the generic probe battery,
the confound report writer, and the seeded generation utilities. The binding
contract for everything here is the private rule spec (not included in this
repository) ``globals``;
the COMMITTED public extract of the per-rule data lives at
data/spec_extract.json (produced by scripts/extract_spec.py).

Module map:
- banks        bank Entry/Bank representation, BANK_QUOTAS contract, self-checks
- banks_data/  one stub module per bank group (filled by the B1-B4 authors)
- schema       the locked item record, jsonl IO, the seeded by-base split assigner
- battery      the 40 frozen generic single-feature predicates + exemption logic
- confound     the confound_report.json writer + distributional asserts
- genutils     seeded RNG plumbing, frame/slot instantiation, count equalizer

Conventions match the rest of the repo: every module raises LOUDLY on a
contract violation (no quiet fallbacks) and never emits markdown into results/
or data/.
"""

from __future__ import annotations
