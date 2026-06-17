"""Bank-content modules, one per bank group (+ core for the reference bank).

Each module exposes ``BANKS = {bank_name: [entries...]}``. banks.py merges them
into a single registry. The group modules ship as empty stubs at B0 stage A;
their authors (phases B1-B4) fill them. core.py is authored at B0 (NUMBER_WORDS).
"""

from __future__ import annotations
