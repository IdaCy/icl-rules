"""Core / reference bank: NUMBER_WORDS, fully authored.

This is the ONE bank populated at B0 stage A. It exists to prove the bank
infrastructure end-to-end: ``banks.check_bank('NUMBER_WORDS')`` must pass
before any group author runs, so the entry contract, the computed tags, and
the quota self-check are all validated against real content.

NUMBER_WORDS (rule-specs banks: size 20, "spelled-out numbers: three..twenty,
thirty, forty, fifty"). The literal "three..twenty + thirty, forty, fifty"
enumeration is three..twenty (18) + 3 = 21, one over the stated size 20. The
resolution that yields EXACTLY 20 and also satisfies the global tier rule
("nothing rarer than tier 2 anywhere", globals.banks header): drop 'nineteen',
which wordfreq ranks rarer than the top-10000 (tier 3) — every other word in
three..twenty + thirty/forty/fifty is tier 1 or tier 2. See banks.py for the
authored-vs-computed tag split; here pos='numeral' and frequency_tier are the
authored tags (tiers verified against wordfreq in the optional verification
test).
"""

from __future__ import annotations

from typing import Any


def _n(word: str, tier: int) -> dict[str, Any]:
    return {"word": word, "pos": "numeral", "frequency_tier": tier}


# tiers from wordfreq top_n_list (1 = top 2000, 2 = top 10000); see module docstring.
NUMBER_WORDS: list[dict[str, Any]] = [
    _n("three", 1),
    _n("four", 1),
    _n("five", 1),
    _n("six", 1),
    _n("seven", 1),
    _n("eight", 1),
    _n("nine", 1),
    _n("ten", 1),
    _n("eleven", 2),
    _n("twelve", 2),
    _n("thirteen", 2),
    _n("fourteen", 2),
    _n("fifteen", 2),
    _n("sixteen", 2),
    _n("seventeen", 2),
    _n("eighteen", 2),
    _n("twenty", 2),
    _n("thirty", 2),
    _n("forty", 2),
    _n("fifty", 2),
]

BANKS: dict[str, list[Any]] = {
    "NUMBER_WORDS": NUMBER_WORDS,
}
