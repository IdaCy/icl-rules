"""Seeded generation plumbing shared by every per-rule generator.

Determinism is a hard requirement (every run logs seed + config):
each generator threads a single ``Gen`` through its work, and the seed it was
built with is what gets logged. Nothing here touches the network or the API.

The load-bearing piece is the word-count equalizer: many rules must make their
True and False variants share an exact word count by padding with place/manner
adjuncts of known word length (ADVERB_PLACE entries are 1, 2, or 3 words). The
equalizer must solve the count EXACTLY (the length_matching policy asserts
|mean_T - mean_F| <= 0.2; per-base exactness is the only way to guarantee it),
so it raises if a target is unreachable rather than getting close.
"""

from __future__ import annotations

import hashlib
import re
from random import Random
from typing import Iterable, Sequence

from .schema import word_count, words

# slot marker in frame templates, e.g. "The {X} was in the garden"
SLOT_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")


class GenError(ValueError):
    """A generation invariant was violated (LOUD; no silent best-effort)."""


class Gen:
    """A seeded RNG wrapper. One per generator; the seed is logged by the caller."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = Random(seed)

    def choice(self, seq: Sequence):
        if not seq:
            raise GenError("choice from an empty sequence")
        return self.rng.choice(seq)

    def sample(self, seq: Sequence, k: int) -> list:
        if k > len(seq):
            raise GenError(f"cannot sample {k} from {len(seq)} items")
        return self.rng.sample(list(seq), k)

    def shuffle(self, seq: list) -> list:
        self.rng.shuffle(seq)
        return seq

    def randint(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)

    def derive(self, tag: str) -> "Gen":
        """A child Gen seeded deterministically from this seed + a label tag.

        Lets a generator give sub-phases independent but reproducible streams
        without consuming the parent's draw order."""
        h = hashlib.sha256(f"{self.seed}:{tag}".encode("utf-8")).hexdigest()
        return Gen(int(h[:16], 16))


# --- frame / slot instantiation -----------------------------------------------


def frame_slots(frame: str) -> list[str]:
    """Slot names in a frame template, in order of appearance (with repeats)."""
    return SLOT_RE.findall(frame)


def fill_frame(frame: str, fillers: dict[str, str]) -> str:
    """Substitute every {slot} in ``frame`` from ``fillers``.

    Raises if a slot has no filler or a filler is empty — a missing slot in a
    paid dataset is a silent corruption, so it must be loud here."""
    needed = set(frame_slots(frame))
    missing = needed - set(fillers)
    if missing:
        raise GenError(f"frame {frame!r} missing fillers for {sorted(missing)}")

    def repl(m: re.Match) -> str:
        val = fillers[m.group(1)]
        if not isinstance(val, str) or val == "":
            raise GenError(f"empty filler for slot {m.group(1)!r} in {frame!r}")
        return val

    return SLOT_RE.sub(repl, frame).strip()


# --- casing transforms --------------------------------------------------------


def to_sentence_case(text: str) -> str:
    """First alphabetic char upper, everything else lower (globals.casing.default).

    Operates on the raw string: leading punctuation is preserved, the first
    LETTER is the one capitalized."""
    lowered = text.lower()
    for i, ch in enumerate(lowered):
        if ch.isalpha():
            return lowered[:i] + ch.upper() + lowered[i + 1 :]
    return lowered  # no letters (cannot occur in training)


def to_lower(text: str) -> str:
    """Fully lowercased (rule 1 True variant)."""
    return text.lower()


def to_title_case(text: str) -> str:
    """Capitalize the first alphabetic char of EVERY whitespace token (rule 4).

    This is 'every word capitalized', NOT editorial title case (stopwords are
    capitalized too) — matches rule 4's canonical articulation."""
    out_tokens = []
    for tok in text.split():
        lowered = tok.lower()
        done = False
        chars = list(lowered)
        for i, ch in enumerate(chars):
            if ch.isalpha():
                chars[i] = ch.upper()
                done = True
                break
        out_tokens.append("".join(chars) if done else lowered)
    return " ".join(out_tokens)


# --- indefinite-article normalizer (central, label-neutral) -------------------

# Vowel-LETTER but consonant-SOUND: these take "a", not "an" ("a university").
_A_BEFORE_VOWEL_LETTER = frozenset(
    {
        "university",
        "unit",
        "use",
        "useful",
        "user",
        "european",
        "one",
        "once",
        "uniform",
        "unique",
    }
)
# Consonant-LETTER but vowel-SOUND: these take "an", not "a" ("an hour").
_AN_BEFORE_CONSONANT_LETTER = frozenset({"hour", "honest", "honor"})
_VOWEL_LETTERS = frozenset("aeiou")
# leading/trailing chars stripped when classifying the following word's sound;
# mirrors schema.PUNCT_STRIP so "(apple" and "apple," classify on "apple".
_ARTICLE_PUNCT = ".,!?;:\"'()[]-–—…"


def _starts_with_vowel_sound(token: str) -> bool:
    """Does the word in ``token`` begin with a vowel SOUND?

    Vowel-LETTER heuristic plus the two small hand-curated exception sets. The
    token is stripped of surrounding punctuation first so "apple," / "(office)"
    classify on the bare word. An empty/punctuation-only token is treated as a
    consonant (no rewrite)."""
    bare = token.strip(_ARTICLE_PUNCT).lower()
    if not bare:
        return False
    if bare in _A_BEFORE_VOWEL_LETTER:
        return False
    if bare in _AN_BEFORE_CONSONANT_LETTER:
        return True
    return bare[0] in _VOWEL_LETTERS


def fix_indefinite_articles(text: str) -> str:
    """Rewrite the indefinite article ``a``/``A`` to ``an``/``An`` when the
    FOLLOWING word starts with a vowel sound (and leave it alone otherwise).

    This is a CENTRAL, label-neutral normalizer applied by the emit pipeline to
    EVERY item's text right after instantiation, so all rules get correct
    "an apple" / "a university" / "an hour" with no per-rule edits.

    Guarantees:
      * Only the standalone article token ``a`` / ``A`` is ever touched; any
        surrounding punctuation on that token is preserved and no OTHER token is
        modified (so it cannot change a rule's label that keys off other words).
      * ``a`` -> ``an`` and ``A`` -> ``An`` preserve the original case; ``an`` is
        a single whitespace token, so the WORD COUNT is unchanged.
      * Classification uses a vowel-LETTER heuristic with two exception sets:
        vowel-letter-but-consonant-sound ("university", "one", ...) keep ``a``;
        consonant-letter-but-vowel-sound ("hour", "honest", "honor") take ``an``.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text
    out = list(tokens)
    for i in range(len(tokens) - 1):
        tok = tokens[i]
        # only a bare article token (allowing surrounding punctuation) is rewritten
        bare = tok.strip(_ARTICLE_PUNCT)
        if bare not in ("a", "A"):
            continue
        if not _starts_with_vowel_sound(tokens[i + 1]):
            continue
        replacement = "an" if bare == "a" else "An"
        # splice "n" in where the bare article sits, keeping any attached punct
        start = tok.find(bare)
        out[i] = tok[:start] + replacement + tok[start + len(bare) :]
    return " ".join(out)


# --- word-count equalizer -----------------------------------------------------


def adjunct_word_lengths(adjuncts: Iterable[str]) -> dict[int, list[str]]:
    """Group adjunct phrases by their word count (1, 2, 3 words ...)."""
    by_len: dict[int, list[str]] = {}
    for phrase in adjuncts:
        n = word_count(phrase)
        if n <= 0:
            raise GenError(f"adjunct {phrase!r} has zero words")
        by_len.setdefault(n, []).append(phrase)
    return by_len


def solve_adjuncts(deficit: int, available_lengths: Sequence[int]) -> list[int]:
    """Pick a multiset of adjunct word-lengths summing EXACTLY to ``deficit``.

    ``available_lengths`` are the distinct phrase word-counts on hand (e.g.
    {1, 2, 3} for ADVERB_PLACE). Returns the chosen lengths (one entry per
    adjunct slot to fill), preferring the FEWEST adjuncts. Raises GenError if
    no exact combination exists (deficit < 0, or deficit unreachable). With a
    1-word phrase available, every non-negative deficit is reachable."""
    if deficit < 0:
        raise GenError(f"negative word-count deficit {deficit}")
    if deficit == 0:
        return []
    lengths = sorted({int(x) for x in available_lengths}, reverse=True)
    if not lengths or any(x <= 0 for x in lengths):
        raise GenError(f"invalid adjunct lengths {available_lengths!r}")
    # minimal-coin search (exact). Lengths are tiny (1..3) so DP over deficit.
    INF = float("inf")
    best: list[float] = [0.0] + [INF] * deficit
    pick: list[int] = [0] * (deficit + 1)
    for target in range(1, deficit + 1):
        for L in lengths:
            if L <= target and best[target - L] + 1 < best[target]:
                best[target] = best[target - L] + 1
                pick[target] = L
    if best[deficit] == INF:
        raise GenError(
            f"cannot reach deficit {deficit} exactly from lengths {lengths}"
        )
    chosen: list[int] = []
    t = deficit
    while t > 0:
        L = pick[t]
        chosen.append(L)
        t -= L
    return chosen


def equalize_word_count(
    text: str,
    target: int,
    adjuncts_by_len: dict[int, list[str]],
    gen: Gen,
    *,
    joiner: str = " ",
) -> str:
    """Append adjunct phrases so ``text`` reaches EXACTLY ``target`` words.

    Picks phrases from ``adjuncts_by_len`` (length -> list of phrases) summing
    to the deficit, draws specific phrases with ``gen`` (deterministic), and
    appends them. Raises if the text already exceeds the target or the deficit
    is unreachable. The chosen phrases are appended in a seeded-shuffled order."""
    have = word_count(text)
    if have > target:
        raise GenError(
            f"text already has {have} words, over target {target}: {text!r}"
        )
    chosen_lengths = solve_adjuncts(target - have, list(adjuncts_by_len))
    phrases: list[str] = []
    for L in chosen_lengths:
        pool = adjuncts_by_len.get(L)
        if not pool:
            raise GenError(f"no adjunct of length {L} available")
        phrases.append(gen.choice(pool))
    gen.shuffle(phrases)
    result = text
    for ph in phrases:
        result = f"{result}{joiner}{ph}"
    if word_count(result) != target:
        raise GenError(
            f"equalizer produced {word_count(result)} words, expected {target}: "
            f"{result!r}"
        )
    return result


# --- dedup + base_id ----------------------------------------------------------


def dedup_texts(items: Sequence[dict], key: str = "text") -> list[dict]:
    """Drop later items whose ``key`` value was already seen (stable order)."""
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        v = it[key]
        if v in seen:
            continue
        seen.add(v)
        out.append(it)
    return out


def base_id(*parts: object) -> str:
    """Stable base_id from its content parts (rule recipes define what goes in).

    Parts are joined and hashed so the id is fixed-length and filesystem-safe
    while still being deterministic for the same content (two variants of a
    base call this with identical parts and share the result)."""
    blob = "␟".join(str(p) for p in parts)  # unit separator, never in text
    return "b" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def item_id(base: str, label: bool, variant_tag: str = "") -> str:
    """Deterministic item_id for a (base, label[, variant]) triple."""
    suffix = "T" if label else "F"
    if variant_tag:
        suffix = f"{suffix}-{variant_tag}"
    return f"{base}-{suffix}"


def alphabetic_length(word: str) -> int:
    """globals.tokenizer.word_length — count of ALPHABETIC chars in a stripped word."""
    return sum(1 for ch in word if ch.isalpha())


def stripped_words(text: str) -> list[str]:
    """Re-export of the global tokenizer for generator convenience."""
    return words(text)
