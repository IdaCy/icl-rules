"""Rule 30: all_words_longer_than_3 (category: hard_articulation).

Canonical articulation: True iff EVERY word in the input has more than 3 letters
(no word of 1, 2, or 3 letters appears). Universal quantifier over ALL words
incl. function words -- that is why the True items are article-free; alphabetic
char count decides length, so a digit token would break the property (never used
in training). The groundtruth verifier ``_r30_all_words_longer_than_3`` recomputes
every label from the surface text (all(alpha_len(tok) > 3)).

Construction (recipe, rule-specs id: all_words_longer_than_3)
------------------------------------------------------------
Each base is ONE article-free telegraphic-but-grammatical sentence (5-9 words)
whose every word is drawn from LONG_ONLY_VOCAB (>= 4 letters): plural bare nouns,
past verbs, adjectives, >= 4-letter prepositions, and the spelled numerals
(three/seven/eight/nine). That long-word sentence is the True variant AND the
``base_id``.

  True  variant = the long-word sentence                          -> label True
  False variant = the SAME sentence with EXACTLY ONE slot replaced -> label False
                  by a <= 3-letter word of the SAME POS from
                  SHORT_WORDS_BY_POS (the spec's 'three'->'two' move),
                  grammar preserved (article-free telegraphic register).

Both variants share every word but the one substituted slot, so their word count
is IDENTICAL -> the 6 word_count battery predicates and the Gate-D length match
sit at exactly 50% by construction. The single short word is the ONLY class
signal; the confound machinery below keeps every other frozen predicate <= 75%.

Confound machinery pinned by the recipe (each earns its keep against a battery
predicate or an multiple-choice distractor):

  * SHORT-WORD DIVERSITY -- no single short word in > 10% of False items; the
    substituted POS is spread noun/verb/adjective/numeral, each >= 15% of False
    items; the substitution POSITION is spread uniformly over the substitutable
    slots. -> no lexical / positional shortcut; keeps the first_word_* and
    last_word_* predicates near 50% (substitution rarely on a fixed end).
  * ARTICLE SALT -- ~10% of False items take 'the' or 'a' as their short-word
    substitution (into a numeral or adjective slot); True items are article-free.
    -> the (non-battery) "contains no articles" distractor disagrees on the ~90%
    of False without an article, and the contains_the / contains_a battery
    predicates stay near 50% (article in only ~5% of False, 0% of True).
  * 4-LETTER QUOTA -- >= 60% of True items contain a word of EXACTLY 4 letters,
    so "every word has more than 4 letters" disagrees on >= 60% of True = 30% of
    items (the FINAL 60% quota).
  * AVG-LENGTH DISCORDANCE -- >= 25% of items where the average word length
    disagrees with the label: True items built mostly of 4-letter words (low avg,
    yet True) and the default False items (one short word among long words). The
    word count also varies 5-9 so the single-short-word char delta is swamped ->
    the char_count and *_len predicates cannot separate the classes (Gate C/D
    verify).

base_id = the True (long-word) sentence's content hash -- shared by both variants,
distinct across bases (the schema split assigner rejects dups).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ... import banks
from ...genutils import Gen, alphabetic_length
from ...schema import PROGRAMMATIC_N_BASES_MIN, word_count

# --- banks (rule-specs generation.banks) --------------------------------------
_LONG_ONLY_VOCAB = "LONG_ONLY_VOCAB"
_SHORT_WORDS_BY_POS = "SHORT_WORDS_BY_POS"

# Build comfortably more than the 340-base floor (100 + 120 + 100 + >= 20 spare).
_N_BASES = 460

# Article-free telegraphic frames as POS-token sequences (5-9 words). POS codes:
#   N   plural bare noun       V   past verb        ADJ adjective
#   P   >= 4-letter preposition (NEVER substituted -- SHORT bank has no preps)
#   NUM spelled numeral (three/seven/eight/nine)
# Every frame is grammatical in the telegraphic register and has the spread of
# lengths that dilutes the single-short-word char delta across the dataset.
_FRAMES: tuple[tuple[str, ...], ...] = (
    ("ADJ", "N", "V", "ADJ", "N"),                       # 5
    ("N", "V", "NUM", "ADJ", "N"),                        # 5
    ("ADJ", "N", "V", "NUM", "N"),                        # 5
    ("N", "V", "P", "ADJ", "N"),                          # 5
    ("ADJ", "N", "V", "N", "P", "N"),                     # 6
    ("N", "V", "ADJ", "N", "P", "N"),                     # 6
    ("N", "V", "NUM", "N", "P", "N"),                     # 6
    ("ADJ", "N", "V", "N", "P", "ADJ", "N"),              # 7
    ("N", "V", "NUM", "ADJ", "N", "P", "N"),              # 7
    ("ADJ", "N", "V", "ADJ", "N", "P", "ADJ", "N"),       # 8
    ("N", "V", "NUM", "ADJ", "N", "P", "ADJ", "N"),       # 8
    ("ADJ", "N", "V", "NUM", "ADJ", "N", "P", "ADJ", "N"),  # 9
)

# the POS codes that have a <= 3-letter same-POS counterpart in SHORT_WORDS_BY_POS
# (prepositions do not, so a P slot is never the substituted one).
_SUBSTITUTABLE = ("N", "V", "ADJ", "NUM")

# article salt: ~10% of False items substitute 'the'/'a' instead of a short
# same-POS word, into a NUM or ADJ slot (rule-specs: articles are part of the
# substitution policy for this rule, never in True items).
_FRAC_ARTICLE_SALT = 0.10
_ARTICLE_WORDS = ("the", "a")
_ARTICLE_SLOTS = ("NUM", "ADJ")

# 4-LETTER QUOTA: >= 60% of True items contain a word of exactly 4 letters. We
# build well above the floor so Gate-free measurement (and the distractor) holds.
_FRAC_FOUR_LETTER = 0.66


@dataclass(frozen=True)
class Base:
    """A rule-30 base.

    ``words`` is the chosen LONG_ONLY_VOCAB filler for each frame slot (the True
    surface = ' '.join(words), sentence-cased first letter). ``sub_index`` is the
    slot the False variant replaces; ``sub_word`` is the <= 3-letter replacement
    (a same-POS SHORT word, or an article when ``article_salt``). ``base_id`` is
    a content hash of the True sentence (shared by both variants, distinct)."""

    base_id: str
    frame: tuple[str, ...]
    words: tuple[str, ...]      # the long-word filler per slot (True surface)
    sub_index: int              # which slot the False variant replaces
    sub_pos: str                # POS code of the substituted slot
    sub_word: str               # the <= 3-letter False replacement (short or article)
    article_salt: bool          # is the False substitution an article ('the'/'a')?


def _cap_first(words: tuple[str, ...]) -> str:
    """Sentence surface from slot words: first letter upper, rest as-is (the
    telegraphic style -- only the leading letter is cased; no terminal punct)."""
    s = " ".join(words)
    return s[:1].upper() + s[1:]


def build_bases(gen: Gen) -> list[Base]:
    """Build >= 340 distinct bases (GENERATOR INTERFACE).

    Deterministic given ``gen``: draw a frame, fill every slot from the matching
    LONG_ONLY_VOCAB POS pool, then pick the False substitution (slot + short
    same-POS word, or an article for the ~10% salt). The POS of the substituted
    slot is round-robin balanced (>= 15% each) and the substitution position is
    spread; no short word is reused in > 10% of False items. Raises (loud) if the
    base floor cannot be reached."""
    long_bank = banks.get_bank(_LONG_ONLY_VOCAB)
    short_bank = banks.get_bank(_SHORT_WORDS_BY_POS)

    # long-word pools by frame POS code. NUM uses the spelled numerals only.
    long_by_pos = {
        "N": [e.word for e in long_bank.entries if e.pos == "noun"],
        "V": [e.word for e in long_bank.entries if e.pos == "verb"],
        "ADJ": [e.word for e in long_bank.entries if e.pos == "adjective"],
        "P": [e.word for e in long_bank.entries if e.pos == "preposition"],
        "NUM": [e.word for e in long_bank.entries if e.pos == "numeral"],
    }
    # exactly-4-letter long words (for the >= 60% quota) by POS.
    four_by_pos = {
        pos: [w for w in ws if alphabetic_length(w) == 4]
        for pos, ws in long_by_pos.items()
    }
    # short <= 3-letter same-POS pools (SHORT_WORDS_BY_POS pos labels).
    _SHORT_POS = {"N": "noun", "V": "verb", "ADJ": "adjective", "NUM": "numeral"}
    short_by_pos = {
        code: [e.word for e in short_bank.entries if e.pos == pos]
        for code, pos in _SHORT_POS.items()
    }
    for code, pool in long_by_pos.items():
        if not pool:
            raise ValueError(f"all_words_longer_than_3: empty long pool for {code}")
    for code, pool in short_by_pos.items():
        if not pool:
            raise ValueError(f"all_words_longer_than_3: empty short pool for {code}")

    g_frame = gen.derive("frame")
    g_fill = gen.derive("fill")
    g_four = gen.derive("four_letter")
    g_subpos = gen.derive("sub_pos")
    g_subword = gen.derive("sub_word")
    g_salt = gen.derive("salt")
    g_artpos = gen.derive("article_pos")

    # round-robin order over substitutable POS so each is >= 15% (= 25% here).
    # The TARGET POS is decided first, then a frame CONTAINING that POS is drawn,
    # so the spread holds even for NUM (which is absent from many frames).
    pos_cycle = list(_SUBSTITUTABLE)
    frames_with_pos = {
        code: [f for f in _FRAMES if code in f] for code in _SUBSTITUTABLE
    }
    for code, fs in frames_with_pos.items():
        if not fs:
            raise ValueError(f"all_words_longer_than_3: no frame contains POS {code}")

    bases: list[Base] = []
    seen: set[str] = set()
    # per-short-word usage cap so no short word lands in > 10% of False items.
    short_usage: Counter[str] = Counter()
    short_cap = max(1, int(_N_BASES * 0.10))

    attempts = 0
    max_attempts = _N_BASES * 200
    n_built = 0
    while len(bases) < _N_BASES and attempts < max_attempts:
        attempts += 1
        # target substituted POS first (round-robin), then a frame that has it.
        target_pos = pos_cycle[n_built % len(pos_cycle)]
        frame = g_frame.choice(frames_with_pos[target_pos])

        # decide whether THIS base must contain an exactly-4-letter word (>=60%).
        want_four = g_four.rng.random() < _FRAC_FOUR_LETTER

        # fill every slot from its POS pool (distinct words within a sentence to
        # keep the surface natural and dedup clean across repeated POS).
        words: list[str] = []
        used: set[str] = set()
        ok = True
        # pick one slot to force an exactly-4-letter word if want_four.
        four_slot = -1
        if want_four:
            cand = [i for i, c in enumerate(frame) if four_by_pos.get(c)]
            if cand:
                four_slot = g_four.choice(cand)
        for i, code in enumerate(frame):
            pool = long_by_pos[code]
            if i == four_slot:
                pool = [w for w in four_by_pos[code] if w not in used] or pool
            choices = [w for w in pool if w not in used]
            if not choices:
                ok = False
                break
            w = g_fill.choice(choices)
            words.append(w)
            used.add(w)
        if not ok:
            continue

        # the substituted slot is one of the target-POS slots in this frame.
        sub_pos = target_pos
        slot_idxs = [i for i, c in enumerate(frame) if c == sub_pos]
        sub_index = g_subpos.choice(slot_idxs)

        # article salt (~10%): replace an ADJ slot (preferred) or NUM slot with
        # 'the'/'a' instead -- ADJ-first so the NUM substitution share is not
        # cannibalized below the 15% POS-spread floor. Article salt is INDEPENDENT
        # of the round-robin POS accounting (it does not count toward the spread).
        article_salt = False
        sub_word = ""
        if g_salt.rng.random() < _FRAC_ARTICLE_SALT:
            adj_idxs = [i for i, c in enumerate(frame) if c == "ADJ"]
            num_idxs = [i for i, c in enumerate(frame) if c == "NUM"]
            art_idxs = adj_idxs or num_idxs
            if art_idxs:
                article_salt = True
                sub_index = g_artpos.choice(art_idxs)
                sub_pos = frame[sub_index]
                sub_word = g_artpos.choice(list(_ARTICLE_WORDS))
        if not article_salt:
            # a short same-POS word, honouring the <=10% per-word usage cap.
            pool = [
                w for w in short_by_pos[sub_pos] if short_usage[w] < short_cap
            ] or short_by_pos[sub_pos]
            sub_word = g_subword.choice(pool)

        words_t = tuple(words)
        true_text = _cap_first(words_t)
        bid = "b" + _content_hash(true_text)
        if bid in seen:
            continue

        seen.add(bid)
        if not article_salt:
            short_usage[sub_word] += 1
        bases.append(
            Base(
                base_id=bid,
                frame=frame,
                words=words_t,
                sub_index=sub_index,
                sub_pos=sub_pos,
                sub_word=sub_word,
                article_salt=article_salt,
            )
        )
        n_built += 1

    if len(bases) < PROGRAMMATIC_N_BASES_MIN:
        raise ValueError(
            f"all_words_longer_than_3: only built {len(bases)} bases, need "
            f">= {PROGRAMMATIC_N_BASES_MIN}"
        )
    return bases


def instantiate(spec: Base, label: bool, gen: Gen) -> tuple[str, dict[str, Any]]:
    """Instantiate ONE variant of a base (GENERATOR INTERFACE).

    True  = the long-word sentence (every word >= 4 letters -> groundtruth True).
    False = the SAME sentence with slot ``sub_index`` replaced by ``sub_word`` (a
    <= 3-letter short same-POS word or an article -> groundtruth False).
    Deterministic; ``gen`` carries no randomness here (the plan is fixed in the
    base spec) but is kept to match the interface."""
    if label:
        out_words = spec.words
        transform = "long_only"
    else:
        lst = list(spec.words)
        lst[spec.sub_index] = spec.sub_word
        out_words = tuple(lst)
        transform = "article_salt" if spec.article_salt else "short_substitution"
    text = _cap_first(out_words)
    meta = {
        "frame": list(spec.frame),
        "words": list(spec.words),
        "sub_index": spec.sub_index,
        "sub_pos": spec.sub_pos,
        "sub_word": spec.sub_word,
        "article_salt": spec.article_salt,
        "transform": transform,
        "word_count": word_count(text),
    }
    return text, meta


# --- helpers ------------------------------------------------------------------


def _content_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
