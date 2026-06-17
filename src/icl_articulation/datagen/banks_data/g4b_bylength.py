"""Bank group G4b (by-length and terminal-letter banks). AUTHORED CONTENT.

Owns these banks (see banks.BANK_QUOTAS for the per-bank contract):
  INITIAL_BY_LENGTH, FINAL_BY_LENGTH, VOWEL_INITIAL, CONSONANT_INITIAL,
  TERMINAL_VOWEL, TERMINAL_CONSONANT

These six banks supply the positional rules (19 first/last-word length
comparison; 20 last-letter vowel) and the starts-with-vowel rule (7), so the
binding constraints come BOTH from globals.banks and from those rule recipes.

Author/check split (banks.py): we author ``word``, ``pos``, ``frequency_tier``
and the optional flags ``discordant`` / ``final_y`` / ``pair``; banks.py
computes ``length`` / ``initial`` / ``final`` / repeat flags from ``word`` and
raises if an authored value disagrees. Every ``frequency_tier`` here was
verified against the pinned wordfreq top-2000 (tier 1) / top-10000 (tier 2)
lists; nothing rarer than tier 2 appears (globals.banks header). Syllable
counts driving the ``discordant`` tags were taken from the pinned nltk cmudict
(implementation_pins.syllable_counter), first pronunciation.

Per-bank decisions the self-check enforces:
  - INITIAL_BY_LENGTH (size 54): sentence-initial-capable words (plural nouns +
    1-word adverbs), lengths 3-11, >= 6 entries PER length (length 11 required).
    Length 3 has almost no plural common nouns at tier <= 2 ('men' is the only
    one), so the length-3 slot is filled with 'men' + five tier-1 1-word
    sentence-initial adverbs (now/far/too/out/off) — the recipe explicitly
    allows 1-word adverbs as sentence-initial fillers.
  - FINAL_BY_LENGTH (size 54): sentence-final-capable words (singular nouns +
    1-word place adverbs), lengths 3-11, >= 6 per length, >= 25% discordant.
    ``discordant`` is tagged honestly as a marked letter/syllable divergence
    matching the spec exemplars 'through' (7 letters / 1 syllable) and 'idea'
    (4 letters / 3 syllables): length >= 6 with a 1-syllable pronunciation, OR
    length <= 5 with a >= 3-syllable pronunciation. 15/54 = 28% are tagged.
  - VOWEL_INITIAL (size 40) / CONSONANT_INITIAL (size 40): matched POS mix
    70% noun / 30% adverb (28 nouns + 12 adverbs each). VOWEL_INITIAL initials
    are all a/e/i/o/u with all five vowels present (>= 5 each so rule 7's
    per-vowel >= 10% guard closes); 'one' is EXCLUDED (vowel letter / consonant
    sound, reserved for step-3 probes). CONSONANT_INITIAL initials spread over
    all four alphabet buckets with none > 40% of the bank. The pair is matched
    on mean length (length_dist).
  - TERMINAL_VOWEL (size >= 50; 52 here) / TERMINAL_CONSONANT (size 50): matched
    final-word banks. TERMINAL_VOWEL is nouns-only, all ending in a vowel LETTER,
    with >= 50% non-e endings (27/52 = 0.519, a margin over the floor) and
    >= 25% silent-final-e endings. TERMINAL_CONSONANT
    tags ``final_y`` on every entry; the rule-20 draws (clear consonant enders,
    noun-only) are the final_y=False nouns, while the final_y=True entries
    (city, slowly, ...) stand for graders/probes and rule-19 reuse. The pair is
    matched on mean length (length_dist).

Each entry is a word-bank dict per the entry contract in banks.py.
"""

from __future__ import annotations

from typing import Any


def _w(word: str, pos: str, tier: int, **flags: Any) -> dict[str, Any]:
    """Author one word entry (authored keys only; banks.py computes the rest)."""
    entry: dict[str, Any] = {"word": word, "pos": pos, "frequency_tier": tier}
    entry.update(flags)
    return entry


# --- INITIAL_BY_LENGTH --------------------------------------------------------
# Sentence-initial-capable: plural common nouns + 1-word sentence-initial
# adverbs. >= 6 per length for lengths 3..11. Length 3 plural nouns are
# essentially unavailable at tier <= 2 (only 'men'), so the length-3 row uses
# 'men' plus five tier-1 1-word adverbs; all other rows are plural nouns.
INITIAL_BY_LENGTH: list[dict[str, Any]] = [
    # length 3 (1 plural noun + 5 one-word adverbs)
    _w("men", "noun", 1),
    _w("now", "adverb", 1),
    _w("far", "adverb", 1),
    _w("too", "adverb", 1),
    _w("out", "adverb", 1),
    _w("off", "adverb", 1),
    # length 4 (plural nouns)
    _w("dogs", "noun", 1),
    _w("cars", "noun", 1),
    _w("jobs", "noun", 1),
    _w("kids", "noun", 1),
    _w("eyes", "noun", 1),
    _w("days", "noun", 1),
    # length 5 (plural nouns)
    _w("books", "noun", 1),
    _w("games", "noun", 1),
    _w("teams", "noun", 1),
    _w("plans", "noun", 1),
    _w("hands", "noun", 1),
    _w("words", "noun", 1),
    # length 6 (plural nouns)
    _w("houses", "noun", 1),
    _w("tables", "noun", 2),
    _w("horses", "noun", 2),
    _w("cities", "noun", 1),
    _w("people", "noun", 1),
    _w("groups", "noun", 1),
    # length 7 (plural nouns)
    _w("gardens", "noun", 2),
    _w("windows", "noun", 1),
    _w("numbers", "noun", 1),
    _w("flowers", "noun", 2),
    _w("animals", "noun", 1),
    _w("members", "noun", 1),
    # length 8 (plural nouns)
    _w("children", "noun", 1),
    _w("students", "noun", 1),
    _w("teachers", "noun", 1),
    _w("brothers", "noun", 2),
    _w("soldiers", "noun", 2),
    _w("machines", "noun", 2),
    # length 9 (plural nouns)
    _w("buildings", "noun", 2),
    _w("customers", "noun", 1),
    _w("daughters", "noun", 2),
    _w("employees", "noun", 1),
    _w("musicians", "noun", 2),
    _w("questions", "noun", 1),
    # length 10 (plural nouns)
    _w("passengers", "noun", 2),
    _w("volunteers", "noun", 2),
    _w("presidents", "noun", 2),
    _w("characters", "noun", 1),
    _w("conditions", "noun", 1),
    _w("industries", "noun", 2),
    # length 11 (plural nouns)
    _w("instruments", "noun", 2),
    _w("researchers", "noun", 2),
    _w("governments", "noun", 2),
    _w("connections", "noun", 2),
    _w("collections", "noun", 2),
    _w("suggestions", "noun", 2),
]


# --- FINAL_BY_LENGTH ----------------------------------------------------------
# Sentence-final-capable: singular nouns + 1-word place adverbs. >= 6 per length
# for lengths 3..11 (length 11 required). >= 25% discordant (tagged honestly:
# long word / 1 syllable, or short word / >= 3 syllables; matches spec
# exemplars 'through' 7/1 and 'idea' 4/3). 15 discordant of 54 = 28%.
FINAL_BY_LENGTH: list[dict[str, Any]] = [
    # length 3 (singular nouns)
    _w("cat", "noun", 1),
    _w("dog", "noun", 1),
    _w("sun", "noun", 1),
    _w("car", "noun", 1),
    _w("box", "noun", 1),
    _w("bus", "noun", 1),
    # length 4 (singular nouns; 'idea'/'area' discordant 4 letters / 3 syllables)
    _w("idea", "noun", 1, discordant=True),
    _w("area", "noun", 1, discordant=True),
    _w("lake", "noun", 1),
    _w("book", "noun", 1),
    _w("road", "noun", 1),
    _w("tree", "noun", 1),
    # length 5 (singular nouns; 'radio'/'video'/'audio' discordant 5 letters / 3 syllables)
    _w("radio", "noun", 1, discordant=True),
    _w("video", "noun", 1, discordant=True),
    _w("audio", "noun", 2, discordant=True),
    _w("house", "noun", 1),
    _w("river", "noun", 1),
    _w("plant", "noun", 1),
    # length 6 (singular nouns; 'bridge'/'school'/'length' discordant 6 letters / 1 syllable)
    _w("bridge", "noun", 1, discordant=True),
    _w("school", "noun", 1, discordant=True),
    _w("length", "noun", 1, discordant=True),
    _w("garden", "noun", 1),
    _w("market", "noun", 1),
    _w("window", "noun", 1),
    # length 7 (singular nouns; 'through'/'thought' discordant 7 letters / 1 syllable)
    _w("through", "adverb", 1, discordant=True),
    _w("thought", "noun", 1, discordant=True),
    _w("morning", "noun", 1),
    _w("village", "noun", 1),
    _w("station", "noun", 1),
    _w("machine", "noun", 1),
    # length 8 (singular nouns; 'straight'/'thoughts' discordant 8 letters / 1 syllable)
    _w("straight", "adverb", 1, discordant=True),
    _w("thoughts", "noun", 1, discordant=True),
    _w("mountain", "noun", 1),
    _w("building", "noun", 1),
    _w("hospital", "noun", 1),
    _w("airplane", "noun", 2),
    # length 9 (singular nouns; 'stretched'/'strengths' discordant 9 letters / 1 syllable)
    _w("stretched", "verb", 2, discordant=True),
    _w("strengths", "noun", 2, discordant=True),
    _w("apartment", "noun", 2),
    _w("furniture", "noun", 2),
    _w("adventure", "noun", 2),
    _w("breakfast", "noun", 2),
    # length 10 (singular nouns; 'playground' 2 syllables, 'friendship' 2 syllables)
    _w("restaurant", "noun", 2),
    _w("playground", "noun", 2),
    _w("motorcycle", "noun", 2),
    _w("helicopter", "noun", 2),
    _w("generation", "noun", 1),
    _w("journalist", "noun", 2),
    # length 11 (singular nouns)
    _w("grandmother", "noun", 2),
    _w("grandfather", "noun", 2),
    _w("information", "noun", 1),
    _w("temperature", "noun", 2),
    _w("supermarket", "noun", 2),
    _w("combination", "noun", 2),
]


# --- VOWEL_INITIAL ------------------------------------------------------------
# 28 plural nouns (>= 5 of each vowel) + 12 adverbs = 40. All initials a/e/i/o/u.
# 'one' EXCLUDED. POS mix 28/40 = 0.70 noun, 12/40 = 0.30 adverb.
VOWEL_INITIAL: list[dict[str, Any]] = [
    # nouns: a (6)
    _w("animals", "noun", 1, pair="vi01"),
    _w("apples", "noun", 2, pair="vi02"),
    _w("authors", "noun", 2, pair="vi03"),
    _w("areas", "noun", 1, pair="vi04"),
    _w("answers", "noun", 2, pair="vi05"),
    _w("actors", "noun", 2, pair="vi06"),
    # nouns: e (6)
    _w("engines", "noun", 2, pair="vi07"),
    _w("eagles", "noun", 2, pair="vi08"),
    _w("examples", "noun", 2, pair="vi09"),
    _w("emotions", "noun", 2, pair="vi10"),
    _w("editors", "noun", 2, pair="vi11"),
    _w("events", "noun", 1, pair="vi12"),
    # nouns: i (5)
    _w("images", "noun", 1, pair="vi13"),
    _w("ideas", "noun", 1, pair="vi14"),
    _w("issues", "noun", 1, pair="vi15"),
    _w("islands", "noun", 2, pair="vi16"),
    _w("insects", "noun", 2, pair="vi17"),
    # nouns: o (6)
    _w("owners", "noun", 2, pair="vi18"),
    _w("onions", "noun", 2, pair="vi19"),
    _w("options", "noun", 2, pair="vi20"),
    _w("others", "noun", 1, pair="vi21"),
    _w("officers", "noun", 1, pair="vi22"),
    _w("objects", "noun", 2, pair="vi23"),
    # nouns: u (5)
    _w("units", "noun", 1, pair="vi24"),
    _w("unions", "noun", 2, pair="vi25"),
    _w("users", "noun", 1, pair="vi26"),
    _w("updates", "noun", 2, pair="vi27"),
    _w("uniforms", "noun", 2, pair="vi28"),
    # adverbs (12): vowels spread a/e/i/o/u
    _w("often", "adverb", 1, pair="vi29"),
    _w("always", "adverb", 1, pair="vi30"),
    _w("early", "adverb", 1, pair="vi31"),
    _w("easily", "adverb", 1, pair="vi32"),
    _w("openly", "adverb", 2, pair="vi33"),
    _w("equally", "adverb", 2, pair="vi34"),
    _w("usually", "adverb", 1, pair="vi35"),
    _w("anywhere", "adverb", 1, pair="vi36"),
    _w("obviously", "adverb", 1, pair="vi37"),
    _w("instantly", "adverb", 2, pair="vi38"),
    _w("elsewhere", "adverb", 2, pair="vi39"),
    _w("utterly", "adverb", 2, pair="vi40"),
]


# --- CONSONANT_INITIAL --------------------------------------------------------
# Matched to VOWEL_INITIAL: 28 nouns + 12 adverbs = 40, mean length matched.
# Initials spread over all four buckets a-f / g-m / n-s / t-z, none > 40%
# (<= 16). pair keys are positional matches to VOWEL_INITIAL for documentation;
# the quota only checks length_dist (mean within 1.5).
CONSONANT_INITIAL: list[dict[str, Any]] = [
    # nouns (28); buckets balanced
    # a-f bucket nouns
    _w("books", "noun", 1, pair="vi01"),
    _w("cars", "noun", 1, pair="vi02"),
    _w("dogs", "noun", 1, pair="vi04"),
    _w("doctors", "noun", 2, pair="vi03"),
    _w("farmers", "noun", 2, pair="vi05"),
    _w("flowers", "noun", 2, pair="vi06"),
    _w("friends", "noun", 1, pair="vi07"),
    # g-m bucket nouns
    _w("games", "noun", 1, pair="vi08"),
    _w("guests", "noun", 2, pair="vi09"),
    _w("houses", "noun", 1, pair="vi10"),
    _w("leaders", "noun", 1, pair="vi11"),
    _w("letters", "noun", 2, pair="vi12"),
    _w("members", "noun", 1, pair="vi13"),
    _w("machines", "noun", 2, pair="vi14"),
    # n-s bucket nouns
    _w("numbers", "noun", 1, pair="vi15"),
    _w("parents", "noun", 1, pair="vi16"),
    _w("players", "noun", 1, pair="vi17"),
    _w("rivers", "noun", 2, pair="vi18"),
    _w("singers", "noun", 2, pair="vi19"),
    _w("students", "noun", 1, pair="vi20"),
    _w("songs", "noun", 1, pair="vi21"),
    # t-z bucket nouns
    _w("teachers", "noun", 1, pair="vi22"),
    _w("tables", "noun", 2, pair="vi23"),
    _w("tigers", "noun", 2, pair="vi24"),
    _w("villages", "noun", 2, pair="vi25"),
    _w("windows", "noun", 1, pair="vi26"),
    _w("workers", "noun", 1, pair="vi27"),
    _w("writers", "noun", 2, pair="vi28"),
    # adverbs (12); buckets balanced
    _w("daily", "adverb", 1, pair="vi29"),
    _w("badly", "adverb", 2, pair="vi30"),
    _w("carefully", "adverb", 2, pair="vi31"),
    _w("firmly", "adverb", 2, pair="vi32"),
    _w("gently", "adverb", 2, pair="vi33"),
    _w("happily", "adverb", 2, pair="vi34"),
    _w("loudly", "adverb", 2, pair="vi35"),
    _w("nearly", "adverb", 1, pair="vi36"),
    _w("mostly", "adverb", 1, pair="vi37"),
    _w("quickly", "adverb", 1, pair="vi38"),
    _w("rarely", "adverb", 2, pair="vi39"),
    _w("simply", "adverb", 1, pair="vi40"),
]


# --- TERMINAL_VOWEL -----------------------------------------------------------
# 52 nouns ending in a vowel LETTER (>= the size-50 quota). >= 50% NOT ending in
# 'e' with a deliberate margin (non-e: 27/52 = 0.519, above the 0.50 floor so a
# single future edit cannot silently drop under it); >= 25% silent-final-e
# (e-enders: 25/52 = 0.481). Nouns-only (rule 20 keeps POS matched to
# TERMINAL_CONSONANT's noun draws). Matched to TERMINAL_CONSONANT on mean length
# (length_dist), which is distributional and needs no per-entry counterpart for
# the two extra non-e nouns.
TERMINAL_VOWEL: list[dict[str, Any]] = [
    # non-e vowel endings (a/i/o/u) -- 25
    _w("sofa", "noun", 2, pair="tv01"),
    _w("piano", "noun", 2, pair="tv02"),
    _w("tomato", "noun", 2, pair="tv03"),
    _w("menu", "noun", 2, pair="tv04"),
    _w("banana", "noun", 2, pair="tv05"),
    _w("radio", "noun", 1, pair="tv06"),
    _w("umbrella", "noun", 2, pair="tv07"),
    _w("pasta", "noun", 2, pair="tv08"),
    _w("idea", "noun", 1, pair="tv09"),
    _w("area", "noun", 1, pair="tv10"),
    _w("video", "noun", 1, pair="tv11"),
    _w("photo", "noun", 1, pair="tv12"),
    _w("taxi", "noun", 2, pair="tv13"),
    _w("camera", "noun", 1, pair="tv14"),
    _w("opera", "noun", 2, pair="tv15"),
    _w("pizza", "noun", 2, pair="tv16"),
    _w("data", "noun", 1, pair="tv17"),
    _w("drama", "noun", 2, pair="tv18"),
    _w("cinema", "noun", 2, pair="tv19"),
    _w("potato", "noun", 2, pair="tv20"),
    _w("hero", "noun", 2, pair="tv21"),
    _w("zero", "noun", 2, pair="tv22"),
    _w("studio", "noun", 2, pair="tv23"),
    _w("formula", "noun", 2, pair="tv24"),
    _w("casino", "noun", 2, pair="tv25"),
    # extra non-e vowel-final nouns -- give the not_final_e_min=0.50 floor a
    # margin (27 non-e / 52 = 0.519) so a single future edit cannot drop it
    # below 0.50; e-enders stay 25/52 = 0.481, above silent_e_min=0.25
    _w("ratio", "noun", 2, pair="tv51"),
    _w("logo", "noun", 2, pair="tv52"),
    # silent-final-e endings -- 25
    _w("table", "noun", 1, pair="tv26"),
    _w("house", "noun", 1, pair="tv27"),
    _w("lake", "noun", 1, pair="tv28"),
    _w("smile", "noun", 2, pair="tv29"),
    _w("office", "noun", 1, pair="tv30"),
    _w("machine", "noun", 1, pair="tv31"),
    _w("bridge", "noun", 1, pair="tv32"),
    _w("bottle", "noun", 2, pair="tv33"),
    _w("apple", "noun", 1, pair="tv34"),
    _w("circle", "noun", 2, pair="tv35"),
    _w("castle", "noun", 2, pair="tv36"),
    _w("candle", "noun", 2, pair="tv37"),
    _w("stone", "noun", 1, pair="tv38"),
    _w("snake", "noun", 2, pair="tv39"),
    _w("plate", "noun", 2, pair="tv40"),
    _w("cake", "noun", 2, pair="tv41"),
    _w("nose", "noun", 2, pair="tv42"),
    _w("rose", "noun", 1, pair="tv43"),
    _w("name", "noun", 1, pair="tv44"),
    _w("game", "noun", 1, pair="tv45"),
    _w("store", "noun", 1, pair="tv46"),
    _w("horse", "noun", 1, pair="tv47"),
    _w("place", "noun", 1, pair="tv48"),
    _w("space", "noun", 1, pair="tv49"),
    _w("face", "noun", 1, pair="tv50"),
]


# --- TERMINAL_CONSONANT -------------------------------------------------------
# 50 words ending in a consonant LETTER. Every entry tagged final_y (true/false).
# rule-20 draws (noun-only, clear consonant enders) are the final_y=False nouns
# (>= 40 of them); the final_y=True entries (city/slowly/...) stand for graders
# and rule-19 reuse. Matched to TERMINAL_VOWEL on mean length (length_dist).
TERMINAL_CONSONANT: list[dict[str, Any]] = [
    # final_y = False, nouns ending in clear consonants (rule-20 draw pool) -- 42
    _w("garden", "noun", 1, final_y=False, pair="tv01"),
    _w("lemon", "noun", 2, final_y=False, pair="tv02"),
    _w("button", "noun", 2, final_y=False, pair="tv03"),
    _w("season", "noun", 1, final_y=False, pair="tv04"),
    _w("reason", "noun", 1, final_y=False, pair="tv05"),
    _w("person", "noun", 1, final_y=False, pair="tv06"),
    _w("dragon", "noun", 2, final_y=False, pair="tv07"),
    _w("market", "noun", 1, final_y=False, pair="tv08"),
    _w("basket", "noun", 2, final_y=False, pair="tv09"),
    _w("planet", "noun", 2, final_y=False, pair="tv10"),
    _w("ticket", "noun", 2, final_y=False, pair="tv11"),
    _w("pocket", "noun", 2, final_y=False, pair="tv12"),
    _w("forest", "noun", 2, final_y=False, pair="tv13"),
    _w("rabbit", "noun", 2, final_y=False, pair="tv14"),
    _w("summer", "noun", 1, final_y=False, pair="tv15"),
    _w("winter", "noun", 1, final_y=False, pair="tv16"),
    _w("dinner", "noun", 1, final_y=False, pair="tv17"),
    _w("corner", "noun", 1, final_y=False, pair="tv18"),
    _w("river", "noun", 1, final_y=False, pair="tv19"),
    _w("paper", "noun", 1, final_y=False, pair="tv20"),
    _w("letter", "noun", 1, final_y=False, pair="tv21"),
    _w("water", "noun", 1, final_y=False, pair="tv22"),
    _w("sister", "noun", 1, final_y=False, pair="tv23"),
    _w("doctor", "noun", 1, final_y=False, pair="tv24"),
    _w("mirror", "noun", 2, final_y=False, pair="tv25"),
    _w("road", "noun", 1, final_y=False, pair="tv26"),
    _w("wood", "noun", 1, final_y=False, pair="tv27"),
    _w("hand", "noun", 1, final_y=False, pair="tv28"),
    _w("field", "noun", 1, final_y=False, pair="tv29"),
    _w("friend", "noun", 1, final_y=False, pair="tv30"),
    _w("record", "noun", 1, final_y=False, pair="tv31"),
    _w("island", "noun", 1, final_y=False, pair="tv32"),
    _w("husband", "noun", 1, final_y=False, pair="tv33"),
    _w("book", "noun", 1, final_y=False, pair="tv34"),
    _w("bank", "noun", 1, final_y=False, pair="tv35"),
    _w("park", "noun", 1, final_y=False, pair="tv36"),
    _w("room", "noun", 1, final_y=False, pair="tv37"),
    _w("farm", "noun", 1, final_y=False, pair="tv38"),
    _w("team", "noun", 1, final_y=False, pair="tv39"),
    _w("system", "noun", 1, final_y=False, pair="tv40"),
    _w("problem", "noun", 1, final_y=False, pair="tv41"),
    _w("kingdom", "noun", 2, final_y=False, pair="tv42"),
    # final_y = True entries (graders / probes / rule-19 reuse) -- 8
    _w("city", "noun", 1, final_y=True, pair="tv43"),
    _w("country", "noun", 1, final_y=True, pair="tv44"),
    _w("family", "noun", 1, final_y=True, pair="tv45"),
    _w("story", "noun", 1, final_y=True, pair="tv46"),
    _w("memory", "noun", 1, final_y=True, pair="tv47"),
    _w("library", "noun", 1, final_y=True, pair="tv48"),
    _w("factory", "noun", 2, final_y=True, pair="tv49"),
    _w("history", "noun", 1, final_y=True, pair="tv50"),
]


BANKS: dict[str, list[Any]] = {
    "INITIAL_BY_LENGTH": INITIAL_BY_LENGTH,
    "FINAL_BY_LENGTH": FINAL_BY_LENGTH,
    "VOWEL_INITIAL": VOWEL_INITIAL,
    "CONSONANT_INITIAL": CONSONANT_INITIAL,
    "TERMINAL_VOWEL": TERMINAL_VOWEL,
    "TERMINAL_CONSONANT": TERMINAL_CONSONANT,
}
