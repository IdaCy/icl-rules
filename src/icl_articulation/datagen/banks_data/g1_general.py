"""Bank group G1 (general-purpose template banks). The workhorse banks used by
almost every rule recipe.

Owns exactly these banks (see banks.BANK_QUOTAS for the per-bank contract):
  NOUN_CONCRETE, VERB_REGULAR, ADJ_PLAIN, ADVERB_PLACE,
  ADVERB_SENT_INITIAL, FRAME_NEUTRAL, FRAME_PROPER

(ER_NONCOMPARATIVE and ADJ_COMPARABLE are owned by g5_numhard, NOT by this
group; they are not populated here so the one-owner-per-bank invariant in
banks._raw_registry holds. This module's BANKS dict exports only the seven
banks listed above.)

Authored-vs-computed tag split (see banks.py): the author supplies only
``word`` / ``pos`` / ``frequency_tier`` (+ the optional keys a quota
references); banks.py computes length / initial / final / has_adjacent_double /
has_nonadjacent_repeat. FRAME_NEUTRAL and FRAME_PROPER are FRAME banks: each
entry is a template string with exactly one '{X}' slot marker and no
word-level tags.

DESIGN DECISIONS (the tensions the spec forces, resolved):

VERB_REGULAR (load-bearing; consumed by rules 8/9/10/12 and many others). The
bank stores the BASE form as ``word``; the generator owns inflection. Every
verb here is chosen so its four forms (base, 3sg -s, past -ed, gerund -ing) are
all DISTINCT, past != base, and NONE is morphologically ambiguous — no
zero-past verbs (put/cut/set/hit/read), no irregular pasts (teach/taught,
catch/caught), no verbs whose -ing/-ed spelling is contentious. We deliberately
EXCLUDE 'agree' and other -ee verbs (gerund 'agreeing' is a naive-generator
trap) so a simple suffix rule yields the canonical spelling for all four forms.
Mix spans -e droppers (move/moves/moved/moving), consonant+y -> -ies/-ied
(carry/study), vowel+y regulars (play/enjoy), sibilant -es (wash/watch/match),
and plain stems (walk/help). pos='verb' throughout.

ADVERB_PLACE (consumed by rules 8/9/10/12/19/24/27 etc.). These are tense-
NEUTRAL place / manner adjunct PHRASES; a phrase's ``word`` may contain spaces
(banks.alphabetic_length still counts only letters, so the entry-contract
checks pass; the phrase_word_counts quota counts whitespace tokens). The
count-equalizer (genutils) needs adjuncts of EVERY length 1/2/3 to solve word
counts exactly, so the bank carries a full spread of 1-, 2-, and 3-word
phrases. Spec line 326 ('fixed word counts (1, 2, 3 words)') and the rule-9
recipe (line 799, '0-3-word ADVERB_PLACE adjunct slots') allow ONLY 1/2/3-word
phrases — a 4-word adjunct would break the count-equalizer's 0-3 adjunct space,
so none appear and the ADVERB_PLACE quota asserts max_phrase_words=3. NO
temporal vocabulary
(yesterday/today/now/soon/often) — those would proxy tense in rule 8/9.
pos='adverb'. frequency_tier for a multiword phrase is tagged by its rarest
content word's tier (1 if every content word is top-2k, else 2); the self-check
only requires tier in {1,2}, and these phrases are built from common vocabulary.

ADVERB_SENT_INITIAL (rules 8/10/17/19): 1-word adverbs that read naturally
sentence-initially, both manner ('quickly close ...') and sentential
('usually she ...'). 'yesterday' is included in the bank but the spec notes it
is EXCLUDED from rule-8 use (temporal); that exclusion is the consumer's job,
not the bank's.

FRAME_NEUTRAL / FRAME_PROPER are substitution-frame templates, NOT word lists.
FRAME_NEUTRAL predicates are natural for ANY concrete-noun filler and contain
NO proper nouns. FRAME_PROPER slots read naturally for BOTH person names AND
cities/months/brands in the same {X} position (rule 17 salts names vs.
non-name proper nouns through one shared frame, so the frame must not presume
person-hood or place-hood).

frequency_tier policy (honest, mirrors core.py / test_banks.py): tier 1 =
wordfreq top_n_list('en', 2000); tier 2 = top 10000 but NOT top 2000; nothing
rarer than tier 2 appears in any bank. Verified during authoring with
wordfreq.
"""

from __future__ import annotations

from typing import Any


def _w(word: str, pos: str, tier: int, **extra: Any) -> dict[str, Any]:
    """A word-bank entry (author keys only; banks.py computes the rest)."""
    return {"word": word, "pos": pos, "frequency_tier": tier, **extra}


# --- NOUN_CONCRETE: 120 everyday concrete nouns (pos noun) --------------------
# Quota: size 120, pos_required {noun}. Used as fillers in almost every rule.

NOUN_CONCRETE: list[dict[str, Any]] = [
    _w("table", "noun", 1),
    _w("garden", "noun", 1),
    _w("basket", "noun", 2),
    _w("road", "noun", 1),
    _w("chair", "noun", 2),
    _w("window", "noun", 1),
    _w("door", "noun", 1),
    _w("house", "noun", 1),
    _w("room", "noun", 1),
    _w("kitchen", "noun", 2),
    _w("garage", "noun", 2),
    _w("fence", "noun", 2),
    _w("gate", "noun", 2),
    _w("wall", "noun", 1),
    _w("floor", "noun", 1),
    _w("ceiling", "noun", 2),
    _w("roof", "noun", 2),
    _w("bridge", "noun", 1),
    _w("street", "noun", 1),
    _w("market", "noun", 1),
    _w("shop", "noun", 1),
    _w("store", "noun", 1),
    _w("office", "noun", 1),
    _w("school", "noun", 1),
    _w("church", "noun", 1),
    _w("station", "noun", 1),
    _w("park", "noun", 1),
    _w("river", "noun", 1),
    _w("lake", "noun", 1),
    _w("field", "noun", 1),
    _w("forest", "noun", 2),
    _w("mountain", "noun", 1),
    _w("beach", "noun", 1),
    _w("island", "noun", 1),
    _w("village", "noun", 1),
    _w("city", "noun", 1),
    _w("town", "noun", 1),
    _w("farm", "noun", 1),
    _w("barn", "noun", 2),
    _w("shed", "noun", 2),
    _w("box", "noun", 1),
    _w("bag", "noun", 1),
    _w("book", "noun", 1),
    _w("paper", "noun", 1),
    _w("pencil", "noun", 2),
    _w("cup", "noun", 1),
    _w("plate", "noun", 2),
    _w("bowl", "noun", 2),
    _w("bottle", "noun", 2),
    _w("glass", "noun", 1),
    _w("spoon", "noun", 2),
    _w("fork", "noun", 2),
    _w("knife", "noun", 2),
    _w("clock", "noun", 2),
    _w("lamp", "noun", 2),
    _w("candle", "noun", 2),
    _w("mirror", "noun", 2),
    _w("picture", "noun", 1),
    _w("painting", "noun", 2),
    _w("photo", "noun", 1),
    _w("camera", "noun", 1),
    _w("phone", "noun", 1),
    _w("radio", "noun", 1),
    _w("letter", "noun", 1),
    _w("envelope", "noun", 2),
    _w("stamp", "noun", 2),
    _w("key", "noun", 1),
    _w("lock", "noun", 2),
    _w("chain", "noun", 2),
    _w("rope", "noun", 2),
    _w("wire", "noun", 2),
    _w("nail", "noun", 2),
    _w("hammer", "noun", 2),
    _w("ladder", "noun", 2),
    _w("bucket", "noun", 2),
    _w("brush", "noun", 2),
    _w("towel", "noun", 2),
    _w("blanket", "noun", 2),
    _w("pillow", "noun", 2),
    _w("bed", "noun", 1),
    _w("sofa", "noun", 2),
    _w("desk", "noun", 2),
    _w("shelf", "noun", 2),
    _w("drawer", "noun", 2),
    _w("cabinet", "noun", 2),
    _w("ball", "noun", 1),
    _w("toy", "noun", 2),
    _w("game", "noun", 1),
    _w("card", "noun", 1),
    _w("coin", "noun", 2),
    _w("ring", "noun", 1),
    _w("watch", "noun", 1),
    _w("hat", "noun", 2),
    _w("coat", "noun", 2),
    _w("shirt", "noun", 2),
    _w("shoe", "noun", 2),
    _w("button", "noun", 2),
    _w("pocket", "noun", 2),
    _w("belt", "noun", 2),
    _w("dress", "noun", 1),
    _w("jacket", "noun", 2),
    _w("bread", "noun", 2),
    _w("cheese", "noun", 2),
    _w("apple", "noun", 1),
    _w("potato", "noun", 2),
    _w("onion", "noun", 2),
    _w("egg", "noun", 2),
    _w("butter", "noun", 2),
    _w("sugar", "noun", 2),
    _w("flower", "noun", 2),
    _w("tree", "noun", 1),
    _w("grass", "noun", 2),
    _w("leaf", "noun", 2),
    _w("branch", "noun", 2),
    _w("seed", "noun", 2),
    _w("stone", "noun", 1),
    _w("rock", "noun", 1),
    _w("sand", "noun", 2),
    _w("brick", "noun", 2),
    _w("wheel", "noun", 2),
    _w("engine", "noun", 1),
    _w("boat", "noun", 1),
    _w("train", "noun", 1),
    _w("truck", "noun", 2),
    _w("tower", "noun", 2),
    _w("hotel", "noun", 1),
    _w("hospital", "noun", 1),
    _w("library", "noun", 1),
    _w("museum", "noun", 2),
    _w("tunnel", "noun", 2),
    _w("factory", "noun", 2),
]


# --- VERB_REGULAR: 60 regular verbs, 4 distinct forms (pos verb) --------------
# Quota: size 60, pos_required {verb}. Base form stored; see module docstring.

VERB_REGULAR: list[dict[str, Any]] = [
    _w("walk", "verb", 1),
    _w("talk", "verb", 1),
    _w("open", "verb", 1),
    _w("close", "verb", 1),
    _w("clean", "verb", 1),
    _w("paint", "verb", 2),
    _w("watch", "verb", 1),
    _w("help", "verb", 1),
    _w("start", "verb", 1),
    _w("finish", "verb", 1),
    _w("climb", "verb", 2),
    _w("pull", "verb", 1),
    _w("push", "verb", 1),
    _w("wash", "verb", 2),
    _w("cook", "verb", 2),
    _w("plant", "verb", 1),
    _w("carry", "verb", 1),
    _w("study", "verb", 1),
    _w("copy", "verb", 1),
    _w("reply", "verb", 2),
    _w("dry", "verb", 1),
    _w("enjoy", "verb", 1),
    _w("play", "verb", 1),
    _w("stay", "verb", 1),
    _w("jump", "verb", 2),
    _w("call", "verb", 1),
    _w("count", "verb", 1),
    _w("move", "verb", 1),
    _w("live", "verb", 1),
    _w("use", "verb", 1),
    _w("wave", "verb", 2),
    _w("smile", "verb", 2),
    _w("arrive", "verb", 2),
    _w("order", "verb", 1),
    _w("offer", "verb", 1),
    _w("answer", "verb", 1),
    _w("enter", "verb", 1),
    _w("visit", "verb", 1),
    _w("repeat", "verb", 2),
    _w("collect", "verb", 2),
    _w("protect", "verb", 1),
    _w("expect", "verb", 1),
    _w("accept", "verb", 1),
    _w("prepare", "verb", 2),
    _w("share", "verb", 1),
    _w("prove", "verb", 1),
    _w("remove", "verb", 2),
    _w("solve", "verb", 2),
    _w("serve", "verb", 1),
    _w("believe", "verb", 1),
    _w("receive", "verb", 1),
    _w("imagine", "verb", 1),
    _w("mention", "verb", 1),
    _w("return", "verb", 1),
    _w("explain", "verb", 1),
    _w("decide", "verb", 1),
    _w("provide", "verb", 1),
    _w("reduce", "verb", 1),
    _w("approach", "verb", 1),
    _w("search", "verb", 1),
]


# --- ADJ_PLAIN: 80 ordinary adjectives, no comparatives (pos adjective) -------
# Quota: size 80, pos_required {adjective}. No -er / 'more' comparative forms.

ADJ_PLAIN: list[dict[str, Any]] = [
    _w("small", "adjective", 1),
    _w("large", "adjective", 1),
    _w("big", "adjective", 1),
    _w("little", "adjective", 1),
    _w("huge", "adjective", 1),
    _w("tiny", "adjective", 2),
    _w("wide", "adjective", 1),
    _w("narrow", "adjective", 2),
    _w("deep", "adjective", 1),
    _w("flat", "adjective", 1),
    _w("round", "adjective", 1),
    _w("square", "adjective", 1),
    _w("heavy", "adjective", 1),
    _w("light", "adjective", 1),
    _w("hard", "adjective", 1),
    _w("soft", "adjective", 2),
    _w("smooth", "adjective", 2),
    _w("rough", "adjective", 2),
    _w("sharp", "adjective", 2),
    _w("clean", "adjective", 1),
    _w("dirty", "adjective", 2),
    _w("wet", "adjective", 2),
    _w("dry", "adjective", 1),
    _w("warm", "adjective", 2),
    _w("cool", "adjective", 1),
    _w("hot", "adjective", 1),
    _w("cold", "adjective", 1),
    _w("bright", "adjective", 2),
    _w("dark", "adjective", 1),
    _w("loud", "adjective", 2),
    _w("quiet", "adjective", 2),
    _w("fast", "adjective", 1),
    _w("slow", "adjective", 1),
    _w("strong", "adjective", 1),
    _w("weak", "adjective", 2),
    _w("full", "adjective", 1),
    _w("empty", "adjective", 2),
    _w("open", "adjective", 1),
    _w("new", "adjective", 1),
    _w("old", "adjective", 1),
    _w("young", "adjective", 1),
    _w("modern", "adjective", 1),
    _w("ancient", "adjective", 2),
    _w("fresh", "adjective", 1),
    _w("plain", "adjective", 2),
    _w("simple", "adjective", 1),
    _w("complex", "adjective", 1),
    _w("easy", "adjective", 1),
    _w("useful", "adjective", 1),
    _w("common", "adjective", 1),
    _w("rare", "adjective", 1),
    _w("quick", "adjective", 1),
    _w("calm", "adjective", 2),
    _w("busy", "adjective", 1),
    _w("lazy", "adjective", 2),
    _w("active", "adjective", 1),
    _w("careful", "adjective", 2),
    _w("honest", "adjective", 1),
    _w("polite", "adjective", 2),
    _w("brave", "adjective", 2),
    _w("gentle", "adjective", 2),
    _w("serious", "adjective", 1),
    _w("nervous", "adjective", 2),
    _w("curious", "adjective", 2),
    _w("famous", "adjective", 1),
    _w("popular", "adjective", 1),
    _w("public", "adjective", 1),
    _w("private", "adjective", 1),
    _w("local", "adjective", 1),
    _w("national", "adjective", 1),
    _w("foreign", "adjective", 1),
    _w("formal", "adjective", 2),
    _w("perfect", "adjective", 1),
    _w("broken", "adjective", 1),
    _w("correct", "adjective", 1),
    _w("sudden", "adjective", 2),
    _w("final", "adjective", 1),
    _w("normal", "adjective", 1),
    _w("strange", "adjective", 2),
    _w("special", "adjective", 1),
]


# --- ADVERB_SENT_INITIAL: 20 one-word sentence-initial adverbs (pos adverb) ---
# Quota: size 20, pos_required {adverb}. Mix of manner + sentential adverbs.

ADVERB_SENT_INITIAL: list[dict[str, Any]] = [
    _w("quickly", "adverb", 1),
    _w("slowly", "adverb", 2),
    _w("carefully", "adverb", 2),
    _w("quietly", "adverb", 2),
    _w("loudly", "adverb", 2),
    _w("suddenly", "adverb", 2),
    _w("usually", "adverb", 1),
    _w("often", "adverb", 1),
    _w("rarely", "adverb", 2),
    _w("sometimes", "adverb", 1),
    _w("always", "adverb", 1),
    _w("apparently", "adverb", 1),
    _w("clearly", "adverb", 1),
    _w("obviously", "adverb", 1),
    _w("certainly", "adverb", 1),
    _w("probably", "adverb", 1),
    _w("luckily", "adverb", 2),
    _w("finally", "adverb", 1),
    _w("recently", "adverb", 1),
    _w("yesterday", "adverb", 1),  # in bank; EXCLUDED from rule-8 use (temporal)
]


# --- ADVERB_PLACE: 30 tense-neutral place/manner adjunct PHRASES (pos adverb) -
# Quota: size 30, phrase_word_counts must cover {1, 2, 3}. ``word`` may contain
# spaces (a phrase). NO temporal vocabulary. Full 1/2/3-word spread so the
# count-equalizer can solve word counts exactly. frequency_tier tagged by the
# phrase's rarest content word (all common -> mostly tier 1/2).

ADVERB_PLACE: list[dict[str, Any]] = [
    # 1-word place/manner adverbs
    _w("downtown", "adverb", 2),
    _w("outside", "adverb", 1),
    _w("upstairs", "adverb", 2),
    _w("abroad", "adverb", 2),
    _w("nearby", "adverb", 2),
    _w("everywhere", "adverb", 2),
    # 2-word phrases
    _w("at home", "adverb", 1),
    _w("at work", "adverb", 1),
    _w("at school", "adverb", 1),
    _w("by hand", "adverb", 1),
    _w("in silence", "adverb", 2),
    _w("with care", "adverb", 1),
    # 3-word phrases
    _w("in the kitchen", "adverb", 1),
    _w("in the garden", "adverb", 1),
    _w("on the table", "adverb", 1),
    _w("near the station", "adverb", 1),
    _w("by the door", "adverb", 1),
    _w("at the market", "adverb", 1),
    _w("along the road", "adverb", 1),
    _w("across the street", "adverb", 1),
    _w("behind the house", "adverb", 1),
    _w("under the bridge", "adverb", 1),
    _w("beside the lake", "adverb", 2),
    _w("around the corner", "adverb", 1),
    # more 1-word place/manner adverbs (replace the deleted 4-word phrases so
    # every entry stays in {1,2,3} words per spec line 326 / rule-9 line 799)
    _w("inside", "adverb", 1),
    _w("below", "adverb", 1),
    _w("overhead", "adverb", 2),
    # more 2-word phrases
    _w("on foot", "adverb", 1),
    _w("in town", "adverb", 1),
    _w("at once", "adverb", 1),
]


# --- FRAME_NEUTRAL: 30 substitution frames, one {X} slot, NO proper nouns -----
# Quota: size 30, frame=True (each must contain exactly one '{X}'). Predicate
# natural for ANY concrete-noun filler; neutral predicates only (no animal-
# typical / chromatic verbs that would leak rules 13/14). No temporal-adverb
# ban here ('yesterday' is fine; only rule 8 bans temporal adverbs).

FRAME_NEUTRAL: list[str] = [
    "The {X} was in the garden yesterday",
    "The {X} was next to the old fence",
    "They found a {X} behind the shed",
    "The {X} appeared in the photo",
    "Everyone noticed the {X} near the entrance",
    "She placed the {X} on the kitchen table",
    "We left the {X} beside the front door",
    "The {X} stood quietly near the window",
    "He carried the {X} across the room",
    "The {X} sat on the wooden shelf",
    "Someone moved the {X} into the corner",
    "The {X} was hidden under the stairs",
    "They photographed the {X} at the market",
    "The {X} rested against the garden wall",
    "A small {X} was lying on the floor",
    "The {X} remained inside the locked cabinet",
    "Children gathered around the {X} after lunch",
    "The {X} was visible from the street",
    "We discovered the {X} in the attic",
    "The {X} appeared near the edge of the field",
    "She pointed at the {X} by the gate",
    "The {X} stayed in the back of the truck",
    "Workers placed the {X} beside the road",
    "The {X} was wrapped in an old blanket",
    "Everyone walked past the {X} without stopping",
    "The {X} sat between the two chairs",
    "He noticed the {X} on the office desk",
    "The {X} was kept inside the garage",
    "They left the {X} near the river bank",
    "A large {X} blocked the narrow path",
]


# --- FRAME_PROPER: 20 frames, one {X} slot, natural for names AND places ------
# Quota: size 20, frame=True. The {X} slot must read naturally for BOTH person
# names (Anna, Lucas) AND non-name proper nouns (cities/months/brands) — rule
# 17 salts the two through one shared frame, so no frame may presume person-hood
# or place-hood. Each tested mentally against a name, a city, a month, a brand.

FRAME_PROPER: list[str] = [
    "The letter from {X} arrived this morning",
    "Everyone talked about {X} during lunch",
    "{X} was mentioned twice in the report",
    "The photo of {X} was on the wall",
    "Nobody had heard about {X} before today",
    "The article described {X} in great detail",
    "She wrote a long note about {X}",
    "The meeting started with a question about {X}",
    "He kept a small file about {X}",
    "The book devoted a chapter to {X}",
    "A short message about {X} reached the office",
    "The map had a mark next to {X}",
    "They argued about {X} for an hour",
    "The headline focused entirely on {X}",
    "Our class studied {X} last term",
    "The record mentioned {X} only once",
    "A rumor about {X} spread through the building",
    "The guide pointed out {X} from the window",
    "The committee voted on {X} after lunch",
    "Her diary referred to {X} several times",
]


BANKS: dict[str, list[Any]] = {
    "NOUN_CONCRETE": NOUN_CONCRETE,
    "VERB_REGULAR": VERB_REGULAR,
    "ADJ_PLAIN": ADJ_PLAIN,
    "ADVERB_SENT_INITIAL": ADVERB_SENT_INITIAL,
    "ADVERB_PLACE": ADVERB_PLACE,
    "FRAME_NEUTRAL": FRAME_NEUTRAL,
    "FRAME_PROPER": FRAME_PROPER,
}
