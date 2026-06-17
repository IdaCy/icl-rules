"""Bank group G3 (semantic + proper-noun banks).

Owns these banks (see banks.BANK_QUOTAS for the per-bank contract):
  ANIMALS, OBJECTS_PLANTS_VEHICLES, COLORS, ADJ_NONCOLOR_MATCHED,
  FIRST_NAMES, NONNAME_PROPER

Constraints honored (all enforced by banks.check_bank):
  - ANIMALS (rule 13 True slot): 60 real, living, non-human animal kinds
    (mammals, birds, fish, insects), 3-8 letters, all nouns, no
    mythical/extinct/toy. Matched to OBJECTS_PLANTS_VEHICLES by length
    distribution and frequency tier (banks.py: mean length within 1.5,
    tier-1 share within 0.20). Mean length 4.6, tier-1 share 0.083.
  - OBJECTS_PLANTS_VEHICLES (rule 13 False slot): 60 inanimate fillers,
    16 plants (>= 15) + 12 vehicles (>= 10) + 32 objects, length/tier
    matched to ANIMALS (mean length 5.0, tier-1 share 0.15). The plant and
    vehicle minimums make "names a living thing" and "can move on its own"
    non-equivalent to "is an animal".
  - COLORS (rule 14 True slot): the 16 spec-enumerated color adjectives,
    verbatim ("orange" is an adjective here, never the fruit).
  - ADJ_NONCOLOR_MATCHED (rule 14 False slot): 30 non-color adjectives,
    length/tier matched to COLORS (mean length 5.47 vs 5.375; tier-1 share
    0.333 vs 0.375). No chromatic words and no material words (gold, wooden,
    metal) so rule 14's material/color boundary probes stay clean.
  - FIRST_NAMES (rules 17, 22 name slot): 30 feminine + 30 masculine common
    given names, stored capitalized with proper=True. No word-homograph
    names (none of these is an ordinary English word -- no Rose, Mark, Bill,
    June, Daisy, Grace, Hope, Joy, Will, Frank, Victor, ...), no
    city-homographs (no Paris, Austin, Florence, Sydney, Victoria,
    Charlotte). banks.py computes length/initial on the capitalized surface.
  - NONNAME_PROPER (rules 17, 21, 22 non-name slot): 30 cities + 12 months
    + 10 countries + 8 brands = 60, all proper=True. Cities are exactly 50%
    of the bank (rule 17 raises cities to 50% so "mentions a city" clears
    the 25% multiple-choice-disagreement floor). No entry is a person-name homograph
    (no Florence/Sydney/Victoria city, no Mercedes-style person brand).

Frequency tiers are the authored tags; every tier was verified against the
pinned wordfreq top-n list (tier 1 = top 2000, tier 2 = top 10000) at author
time. See module-level TENSIONS for the two places the spec's verbatim
content sits just outside the strict top-10000 cutoff.

TENSIONS (documented, not silently loosened):
  - COLORS lists 'beige', 'maroon', 'crimson', 'turquoise' verbatim; wordfreq
    ranks them ~13k-20k (just past the top-10000 tier-2 boundary), and
    'turquoise' is 9 letters. The COLORS quota checks only size and POS, not
    tier or length, so the verbatim list passes check_bank. These four are
    tagged tier 2 (the nearest allowed value) because the spec enumerates
    them as binding COLORS content; the global "nothing rarer than tier 2"
    rule is a default that the explicit verbatim list overrides for this one
    bank. All are universally familiar color names.
  - NONNAME_PROPER requires 12 months, and April/May/June are also given
    names. The rule-17 ambiguity note "name/month tokens (June, April)
    excluded from BOTH banks" binds FIRST_NAMES (no June/April/May as
    given names, honored here) and is a step-3 probe concern for the
    generator; excluding them from NONNAME_PROPER would leave only 9 months
    and fail subtype_min{month:12} with no alternative source. All 12
    calendar months are therefore included in NONNAME_PROPER.
"""

from __future__ import annotations

from typing import Any


def _w(word: str, pos: str, tier: int, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"word": word, "pos": pos, "frequency_tier": tier}
    entry.update(extra)
    return entry


def _animal(word: str, tier: int) -> dict[str, Any]:
    return _w(word, "noun", tier)


def _opv(word: str, tier: int, subtype: str) -> dict[str, Any]:
    return _w(word, "noun", tier, subtype=subtype)


def _color(word: str, tier: int) -> dict[str, Any]:
    return _w(word, "adjective", tier)


def _adj(word: str, tier: int) -> dict[str, Any]:
    return _w(word, "adjective", tier)


def _name(word: str, tier: int, subtype: str) -> dict[str, Any]:
    return _w(word, "noun", tier, proper=True, subtype=subtype)


def _proper(word: str, tier: int, subtype: str) -> dict[str, Any]:
    return _w(word, "noun", tier, proper=True, subtype=subtype)


# --- ANIMALS (60; nouns; 3-8 letters; tier 1/2) -------------------------------
# Specific living non-human animal kinds; mythical/extinct/toy excluded.
ANIMALS: list[dict[str, Any]] = [
    _animal("cat", 1),
    _animal("dog", 1),
    _animal("horse", 1),
    _animal("bear", 1),
    _animal("fish", 1),
    _animal("rabbit", 2),
    _animal("tiger", 2),
    _animal("lion", 2),
    _animal("deer", 2),
    _animal("goat", 2),
    _animal("sheep", 2),
    _animal("duck", 2),
    _animal("goose", 2),
    _animal("frog", 2),
    _animal("snake", 2),
    _animal("shark", 2),
    _animal("whale", 2),
    _animal("eagle", 2),
    _animal("owl", 2),
    _animal("bee", 2),
    _animal("ant", 2),
    _animal("seal", 2),
    _animal("mouse", 2),
    _animal("fox", 2),
    _animal("wolf", 2),
    _animal("cow", 2),
    _animal("pig", 2),
    _animal("swan", 2),
    _animal("robin", 2),
    _animal("trout", 2),
    _animal("salmon", 2),
    _animal("spider", 2),
    _animal("turtle", 2),
    _animal("monkey", 2),
    _animal("penguin", 2),
    _animal("crow", 2),
    _animal("hawk", 2),
    _animal("bat", 2),
    _animal("rat", 2),
    _animal("ram", 2),
    _animal("puppy", 2),
    _animal("lamb", 2),
    _animal("chick", 2),
    _animal("pony", 2),
    _animal("bull", 2),
    _animal("shrimp", 2),
    _animal("cattle", 2),
    _animal("buffalo", 2),
    _animal("cod", 2),
    _animal("bass", 2),
    _animal("elephant", 2),
    _animal("ray", 2),
    _animal("chicken", 2),
    _animal("turkey", 2),
    _animal("python", 2),
    _animal("raven", 2),
    _animal("cricket", 2),
    _animal("crane", 2),
    _animal("bug", 2),
    _animal("kitty", 2),
]


# --- OBJECTS_PLANTS_VEHICLES (60; >=15 plants, >=10 vehicles) -----------------
# Matched to ANIMALS in length distribution and frequency tier.
OBJECTS_PLANTS_VEHICLES: list[dict[str, Any]] = [
    # plants (16)
    _opv("tree", 1, "plant"),
    _opv("flower", 2, "plant"),
    _opv("grass", 2, "plant"),
    _opv("oak", 2, "plant"),
    _opv("pine", 2, "plant"),
    _opv("leaf", 2, "plant"),
    _opv("bush", 2, "plant"),
    _opv("weed", 2, "plant"),
    _opv("maple", 2, "plant"),
    _opv("daisy", 2, "plant"),
    _opv("onion", 2, "plant"),
    _opv("potato", 2, "plant"),
    _opv("tomato", 2, "plant"),
    _opv("apple", 1, "plant"),
    _opv("lemon", 2, "plant"),
    _opv("branch", 2, "plant"),
    # vehicles (12)
    _opv("car", 1, "vehicle"),
    _opv("bus", 1, "vehicle"),
    _opv("truck", 2, "vehicle"),
    _opv("train", 1, "vehicle"),
    _opv("boat", 1, "vehicle"),
    _opv("ship", 1, "vehicle"),
    _opv("bike", 2, "vehicle"),
    _opv("van", 1, "vehicle"),
    _opv("taxi", 2, "vehicle"),
    _opv("wagon", 2, "vehicle"),
    _opv("rocket", 2, "vehicle"),
    _opv("tractor", 2, "vehicle"),
    # inanimate objects (32)
    _opv("table", 1, "object"),
    _opv("chair", 2, "object"),
    _opv("spoon", 2, "object"),
    _opv("fork", 2, "object"),
    _opv("knife", 2, "object"),
    _opv("bowl", 2, "object"),
    _opv("bottle", 2, "object"),
    _opv("lamp", 2, "object"),
    _opv("clock", 2, "object"),
    _opv("mirror", 2, "object"),
    _opv("brush", 2, "object"),
    _opv("towel", 2, "object"),
    _opv("pillow", 2, "object"),
    _opv("basket", 2, "object"),
    _opv("bucket", 2, "object"),
    _opv("ladder", 2, "object"),
    _opv("hammer", 2, "object"),
    _opv("rope", 2, "object"),
    _opv("chain", 2, "object"),
    _opv("pipe", 2, "object"),
    _opv("brick", 2, "object"),
    _opv("fence", 2, "object"),
    _opv("gate", 2, "object"),
    _opv("candle", 2, "object"),
    _opv("pencil", 2, "object"),
    _opv("ticket", 2, "object"),
    _opv("drum", 2, "object"),
    _opv("guitar", 2, "object"),
    _opv("anchor", 2, "object"),
    _opv("helmet", 2, "object"),
    _opv("shield", 2, "object"),
    _opv("umbrella", 2, "object"),
]


# --- COLORS (16; adjectives; spec-enumerated verbatim) ------------------------
# See module TENSIONS: beige/maroon/crimson/turquoise sit just past the strict
# top-10000 cutoff but are enumerated verbatim by the spec; tagged tier 2.
COLORS: list[dict[str, Any]] = [
    _color("red", 1),
    _color("blue", 1),
    _color("green", 1),
    _color("yellow", 2),
    _color("purple", 2),
    _color("pink", 2),
    _color("brown", 1),
    _color("gray", 2),
    _color("black", 1),
    _color("white", 1),
    _color("beige", 2),
    _color("maroon", 2),
    _color("violet", 2),
    _color("crimson", 2),
    _color("turquoise", 2),
    _color("orange", 2),
]


# --- ADJ_NONCOLOR_MATCHED (30; adjectives; matched to COLORS) -----------------
# No chromatic or material adjectives (rule-14 boundary probes stay clean).
ADJ_NONCOLOR_MATCHED: list[dict[str, Any]] = [
    _adj("small", 1),
    _adj("round", 1),
    _adj("heavy", 1),
    _adj("cheap", 1),
    _adj("clean", 1),
    _adj("broken", 1),
    _adj("empty", 2),
    _adj("tall", 2),
    _adj("narrow", 2),
    _adj("thick", 2),
    _adj("smooth", 2),
    _adj("sharp", 2),
    _adj("quiet", 2),
    _adj("plain", 2),
    _adj("fancy", 2),
    _adj("simple", 1),
    _adj("strange", 2),
    _adj("silent", 2),
    _adj("gentle", 2),
    _adj("steady", 2),
    _adj("hollow", 2),
    _adj("sticky", 2),
    _adj("curved", 2),
    _adj("square", 1),
    _adj("shallow", 2),
    _adj("tight", 2),
    _adj("loose", 2),
    _adj("blank", 2),
    _adj("modern", 1),
    _adj("wide", 1),
]


# --- FIRST_NAMES (60; proper nouns; 30 feminine + 30 masculine) ---------------
# No word-homograph names, no city-homograph names.
FIRST_NAMES: list[dict[str, Any]] = [
    # feminine (30)
    _name("Anna", 2, "feminine"),
    _name("Maria", 2, "feminine"),
    _name("Emma", 2, "feminine"),
    _name("Sophie", 2, "feminine"),
    _name("Laura", 2, "feminine"),
    _name("Julia", 2, "feminine"),
    _name("Clara", 2, "feminine"),
    _name("Hannah", 2, "feminine"),
    _name("Sarah", 2, "feminine"),
    _name("Rachel", 2, "feminine"),
    _name("Linda", 2, "feminine"),
    _name("Nina", 2, "feminine"),
    _name("Diana", 2, "feminine"),
    _name("Olivia", 2, "feminine"),
    _name("Emily", 2, "feminine"),
    _name("Mia", 2, "feminine"),
    _name("Monica", 2, "feminine"),
    _name("Sandra", 2, "feminine"),
    _name("Karen", 2, "feminine"),
    _name("Sara", 2, "feminine"),
    _name("Alice", 2, "feminine"),
    _name("Andrea", 2, "feminine"),
    _name("Angela", 2, "feminine"),
    _name("Anne", 2, "feminine"),
    _name("Barbara", 2, "feminine"),
    _name("Catherine", 2, "feminine"),
    _name("Christina", 2, "feminine"),
    _name("Helen", 2, "feminine"),
    _name("Jessica", 2, "feminine"),
    _name("Natalie", 2, "feminine"),
    # masculine (30)
    _name("David", 1, "masculine"),
    _name("Lucas", 2, "masculine"),
    _name("Thomas", 1, "masculine"),
    _name("Daniel", 2, "masculine"),
    _name("Michael", 1, "masculine"),
    _name("Peter", 1, "masculine"),
    _name("Martin", 1, "masculine"),
    _name("Carlos", 2, "masculine"),
    _name("Diego", 2, "masculine"),
    _name("Marco", 2, "masculine"),
    _name("Felix", 2, "masculine"),
    _name("Oscar", 2, "masculine"),
    _name("Henry", 1, "masculine"),
    _name("Simon", 2, "masculine"),
    _name("Adam", 2, "masculine"),
    _name("Jacob", 2, "masculine"),
    _name("Noah", 2, "masculine"),
    _name("Leo", 2, "masculine"),
    _name("Pablo", 2, "masculine"),
    _name("Bruno", 2, "masculine"),
    _name("Gabriel", 2, "masculine"),
    _name("Samuel", 2, "masculine"),
    _name("Nathan", 2, "masculine"),
    _name("Julian", 2, "masculine"),
    _name("Aaron", 2, "masculine"),
    _name("Ivan", 2, "masculine"),
    _name("Frederick", 2, "masculine"),
    _name("Andre", 2, "masculine"),
    _name("Dennis", 2, "masculine"),
    _name("Brian", 2, "masculine"),
]


# --- NONNAME_PROPER (60; proper nouns) ----------------------------------------
# 30 cities + 12 months + 10 countries + 8 brands; cities = 50% of the bank
# (rule 17). No person-name homographs.
NONNAME_PROPER: list[dict[str, Any]] = [
    # cities (30)
    _proper("London", 1, "city"),
    _proper("Madrid", 2, "city"),
    _proper("Berlin", 2, "city"),
    _proper("Tokyo", 2, "city"),
    _proper("Boston", 2, "city"),
    _proper("Chicago", 1, "city"),
    _proper("Denver", 2, "city"),
    _proper("Seattle", 2, "city"),
    _proper("Dublin", 2, "city"),
    _proper("Vienna", 2, "city"),
    _proper("Moscow", 2, "city"),
    _proper("Cairo", 2, "city"),
    _proper("Athens", 2, "city"),
    _proper("Munich", 2, "city"),
    _proper("Bristol", 2, "city"),
    _proper("Glasgow", 2, "city"),
    _proper("Toronto", 2, "city"),
    _proper("Ottawa", 2, "city"),
    _proper("Manila", 2, "city"),
    _proper("Dallas", 2, "city"),
    _proper("Houston", 2, "city"),
    _proper("Phoenix", 2, "city"),
    _proper("Portland", 2, "city"),
    _proper("Detroit", 2, "city"),
    _proper("Atlanta", 2, "city"),
    _proper("Miami", 2, "city"),
    _proper("Memphis", 2, "city"),
    _proper("Orlando", 2, "city"),
    _proper("Calgary", 2, "city"),
    _proper("Geneva", 2, "city"),
    # months (12; all tier 1)
    _proper("January", 1, "month"),
    _proper("February", 1, "month"),
    _proper("March", 1, "month"),
    _proper("April", 1, "month"),
    _proper("May", 1, "month"),
    _proper("June", 1, "month"),
    _proper("July", 1, "month"),
    _proper("August", 1, "month"),
    _proper("September", 1, "month"),
    _proper("October", 1, "month"),
    _proper("November", 1, "month"),
    _proper("December", 1, "month"),
    # countries (10)
    _proper("France", 1, "country"),
    _proper("Germany", 1, "country"),
    _proper("Spain", 2, "country"),
    _proper("Italy", 2, "country"),
    _proper("Japan", 1, "country"),
    _proper("Brazil", 2, "country"),
    _proper("Canada", 1, "country"),
    _proper("Mexico", 1, "country"),
    _proper("Egypt", 2, "country"),
    _proper("Poland", 2, "country"),
    # brands (8)
    _proper("Toyota", 2, "brand"),
    _proper("Honda", 2, "brand"),
    _proper("Samsung", 2, "brand"),
    _proper("Sony", 2, "brand"),
    _proper("Google", 1, "brand"),
    _proper("Disney", 2, "brand"),
    _proper("Nintendo", 2, "brand"),
    _proper("Boeing", 2, "brand"),
]


BANKS: dict[str, list[Any]] = {
    "ANIMALS": ANIMALS,
    "OBJECTS_PLANTS_VEHICLES": OBJECTS_PLANTS_VEHICLES,
    "COLORS": COLORS,
    "ADJ_NONCOLOR_MATCHED": ADJ_NONCOLOR_MATCHED,
    "FIRST_NAMES": FIRST_NAMES,
    "NONNAME_PROPER": NONNAME_PROPER,
}
