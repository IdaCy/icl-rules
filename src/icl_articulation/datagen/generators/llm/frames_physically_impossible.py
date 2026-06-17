"""Authored minimal-pair frame bank for rule 18 (physically_impossible).

This is the BULK of the rule-18 build: a systematically authored bank of >= 440
minimal-pair frames. Each frame carries a single ``{S}`` slot and is shared
VERBATIM across the two classes. Exactly one word changes between the True
(physically impossible) and False (ordinary, plausible) variant, so the two
surface strings of a base differ in EXACTLY one slot word.

Why a CONSTRUCTED bank rather than 440 hand-typed pairs
-------------------------------------------------------
The hardest rule-18 guard is CROSS-CLASS WORD REUSE >= 80%: >= 80% of the
distinct slot words must appear in BOTH classes in an ordinary role, so no slot
word is class-predictive ("no lexical proxy"). Hand-typing pairs makes that
fraction drift low (an impossible filler like 'boulder' tends to never appear as
a plausible filler). So instead the bank is built from a few SHARED WORD POOLS,
filled into authored frame TEMPLATES, with the construction guaranteeing reuse:
every word that is ever an impossible filler is ALSO used as a plausible filler
somewhere, and vice versa. The templates are still authored by hand (they carry
the linguistic content of the four impossibility types); the pools are what make
the reuse guard hold by construction.

Construction invariants (rule-specs rule 18 recipe + distribution_guards),
asserted LOUDLY by ``audit`` at import:

  * SINGLE-WORD fillers -> word_count is identical across a base's two variants
    BY CONSTRUCTION (confound |mean_wc(T)-mean_wc(F)| is exactly 0; every
    frame-determined battery predicate sits at 0.5; the slot animacy/identity is
    the only signal).
  * LENGTH-MATCHED +/- 2 alphabetic chars on the two fillers (recipe).
  * IMPOSSIBILITY TYPE MIX each in [20%, 35%] of frames:
      (a) INANIMATE_AGENT  -- inanimate agent of a biological action
      (b) SCALE_STRENGTH   -- scale/strength violation, HUMAN agent
      (c) CONTAINER_SIZE   -- container/size violation
      (d) MATERIAL_STATE   -- material/state violation
  * CROSS-CLASS WORD REUSE >= 80% (the key guard).
  * MUNDANE VOCABULARY ONLY; a frozen fantasy ban-list is asserted.
  * STYLE: 4-14 words, ASCII, no comma/terminal punctuation, no banned 'I' after
    the generator sentence-cases each frame.

The labels here are the AUTHOR's INTENT only. rule 18 is validator-derived, so
the two-pass LLM gate is what promotes an intended label to a stored label
(slots_meta['validated_agreement']); this bank only needs to be a high-yield set
of candidate minimal pairs covering the four types with no lexical proxy.
"""

from __future__ import annotations

from dataclasses import dataclass


# impossibility-type tags (rule-18 recipe type mix (a)-(d))
TYPE_INANIMATE_AGENT = "inanimate_agent"      # (a)
TYPE_SCALE_STRENGTH = "scale_strength"        # (b)
TYPE_CONTAINER_SIZE = "container_size"        # (c)
TYPE_MATERIAL_STATE = "material_state"        # (d)

ALL_TYPES = (
    TYPE_INANIMATE_AGENT,
    TYPE_SCALE_STRENGTH,
    TYPE_CONTAINER_SIZE,
    TYPE_MATERIAL_STATE,
)

# each impossibility type must hold this share of the bank (recipe: each 20-35%).
TYPE_SHARE_MIN = 0.20
TYPE_SHARE_MAX = 0.35

# cross-class word reuse floor (recipe: ">= 80% of slot words ... opposite class").
CROSS_CLASS_REUSE_MIN = 0.80

# length-match tolerance on the two fillers' alphabetic lengths (recipe +/- 2).
LENGTH_MATCH_TOL = 2

# minimum authored frames (recipe: ">= 440 written frames" for ~340 surviving).
MIN_FRAMES = 440

# fantasy / non-mundane lexicon that may NEVER appear in a frame or filler.
BANNED_LEXICON = frozenset(
    """magic magical wizard witch dragon ghost goblin fairy unicorn spell potion
    sorcerer sorcery enchanted enchantment vampire werewolf zombie demon angel
    spirit phantom monster troll elf elves dwarf giant mermaid phoenix griffin
    curse cursed haunted supernatural mythical fantasy fairytale""".split()
)


@dataclass(frozen=True)
class Frame:
    """One authored minimal-pair frame.

    ``template`` carries exactly one ``{S}`` slot. ``plausible`` fills it for the
    False (ordinary) variant; ``impossible`` fills it for the True (physically
    impossible) variant. ``itype`` is one of the four impossibility types. Both
    fillers are single words, length-matched +/- 2."""

    template: str
    plausible: str
    impossible: str
    itype: str


# =============================================================================
# SHARED WORD POOLS
# Each word below is used as BOTH a plausible filler and an impossible filler
# somewhere in the bank, which is what makes cross-class reuse >= 80% hold by
# construction. The pools are grouped only for authoring clarity; the audit does
# not depend on the grouping.
# =============================================================================

# TINY objects -- genuinely fit a pocket / matchbox / wallet / teacup. Used as:
#   - the PLAUSIBLE inserted object in type (c) container frames (they FIT, so the
#     plausible variant validates as possible), and
#   - the IMPOSSIBLE agent in type (a) biological frames ('the pebble ate ...').
# So each appears in BOTH classes. (The earlier yield run showed the plausible
# container variant FAILS when the object is too big for the container -- 'slipped
# the shelf into his pocket' is itself impossible -- so the plausible container
# object MUST be pocket-sized. Hence this dedicated tiny pool.)
TINY_OBJECTS = [
    "coin", "pebble", "button", "ring", "bead", "stamp", "key", "marble",
    "screw", "nail", "cork", "eraser", "pin", "seed", "crumb", "chip", "clip",
    "token", "dime", "penny", "thimble", "acorn", "pellet", "stud", "bolt",
    "washer", "berry", "olive", "raisin", "pebble",
]

# CARRYABLE objects -- a person can lift/carry one by hand. Used as:
#   - the PLAUSIBLE carried object in type (b) scale frames ('carried the kettle
#     home' -> possible), and
#   - the IMPOSSIBLE agent in type (a) biological frames ('the kettle ate ...').
# So each appears in BOTH classes.
CARRY_OBJECTS = [
    "kettle", "bucket", "lamp", "bottle", "broom", "pillow", "satchel",
    "ladle", "mirror", "hammer", "teapot", "candle", "saucer", "spoon",
    "shovel", "trowel", "wrench", "lantern", "basket", "jacket", "blanket",
    "cushion", "skillet", "vase", "rock", "brick", "stone", "plank", "drum",
    "spade",
]

# HUGE things -- a single person canNOT carry one, and one canNOT fit into a
# pocket-sized container. Used as:
#   - the IMPOSSIBLE object in type (b)/(c) frames ('carried the cottage home',
#     'slid the cottage into the matchbox' -> impossible), and
#   - the PLAUSIBLE object of a crane-LIFT frame in the reuse-closure pass ('the
#     crane lifted the cottage' -> possible).
# So each appears in BOTH classes. These are chosen to be UNAMBIGUOUSLY un-
# carryable (the earlier yield run showed borderline items like 'piano' get judged
# 'possible' as exaggeration, so the pool leans to clearly-immovable structures).
HUGE_THINGS = [
    "cottage", "bridge", "tractor", "building", "warehouse", "locomotive",
    "wardrobe", "staircase", "harvester", "chimney", "dumpster", "furnace",
    "bookcase", "mountain", "barn", "cathedral", "hillside", "rooftop",
    "monument", "scaffold", "bungalow", "windmill", "boulder", "anchor",
]

# Animate agents (people / animals). Used as:
#   - the PLAUSIBLE agent in type (a) frames ('the boy ate breakfast'),
#   - the PLAUSIBLE subject in ordinary type (d) frames ('the boy stood by the
#     fence'). Animate words are NOT used as impossible fillers (an animate agent
#     of a biological action is plausible), so they would be plausible-only.
#   To keep them inside the reuse pool we ALSO use a subset of them as impossible
#   fillers in type (d) state frames where an animal is in an impossible state.
ANIMATES = [
    "boy", "girl", "baby", "woman", "man", "nurse", "driver", "child",
    "patient", "hiker", "runner", "guest", "farmer", "teacher", "sailor",
    "dog", "cat", "horse", "rabbit", "duck", "bear", "swimmer", "cook",
    "actor", "singer", "diner", "waiter", "uncle", "infant", "player",
]


def _f(template: str, plausible: str, impossible: str, itype: str) -> Frame:
    return Frame(template=template, plausible=plausible, impossible=impossible, itype=itype)


def _alpha_len(word: str) -> int:
    return sum(1 for ch in word if ch.isalpha())


# =============================================================================
# FRAME TEMPLATES, by impossibility type. Each template's two fillers come from
# the shared pools above so reuse holds. The templates carry the linguistic
# content; ``build_frames`` instantiates the bank from them.
#
# A template is (sentence, plausible_pool, impossible_pool). The builder pairs a
# plausible word with a length-matched impossible word, advancing through the
# pools so each pool word is reused across many templates.
# =============================================================================

# (a) INANIMATE AGENT: biological-verb predicate (frame-shared); slot = agent.
#     plausible = ANIMATE agent, impossible = OBJECT.
_A_TEMPLATES = [
    "the {S} ate breakfast before the long meeting",
    "the {S} drank water after the morning run",
    "the {S} slept soundly through the loud storm",
    "the {S} breathed slowly during the quiet exam",
    "the {S} chewed the bread at the kitchen table",
    "the {S} swallowed the pill without any water",
    "the {S} sneezed twice during the long lecture",
    "the {S} yawned widely after the dull movie",
    "the {S} coughed loudly in the silent library",
    "the {S} blinked at the bright morning sun",
    "the {S} shivered in the cold winter wind",
    "the {S} sweated under the hot summer sun",
    "the {S} sipped the tea on the cold porch",
    "the {S} dreamed quietly all through the night",
    "the {S} healed slowly after the bad fall",
    "the {S} licked the bowl after the warm soup",
    "the {S} snored gently during the afternoon nap",
    "the {S} panted hard after the steep climb",
    "the {S} winked at the camera near the door",
    "the {S} hummed a tune while washing the dishes",
    "the {S} wept softly at the end of the play",
    "the {S} laughed at the joke during the meal",
    "the {S} gulped the milk before the bus came",
    "the {S} digested the heavy meal very slowly",
    "the {S} dozed off during the warm bus ride",
    "the {S} salivated at the smell of the bread",
    "the {S} stretched after the long quiet drive",
    "the {S} bit into the apple near the gate",
    "the {S} tasted the soup at the busy counter",
    "the {S} grew thirsty after the long dry walk",
]

# (b) SCALE/STRENGTH: human verb (frame-shared); slot = carried/lifted object.
#     plausible = small OBJECT, impossible = BIG_THING.
_B_TEMPLATES = [
    "the man carried the {S} home before dark",
    "the woman lifted the {S} onto the high shelf",
    "the boy pushed the {S} across the wide yard",
    "the girl dragged the {S} into the small room",
    "the farmer hauled the {S} up the steep hill",
    "the worker raised the {S} above his tired head",
    "the nurse carried the {S} down the long hall",
    "the driver loaded the {S} into the small trunk",
    "the clerk stacked the {S} beside the front desk",
    "the teacher moved the {S} across the small stage",
    "the climber slung the {S} over his sore shoulder",
    "the sailor heaved the {S} onto the wooden deck",
    "the woman tucked the {S} under her left arm",
    "the worker shifted the {S} onto the flat cart",
    "the porter wheeled the {S} through the wide lobby",
    "the runner clutched the {S} during the whole race",
    "the cook carried the {S} from the back pantry",
    "the boy slipped the {S} into his coat pocket",
    "the helper carried the {S} up the narrow stairs",
    "the man squeezed the {S} into the tiny gap",
    "the woman lifted the {S} onto the kitchen counter",
    "the man pushed the {S} up the loading ramp",
    "the diver hauled the {S} back onto the boat",
    "the woman carried the {S} through the busy station",
    "the man lifted the {S} onto the wooden roof",
    "the boy carried the {S} to the recycling bin",
    "the man balanced the {S} on the narrow beam",
    "the girl carried the {S} from the parked car",
    "the man tucked the {S} beneath the front seat",
    "the woman raised the {S} onto the top bunk",
]

# (c) CONTAINER/SIZE: small container (frame-shared); slot = inserted object.
#     plausible = small OBJECT, impossible = BIG_THING.
_C_TEMPLATES = [
    "she slid the {S} into the small matchbox",
    "he packed the {S} inside the tiny envelope",
    "he stuffed the {S} into the small wallet",
    "she fit the {S} inside the thin pencil case",
    "he squeezed the {S} into the little teacup",
    "she tucked the {S} inside the small locket",
    "he placed the {S} inside the small jewel box",
    "he loaded the {S} into the tiny toy truck",
    "she fit the {S} inside the slim phone case",
    "he stored the {S} inside the small spice jar",
    "she pushed the {S} into the narrow mail slot",
    "he zipped the {S} inside the small pencil pouch",
    "he packed the {S} into the small lunch box",
    "he tucked the {S} inside the slim book sleeve",
    "she slid the {S} into the narrow guitar case",
    "she fit the {S} inside the tiny ring box",
    "she packed the {S} into the slim laptop bag",
    "he stuffed the {S} into the small sock drawer",
    "she fit the {S} inside the small camera case",
    "she tucked the {S} into the small first aid box",
    "he packed the {S} inside the slim violin case",
    "he loaded the {S} into the small toy chest",
    "she stuffed the {S} into the small makeup bag",
    "she packed the {S} into the slim document folder",
    "he stuffed the {S} into the small glove box",
    "she slid the {S} into the small toaster oven",
    "she fit the {S} into the tiny dollhouse room",
    "she packed the {S} into the slim umbrella sleeve",
    "he fit the {S} inside the small sewing kit box",
    "she slid the {S} into the small mailbox door",
]

# (d) MATERIAL/STATE: ordinary vs impossible-state place/object.
#     Type (d) covers material/state violations. The slot is the APPLIANCE / place
#     whose temperature determines the outcome. The TEMPLATE states only the
#     outcome; the appliance is the single variable:
#       - a COLD appliance (freezer/fridge/cellar) keeps food cold/frozen and is
#         the PLAUSIBLE filler for a 'stayed frozen' outcome, and the IMPOSSIBLE
#         filler for a 'baked/melted' outcome;
#       - a HOT appliance (oven/furnace/stove) does the reverse.
#     The two pools are each used in BOTH roles across the two template groups
#     (d1a 'cold outcome' and d1b 'hot outcome'), so the appliance words are not
#     class-predictive. NO temperature adjective sits next to the slot (that would
#     contradict the appliance, e.g. 'a warm freezer'); the appliance alone
#     carries the (im)possibility.
COLD_APPLIANCES = [
    "freezer", "fridge", "cellar", "icebox", "cooler", "larder", "pantry",
    "chiller", "fridge", "freezer",
]
HOT_APPLIANCES = [
    "oven", "furnace", "stove", "burner", "boiler", "skillet", "heater",
    "grill", "toaster", "kiln", "hearth", "kettle",
]

# (d1a) COLD outcome: plausible = COLD appliance, impossible = HOT appliance.
_D1A_TEMPLATES = [
    "the ice stayed frozen inside the kitchen {S}",
    "the milk stayed cold inside the closed {S}",
    "the cream stayed chilled inside the steel {S}",
    "the butter stayed hard inside the small {S}",
    "the juice stayed icy inside the white {S}",
    "the cube stayed frozen inside the metal {S}",
    "the frost stayed thick inside the old {S}",
    "the sorbet stayed solid inside the wide {S}",
    "the soda stayed cold inside the tall {S}",
    "the yogurt stayed firm inside the quiet {S}",
    "the popsicle stayed frozen inside the gray {S}",
    "the meat stayed fresh inside the clean {S}",
]
# (d1b) HOT outcome: plausible = HOT appliance, impossible = COLD appliance.
_D1B_TEMPLATES = [
    "the bread baked golden inside the kitchen {S}",
    "the stew simmered slowly inside the iron {S}",
    "the roast browned evenly inside the steel {S}",
    "the pie crisped nicely inside the small {S}",
    "the cake rose fully inside the warm {S}",
    "the soup boiled hard inside the deep {S}",
    "the buns toasted brown inside the wide {S}",
    "the sauce thickened fast inside the hot {S}",
    "the chicken cooked through inside the large {S}",
    "the dough baked firm inside the round {S}",
    "the rice steamed soft inside the tall {S}",
    "the fish grilled crisp inside the open {S}",
]

# (d1c) ANIMATE impossible STATE: an inanimate OBJECT persisting unchanged /
#     motionless is ordinary; a LIVING animate doing the same (no food, no
#     movement, no breath for weeks, or shrinking to nothing) is physically
#     impossible. slot = the SUBJECT (plausible OBJECT vs impossible ANIMATE).
_D1C_TEMPLATES = [
    "the {S} lay motionless for the entire long month",
    "the {S} stayed perfectly still for many silent weeks",
    "the {S} sat unchanged for years in the dry attic",
    "the {S} remained frozen stiff for the whole warm summer",
    "the {S} stood without moving for the full long winter",
]


class FrameBankError(ValueError):
    """An authored-frame-bank invariant was violated (LOUD; no silent skip)."""


def build_frames() -> list[Frame]:
    """Instantiate the full minimal-pair bank from templates + shared pools.

    Deterministic (no RNG): the pairing walks the pools by index, length-matching
    each plausible filler to an impossible filler, so the same word is reused
    across many frames in BOTH roles. Returns >= MIN_FRAMES frames; ``audit``
    then asserts the type-mix, length-match, reuse and ban-list invariants."""
    frames: list[Frame] = []
    seen_templates: set[str] = set()

    def _add(template: str, plausible: str, impossible: str, itype: str) -> None:
        # the SLOT-filled instance must be unique; the template carries {S} so two
        # different (plausible/impossible) draws on the same template are distinct
        # frames only if they fill differently. We key uniqueness on the filled
        # impossible+plausible surface to forbid identical bases.
        key = f"{template}||{plausible}||{impossible}"
        if key in seen_templates:
            return
        seen_templates.add(key)
        frames.append(_f(template, plausible, impossible, itype))

    # combined inanimate-object pool for type (a): every TINY or CARRYABLE object
    # is a valid IMPOSSIBLE biological agent ('the pebble ate breakfast'), and each
    # is ALSO a plausible object in (b)/(c), so this keeps reuse high.
    inanimate_pool = TINY_OBJECTS + CARRY_OBJECTS

    # (a) inanimate agent: plausible = ANIMATE agent, impossible = inanimate OBJECT.
    for ti, template in enumerate(_A_TEMPLATES):
        for k in range(5):
            animate = ANIMATES[(ti * 5 + k) % len(ANIMATES)]
            obj = _pick(inanimate_pool, ti * 7 + k, _alpha_len(animate))
            if obj is None:
                continue
            _add(template, animate, obj, TYPE_INANIMATE_AGENT)

    # (b) scale/strength: plausible = a CARRYABLE object (a person can carry it),
    #     impossible = a HUGE thing (no person can carry/lift it).
    for ti, template in enumerate(_B_TEMPLATES):
        for k in range(5):
            obj = CARRY_OBJECTS[(ti * 5 + k) % len(CARRY_OBJECTS)]
            big = _pick(HUGE_THINGS, ti * 5 + k, _alpha_len(obj))
            if big is None:
                continue
            _add(template, obj, big, TYPE_SCALE_STRENGTH)

    # (c) container/size: plausible = a TINY object (it FITS the small container),
    #     impossible = a HUGE thing (it cannot fit).
    for ti, template in enumerate(_C_TEMPLATES):
        for k in range(5):
            obj = TINY_OBJECTS[(ti * 5 + k) % len(TINY_OBJECTS)]
            big = _pick(HUGE_THINGS, ti * 5 + k + 3, _alpha_len(obj))
            if big is None:
                continue
            _add(template, obj, big, TYPE_CONTAINER_SIZE)

    # (d1a) material/state COLD outcome: plausible cold appliance, impossible hot.
    for ti, template in enumerate(_D1A_TEMPLATES):
        for k in range(5):
            cold = COLD_APPLIANCES[(ti * 5 + k) % len(COLD_APPLIANCES)]
            hot = _pick(HOT_APPLIANCES, ti * 5 + k, _alpha_len(cold))
            if hot is None:
                continue
            # cold outcome: COLD appliance plausible, HOT appliance impossible.
            _add(template, cold, hot, TYPE_MATERIAL_STATE)

    # (d1b) HOT outcome: plausible = HOT appliance, impossible = COLD appliance.
    for ti, template in enumerate(_D1B_TEMPLATES):
        for k in range(5):
            hot = HOT_APPLIANCES[(ti * 5 + k) % len(HOT_APPLIANCES)]
            cold = _pick(COLD_APPLIANCES, ti * 5 + k, _alpha_len(hot))
            if cold is None:
                continue
            _add(template, hot, cold, TYPE_MATERIAL_STATE)

    # (d1c) ANIMATE impossible-STATE: an animate frozen solid / turned to stone is
    #     impossible; an inanimate OBJECT in that same state is ordinary. So the
    #     slot is the SUBJECT: plausible = an OBJECT (a statue/pebble can be stone-
    #     hard or frozen), impossible = an ANIMATE. This is the clean way to give
    #     every ANIMATE a genuine IMPOSSIBLE occurrence (so animates are both-class
    #     without the incoherent 'baked inside the kitchen cat' closure pairing).
    for ti, template in enumerate(_D1C_TEMPLATES):
        for k in range(6):
            animate = ANIMATES[(ti * 6 + k) % len(ANIMATES)]
            obj = _pick(TINY_OBJECTS + CARRY_OBJECTS, ti * 6 + k, _alpha_len(animate))
            if obj is None:
                continue
            _add(template, obj, animate, TYPE_MATERIAL_STATE)

    # REUSE-CLOSURE pass (the key guard: cross-class reuse >= 80%). After the
    # main blocks, some slot words sit in ONE class only -- HUGE_THINGS that were
    # only ever impossible ('cottage', 'barn'), and any appliance/word that landed
    # in only one class. For each such word we author extra, SURVIVABLE minimal-
    # pair frames that put it in the OPPOSITE class in an ordinary role, length-
    # matched against a word already established in that opposite class. This is
    # exactly the spec's reuse construction ('The rock was next to the path as a
    # False item elsewhere'); it makes no slot word class-predictive. The closure
    # is deterministic and order-stable.
    _close_reuse(frames, _add)

    return frames


# Intangibles: things with no solid body. Used as the IMPOSSIBLE object of a
# physical lift/carry frame ('the crane lifted the cloud' is impossible) and ALSO
# as a PLAUSIBLE subject of a motion frame ('the cloud drifted over the hills'),
# so they too are both-class and never class-predictive.
INTANGIBLES = [
    "cloud", "shadow", "rainbow", "breeze", "mist", "fog", "sunbeam",
    "ripple", "echo", "glow", "haze", "smoke", "draft", "gust",
]

# Carrier templates for the reuse-closure pass. Each is a genuine, survivable
# minimal pair (one variant ordinary, the other physically impossible).
#
# (i) put an impossible-only HUGE_THING into a PLAUSIBLE role: a crane/forklift
#     LIFT frame. A crane plausibly lifts a cottage/bridge/tractor (HUGE_THING ->
#     plausible); a crane cannot lift a cloud/shadow/rainbow (INTANGIBLE ->
#     impossible). So HUGE_THING enters the PLAUSIBLE class and the pair survives.
_CLOSE_PLAUSIBLE_TEMPLATES = [
    "the crane lifted the {S} above the building site",
    "the forklift raised the {S} onto the loading dock",
    "the tall crane hoisted the {S} over the wide yard",
    "the heavy crane swung the {S} across the river",
    "the dockside crane lowered the {S} onto the barge",
]

# (ii) put a plausible-only COLD appliance into an IMPOSSIBLE role: a HOT-process
#      frame. Bread baking inside an oven (HOT -> plausible); bread baking inside
#      a fridge/cellar (COLD -> impossible). So the cold appliance enters the
#      IMPOSSIBLE class and the pair survives. No 'freezing/icy' adjective: the
#      appliance ITSELF carries the (im)possible state.
_CLOSE_IMPOSSIBLE_TEMPLATES = [
    "the loaf baked golden brown inside the kitchen {S}",
    "the stew simmered for hours inside the iron {S}",
    "the pie crisped to a crust inside the small {S}",
    "the roast browned slowly inside the steel {S}",
    "the buns rose and baked inside the heavy {S}",
]


def _close_reuse(frames: list[Frame], add) -> None:
    """Author extra, SURVIVABLE minimal-pair frames so every slot word reaches
    BOTH classes (cross-class reuse guard, on the emitted data).

    For each currently single-class slot word, emit ordinary-role frames in the
    opposite class with a length-matched, correctly-signed partner, so the base
    can genuinely survive validation (both variants validate as intended).
    Deterministic and order-stable."""
    plausible = {fr.plausible.lower() for fr in frames}
    impossible = {fr.impossible.lower() for fr in frames}

    # First seed the INTANGIBLES as PLAUSIBLE subjects of a MOTION frame, paired
    # with a HUGE_THING as the impossible subject: 'the cloud drifted over the
    # hills' (plausible) vs 'the cottage drifted over the hills' (impossible). So
    # intangibles become both-class (plausible here, impossible in the crane
    # closure below) before being used as partners.
    intangible_motion = [
        "the {S} drifted slowly over the distant hills",
        "the {S} floated gently across the evening sky",
        "the {S} spread softly above the quiet valley",
    ]
    for idx, word in enumerate(INTANGIBLES):
        template = intangible_motion[idx % len(intangible_motion)]
        # impossible partner: a HUGE_THING cannot drift/float through the sky.
        partner = _pick(HUGE_THINGS, idx, _alpha_len(word))
        if partner is None or partner == word:
            continue
        add(template, word, partner, TYPE_MATERIAL_STATE)

    # recompute classes after seeding intangibles
    plausible = {fr.plausible.lower() for fr in frames}
    impossible = {fr.impossible.lower() for fr in frames}

    # (i) impossible-only words (HUGE_THINGS) -> PLAUSIBLE crane-LIFT frame vs an
    #     INTANGIBLE (a crane cannot lift a cloud/shadow). HUGE_THING -> plausible.
    impossible_only = sorted(impossible - plausible)
    for idx, word in enumerate(impossible_only):
        template = _CLOSE_PLAUSIBLE_TEMPLATES[idx % len(_CLOSE_PLAUSIBLE_TEMPLATES)]
        partner = _pick(INTANGIBLES, idx, _alpha_len(word))
        if partner is None or partner == word:
            continue
        add(template, word, partner, TYPE_MATERIAL_STATE)

    # (ii) plausible-only words (e.g. an appliance only ever used plausibly) ->
    #      IMPOSSIBLE bake frame vs a HOT appliance. The plausible-only word is
    #      placed in the IMPOSSIBLE slot (baking inside a fridge -> impossible).
    plausible_only = sorted(plausible - impossible)
    for idx, word in enumerate(plausible_only):
        template = _CLOSE_IMPOSSIBLE_TEMPLATES[idx % len(_CLOSE_IMPOSSIBLE_TEMPLATES)]
        partner = _pick(HOT_APPLIANCES, idx, _alpha_len(word))
        if partner is None or partner == word:
            continue
        add(template, partner, word, TYPE_MATERIAL_STATE)


def _pick(pool: list[str], offset: int, target_len: int) -> str | None:
    """Pick a pool word within +/- LENGTH_MATCH_TOL of ``target_len``, scanning
    from ``offset`` (cyclic). Returns None if no length-matched word exists."""
    n = len(pool)
    for i in range(n):
        w = pool[(offset + i) % n]
        if abs(_alpha_len(w) - target_len) <= LENGTH_MATCH_TOL:
            return w
    return None


# the full authored bank (deterministic).
FRAMES: list[Frame] = build_frames()


# =============================================================================
# audit (LOUD): every construction invariant is asserted at import time.
# =============================================================================


def audit(frames: list[Frame] = FRAMES) -> dict[str, object]:
    """Assert every rule-18 authoring invariant; return a measured summary.

    Raises FrameBankError on the FIRST violated invariant (loud)."""
    if len(frames) < MIN_FRAMES:
        raise FrameBankError(
            f"authored only {len(frames)} frames, rule-18 recipe needs >= {MIN_FRAMES}"
        )

    # 1) structural: one {S} slot, single-word fillers, length-matched, distinct.
    seen: set[tuple[str, str, str]] = set()
    for fr in frames:
        if fr.template.count("{S}") != 1 or "{" in fr.template.replace("{S}", ""):
            raise FrameBankError(f"frame must carry exactly one {{S}} slot: {fr.template!r}")
        key = (fr.template, fr.plausible, fr.impossible)
        if key in seen:
            raise FrameBankError(f"duplicate frame: {key!r}")
        seen.add(key)
        for filler in (fr.plausible, fr.impossible):
            if len(filler.split()) != 1 or not filler.isascii() or not filler.isalpha():
                raise FrameBankError(f"filler must be a single ascii word: {filler!r}")
        dlen = abs(_alpha_len(fr.plausible) - _alpha_len(fr.impossible))
        if dlen > LENGTH_MATCH_TOL:
            raise FrameBankError(
                f"fillers {fr.plausible!r}/{fr.impossible!r} differ in length by "
                f"{dlen} > {LENGTH_MATCH_TOL}"
            )
        if fr.plausible == fr.impossible:
            raise FrameBankError(f"plausible == impossible in {fr.template!r}")
        if fr.itype not in ALL_TYPES:
            raise FrameBankError(f"unknown impossibility type {fr.itype!r}")

    # also: no duplicate FILLED impossible/plausible surface within a base would
    # be caught downstream; here we only forbid a frame whose two variants are
    # identical, already checked above.

    # 2) banned (fantasy) lexicon nowhere.
    for fr in frames:
        tokens = set(fr.template.lower().replace("{s}", "").split())
        tokens |= {fr.plausible.lower(), fr.impossible.lower()}
        bad = tokens & BANNED_LEXICON
        if bad:
            raise FrameBankError(f"banned fantasy lexicon {sorted(bad)} in {fr.template!r}")

    # 3) impossibility type mix: each type within [20%, 35%].
    from collections import Counter

    tc = Counter(fr.itype for fr in frames)
    n = len(frames)
    shares = {t: tc[t] / n for t in ALL_TYPES}
    for t, share in shares.items():
        if not (TYPE_SHARE_MIN <= share <= TYPE_SHARE_MAX):
            raise FrameBankError(
                f"impossibility type {t!r} share {share:.3f} outside "
                f"[{TYPE_SHARE_MIN}, {TYPE_SHARE_MAX}] ({tc[t]}/{n})"
            )

    # 4) cross-class word reuse >= 80%.
    pl = {fr.plausible.lower() for fr in frames}
    im = {fr.impossible.lower() for fr in frames}
    allw = pl | im
    reused = pl & im
    reuse_fraction = len(reused) / len(allw)
    if reuse_fraction < CROSS_CLASS_REUSE_MIN:
        raise FrameBankError(
            f"cross-class word reuse {reuse_fraction:.3f} < {CROSS_CLASS_REUSE_MIN} "
            f"({len(reused)}/{len(allw)} distinct slot words in BOTH classes). "
            f"plausible-only={sorted(pl - im)[:10]}, "
            f"impossible-only={sorted(im - pl)[:10]}"
        )

    return {
        "n_frames": n,
        "type_counts": dict(tc),
        "type_shares": shares,
        "n_distinct_slot_words": len(allw),
        "n_reused_both_classes": len(reused),
        "cross_class_reuse_fraction": reuse_fraction,
    }


# run the audit at import — a malformed bank must fail fast, never ship.
AUDIT_SUMMARY = audit()
