"""Per-rule generator modules (one file per rule).

Each module exposes the GENERATOR INTERFACE documented in
``icl_articulation.datagen.generators.base``:

    build_bases(gen) -> list[BaseSpec]
    instantiate(spec, label, gen) -> (text, slots_meta)

and is wired into ``registry._REGISTRY``. The reference rule is
``all_lowercase``; the 26 fan-out agents add their rule's module here.
"""

from __future__ import annotations
