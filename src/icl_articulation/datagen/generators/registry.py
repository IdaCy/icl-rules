"""rule_id -> per-rule generator module dispatch (AUTO-DISCOVERED).

Each registered module exposes the GENERATOR INTERFACE (``build_bases`` and
``instantiate``; see ``base``). The registry maps a rule_id to its module and
provides ``run`` to drive that module through the shared gated pipeline
(``base.emit_rule``).

The registry is built by SCANNING the ``rules`` package at import time: every
module under ``rules/`` is imported and registered under its rule_id. The
rule_id is the module's ``RULE_ID`` constant if present, else the module
filename. This removes the race-prone hand-maintained ``_REGISTRY`` dict --
adding a rule is now purely "drop a module in ``rules/``", with no shared edit
to coordinate, so parallel fan-out cannot drop registrations.

Lookups are LOUD: an unknown rule_id raises (no quiet fallback), and a module
that does not expose both required callables raises at discovery (so a malformed
generator fails fast instead of silently disappearing from ``--list``).
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Callable

from . import rules as _rules_pkg
from .base import EmitSummary, emit_rule

# the two callables a generator module MUST expose (the GENERATOR INTERFACE)
_REQUIRED_ATTRS = ("build_bases", "instantiate")

# default seed for a CLI run (overridable via --seed); logged in the summary.
DEFAULT_SEED = 1234


class RegistryError(KeyError):
    """rule_id is not registered, or its module is malformed."""


def _rule_id_of(module: ModuleType, module_name: str) -> str:
    """The rule_id a module registers under: its ``RULE_ID`` constant if it
    exposes one (a non-empty str), else the module filename."""
    rule_id = getattr(module, "RULE_ID", None)
    if isinstance(rule_id, str) and rule_id:
        return rule_id
    return module_name


def _discover() -> dict[str, str]:
    """Scan the ``rules`` package, import every module, and map rule_id ->
    dotted module path. LOUD on a malformed module (missing a required
    callable) and on a rule_id collision (two modules claiming one id)."""
    registry: dict[str, str] = {}
    pkg_path = _rules_pkg.__path__
    pkg_name = _rules_pkg.__name__
    for mod_info in pkgutil.iter_modules(pkg_path):
        if mod_info.ispkg:
            continue
        module_name = mod_info.name
        dotted = f"{pkg_name}.{module_name}"
        module = importlib.import_module(dotted)
        missing = [a for a in _REQUIRED_ATTRS if not callable(getattr(module, a, None))]
        if missing:
            raise RegistryError(
                f"generator module {dotted!r} is missing required callable(s): "
                f"{missing} (GENERATOR INTERFACE: {list(_REQUIRED_ATTRS)})"
            )
        rule_id = _rule_id_of(module, module_name)
        if rule_id in registry:
            raise RegistryError(
                f"duplicate rule_id {rule_id!r}: both {registry[rule_id]!r} and "
                f"{dotted!r} register it (set a unique RULE_ID or rename a module)"
            )
        registry[rule_id] = dotted
    return registry


# rule_id -> dotted module path of the generator module, discovered at import.
_REGISTRY: dict[str, str] = _discover()


def registered_rules() -> list[str]:
    """The rule_ids with a registered generator module (sorted)."""
    return sorted(_REGISTRY)


def get_module(rule_id: str) -> ModuleType:
    """Import and return the generator module for ``rule_id`` (LOUD if absent /
    missing a required callable)."""
    if rule_id not in _REGISTRY:
        raise RegistryError(
            f"no generator registered for rule_id {rule_id!r}; "
            f"registered: {registered_rules()}"
        )
    module = importlib.import_module(_REGISTRY[rule_id])
    missing = [a for a in _REQUIRED_ATTRS if not callable(getattr(module, a, None))]
    if missing:
        raise RegistryError(
            f"generator module {_REGISTRY[rule_id]!r} for rule {rule_id!r} is "
            f"missing required callable(s): {missing} "
            f"(GENERATOR INTERFACE: {list(_REQUIRED_ATTRS)})"
        )
    return module


def get_generator(rule_id: str) -> tuple[Callable, Callable]:
    """Return (build_bases, instantiate) for ``rule_id``."""
    module = get_module(rule_id)
    return module.build_bases, module.instantiate


def run(
    rule_id: str,
    seed: int = DEFAULT_SEED,
    *,
    write: bool = True,
    run_pos: bool = True,
    data_dir=None,
    output_rule_id: str | None = None,
    stored_rule_id: str | None = None,
) -> EmitSummary:
    """Dispatch ``rule_id`` to its generator and run the full gated pipeline.

    A module may optionally expose ``STYLE_RULE_ID`` (to alias the sentence_style
    policy) and ``base_id_of`` (if its BaseSpec is not a str / lacks base_id);
    both are read here so a generator can stay a two-function module."""
    module = get_module(rule_id)
    return emit_rule(
        rule_id,
        module.build_bases,
        module.instantiate,
        seed,
        style_rule_id=getattr(module, "STYLE_RULE_ID", None),
        base_id_of=getattr(module, "base_id_of", None),
        data_dir=data_dir,
        output_rule_id=output_rule_id,
        stored_rule_id=stored_rule_id,
        write=write,
        run_pos=run_pos,
    )
