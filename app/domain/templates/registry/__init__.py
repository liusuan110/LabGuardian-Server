"""Topology template registry — single source of truth for all CADx templates.

Each template module under this package exports a module-level ``TEMPLATE``
constant of type :class:`app.domain.templates.base.TopologyTemplate`. The
:func:`get_template_registry` function lazily collects them into a dict keyed
by ``template_id``.

To add a new template, drop a file in this directory with a top-level
``TEMPLATE = TopologyTemplate(...)`` and append its module name to
:data:`_TEMPLATE_MODULES` below.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Mapping

from app.domain.templates.base import TopologyTemplate


# Order here also defines the iteration order of ``get_template_registry``
# (Python 3.7+ dict insertion order). Templates ranked first by
# "demo importance" — single-IC op-amp circuits, then BJT, then passives.
_TEMPLATE_MODULES: tuple[str, ...] = (
    "inverting_amp_opamp",
    "summing_amp_opamp",
    "integrator_opamp",
    "common_emitter",
    "differential_pair",
    "rc_first_order",
)


@lru_cache(maxsize=1)
def get_template_registry() -> Mapping[str, TopologyTemplate]:
    """Return the global registry of topology templates.

    Cached: the first call imports all six template modules; subsequent
    calls return the same mapping. Tests can reset via
    ``get_template_registry.cache_clear()``.

    Returns:
        An immutable-feeling ``Mapping[str, TopologyTemplate]``. The
        returned object is a regular ``dict`` for simplicity; callers
        are expected not to mutate it.

    Raises:
        ImportError: If any template module fails to import.
        ValueError: If a template fails its own ``validate()`` self-check.
    """
    registry: dict[str, TopologyTemplate] = {}
    for module_name in _TEMPLATE_MODULES:
        module = importlib.import_module(
            f"app.domain.templates.registry.{module_name}"
        )
        template = getattr(module, "TEMPLATE", None)
        if not isinstance(template, TopologyTemplate):
            raise ImportError(
                f"template module {module_name!r} did not export a valid "
                "TEMPLATE: TopologyTemplate"
            )
        errors = template.validate()
        if errors:
            raise ValueError(
                f"template {template.template_id!r} failed validation: {errors}"
            )
        if template.template_id in registry:
            raise ValueError(
                f"duplicate template_id detected: {template.template_id!r}"
            )
        registry[template.template_id] = template
    return registry


__all__ = ["get_template_registry"]
