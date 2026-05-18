"""Phase 3 of plan §十 R6 — load real student netlist_v2 exports.

The synthetic pipeline produces ``(netlist.json, meta.json)`` pairs
on disk (see ``scripts/gnn_export_pseudo_real.py``). When the
frontend / production S3 starts emitting real student data into a
matching directory layout, **this loader is the only new code path
needed** — the evaluator already consumes pre-baked netlist_v2 dicts
via ``evaluate_split(..., netlist_v2_dir=...)``. This module's job
is to read those pairs, validate them, and hand a uniform
``list[RealSample]`` to the evaluator.

Expected directory layout (mirrors the synthetic ``pseudo_real``
layout so a single command works against either)::

    <real_dir>/
        opamp_buffer/
            student_001.json          # netlist_v2
            student_001.meta.json     # ref_id + expected_outcome + notes
            student_002.json
            student_002.meta.json
        rc_lowpass/
            student_007.json
            student_007.meta.json
        manifest.json                  # optional, total counts

The ``.meta.json`` sidecar carries the ground truth (and any teacher
notes) the evaluator needs to score the sample::

    {
        "sample_id": "student_001",
        "ref_id": "opamp_buffer",
        "expected_outcome": "positive" | "wrong_observed" | "missing_required",
        "annotation_source": "teacher" | "self_report" | "auto",
        "notes": "...",
        "perturbation_chain": []     # optional — usually empty for real
    }

Real exports are **schema-permissive**: extra fields (``observations``,
``confidence``, vendor-specific keys) pass through untouched. Required
fields are minimal: each component needs ``component_id`` +
``component_type``, each pin needs ``pin_name`` +
``electrical_net_id``, each net needs ``electrical_net_id``. Missing
any of those → the sample is skipped with a warning, not a crash.

See ``docs/REAL_STUDENT_NETLIST.md`` for the full export contract +
how teachers should annotate samples.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("gnn.real_loader")


# Outcomes the evaluator understands. Mirrors
# ``PerturbedCur.expected_outcome`` so the same downstream code works.
ALLOWED_OUTCOMES = frozenset({
    "positive",
    "wrong_observed",
    "missing_required",
})


@dataclass(frozen=True)
class RealSample:
    """One real student export, fully validated and ready for
    :func:`app.domain.gnn.evaluator.evaluate_real_samples`."""

    sample_id: str
    ref_id: str
    expected_outcome: str         # one of ALLOWED_OUTCOMES
    netlist_v2: dict[str, Any]
    meta: dict[str, Any]
    netlist_path: Path
    meta_path: Path
    annotation_source: str = "unknown"
    perturbation_chain: tuple[str, ...] = ()
    notes: str = ""


@dataclass
class LoadStats:
    """Counters returned by :func:`load_real_samples` for observability."""

    n_loaded: int = 0
    n_skipped_no_meta: int = 0
    n_skipped_bad_outcome: int = 0
    n_skipped_invalid_schema: int = 0
    n_skipped_other: int = 0
    skipped_paths: list[str] = field(default_factory=list)


def _validate_netlist_v2_schema(doc: dict[str, Any]) -> list[str]:
    """Return a list of human-readable schema errors. Empty list = ok.

    The check is intentionally minimal — only the fields the rule
    comparator actually consumes are required. Unknown extras are fine.
    """

    errors: list[str] = []
    if not isinstance(doc.get("components"), list):
        errors.append("missing or non-list ``components``")
    if not isinstance(doc.get("nets"), list):
        errors.append("missing or non-list ``nets``")
    for i, comp in enumerate(doc.get("components", []) or []):
        if not isinstance(comp, dict):
            errors.append(f"components[{i}] is not a dict")
            continue
        if not str(comp.get("component_id") or "").strip():
            errors.append(f"components[{i}].component_id missing")
        if not str(comp.get("component_type") or "").strip():
            errors.append(f"components[{i}].component_type missing")
        # pins may be empty (a disconnected detected component is fine)
        for j, pin in enumerate(comp.get("pins", []) or []):
            if not isinstance(pin, dict):
                errors.append(
                    f"components[{i}].pins[{j}] is not a dict"
                )
                continue
            if not str(pin.get("pin_name") or pin.get("pin") or "").strip():
                errors.append(
                    f"components[{i}].pins[{j}].pin_name missing"
                )
    for k, net in enumerate(doc.get("nets", []) or []):
        if not isinstance(net, dict):
            errors.append(f"nets[{k}] is not a dict")
            continue
        if not str(net.get("electrical_net_id") or "").strip():
            errors.append(f"nets[{k}].electrical_net_id missing")
    return errors


def _read_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (OSError, ValueError) as e:
        log.warning("meta read failed for %s: %s", meta_path, e)
        return None


def _build_sample(
    netlist_path: Path,
    stats: LoadStats,
) -> RealSample | None:
    meta_path = netlist_path.with_name(netlist_path.stem + ".meta.json")
    meta = _read_meta(meta_path)
    if meta is None:
        stats.n_skipped_no_meta += 1
        stats.skipped_paths.append(str(netlist_path))
        log.warning(
            "skip %s: no sidecar meta at %s",
            netlist_path.name, meta_path.name,
        )
        return None

    ref_id = str(meta.get("ref_id") or "").strip()
    sample_id = str(
        meta.get("sample_id") or netlist_path.stem
    ).strip()
    outcome = str(meta.get("expected_outcome") or "").strip()
    if outcome not in ALLOWED_OUTCOMES:
        stats.n_skipped_bad_outcome += 1
        stats.skipped_paths.append(str(netlist_path))
        log.warning(
            "skip %s: meta.expected_outcome=%r not in %s",
            sample_id, outcome, sorted(ALLOWED_OUTCOMES),
        )
        return None
    if not ref_id:
        stats.n_skipped_bad_outcome += 1  # bucket: meta-level rejection
        stats.skipped_paths.append(str(netlist_path))
        log.warning("skip %s: meta.ref_id missing", sample_id)
        return None

    try:
        netlist_v2 = json.loads(netlist_path.read_text())
    except (OSError, ValueError) as e:
        stats.n_skipped_other += 1
        stats.skipped_paths.append(str(netlist_path))
        log.warning("skip %s: netlist read failed: %s", sample_id, e)
        return None
    if not isinstance(netlist_v2, dict):
        stats.n_skipped_invalid_schema += 1
        stats.skipped_paths.append(str(netlist_path))
        log.warning("skip %s: netlist top-level is not an object", sample_id)
        return None

    errors = _validate_netlist_v2_schema(netlist_v2)
    if errors:
        stats.n_skipped_invalid_schema += 1
        stats.skipped_paths.append(str(netlist_path))
        log.warning(
            "skip %s: schema invalid — first errors: %s",
            sample_id, errors[:3],
        )
        return None

    return RealSample(
        sample_id=sample_id,
        ref_id=ref_id,
        expected_outcome=outcome,
        netlist_v2=netlist_v2,
        meta=meta,
        netlist_path=netlist_path,
        meta_path=meta_path,
        annotation_source=str(meta.get("annotation_source") or "unknown"),
        perturbation_chain=tuple(
            str(x) for x in (meta.get("perturbation_chain") or ())
        ),
        notes=str(meta.get("notes") or ""),
    )


def load_real_samples(
    real_dir: Path,
    *,
    limit: int | None = None,
) -> tuple[list[RealSample], LoadStats]:
    """Walk ``real_dir`` and load every ``(netlist, meta)`` pair.

    Args:
        real_dir: root containing ``<ref_id>/<sample_id>.json`` files
            with sidecar ``<sample_id>.meta.json``.
        limit: cap on samples returned (for smoke tests).

    Returns:
        (samples, stats) — ``samples`` is sorted by ``(ref_id,
        sample_id)`` for reproducibility. ``stats`` carries
        skip-reason counters for the call site to surface in the
        report.
    """

    if not real_dir.is_dir():
        raise FileNotFoundError(f"real_dir not found: {real_dir}")

    stats = LoadStats()
    samples: list[RealSample] = []
    for netlist_path in sorted(real_dir.rglob("*.json")):
        # Skip sidecars / manifest
        if netlist_path.name.endswith(".meta.json"):
            continue
        if netlist_path.name == "manifest.json":
            continue
        s = _build_sample(netlist_path, stats)
        if s is not None:
            samples.append(s)
            stats.n_loaded += 1
            if limit is not None and stats.n_loaded >= limit:
                break

    samples.sort(key=lambda r: (r.ref_id, r.sample_id))
    return samples, stats


__all__ = [
    "ALLOWED_OUTCOMES",
    "RealSample",
    "LoadStats",
    "load_real_samples",
]
