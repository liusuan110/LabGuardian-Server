"""CADx Phase 0 acceptance — run all real_student fixtures through the
template matcher and emit a markdown comparison report.

Usage::

    python scripts/cadx/phase0_comparison_report.py \\
        --fixtures tests/fixtures/real_student/ \\
        --output reports/cadx_phase0_comparison.md

The report tabulates per-fixture:
  * fixture name, expected topology (heuristic from filename)
  * top-3 template matches with confidence
  * pass/fail (top-1 matches expected, confidence > 0.5)
  * role assignments for top-1 (sample of 5)
  * notes — any forbidden violations, missing required roles, variant

Verification stays in this script (not pytest) because it doubles as a
human-readable acceptance report attached to the Phase 0 milestone in
the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.domain.logical_reference import current_netlist_v2_to_graph  # noqa: E402
from app.domain.templates import (  # noqa: E402
    get_template_registry,
    match_all_templates,
)


# Map filename prefix -> expected template_id. Used to score acceptance.
#
# NOTE: `opamp_inverting_lpf` → `integrator_ua741_v1` looks counterintuitive
# but is the **structurally correct** classification: an LPF with R_f ∥ C1 in
# the feedback path *is* a lossy integrator. The integrator template's
# ``with_leak_resistor`` variant explicitly captures the R+C feedback, so
# it (legitimately) outranks the plain inverting_amp template. The
# "wrong" LPF fixture missing C1's full connection correctly falls back
# to inverting_amp.
EXPECTED_TEMPLATE_BY_PREFIX = {
    "inverting_amp": "inverting_amp_ua741_v1",
    "opamp_inverting_lpf_correct": "integrator_ua741_v1",
    "opamp_inverting_lpf_wrong": "inverting_amp_ua741_v1",
    "opamp_summing": "summing_amp_ua741_v1",
    "bjt_diff_amp": "differential_pair_v1",
}


@dataclass
class FixtureResult:
    fixture_name: str
    expected_template_id: str | None
    top_3: list[tuple[str, float, str | None]] = field(default_factory=list)
    role_assignments_top1: dict[str, str] = field(default_factory=dict)
    forbidden_top1: int = 0
    missing_required_top1: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def top1_template_id(self) -> str | None:
        return self.top_3[0][0] if self.top_3 else None

    @property
    def top1_confidence(self) -> float:
        return self.top_3[0][1] if self.top_3 else 0.0

    @property
    def passed(self) -> bool:
        if self.expected_template_id is None:
            return False
        return (
            self.top1_template_id == self.expected_template_id
            and self.top1_confidence >= 0.5
        )


def _infer_expected(fixture_basename: str) -> str | None:
    """Heuristic: longest matching prefix wins."""
    matches = [
        (prefix, tid)
        for prefix, tid in EXPECTED_TEMPLATE_BY_PREFIX.items()
        if fixture_basename.startswith(prefix)
    ]
    if not matches:
        return None
    matches.sort(key=lambda x: -len(x[0]))
    return matches[0][1]


def _run_fixture(path: Path) -> FixtureResult:
    name = path.stem  # drop .json
    expected = _infer_expected(name)
    result = FixtureResult(fixture_name=name, expected_template_id=expected)

    try:
        payload = json.loads(path.read_text())
        graph = current_netlist_v2_to_graph(payload)
        matches = match_all_templates(graph)
        for m in matches[:3]:
            result.top_3.append((m.template_id, m.confidence, m.matched_variant))
        if matches:
            top = matches[0]
            result.role_assignments_top1 = dict(top.role_assignments)
            result.forbidden_top1 = len(top.forbidden_violations)
            result.missing_required_top1 = list(top.missing_required)
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def _gather_fixtures(root: Path) -> Iterable[Path]:
    if root.is_file():
        return [root]
    # Drop .expected.json fixtures (those are golden output files).
    return sorted(
        p for p in root.glob("*.json")
        if not p.name.endswith(".expected.json")
    )


def _render_markdown(results: list[FixtureResult]) -> str:
    lines: list[str] = []
    lines.append("# CADx Phase 0 — Template Match Comparison Report\n")
    registry = get_template_registry()
    lines.append(
        f"Registry: **{len(registry)} templates** "
        f"({', '.join(registry.keys())}).\n"
    )
    lines.append("")
    passed = sum(1 for r in results if r.passed)
    has_expected = sum(1 for r in results if r.expected_template_id)
    skipped = sum(1 for r in results if r.expected_template_id is None)
    lines.append(
        f"## Acceptance: **{passed} / {has_expected} fixtures with expected topology pass** "
        f"(top-1 matches expected & confidence ≥ 0.5)"
    )
    if skipped:
        lines.append(
            f"_{skipped} fixture(s) skipped — no expected topology inferable from filename._\n"
        )
    lines.append("")

    lines.append("## Per-fixture results\n")
    lines.append(
        "| Fixture | Expected | Top-1 (conf) | Top-2 (conf) | Top-3 (conf) | Pass |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        top_cells = []
        for i in range(3):
            if i < len(r.top_3):
                tid, conf, var = r.top_3[i]
                cell = f"`{tid}` ({conf:.2f})"
                if var:
                    cell += f" [{var}]"
            else:
                cell = "—"
            top_cells.append(cell)
        mark = "✅" if r.passed else ("⏭️" if r.expected_template_id is None else "❌")
        lines.append(
            f"| `{r.fixture_name}` | `{r.expected_template_id or '—'}` "
            f"| {top_cells[0]} | {top_cells[1]} | {top_cells[2]} | {mark} |"
        )
    lines.append("")

    # Detail blocks for the passing cases — useful for showing role-assignment quality.
    lines.append("## Top-1 role assignments (passing cases)\n")
    for r in results:
        if not r.passed or not r.role_assignments_top1:
            continue
        lines.append(f"### `{r.fixture_name}` → `{r.top1_template_id}`")
        for student_id, role in list(r.role_assignments_top1.items())[:8]:
            short_id = student_id.replace("cur_comp:", "")
            lines.append(f"- `{short_id}` → **{role}**")
        if r.missing_required_top1:
            lines.append(
                f"- ⚠️ Missing required: {', '.join(f'`{x}`' for x in r.missing_required_top1)}"
            )
        if r.forbidden_top1:
            lines.append(f"- 🚫 {r.forbidden_top1} forbidden violations")
        lines.append("")

    # Failures — diagnose what went wrong.
    failures = [r for r in results if not r.passed and r.expected_template_id]
    if failures:
        lines.append("## Failures — needs investigation\n")
        for r in failures:
            lines.append(f"### `{r.fixture_name}`")
            lines.append(f"- Expected: `{r.expected_template_id}`")
            lines.append(f"- Got top-1: `{r.top1_template_id}` ({r.top1_confidence:.3f})")
            if r.error:
                lines.append(f"- Error: `{r.error}`")
            if r.role_assignments_top1:
                lines.append(
                    f"- Role assignments: "
                    f"{', '.join(f'`{k.replace(chr(0x3a), str())}`→{v}' for k, v in r.role_assignments_top1.items())}"
                )
            lines.append("")

    # Hypothesis distribution — what does the matcher most often think?
    lines.append("## Hypothesis distribution (top-1 across all fixtures)\n")
    counter = Counter(r.top1_template_id for r in results if r.top1_template_id)
    for tid, count in counter.most_common():
        lines.append(f"- `{tid}`: {count}")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fixtures",
        type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "real_student",
        help="Directory (or single .json file) of fixtures to evaluate.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "cadx_phase0_comparison.md",
        help="Markdown report output path.",
    )
    args = p.parse_args()

    fixtures = list(_gather_fixtures(args.fixtures))
    if not fixtures:
        print(f"❌ No .json fixtures found under {args.fixtures}", file=sys.stderr)
        return 2

    print(f"Evaluating {len(fixtures)} fixtures...")
    results = [_run_fixture(path) for path in fixtures]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render_markdown(results))

    passed = sum(1 for r in results if r.passed)
    has_expected = sum(1 for r in results if r.expected_template_id)
    print(f"✅ Wrote report: {args.output}")
    print(f"   Acceptance: {passed}/{has_expected} fixtures pass top-1 expected check")
    return 0 if passed == has_expected else 1


if __name__ == "__main__":
    sys.exit(main())
