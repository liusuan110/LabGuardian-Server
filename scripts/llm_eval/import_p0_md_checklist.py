from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


HEADER_RE = re.compile(r"^##\s+([A-Za-z0-9\\_]+)\s+·\s+")
CHECKBOX_RE = re.compile(r"^- \[(?P<mark>[ xX])\]\s+(?P<label>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import manual checklist ticks from P0 markdown report into CSV score sheet."
    )
    parser.add_argument("--report-md", type=Path, required=True, help="Markdown report with manual checklists.")
    parser.add_argument("--score-sheet", type=Path, required=True, help="Existing CSV score sheet to update.")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV with imported hits.")
    return parser.parse_args()


def extract_hits(report_md: Path) -> dict[str, list[str]]:
    text = report_md.read_text(encoding="utf-8")
    hits_by_qid: dict[str, list[str]] = {}
    current_qid: str | None = None
    in_checklist = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        header = HEADER_RE.match(line)
        if header:
            current_qid = header.group(1).replace("\\_", "_")
            in_checklist = False
            continue

        if line == "**Manual Checklist**:":
            in_checklist = True
            hits_by_qid.setdefault(current_qid or "", [])
            continue

        if not in_checklist:
            continue

        if line == "***":
            in_checklist = False
            current_qid = None
            continue

        match = CHECKBOX_RE.match(line)
        if match and current_qid:
            mark = match.group("mark")
            hits_by_qid.setdefault(current_qid, []).append("1" if mark.lower() == "x" else "0")

    return {qid: values[:3] for qid, values in hits_by_qid.items() if qid}


def update_score_sheet(score_sheet: Path, output_csv: Path, hits_by_qid: dict[str, list[str]]) -> None:
    with score_sheet.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise SystemExit("Score sheet is missing CSV headers.")

    for row in rows:
        qid = row.get("qid", "")
        hits = hits_by_qid.get(qid)
        if not hits:
            continue
        for idx in range(3):
            row[f"hit_{idx + 1}"] = hits[idx] if idx < len(hits) else ""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    report_md = args.report_md.resolve()
    score_sheet = args.score_sheet.resolve()
    output_csv = args.output_csv.resolve()

    if not report_md.exists():
        raise SystemExit(f"Markdown report not found: {report_md}")
    if not score_sheet.exists():
        raise SystemExit(f"Score sheet not found: {score_sheet}")

    hits_by_qid = extract_hits(report_md)
    update_score_sheet(score_sheet, output_csv, hits_by_qid)
    print(f"imported checklist for {len(hits_by_qid)} questions")
    print(f"wrote {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
