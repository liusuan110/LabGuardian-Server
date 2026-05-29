"""Export the merged student model to OpenVINO INT4 IR.

This is the local preparation step before copying the model directory to the
board. It wraps ``optimum-cli export openvino`` with the task configuration
required by ``openvino_genai.LLMPipeline``.

Examples:
    python scripts/openvino_export/export_student_openvino.py

    python scripts/openvino_export/export_student_openvino.py \
        --source-dir models/labguardian-student-1p5-merged \
        --output-dir models/labguardian-student-1p5-int4-ov \
        --overwrite
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "models" / "labguardian-student-1p5-merged"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "labguardian-student-1p5-int4-ov"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the merged student model to OpenVINO IR for board deployment."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Merged Hugging Face model directory (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output OpenVINO model directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--weight-format",
        default="int4",
        help="OpenVINO weight format, usually int4 for board deployment.",
    )
    parser.add_argument(
        "--task",
        default="text-generation-with-past",
        help="Export task passed to optimum-cli. Keep text-generation-with-past for GenAI.",
    )
    parser.add_argument(
        "--optimum-cli",
        default="optimum-cli",
        help="optimum-cli executable name or full path.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward --trust-remote-code to optimum-cli (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before exporting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    return parser.parse_args()


def build_export_command(args: argparse.Namespace, source_dir: Path, output_dir: Path) -> list[str]:
    command = [
        args.optimum_cli,
        "export",
        "openvino",
        "--model",
        str(source_dir),
        "--weight-format",
        args.weight_format,
        "--task",
        args.task,
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    command.append(str(output_dir))
    return command


def ensure_source(source_dir: Path) -> None:
    if not source_dir.exists():
        raise SystemExit(f"Source model directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise SystemExit(f"Source model path is not a directory: {source_dir}")


def ensure_output(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not overwrite:
        raise SystemExit(
            "Output directory already exists. Use --overwrite to recreate it: "
            f"{output_dir}"
        )
    shutil.rmtree(output_dir)


def verify_export(output_dir: Path) -> None:
    required_files = [
        output_dir / "openvino_model.xml",
        output_dir / "openvino_model.bin",
        output_dir / "openvino_tokenizer.xml",
        output_dir / "openvino_tokenizer.bin",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise SystemExit(
            "Export completed but required OpenVINO files are missing:\n- "
            + "\n- ".join(missing)
        )


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    ensure_source(source_dir)

    command = build_export_command(args, source_dir, output_dir)
    print("Resolved command:")
    print("  " + " ".join(f'"{part}"' if " " in part else part for part in command))
    print()
    print(f"Source : {source_dir}")
    print(f"Output : {output_dir}")
    print(f"Task   : {args.task}")
    print(f"Format : {args.weight_format}")

    if args.dry_run:
        return 0

    ensure_output(output_dir, args.overwrite)

    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode

    verify_export(output_dir)
    print()
    print("Export finished successfully.")
    print(f"Model dir ready for board copy: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
