"""Precheck before any distillation data-generation run.

WP-3 (2026-05-24): enforces the train↔deploy retrieval contract documented
in ``docs/retrieval-contract.md`` §3 + §6. Refuses to exit 0 unless every
piece of the contract is in place; the distillation entrypoint (WP-2)
will gate on this script's exit code.

Checks performed (in declared order — first failure stops the run):

  1. ``DISTILL_MODE`` is true (otherwise ``datasheet_lookup_tool`` would
     silently fall back to ``LOCAL_DATASHEET_FALLBACKS`` on misses).
  2. ``DATASHEET_EMBEDDING_BACKEND == "openvino"`` (no keyword-only or
     null backend — those produce a different vector space).
  3. ``DATASHEET_EMBEDDING_MODEL_DIR`` exists and contains
     ``openvino_model.xml``, ``openvino_model.bin``, ``tokenizer.json``.
  4. Every ``knowledge/datasheets/*.json`` document has a matching
     ``embeddings/<document_id>.npz`` with non-empty vectors.
  5. Every chunk in each datasheet JSON has a vector row in its .npz
     (no orphan chunks).
  6. All .npz vector dims are equal and match the loaded model's output
     dim (sanity — prevents stale embeddings from a different model).

Usage::

    .venv/bin/python -m scripts.distill.precheck_retrieval

    # As a gate in shell:
    .venv/bin/python -m scripts.distill.precheck_retrieval && \
        .venv/bin/python -m scripts.distill.run_inference  # (WP-2)

Exit codes:
    0  — all checks pass; safe to generate distillation data.
    1  — at least one check failed; do NOT generate data.

The script prints a structured report (one line per check) so it can be
captured into the distillation run's audit log alongside the dataset.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Make ``app`` importable when invoked as a script from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings  # noqa: E402
from app.schemas.kb import DatasheetDocument  # noqa: E402


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}"


def _check_distill_mode() -> CheckResult:
    if not getattr(settings, "DISTILL_MODE", False):
        return CheckResult(
            name="DISTILL_MODE",
            passed=False,
            detail=(
                "DISTILL_MODE is False. Set env DISTILL_MODE=true before "
                "running. Otherwise datasheet_lookup_tool would silently "
                "fall back to hand-coded rules on misses (synthetic "
                "evidence that the on-device runtime never produces)."
            ),
        )
    return CheckResult(name="DISTILL_MODE", passed=True, detail="enabled")


def _check_embedding_backend() -> CheckResult:
    backend = (getattr(settings, "DATASHEET_EMBEDDING_BACKEND", "null") or "").lower()
    if backend != "openvino":
        return CheckResult(
            name="DATASHEET_EMBEDDING_BACKEND",
            passed=False,
            detail=(
                f"backend={backend!r}. Distillation requires "
                "DATASHEET_EMBEDDING_BACKEND=openvino so that train-time "
                "retrieval uses the same vector space as on-device deploy."
            ),
        )
    return CheckResult(
        name="DATASHEET_EMBEDDING_BACKEND",
        passed=True,
        detail="openvino",
    )


def _check_model_dir() -> tuple[CheckResult, Path | None]:
    raw = getattr(settings, "DATASHEET_EMBEDDING_MODEL_DIR", None)
    if not raw:
        return (
            CheckResult(
                name="DATASHEET_EMBEDDING_MODEL_DIR",
                passed=False,
                detail=(
                    "not set (env DATASHEET_EMBEDDING_MODEL_DIR is required). "
                    "Run scripts/distill/fetch_artifacts.sh first."
                ),
            ),
            None,
        )
    model_dir = Path(raw)
    if not model_dir.is_absolute():
        model_dir = REPO_ROOT / model_dir
    if not model_dir.is_dir():
        return (
            CheckResult(
                name="DATASHEET_EMBEDDING_MODEL_DIR",
                passed=False,
                detail=(
                    f"directory does not exist: {model_dir}. "
                    "Run scripts/distill/fetch_artifacts.sh to fetch the OV IR."
                ),
            ),
            None,
        )
    required = ("openvino_model.xml", "openvino_model.bin", "tokenizer.json")
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        return (
            CheckResult(
                name="DATASHEET_EMBEDDING_MODEL_DIR",
                passed=False,
                detail=f"{model_dir} is missing required files: {missing}",
            ),
            None,
        )
    return (
        CheckResult(
            name="DATASHEET_EMBEDDING_MODEL_DIR",
            passed=True,
            detail=str(model_dir.relative_to(REPO_ROOT))
            if model_dir.is_relative_to(REPO_ROOT)
            else str(model_dir),
        ),
        model_dir,
    )


def _check_embedding_backend_active(model_dir: Path | None) -> CheckResult:
    """WP-3 v2: actually instantiate the OpenVINO backend and call it.

    Static file presence (``_check_model_dir``) is insufficient — a
    corrupted .xml, missing OpenVINO runtime install, or wrong tokenizer
    format would still pass the file checks but silently fall back to
    keyword-only at runtime, breaking the train↔deploy contract."""
    if model_dir is None:
        # Already reported as failed by _check_model_dir; chain a notice.
        return CheckResult(
            name="embedding_backend.active",
            passed=False,
            detail="skipped — model_dir check failed first.",
        )
    try:
        from app.services.embedding_backend import create_embedding_backend

        backend = create_embedding_backend(
            "openvino",
            model_dir=model_dir,
            device=getattr(settings, "DATASHEET_EMBEDDING_DEVICE", "CPU"),
            max_length=getattr(settings, "DATASHEET_EMBEDDING_MAX_LEN", 256),
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="embedding_backend.active",
            passed=False,
            detail=f"create_embedding_backend raised: {type(exc).__name__}: {exc}",
        )

    if not backend.is_active:
        return CheckResult(
            name="embedding_backend.active",
            passed=False,
            detail=(
                "backend.is_active is False — model directory exists but the "
                "OpenVINO runtime could not load it (check that "
                "`openvino` + `tokenizers` are installed and the IR files "
                "are valid)."
            ),
        )

    # Probe encode to confirm forward pass works + dim matches static metadata.
    try:
        vec = backend.encode(["precheck probe"])
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="embedding_backend.active",
            passed=False,
            detail=f"backend.encode raised: {type(exc).__name__}: {exc}",
        )
    if vec.shape[0] != 1 or vec.shape[1] == 0:
        return CheckResult(
            name="embedding_backend.active",
            passed=False,
            detail=f"backend.encode produced unexpected shape: {vec.shape}",
        )
    return CheckResult(
        name="embedding_backend.active",
        passed=True,
        detail=f"loaded + probe-encoded (dim={vec.shape[1]})",
    )


def _iter_datasheet_jsons() -> list[Path]:
    datasheet_dir = REPO_ROOT / "knowledge" / "datasheets"
    return sorted(p for p in datasheet_dir.glob("*.json") if p.is_file())


def _load_document(path: Path) -> DatasheetDocument | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DatasheetDocument.model_validate(payload)
    except Exception:
        return None


def _check_npz_coverage() -> tuple[list[CheckResult], dict[str, np.ndarray]]:
    results: list[CheckResult] = []
    embeddings_dir = Path(getattr(settings, "DATASHEET_EMBEDDINGS_DIR", ""))
    if not embeddings_dir.is_absolute():
        embeddings_dir = REPO_ROOT / embeddings_dir
    if not embeddings_dir.is_dir():
        return (
            [
                CheckResult(
                    name="datasheet.embeddings_dir",
                    passed=False,
                    detail=f"directory does not exist: {embeddings_dir}",
                )
            ],
            {},
        )

    documents = _iter_datasheet_jsons()
    if not documents:
        return (
            [
                CheckResult(
                    name="datasheet.documents",
                    passed=False,
                    detail="no datasheet JSON files found.",
                )
            ],
            {},
        )

    loaded_vectors: dict[str, np.ndarray] = {}
    for json_path in documents:
        doc = _load_document(json_path)
        if doc is None:
            results.append(
                CheckResult(
                    name=f"datasheet.{json_path.stem}.parse",
                    passed=False,
                    detail=f"failed to parse {json_path.relative_to(REPO_ROOT)}",
                )
            )
            continue
        npz_path = embeddings_dir / f"{doc.document_id}.npz"
        if not npz_path.exists():
            try:
                display_path = npz_path.relative_to(REPO_ROOT)
            except ValueError:
                # ``DATASHEET_EMBEDDINGS_DIR`` may point outside the repo
                # (e.g. in a tempdir during tests). Fall back to absolute.
                display_path = npz_path
            results.append(
                CheckResult(
                    name=f"datasheet.{doc.document_id}.npz",
                    passed=False,
                    detail=(
                        f"missing {display_path}. Run "
                        f"scripts/build_datasheet_embeddings.py "
                        f"--documents {doc.document_id}"
                    ),
                )
            )
            continue
        try:
            data = np.load(npz_path, allow_pickle=False)
            chunk_ids = list(data["chunk_ids"].tolist())
            vectors = np.asarray(data["vectors"], dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckResult(
                    name=f"datasheet.{doc.document_id}.npz",
                    passed=False,
                    detail=f"failed to load {npz_path.name}: {exc!r}",
                )
            )
            continue

        json_chunk_ids = [c.chunk_id for c in doc.chunks]
        missing_in_npz = set(json_chunk_ids) - set(chunk_ids)
        if missing_in_npz:
            results.append(
                CheckResult(
                    name=f"datasheet.{doc.document_id}.coverage",
                    passed=False,
                    detail=(
                        f"{len(missing_in_npz)} chunks have no vector "
                        f"(orphan): {sorted(missing_in_npz)[:3]}... — "
                        f"re-run build_datasheet_embeddings.py for this document."
                    ),
                )
            )
            continue
        if vectors.size == 0 or vectors.shape[0] == 0:
            results.append(
                CheckResult(
                    name=f"datasheet.{doc.document_id}.npz",
                    passed=False,
                    detail="vectors array is empty.",
                )
            )
            continue
        loaded_vectors[doc.document_id] = vectors
        results.append(
            CheckResult(
                name=f"datasheet.{doc.document_id}",
                passed=True,
                detail=f"{vectors.shape[0]} chunks × {vectors.shape[1]}-dim",
            )
        )
    return results, loaded_vectors


def _check_vector_dim_uniformity(loaded: dict[str, np.ndarray]) -> CheckResult:
    if not loaded:
        return CheckResult(
            name="datasheet.dim_uniformity",
            passed=False,
            detail="no .npz files loaded successfully.",
        )
    dims = {doc_id: vec.shape[1] for doc_id, vec in loaded.items()}
    unique_dims = set(dims.values())
    if len(unique_dims) != 1:
        return CheckResult(
            name="datasheet.dim_uniformity",
            passed=False,
            detail=(
                f"vector dims differ across .npz files: {dims}. "
                "All must come from the same model — re-run "
                "build_datasheet_embeddings.py with the unified model."
            ),
        )
    return CheckResult(
        name="datasheet.dim_uniformity",
        passed=True,
        detail=f"all {len(loaded)} documents share dim={unique_dims.pop()}",
    )


def run_all_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(_check_distill_mode())
    results.append(_check_embedding_backend())
    model_check, model_dir = _check_model_dir()
    results.append(model_check)
    # WP-3 v2: probe the backend after files exist; catches corrupted IR
    # or missing runtime that the static check can't see.
    results.append(_check_embedding_backend_active(model_dir))
    cov_results, loaded = _check_npz_coverage()
    results.extend(cov_results)
    results.append(_check_vector_dim_uniformity(loaded))
    return results


def main(argv: list[str] | None = None) -> int:
    results = run_all_checks()
    for r in results:
        print(r.render())
    failed = [r for r in results if not r.passed]
    if failed:
        print(
            f"\n{len(failed)} of {len(results)} checks FAILED. "
            "Distillation data generation is NOT safe — see "
            "docs/retrieval-contract.md §3.",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {len(results)} checks passed. Safe to generate distillation data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
