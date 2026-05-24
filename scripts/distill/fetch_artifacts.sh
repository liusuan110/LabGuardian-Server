#!/usr/bin/env bash
#
# Fetch the binary artifacts required for distillation reproducibility.
#
# WP-3.1 (2026-05-24): the embedding model (~24 MB INT8 IR) and the
# pre-computed chunk .npz files are excluded from git (see .gitignore)
# because they are derivable and large-ish. A fresh checkout must run
# this script before `python -m scripts.distill.precheck_retrieval`.
#
# Steps:
#   1. Ensure the OV INT8 embedding model is present (download once
#      from HuggingFace if missing).
#   2. Rebuild every datasheet's .npz cache by running
#      scripts/build_datasheet_embeddings.py against all JSONs.
#
# After this script exits 0, run:
#   DISTILL_MODE=true \
#   DATASHEET_EMBEDDING_BACKEND=openvino \
#   DATASHEET_EMBEDDING_MODEL_DIR=models/bge-small-zh-v1.5-int8-ov \
#   .venv/bin/python -m scripts.distill.precheck_retrieval

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="models/bge-small-zh-v1.5-int8-ov"
# Pinning the HF revision is intentional — same artifact on every
# reproduction. Update the SHA + the docs/retrieval-contract.md change
# log together when you bump the model version.
HF_REPO="OpenVINO/bge-small-zh-v1.5-int8-ov"
HF_REVISION="main"   # TODO: pin to a specific commit SHA for full reproducibility

step() { printf '\n==> %s\n' "$*"; }

step "[1/2] Checking embedding model directory: $MODEL_DIR"
required_files=(openvino_model.xml openvino_model.bin tokenizer.json)
missing=()
for f in "${required_files[@]}"; do
    if [[ ! -f "$MODEL_DIR/$f" ]]; then
        missing+=("$f")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing model files: ${missing[*]}"
    echo "Downloading from HuggingFace ($HF_REPO @ $HF_REVISION) ..."
    if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "ERROR: neither 'hf' nor 'huggingface-cli' on PATH." >&2
        echo "       Install: pip install -U 'huggingface_hub[cli]'" >&2
        exit 2
    fi
    mkdir -p "$MODEL_DIR"
    if command -v hf >/dev/null 2>&1; then
        hf download "$HF_REPO" --revision "$HF_REVISION" --local-dir "$MODEL_DIR"
    else
        huggingface-cli download "$HF_REPO" --revision "$HF_REVISION" --local-dir "$MODEL_DIR"
    fi
    echo "Model downloaded to $MODEL_DIR"
else
    echo "Model directory already populated. Skipping download."
fi

step "[2/2] (Re)building datasheet .npz cache"
.venv/bin/python scripts/build_datasheet_embeddings.py \
    --backend openvino \
    --model-dir "$MODEL_DIR" \
    --device CPU

step "Done."
echo "Artifacts ready. Next: run"
echo "    DISTILL_MODE=true DATASHEET_EMBEDDING_BACKEND=openvino \\"
echo "    DATASHEET_EMBEDDING_MODEL_DIR=$MODEL_DIR \\"
echo "    .venv/bin/python -m scripts.distill.precheck_retrieval"
