#!/usr/bin/env bash
set -euo pipefail

# Run experiment pipeline (after dataset generation via run.sh)
#
# Usage:
#   ./run_experiments.sh                  # run all steps
#   ./run_experiments.sh --model llama    # only Llama
#   ./run_experiments.sh --skip-extract   # skip extraction (already done)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_FLAG=""
SKIP_EXTRACT=false
AGG="mean"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_FLAG="--model $2"; shift 2 ;;
        --skip-extract) SKIP_EXTRACT=true; shift ;;
        --agg) AGG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Verify dataset exists
if [ ! -f "data/stimuli/stimuli_final.jsonl" ]; then
    echo "ERROR: Dataset not found. Run ./run.sh first to generate the dataset."
    exit 1
fi

echo "============================================"
echo "GL-Qualia Experiment Pipeline"
echo "============================================"
echo ""

# Step 1: Extract reference representations
if [ "$SKIP_EXTRACT" = false ]; then
    echo "── Step 1/4: Extracting reference representations ──"
    uv run python experiments/references.py $MODEL_FLAG
    echo ""

    # Step 2: Extract stimulus representations
    echo "── Step 2/4: Extracting stimulus representations ──"
    uv run python experiments/extract.py $MODEL_FLAG
    echo ""
else
    echo "── Skipping extraction (--skip-extract) ──"
    echo ""
fi

# Step 3: Similarity analysis + GL hypothesis tests
echo "── Step 3/4: Similarity analysis + GL hypothesis tests ──"
uv run python experiments/rsa.py --agg "$AGG" $MODEL_FLAG
echo ""

# Step 4: Probing classifiers
echo "── Step 4/4: Probing classifiers ──"
uv run python experiments/probing.py --agg "$AGG" $MODEL_FLAG
echo ""

echo "============================================"
echo "Done. Results saved to data/analysis/ and data/probing_results/"
echo "============================================"
