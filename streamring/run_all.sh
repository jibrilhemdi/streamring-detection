#!/usr/bin/env bash
# StreamRing reproducibility entry point.
#
# Usage:
#   bash run_all.sh              # Run all published experiments and figures
#   bash run_all.sh --core       # Run only the v3 experiment suite + figures
#   bash run_all.sh --figures    # Regenerate figures/tables from existing JSON

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
    PYTHON="${PYTHON:-python}"
fi
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export STREAMRING_DETERMINISTIC="${STREAMRING_DETERMINISTIC:-1}"
export STREAMRING_NUM_THREADS="${STREAMRING_NUM_THREADS:-1}"

if ! "$PYTHON" -c "import torch; import torch_geometric" 2>/dev/null; then
    echo "ERROR: torch and torch_geometric must be installed."
    echo "  pip install -r requirements.txt"
    exit 1
fi

MODE="${1:-full}"
FIGURES_ONLY=false
CORE_ONLY=false
FULL_PAPER=false

case "$MODE" in
    --figures)
        FIGURES_ONLY=true
        ;;
    --core)
        CORE_ONLY=true
        ;;
    --full|--paper|full)
        FULL_PAPER=true
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: bash run_all.sh [--core|--figures|--full]"
        exit 1
        ;;
esac

format_runtime() {
    local seconds="$1"
    printf '%02d:%02d:%02d' $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
}

run_python() {
    local start end elapsed
    echo ""
    echo "Running: $*"
    start=$(date +%s)
    "$PYTHON" "$@"
    end=$(date +%s)
    elapsed=$((end - start))
    echo "Completed: $*"
    echo "Runtime: $(format_runtime "$elapsed")"
}

ensure_data_inputs() {
    echo ""
    echo "Preparing pattern features from available graphs..."
    run_python scripts/generate_patterns.py --period all --base-dir "$SCRIPT_DIR"

    echo ""
    echo "Preparing labels from available graphs and pattern features..."
    run_python -m src.labeling.generate_labels --period all --base-dir "$SCRIPT_DIR"

    local missing=()
    for period in dao_hack pre_dao post_fork attack_51_v1 attack_51_v2; do
        if [ ! -f "$SCRIPT_DIR/data/graphs/${period}_graph.pt" ]; then
            missing+=("data/graphs/${period}_graph.pt")
        fi
        if [ ! -f "$SCRIPT_DIR/data/processed/${period}_patterns.pt" ]; then
            missing+=("data/processed/${period}_patterns.pt")
        fi
        if [ ! -f "$SCRIPT_DIR/data/processed/${period}_labels.pt" ]; then
            missing+=("data/processed/${period}_labels.pt")
        fi
    done

    if [ "${#missing[@]}" -gt 0 ]; then
        echo ""
        echo "ERROR: Missing generated data required for experiments:"
        printf '  - %s\n' "${missing[@]}"
        echo ""
        echo "Rebuild the missing data before running experiments:"
        echo "  1. bash scripts/extract_data.sh"
        echo "  2. bash scripts/build_graphs.sh"
        echo "  3. python scripts/generate_patterns.py --period all --base-dir ."
        echo "  4. python -m src.labeling.generate_labels --period all --base-dir ."
        exit 1
    fi
}

START=$(date +%s)

if [[ "$FIGURES_ONLY" == false ]]; then
    ensure_data_inputs

    echo "============================================"
    echo " StreamRing deterministic experiment run"
    echo "============================================"
    echo "Environment:"
    echo "  PYTHONHASHSEED=$PYTHONHASHSEED"
    echo "  CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
    echo "  STREAMRING_DETERMINISTIC=$STREAMRING_DETERMINISTIC"
    echo "  STREAMRING_NUM_THREADS=$STREAMRING_NUM_THREADS"
    echo ""

    run_python experiments/allout_v3.py

    if [[ "$CORE_ONLY" == false ]]; then
        run_python experiments/baselines.py
        run_python experiments/temporal_baselines.py
        run_python experiments/lstm_tier3.py
        run_python experiments/graphsage_tier3.py
        run_python experiments/streaming_simulation.py
        run_python experiments/compute_rdt.py
        run_python experiments/accuracy_at_latency.py
        run_python experiments/augmentation_ablation.py
        run_python experiments/case_study_visualization.py
    fi
fi

echo ""
echo "============================================"
echo " Generating publication figures and tables"
echo "============================================"
run_python experiments/generate_final_figures_v2.py

END=$(date +%s)
echo ""
echo "============================================"
echo " Done"
echo "============================================"
echo "Elapsed: $(( (END - START) / 60 )) minutes"
echo "Tables:  results/tables/*.json, results/tables/*.tex"
echo "Figures: results/figures/fig{1..16}_*.{png,pdf}"
