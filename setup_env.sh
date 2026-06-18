#!/bin/bash
# ============================================================
# StreamRing — Environment Setup Script
# Jalankan: chmod +x setup_env.sh && ./setup_env.sh
# ============================================================

set -euo pipefail

export STREAMRING_DETERMINISTIC="${STREAMRING_DETERMINISTIC:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export STREAMRING_NUM_THREADS="${STREAMRING_NUM_THREADS:-1}"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${CYAN}[SETUP]${NC} $1"; }
ok()  { echo -e "${GREEN}[OK]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; }

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
STREAMRING_DIR="${PROJECT_DIR}/streamring"
ETL_DIR="${PROJECT_DIR}/ethereum-etl-develop"
VENV_DIR="${STREAMRING_DIR}/.venv"

# ---- Step 1: Create Virtual Environment ----
log "Step 1/6: Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    log "Virtual environment already exists at ${VENV_DIR}"
else
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
ok "Activated: $(python3 --version) at $(which python3)"

# ---- Step 2: Upgrade pip ----
log "Step 2/6: Upgrading pip..."
pip install --upgrade pip wheel "setuptools<81" -q
ok "pip upgraded"

# ---- Step 3: Install ethereum-etl ----
log "Step 3/6: Installing ethereum-etl..."
if [ -d "${ETL_DIR}" ]; then
    cd "${ETL_DIR}"
    pip install -e ".[streaming]" -q 2>&1 | tail -5
    cd "${PROJECT_DIR}"
    ok "ethereum-etl installed"
else
    err "ethereum-etl-develop directory not found!"
    exit 1
fi

# ---- Step 4: Install StreamRing dependencies ----
log "Step 4/6: Installing StreamRing dependencies..."
if [ -f "${STREAMRING_DIR}/requirements.txt" ]; then
    pip install -r "${STREAMRING_DIR}/requirements.txt" -q 2>&1 | tail -5
    ok "StreamRing dependencies installed"
else
    err "streamring/requirements.txt not found!"
    exit 1
fi

# ---- Step 5: Verify imports ----
log "Step 5/6: Verifying imports..."
cd "${STREAMRING_DIR}"

IMPORT_TEST=$(python3 -c "
import sys
errors = []
try:
    import torch
    print(f'  PyTorch: {torch.__version__}')
except ImportError as e:
    errors.append(f'torch: {e}')

try:
    import torch_geometric
    print(f'  PyG: {torch_geometric.__version__}')
except ImportError as e:
    errors.append(f'torch_geometric: {e}')

try:
    import xgboost
    print(f'  XGBoost: {xgboost.__version__}')
except ImportError as e:
    errors.append(f'xgboost: {e}')

try:
    import web3
    print(f'  Web3: {web3.__version__}')
except ImportError as e:
    errors.append(f'web3: {e}')

try:
    from google.cloud import bigquery
    print(f'  BigQuery: OK')
except ImportError as e:
    errors.append(f'bigquery: {e}')

try:
    from src.etl.bigquery_extractor import extract_period
    from src.graph_construction.temporal_graph import build_temporal_graph
    from src.subgraph_matching.ifasi_index import IFASIIndex
    from src.gnn_models.tgn_backbone import TGNBackbone
    from src.gnn_models.subgnn_encoder import SubGNNEncoder
    from src.fraud_detection.streamring import StreamRingPipeline
    from src.evaluation.metrics import compute_all_metrics
    from src.streaming.stream_simulator import StreamSimulator
    print(f'  StreamRing modules: ALL OK')
except ImportError as e:
    errors.append(f'streamring: {e}')

if errors:
    print(f'\nFailed imports:')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print(f'\nAll imports successful!')
" 2>&1) || true

echo "$IMPORT_TEST"

# ---- Step 6: Check Google Cloud ----
log "Step 6/6: Checking Google Cloud credentials..."
if command -v gcloud &> /dev/null; then
    ACCOUNT=$(gcloud config get-value account 2>/dev/null || echo "none")
    if [ "$ACCOUNT" != "none" ] && [ -n "$ACCOUNT" ]; then
        ok "Google Cloud: logged in as ${ACCOUNT}"
    else
        echo -e "${RED}  Google Cloud: not logged in${NC}"
        echo "  Run: gcloud auth login && gcloud auth application-default login"
    fi
else
    echo -e "${RED}  gcloud CLI not installed${NC}"
    echo "  Run: brew install google-cloud-sdk"
fi

# ---- Summary ----
cd "${PROJECT_DIR}"
echo ""
echo "============================================================"
echo -e "${GREEN}Environment Setup Complete!${NC}"
echo "============================================================"
echo ""
echo "To activate the environment in the future:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Next steps:"
echo "  1. cd streamring"
echo "  2. python -m src.etl.bigquery_extractor --period dao_hack"
echo "  3. Read streamring/REPRODUCIBILITY.md for public/reviewer reproduction steps"
echo ""
