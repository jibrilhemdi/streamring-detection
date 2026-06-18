#!/bin/bash
# =============================================================================
# StreamRing Data Extraction Script
# Extracts Ethereum Classic data from BigQuery for all experimental periods
# =============================================================================

set -e

echo "============================================="
echo "StreamRing Data Extraction Pipeline"
echo "============================================="

# Check prerequisites
command -v bq >/dev/null 2>&1 || { echo "Error: Google Cloud SDK (bq) not installed. Run: brew install google-cloud-sdk"; exit 1; }

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$BASE_DIR/data/raw"

mkdir -p "$RAW_DIR"

echo ""
echo "Step 1: Extracting data via Python BigQuery client..."
echo "Make sure you have authenticated: gcloud auth application-default login"
echo ""

cd "$BASE_DIR"

# Run the Python extractor for all periods
python -m src.etl.bigquery_extractor --config configs/default.yaml --period all --base-dir "$BASE_DIR"

echo ""
echo "Step 2: Computing dataset statistics..."
echo ""

python -c "
import os, glob
import pandas as pd

raw_dir = '$RAW_DIR'
for period in os.listdir(raw_dir):
    period_dir = os.path.join(raw_dir, period)
    if not os.path.isdir(period_dir):
        continue
    print(f'\\n=== Period: {period} ===')
    for f in sorted(glob.glob(os.path.join(period_dir, '*.csv'))):
        df = pd.read_csv(f, nrows=1)
        rows = sum(1 for _ in open(f)) - 1  # minus header
        print(f'  {os.path.basename(f):30s} {rows:>12,} rows, {len(df.columns)} columns')
"

echo ""
echo "Data extraction complete!"
echo "Raw data saved to: $RAW_DIR"
echo ""
echo "Next step: Build graphs with:"
echo "  bash scripts/build_graphs.sh"
