#!/bin/bash
# =============================================================================
# StreamRing Graph Construction Script
# Builds temporal transaction graphs from raw CSV data
# =============================================================================

set -e

echo "============================================="
echo "StreamRing Graph Construction Pipeline"  
echo "============================================="

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="$BASE_DIR/data/raw"
GRAPH_DIR="$BASE_DIR/data/graphs"

mkdir -p "$GRAPH_DIR"

cd "$BASE_DIR"

# Build graph for each period
for PERIOD_DIR in "$RAW_DIR"/*/; do
    PERIOD=$(basename "$PERIOD_DIR")
    echo ""
    echo "Building graph for period: $PERIOD"
    
    TX_FILE="$PERIOD_DIR/transactions.csv"
    TRACE_FILE="$PERIOD_DIR/traces.csv"
    TT_FILE="$PERIOD_DIR/token_transfers.csv"
    
    if [ ! -f "$TX_FILE" ]; then
        echo "  Skipping: no transactions.csv found"
        continue
    fi
    
    CMD="python -m src.graph_construction.temporal_graph --input $TX_FILE --output $GRAPH_DIR/${PERIOD}_graph.pt"
    
    [ -f "$TRACE_FILE" ] && CMD="$CMD --traces $TRACE_FILE"
    [ -f "$TT_FILE" ] && CMD="$CMD --token-transfers $TT_FILE"
    
    echo "  Running: $CMD"
    eval "$CMD"
done

echo ""
echo "Graph construction complete!"
echo "Graphs saved to: $GRAPH_DIR"
echo ""
echo "Next step: run experiments or regenerate figures with:"
echo "  bash run_all.sh"
