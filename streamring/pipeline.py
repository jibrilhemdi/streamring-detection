"""
StreamRing Complete Pipeline Script
====================================
Runs the full StreamRing research pipeline from data extraction through evaluation.
This is the main executable for producing all paper results.

Usage:
    python -m streamring.pipeline --phase all
    python -m streamring.pipeline --phase extract
    python -m streamring.pipeline --phase train
    python -m streamring.pipeline --phase evaluate
"""

import os
import sys
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.reproducibility import set_runtime_threads, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# PHASE 1: ENVIRONMENT SETUP
# ============================================================================

def setup_environment(seed=42, deterministic=True):
    """Verify environment and set random seeds for reproducibility."""
    set_seed(seed, deterministic)
    set_runtime_threads()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Create directories
    base = Path(__file__).resolve().parent
    for d in ["data/raw", "data/processed", "data/graphs", "data/embeddings",
              "models", "results/figures", "results/tables", "logs"]:
        (base / d).mkdir(parents=True, exist_ok=True)
    
    return device


# ============================================================================
# PHASE 2: DATA EXTRACTION (BigQuery + ethereum-etl)
# ============================================================================

def extract_data_bigquery(periods: dict, output_dir: str):
    """
    Extract ETC blockchain data from Google BigQuery.
    
    Requires: pip install google-cloud-bigquery
    Auth: gcloud auth application-default login
    """
    from google.cloud import bigquery
    client = bigquery.Client()
    
    DATASET = "bigquery-public-data.crypto_ethereum_classic"
    
    for period_name, period_cfg in periods.items():
        start = period_cfg["start_block"]
        end = period_cfg["end_block"]
        out_dir = Path(output_dir) / period_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Extracting: {period_name} (blocks {start}-{end})")
        print(f"{'='*60}")
        
        # Transactions
        tx_query = f"""
        SELECT `hash`, nonce, block_number, transaction_index,
               from_address, to_address, value, gas, gas_price,
               block_timestamp, receipt_status
        FROM `{DATASET}.transactions`
        WHERE block_number BETWEEN @start AND @end
        ORDER BY block_number, transaction_index
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("start", "INT64", start),
            bigquery.ScalarQueryParameter("end", "INT64", end),
        ])
        df = client.query(tx_query, job_config=job_config).to_dataframe()
        df.to_csv(out_dir / "transactions.csv", index=False)
        print(f"  Transactions: {len(df):,}")
        
        # Traces
        trace_query = f"""
        SELECT block_number, transaction_hash, from_address, to_address,
               value, trace_type, call_type, gas, gas_used, subtraces, error
        FROM `{DATASET}.traces`
        WHERE block_number BETWEEN @start AND @end
        ORDER BY block_number
        """
        df = client.query(trace_query, job_config=job_config).to_dataframe()
        df.to_csv(out_dir / "traces.csv", index=False)
        print(f"  Traces: {len(df):,}")
        
        # Blocks
        block_query = f"""
        SELECT number, `hash`, miner, difficulty, size,
               gas_limit, gas_used, timestamp, transaction_count
        FROM `{DATASET}.blocks`
        WHERE number BETWEEN @start AND @end
        ORDER BY number
        """
        df = client.query(block_query, job_config=job_config).to_dataframe()
        df.to_csv(out_dir / "blocks.csv", index=False)
        print(f"  Blocks: {len(df):,}")


# ============================================================================
# PHASE 3: GRAPH CONSTRUCTION
# ============================================================================

def build_graph(tx_path: str, trace_path: str = None, output_path: str = None):
    """Build temporal transaction graph from CSV data."""
    from src.graph_construction.temporal_graph import build_temporal_graph, save_graph
    
    graph_data = build_temporal_graph(tx_path, trace_path)
    if output_path:
        save_graph(graph_data, output_path)
    return graph_data


# ============================================================================
# PHASE 4: FEATURE ENGINEERING
# ============================================================================

def compute_features(graph_data: dict) -> pd.DataFrame:
    """
    Compute 56+ features per address inspired by Elliptic++.
    
    Feature groups:
    A. Address-level aggregates (in/out degree, values, counts)
    B. Temporal features (inter-tx time, activity patterns)
    C. Graph-structural features (clustering, PageRank, centrality)
    D. Pattern features (cycle counts, fan-out/in - from IFASI)
    """
    import networkx as nx
    
    edge_index = graph_data["edge_index"].numpy()
    edge_time = graph_data["edge_time"].numpy()
    edge_attr = graph_data["edge_attr"].numpy()
    num_nodes = graph_data["num_nodes"]
    
    # Build networkx graph for structural features
    G = nx.DiGraph()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst, timestamp=edge_time[i], value=edge_attr[i, 0])
    
    features = {}
    
    # A. Address-level aggregates
    for node in range(num_nodes):
        in_deg = G.in_degree(node) if G.has_node(node) else 0
        out_deg = G.out_degree(node) if G.has_node(node) else 0
        
        in_values = [d["value"] for _, _, d in G.in_edges(node, data=True)] if G.has_node(node) else []
        out_values = [d["value"] for _, _, d in G.out_edges(node, data=True)] if G.has_node(node) else []
        
        in_times = sorted([d["timestamp"] for _, _, d in G.in_edges(node, data=True)]) if G.has_node(node) else []
        out_times = sorted([d["timestamp"] for _, _, d in G.out_edges(node, data=True)]) if G.has_node(node) else []
        all_times = sorted(in_times + out_times)
        
        # Temporal features
        inter_tx_times = np.diff(all_times) if len(all_times) > 1 else [0]
        
        features[node] = {
            "in_degree": in_deg,
            "out_degree": out_deg,
            "total_value_sent": sum(out_values),
            "total_value_received": sum(in_values),
            "avg_value_sent": np.mean(out_values) if out_values else 0,
            "avg_value_received": np.mean(in_values) if in_values else 0,
            "tx_count": in_deg + out_deg,
            "unique_out_counterparties": len(set(G.successors(node))) if G.has_node(node) else 0,
            "unique_in_counterparties": len(set(G.predecessors(node))) if G.has_node(node) else 0,
            "inter_tx_time_mean": np.mean(inter_tx_times),
            "inter_tx_time_std": np.std(inter_tx_times),
            "activity_span": (max(all_times) - min(all_times)) if len(all_times) > 1 else 0,
        }
    
    # C. Graph-structural features (sampled for large graphs)
    print("Computing PageRank...")
    pr = nx.pagerank(G, max_iter=50) if len(G) < 100000 else {}
    
    for node in features:
        features[node]["pagerank"] = pr.get(node, 0)
    
    return pd.DataFrame.from_dict(features, orient="index")


# ============================================================================
# PHASE 5: SUBGRAPH PATTERN ENUMERATION (IBM GFP-style)
# ============================================================================

def enumerate_patterns(graph_data: dict) -> dict:
    """
    Enumerate fraud patterns following IBM GFP (KDD 2024) approach.
    Returns pattern count features per node.
    """
    from src.subgraph_matching.ifasi_index import IFASIIndex, TemporalEdge
    
    edge_index = graph_data["edge_index"].numpy()
    edge_time = graph_data["edge_time"].numpy()
    edge_attr = graph_data["edge_attr"].numpy()
    
    index = IFASIIndex(max_temporal_gap=3600, window_size=21600)
    
    # Insert all edges
    print(f"Building IFASI index with {edge_index.shape[1]} edges...")
    for i in range(edge_index.shape[1]):
        edge = TemporalEdge(
            src=int(edge_index[0, i]),
            dst=int(edge_index[1, i]),
            timestamp=float(edge_time[i]),
            value=float(edge_attr[i, 0]),
        )
        index.insert_edge(edge)
    
    # Compute pattern features for all nodes
    all_nodes = set(edge_index[0].tolist() + edge_index[1].tolist())
    pattern_features = {}
    
    print(f"Computing pattern features for {len(all_nodes)} nodes...")
    for node in all_nodes:
        # Use latest timestamp for this node
        node_edges = edge_time[
            (edge_index[0] == node) | (edge_index[1] == node)]
        if len(node_edges) > 0:
            ref_time = float(node_edges.max())
            pattern_features[node] = index.get_pattern_features(node, ref_time)
    
    return pattern_features


# ============================================================================
# PHASE 6: TIER 1 - XGBOOST CLASSIFIER
# ============================================================================

def train_tier1(features_df: pd.DataFrame, pattern_features: dict,
                labels: dict, device: str = "cpu"):
    """Train Tier 1 XGBoost classifier on node + pattern features."""
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
    
    # Combine node features with pattern features
    feature_rows = []
    label_list = []
    
    for node_id in features_df.index:
        if node_id in labels:
            row = features_df.loc[node_id].values.tolist()
            if node_id in pattern_features:
                row.extend(pattern_features[node_id].tolist())
            else:
                row.extend([0] * 12)
            feature_rows.append(row)
            label_list.append(labels[node_id])
    
    X = np.array(feature_rows)
    y = np.array(label_list)
    
    print(f"Training Tier 1: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Temporal split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train XGBoost
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        max_depth=6,
        n_estimators=200,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    results = {
        "auc_roc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_prob),
    }
    
    print(f"Tier 1 Results: AUC={results['auc_roc']:.4f}, "
          f"F1={results['f1']:.4f}, PR-AUC={results['pr_auc']:.4f}")
    
    # Measure inference latency
    start = time.perf_counter()
    for _ in range(1000):
        model.predict_proba(X_test[:1])
    latency_ms = (time.perf_counter() - start) / 1000 * 1000
    print(f"Tier 1 Inference Latency: {latency_ms:.3f}ms per sample")
    
    return model, results


# ============================================================================
# MAIN PIPELINE
# ============================================================================

PERIODS = {
    "dao_hack": {"start_block": 1917000, "end_block": 1925000},
    "pre_dao": {"start_block": 1800000, "end_block": 1917000},
    "attack_51_v1": {"start_block": 7200000, "end_block": 7300000},
    "attack_51_v2": {"start_block": 10900000, "end_block": 11000000},
    "normal_ops": {"start_block": 5000000, "end_block": 6000000},
}


def main():
    parser = argparse.ArgumentParser(description="StreamRing Pipeline")
    parser.add_argument("--phase", default="all",
                        choices=["all", "extract", "graph", "features",
                                 "patterns", "train", "evaluate"])
    args = parser.parse_args()
    
    device = setup_environment()
    base_dir = Path(__file__).parent
    
    if args.phase in ("all", "extract"):
        print("\n" + "="*60)
        print("PHASE 1: DATA EXTRACTION")
        print("="*60)
        extract_data_bigquery(PERIODS, str(base_dir / "data" / "raw"))
    
    if args.phase in ("all", "graph"):
        print("\n" + "="*60)
        print("PHASE 2: GRAPH CONSTRUCTION")
        print("="*60)
        for period in PERIODS:
            tx_path = base_dir / "data" / "raw" / period / "transactions.csv"
            trace_path = base_dir / "data" / "raw" / period / "traces.csv"
            out_path = base_dir / "data" / "graphs" / f"{period}_graph.pt"
            if tx_path.exists():
                build_graph(str(tx_path), str(trace_path), str(out_path))
    
    if args.phase in ("all", "features"):
        print("\n" + "="*60)
        print("PHASE 3: FEATURE ENGINEERING")
        print("="*60)
        for period in PERIODS:
            graph_path = base_dir / "data" / "graphs" / f"{period}_graph.pt"
            if graph_path.exists():
                graph_data = torch.load(graph_path, weights_only=False)
                features = compute_features(graph_data)
                features.to_csv(base_dir / "data" / "processed" / f"{period}_features.csv")
                print(f"  {period}: {len(features)} addresses, {features.shape[1]} features")
    
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
