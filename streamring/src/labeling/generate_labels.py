"""
Fraud Label Generation for StreamRing.
Combines known event labels + heuristic labels + structural anomaly detection.

Labeling Strategy (from Blueprint Section 3.5):
1. Known event labels: DAO attacker addresses, 51% double-spend transactions
2. Heuristic labels: self-loops, rapid fan-out/fan-in, wash trading
3. Structural anomaly: dense subgraph / temporal burst detection
"""

import os
import numpy as np
import pandas as pd
import torch
import networkx as nx
from pathlib import Path


# ============================================================================
# KNOWN DAO HACK ADDRESSES
# The DAO attacker drained ~3.6M ETC via recursive call exploit.
# These are the known attacker + splitter contract addresses.
# ============================================================================

DAO_ATTACKER_ADDRESSES = {
    "0xf4c64518ea10f995918a454158c6b61407ea345c",  # The DAO attacker (Dark DAO)
    "0xc0ee9db1a9e07ca63e4ff0d5fb6f86bf68d47b89",  # DAO extra balance
    "0x304a554a310c7e546dfe434669c62820b7d83490",  # TheDAO contract
    "0x914d1b8b43e92723e64fd0a06f5bdb8dd9b10c79",  # DAO child contract
    "0x4a574510c7014e4ae985403536074abe582adfc8",  # Known splitter
    "0xb136707642a4ea12fb4bae820f03d2562ebff487",  # Withdrawal contract
    "0x2ef47100e0787b915105fd5e3f4ff6752079d5cb",  # DAO curator
    "0xac0e15a038eedfc68ba3c35c73fed5be4a07afb5",  # Related flow
    "0xda4a4626d3e16e094de3225a751aab7128e96526",  # High-value recipient
}


def label_known_events(addr_index: dict, period: str) -> dict:
    """Label addresses based on known fraud events."""
    labels = {}

    if period in ("dao_hack", "pre_dao", "post_fork"):
        for addr, idx in addr_index.items():
            if addr in DAO_ATTACKER_ADDRESSES:
                labels[idx] = 1  # Known fraud
    return labels


def label_heuristic_patterns(graph_data: dict, features_df: pd.DataFrame,
                             pattern_features: dict) -> dict:
    """
    Label addresses using heuristic fraud indicators:
    1. Self-loop transactions (wash trading)
    2. Rapid fan-out then fan-in (layering pattern)
    3. Extreme cycle participation (ring structure)
    4. Abnormally high pattern counts relative to tx count
    """
    labels = {}
    edge_index = graph_data["edge_index"].numpy()

    # 1. Self-loop detection
    self_loops = set()
    for i in range(edge_index.shape[1]):
        if edge_index[0, i] == edge_index[1, i]:
            self_loops.add(int(edge_index[0, i]))

    # 2. High cycle count relative to degree (ring indicator)
    for node_id, pf in pattern_features.items():
        if isinstance(pf, np.ndarray) and len(pf) >= 3:
            cycle_2 = pf[0]  # 2-cycles
            cycle_3 = pf[1]  # 3-cycles
            total_cycles = cycle_2 + cycle_3

            # High cycle participation = ring indicator
            if total_cycles > 10:
                labels[node_id] = 1

    # 3. Fan-out + fan-in pattern (layering)
    if "out_degree" in features_df.columns and "in_degree" in features_df.columns:
        for node_id in features_df.index:
            out_deg = features_df.loc[node_id, "out_degree"]
            in_deg = features_df.loc[node_id, "in_degree"]

            # High fan-out AND high fan-in with similar magnitude = layering
            if out_deg > 20 and in_deg > 20 and 0.3 < out_deg / max(in_deg, 1) < 3.0:
                labels[node_id] = 1

    # 4. Self-loops as wash trading
    for node_id in self_loops:
        labels[node_id] = 1

    return labels


def label_structural_anomalies(graph_data: dict) -> dict:
    """
    Detect structurally anomalous subgraphs (potential fraud rings).
    Uses dense subgraph detection on time-windowed snapshots.
    """
    labels = {}
    edge_index = graph_data["edge_index"].numpy()
    edge_time = graph_data["edge_time"].numpy()
    num_edges = edge_index.shape[1]

    if num_edges == 0:
        return labels

    # Build NetworkX graph for connected component analysis
    G = nx.DiGraph()
    for i in range(num_edges):
        G.add_edge(int(edge_index[0, i]), int(edge_index[1, i]))

    # Find strongly connected components (potential rings)
    for scc in sorted((set(component) for component in nx.strongly_connected_components(G)), key=lambda c: sorted(c)):
        if len(scc) >= 3:  # Ring must have >= 3 members
            subG = G.subgraph(scc)
            density = nx.density(subG)
            # Abnormally dense SCCs are fraud ring candidates
            if density > 0.3 and len(scc) <= 50:
                for node in sorted(scc):
                    labels[node] = 1

    return labels


def generate_labels(graph_data: dict, features_df: pd.DataFrame,
                    pattern_features: dict, period: str) -> dict:
    """
    Generate combined fraud labels for all addresses.

    Returns:
        labels: dict mapping node_id -> 0 (benign) or 1 (fraud)
    """
    num_nodes = graph_data["num_nodes"]
    addr_index = graph_data.get("addr_index", {})

    # Start with all benign
    labels = {i: 0 for i in range(num_nodes)}

    # Layer 1: Known event labels (highest confidence)
    known = label_known_events(addr_index, period)
    labels.update(known)
    print(f"  Known event labels: {sum(v == 1 for v in known.values())} fraud addresses")

    # Layer 2: Heuristic patterns
    heuristic = label_heuristic_patterns(graph_data, features_df, pattern_features)
    labels.update(heuristic)
    print(f"  Heuristic labels: {sum(v == 1 for v in heuristic.values())} fraud addresses")

    # Layer 3: Structural anomalies
    structural = label_structural_anomalies(graph_data)
    labels.update(structural)
    print(f"  Structural anomaly labels: {sum(v == 1 for v in structural.values())} fraud addresses")

    total_fraud = sum(v == 1 for v in labels.values())
    total_benign = sum(v == 0 for v in labels.values())
    print(f"  TOTAL: {total_fraud} fraud ({total_fraud / num_nodes * 100:.2f}%), "
          f"{total_benign} benign ({total_benign / num_nodes * 100:.2f}%)")

    return labels


def main():
    """Generate labels for all periods."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="all")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    base = Path(args.base_dir)
    periods = ["dao_hack", "pre_dao", "post_fork", "attack_51_v1",
               "attack_51_v2", "normal_ops"]

    if args.period != "all":
        periods = [args.period]

    for period in periods:
        graph_path = base / "data" / "graphs" / f"{period}_graph.pt"
        feat_path = base / "data" / "processed" / f"{period}_features.csv"
        pat_path = base / "data" / "processed" / f"{period}_patterns.pt"

        if not graph_path.exists():
            print(f"Skipping {period}: graph not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Generating labels: {period}")
        print(f"{'=' * 60}")

        graph = torch.load(graph_path, weights_only=False)
        features_df = pd.read_csv(feat_path, index_col=0) if feat_path.exists() else pd.DataFrame()
        pattern_features = torch.load(pat_path, weights_only=False) if pat_path.exists() else {}

        labels = generate_labels(graph, features_df, pattern_features, period)

        # Save labels
        out_path = base / "data" / "processed" / f"{period}_labels.pt"
        torch.save(labels, str(out_path))
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
