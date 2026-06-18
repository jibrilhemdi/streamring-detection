"""
Temporal Graph Construction from Ethereum Classic blockchain data.
Builds a heterogeneous temporal graph from transactions, traces, and token transfers.
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def load_transactions(tx_path: str) -> pd.DataFrame:
    """Load and clean transaction data."""
    df = pd.read_csv(tx_path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["gas_price"] = pd.to_numeric(df["gas_price"], errors="coerce").fillna(0)
    df["gas"] = pd.to_numeric(df["gas"], errors="coerce").fillna(0)
    df["block_timestamp"] = pd.to_datetime(df["block_timestamp"], errors="coerce", utc=True).astype("int64") // 10**9
    # Drop rows without from/to addresses
    df = df.dropna(subset=["from_address", "to_address"])
    return df


def load_traces(trace_path: str) -> pd.DataFrame:
    """Load and clean trace (internal transaction) data."""
    if not os.path.exists(trace_path):
        return pd.DataFrame()
    df = pd.read_csv(trace_path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df = df.dropna(subset=["from_address", "to_address"])
    # Keep only call-type traces with non-zero value (fund transfers)
    df = df[(df["trace_type"] == "call") & (df["value"] > 0)]
    return df


def load_token_transfers(tt_path: str) -> pd.DataFrame:
    """Load and clean token transfer data."""
    if not os.path.exists(tt_path):
        return pd.DataFrame()
    df = pd.read_csv(tt_path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df = df.dropna(subset=["from_address", "to_address"])
    return df


def build_address_index(transactions: pd.DataFrame, traces: pd.DataFrame,
                        token_transfers: pd.DataFrame) -> dict:
    """Create a mapping from address string to integer index."""
    addresses = set()
    for df in [transactions, traces, token_transfers]:
        if len(df) > 0:
            addresses.update(df["from_address"].unique())
            addresses.update(df["to_address"].unique())
    return {addr: idx for idx, addr in enumerate(sorted(addresses))}


def compute_node_features(addr_index: dict, transactions: pd.DataFrame,
                          traces: pd.DataFrame) -> torch.Tensor:
    """Compute feature vector for each address node (vectorized)."""
    num_nodes = len(addr_index)
    features = np.zeros((num_nodes, 8), dtype=np.float32)

    for df in [transactions, traces]:
        if len(df) == 0:
            continue
        src_idx = df["from_address"].map(addr_index)
        dst_idx = df["to_address"].map(addr_index)
        val = df["value"].astype(np.float64) / 1e18

        valid_src = src_idx.dropna().astype(int)
        valid_dst = dst_idx.dropna().astype(int)

        np.add.at(features[:, 1], valid_src.values, 1)  # out_degree
        np.add.at(features[:, 2], valid_src.values, val.loc[valid_src.index].values)  # total_sent
        np.add.at(features[:, 5], valid_src.values, 1)  # tx_count

        np.add.at(features[:, 0], valid_dst.values, 1)  # in_degree
        np.add.at(features[:, 3], valid_dst.values, val.loc[valid_dst.index].values)  # total_received
        np.add.at(features[:, 5], valid_dst.values, 1)  # tx_count

    # Normalize features (log-scale for heavy-tailed distributions)
    nonzero = features > 0
    features[nonzero] = np.log1p(features[nonzero])

    return torch.tensor(features, dtype=torch.float32)


def build_edge_index_and_features(df: pd.DataFrame, addr_index: dict,
                                  edge_type: str) -> tuple:
    """Build edge index and edge feature tensors from a dataframe (vectorized)."""
    if len(df) == 0:
        return (torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, 4, dtype=torch.float32),
                torch.zeros(0, dtype=torch.float64))

    src_idx = df["from_address"].map(addr_index)
    dst_idx = df["to_address"].map(addr_index)

    # Keep only rows where both src and dst are in addr_index
    valid = src_idx.notna() & dst_idx.notna()
    src_idx = src_idx[valid].astype(int).values
    dst_idx = dst_idx[valid].astype(int).values

    val = df.loc[valid, "value"].astype(np.float64).values / 1e18
    gas = df.loc[valid, "gas"].astype(np.float64).values / 1e6 if "gas" in df.columns else np.zeros(len(src_idx))
    gas_price = df.loc[valid, "gas_price"].astype(np.float64).values / 1e9 if "gas_price" in df.columns else np.zeros(len(src_idx))
    if "block_timestamp" in df.columns:
        ts = df.loc[valid, "block_timestamp"].astype(np.float64).fillna(0).values
    else:
        ts = np.zeros(len(src_idx))
    is_trace = np.full(len(src_idx), 1.0 if edge_type == "trace" else 0.0)

    edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)
    edge_attr = torch.tensor(np.column_stack([val, gas, gas_price, is_trace]), dtype=torch.float32)
    edge_time = torch.tensor(ts, dtype=torch.float64)

    return edge_index, edge_attr, edge_time


def build_temporal_graph(tx_path: str, trace_path: str = None,
                         tt_path: str = None) -> dict:
    """
    Build a temporal transaction graph from raw CSV data.
    
    Returns dict with:
        - node_features: Tensor [num_nodes, feat_dim]
        - edge_index: Tensor [2, num_edges]
        - edge_attr: Tensor [num_edges, edge_feat_dim]
        - edge_time: Tensor [num_edges]
        - addr_index: dict mapping address -> node_id
        - num_nodes, num_edges: int
    """
    transactions = load_transactions(tx_path)
    traces = load_traces(trace_path) if trace_path else pd.DataFrame()
    token_transfers = load_token_transfers(tt_path) if tt_path else pd.DataFrame()

    print(f"Loaded: {len(transactions)} transactions, {len(traces)} traces, "
          f"{len(token_transfers)} token transfers")

    # Build address index
    addr_index = build_address_index(transactions, traces, token_transfers)
    print(f"Unique addresses: {len(addr_index)}")

    # Compute node features
    node_features = compute_node_features(addr_index, transactions, traces)

    # Build edges from all sources
    tx_ei, tx_ef, tx_et = build_edge_index_and_features(
        transactions, addr_index, "transaction")
    tr_ei, tr_ef, tr_et = build_edge_index_and_features(
        traces, addr_index, "trace")
    tt_ei, tt_ef, tt_et = build_edge_index_and_features(
        token_transfers, addr_index, "token_transfer")

    # Concatenate all edges
    edge_index = torch.cat([tx_ei, tr_ei, tt_ei], dim=1)
    edge_attr = torch.cat([tx_ef, tr_ef, tt_ef], dim=0)
    edge_time = torch.cat([tx_et, tr_et, tt_et], dim=0)

    # Sort by timestamp
    sort_idx = torch.argsort(edge_time)
    edge_index = edge_index[:, sort_idx]
    edge_attr = edge_attr[sort_idx]
    edge_time = edge_time[sort_idx]

    print(f"Graph: {len(addr_index)} nodes, {edge_index.shape[1]} edges")
    print(f"Time range: {edge_time.min().item():.0f} - {edge_time.max().item():.0f}")

    graph_data = {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_time": edge_time,
        "addr_index": addr_index,
        "num_nodes": len(addr_index),
        "num_edges": edge_index.shape[1],
    }

    return graph_data


def save_graph(graph_data: dict, output_path: str):
    """Save graph data to a .pt file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graph_data, output_path)
    print(f"Graph saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build temporal graph from ETC data")
    parser.add_argument("--input", required=True, help="Path to transactions CSV")
    parser.add_argument("--traces", default=None, help="Path to traces CSV")
    parser.add_argument("--token-transfers", default=None, help="Path to token transfers CSV")
    parser.add_argument("--output", required=True, help="Output .pt file path")
    args = parser.parse_args()

    graph_data = build_temporal_graph(
        tx_path=args.input,
        trace_path=args.traces,
        tt_path=args.token_transfers,
    )
    save_graph(graph_data, args.output)


if __name__ == "__main__":
    main()
