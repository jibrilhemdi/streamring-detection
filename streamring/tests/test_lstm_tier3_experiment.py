"""Smoke tests for lstm_tier3.py helper functions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch


def test_build_node_timestamps_basic():
    """Nodes with incident edges get their max timestamp."""
    from experiments.lstm_tier3 import build_node_timestamps
    # subset = global IDs [10, 20, 30]; edges: 10-20 at t=5, 20-30 at t=8
    subset = torch.tensor([10, 20, 30])
    ei = torch.tensor([[10, 20], [20, 30]])
    et = torch.tensor([5.0, 8.0])
    result = build_node_timestamps(subset, ei, et)
    assert result.shape == (3,)
    assert float(result[0]) == 5.0   # node 10: max(5) = 5
    assert float(result[1]) == 8.0   # node 20: max(5, 8) = 8
    assert float(result[2]) == 8.0   # node 30: max(8) = 8


def test_build_node_timestamps_no_edges():
    """Node with no incident edges (in both-endpoints check) gets float('inf')."""
    from experiments.lstm_tier3 import build_node_timestamps
    subset = torch.tensor([5, 99])   # node 99 has no edges
    ei = torch.tensor([[5], [5]])    # self-loop on 5, neither endpoint is 99
    et = torch.tensor([3.0])
    result = build_node_timestamps(subset, ei, et)
    assert result[1].item() == float("inf")


def test_build_node_timestamps_edges_outside_subset_ignored():
    """Edges where only one endpoint is in subset are ignored."""
    from experiments.lstm_tier3 import build_node_timestamps
    subset = torch.tensor([1, 2])
    # Edge 1-2 at t=10 (both in subset) counts
    # Edge 1-3 at t=99 (node 3 NOT in subset) ignored
    ei = torch.tensor([[1, 1], [2, 3]])
    et = torch.tensor([10.0, 99.0])
    result = build_node_timestamps(subset, ei, et)
    assert float(result[0]) == 10.0  # node 1: only edge 1-2 counts
    assert float(result[1]) == 10.0  # node 2: only edge 1-2 counts


def test_extract_subgraphs_attaches_node_timestamps():
    """extract_subgraphs_with_timestamps attaches node_timestamps to each Data object."""
    from experiments.lstm_tier3 import extract_subgraphs_with_timestamps
    num_nodes = 20
    src = list(range(19)) + list(range(1, 20))
    dst = list(range(1, 20)) + list(range(19))
    edge_index = torch.tensor([src, dst])
    edge_time = torch.arange(len(src), dtype=torch.float)
    node_features = torch.randn(num_nodes, 8)
    graph_data = {
        "edge_index": edge_index,
        "node_features": node_features,
        "num_nodes": num_nodes,
    }
    # Provide enough labels (at least 5 per class for max_per_class=5)
    labels = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0}
    subs, labs = extract_subgraphs_with_timestamps(
        graph_data, labels, edge_time, max_per_class=5, num_hops=1)
    assert len(subs) > 0
    for d in subs:
        assert hasattr(d, "node_timestamps"), "Missing node_timestamps on Data object"
        assert d.node_timestamps.shape == (d.x.shape[0],), (
            f"node_timestamps shape {d.node_timestamps.shape} != num_nodes {d.x.shape[0]}")
        assert d.node_timestamps.dtype == torch.float32


def test_train_and_evaluate_lstm_smoke():
    """train_lstm + evaluate_lstm run without error on synthetic data."""
    from experiments.lstm_tier3 import train_lstm, evaluate_lstm
    from src.gnn_models.lstm_classifier import LSTMFraudClassifier
    from torch_geometric.data import Data

    torch.manual_seed(0)
    np.random.seed(0)
    feat_dim = 16
    n_nodes = 8

    def make_subgraph(label):
        x = torch.randn(n_nodes, feat_dim)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        ts = torch.arange(n_nodes, dtype=torch.float) * (1.0 if label == 1 else -1.0)
        d = Data(x=x, edge_index=ei)
        d.node_timestamps = ts
        return d

    subs = [make_subgraph(i % 2) for i in range(20)]
    labs = [i % 2 for i in range(20)]

    model = LSTMFraudClassifier(feat_dim, hidden_dim=32, num_layers=1,
                                output_dim=16, dropout=0.0)
    train_lstm(model, subs[:14], labs[:14], subs[14:17], labs[14:17],
               epochs=5, lr=1e-3, patience=10)
    result = evaluate_lstm(model, subs[17:], labs[17:])

    assert "auc_roc" in result
    assert "f1" in result
    assert 0.0 <= result["auc_roc"] <= 1.0


def test_run_streaming_smoke():
    """run_streaming_phase completes without error on tiny synthetic graph."""
    from experiments.lstm_tier3 import run_streaming_phase

    torch.manual_seed(0)
    np.random.seed(0)

    num_nodes = 30
    num_edges = 60
    feat_dim = 8
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst])
    edge_time = torch.arange(num_edges, dtype=torch.float)
    node_features = torch.randn(num_nodes, feat_dim)
    patterns = {i: np.zeros(12) for i in range(num_nodes)}
    fraud_set = {0, 1, 2}
    labels = {i: (1 if i in fraud_set else 0) for i in range(num_nodes)}

    graph_data = {
        "edge_index": edge_index,
        "node_features": node_features,
        "edge_time": edge_time,
        "num_nodes": num_nodes,
    }

    results = run_streaming_phase(
        graph_data, labels, patterns,
        warmup_frac=0.6, tier3_epochs=2, synthetic_mode=True)

    assert "SubGNN+CSP" in results
    assert "LSTM-BiLSTM" in results
    for key in ["throughput", "tier3_latency_p50", "confusion"]:
        assert key in results["SubGNN+CSP"], f"Missing key '{key}' in SubGNN+CSP"
        assert key in results["LSTM-BiLSTM"], f"Missing key '{key}' in LSTM-BiLSTM"


def test_print_comparison_tables_smoke():
    """print_comparison_tables runs without error on dummy data."""
    from experiments.lstm_tier3 import print_comparison_tables

    offline_results = {
        "LSTM-BiLSTM": {
            "mean": {"auc_roc": 0.85, "f1": 0.80, "pr_auc": 0.88, "mcc": 0.60},
            "std": {"auc_roc": 0.01, "f1": 0.01, "pr_auc": 0.01, "mcc": 0.02},
        }
    }
    prior_subgnn = {
        "mean": {"auc_roc": 0.888, "f1": 0.849, "pr_auc": 0.914, "mcc": 0.650},
        "std": {"auc_roc": 0.020, "f1": 0.020, "pr_auc": 0.006, "mcc": 0.030},
    }
    streaming_results = {
        "SubGNN+CSP": {
            "throughput": 300.0, "tier3_latency_p50": 8.0,
            "tier3_latency_p99": 50.0, "tier3_meets_target": True,
            "detections": 10, "precision": 80.0,
            "confusion": {"tp": 8, "fp": 2, "tn": 100, "fn": 3},
            "filter_rate_tier1": 60.0, "filter_rate_tier12": 80.0,
        },
        "LSTM-BiLSTM": {
            "throughput": 280.0, "tier3_latency_p50": 12.0,
            "tier3_latency_p99": 60.0, "tier3_meets_target": True,
            "detections": 9, "precision": 77.0,
            "confusion": {"tp": 7, "fp": 2, "tn": 100, "fn": 4},
            "filter_rate_tier1": 60.0, "filter_rate_tier12": 80.0,
        },
        "shared_config": {"subgnn_train_time_s": 30.0, "lstm_train_time_s": 25.0},
    }
    print_comparison_tables(offline_results, prior_subgnn, streaming_results)
    # No assertion needed — just must not raise
