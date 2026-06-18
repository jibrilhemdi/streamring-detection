"""
StreamRing Streaming Simulation — Full 3-tier cascading evaluation on DAO Hack period.

Simulates real-time transaction processing with:
- Tier 1: XGBoost on IFASI pattern features (<5ms target)
- Tier 2: SAGEConv node scoring (<50ms target)
- Tier 3: SubGNN+CSP subgraph classification (<500ms target)

Measures per-tier latency, throughput, filter rates, and fraud detection accuracy.
"""

import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, Batch

import xgboost as xgb

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.reproducibility import set_seed
from src.gnn_models.subgnn_encoder import SubGNNEncoder, FraudRingClassifier

base = project_root
table_dir = base / "results" / "tables"
fig_dir = base / "results" / "figures"

SEED = 42
set_seed(SEED)


# ============================================================================
# POSITION ENCODING
# ============================================================================

def compute_position_encoding(data, n_anchors=16):
    """Anchor-based BFS position encoding."""
    from torch_geometric.utils import to_networkx
    import networkx as nx
    G = to_networkx(data, to_undirected=True)
    n = data.x.size(0)
    if n < 2:
        return torch.zeros(n, n_anchors)
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    anchors = sorted_nodes[:min(n_anchors, len(sorted_nodes))]
    pos_enc = torch.full((n, n_anchors), float(n), dtype=torch.float)
    for ai, anchor in enumerate(anchors):
        try:
            dists = nx.single_source_shortest_path_length(G, anchor)
            for node, dist in dists.items():
                if node < n:
                    pos_enc[node, ai] = dist
        except Exception:
            pass
    max_dist = pos_enc.max()
    if max_dist > 0:
        pos_enc = 1.0 - (pos_enc / max_dist)
    pos_enc[pos_enc < 0] = 0
    return pos_enc


# ============================================================================
# TIER 2: Simple SAGEConv Node Classifier
# ============================================================================

class SAGENodeClassifier(nn.Module):
    """2-layer GraphSAGE for node-level fraud scoring."""
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, 0.2, training=self.training)
        return self.conv2(h, edge_index).squeeze(-1)


# ============================================================================
# STREAMING SIMULATION
# ============================================================================

def run_streaming_simulation():
    print("=" * 70)
    print("StreamRing Streaming Simulation — DAO Hack Period")
    print("=" * 70, flush=True)

    # Load DAO Hack graph
    gd = torch.load(base / "data" / "graphs" / "dao_hack_graph.pt", weights_only=False)
    labels = torch.load(base / "data" / "processed" / "dao_hack_labels.pt", weights_only=False)
    patterns = torch.load(base / "data" / "processed" / "dao_hack_patterns.pt", weights_only=False)

    edge_index = gd["edge_index"]
    node_features = gd["node_features"]
    edge_time = gd["edge_time"]
    num_nodes = gd["num_nodes"]
    num_edges = edge_index.shape[1]

    # Node-level labels
    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    for nid, lbl in labels.items():
        if nid < num_nodes:
            node_labels[nid] = lbl
    fraud_set = set(n for n, l in labels.items() if l == 1 and n < num_nodes)

    print(f"Graph: {num_nodes} nodes, {num_edges} edges")
    print(f"Fraud nodes: {len(fraud_set)}")
    print(f"Edge time range: {edge_time.min():.0f} to {edge_time.max():.0f}", flush=True)

    # Sort edges by timestamp for streaming
    time_order = edge_time.argsort()
    sorted_src = edge_index[0][time_order]
    sorted_dst = edge_index[1][time_order]
    sorted_time = edge_time[time_order]

    # ========================================================================
    # PHASE 1: Train models on first 60% of edges (warmup), evaluate on rest
    # ========================================================================

    warmup_n = int(num_edges * 0.6)
    eval_n = num_edges - warmup_n
    print(f"\nPhase 1: Training on {warmup_n} warmup edges, evaluating on {eval_n} edges", flush=True)

    # --- Tier 1: XGBoost on pattern features ---
    print("\nTraining Tier 1 (XGBoost)...", flush=True)

    # Build pattern features for edges
    def get_edge_features(src, dst, patterns_dict, nf):
        """Build Tier 1 features: concatenated src+dst pattern features."""
        src_p = patterns_dict.get(int(src), np.zeros(12))
        dst_p = patterns_dict.get(int(dst), np.zeros(12))
        if hasattr(src_p, 'numpy'):
            src_p = src_p
        src_nf = nf[src].numpy() if src < nf.shape[0] else np.zeros(nf.shape[1])
        dst_nf = nf[dst].numpy() if dst < nf.shape[0] else np.zeros(nf.shape[1])
        return np.concatenate([src_p, dst_p, src_nf, dst_nf])

    # Build training data for Tier 1
    tier1_X, tier1_y = [], []
    for i in range(warmup_n):
        src_i = int(sorted_src[i])
        dst_i = int(sorted_dst[i])
        feat = get_edge_features(src_i, dst_i, patterns, node_features)
        is_fraud = 1 if (src_i in fraud_set or dst_i in fraud_set) else 0
        tier1_X.append(feat)
        tier1_y.append(is_fraud)

    tier1_X = np.array(tier1_X)
    tier1_y = np.array(tier1_y)
    n_fraud_train = tier1_y.sum()
    n_benign_train = len(tier1_y) - n_fraud_train
    scale_pos = n_benign_train / max(n_fraud_train, 1)

    tier1_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        random_state=SEED, verbosity=0, n_jobs=1, tree_method="hist"
    )
    tier1_model.fit(tier1_X, tier1_y)
    print(f"  Tier 1 trained: {n_fraud_train} fraud / {n_benign_train} benign edges", flush=True)

    # --- Tier 2: SAGEConv node classifier ---
    print("Training Tier 2 (SAGEConv)...", flush=True)
    warmup_ei = torch.stack([sorted_src[:warmup_n], sorted_dst[:warmup_n]])

    tier2_model = SAGENodeClassifier(node_features.shape[1], hidden=64)
    tier2_optimizer = torch.optim.Adam(tier2_model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_pos = node_labels.sum().item()
    n_neg = num_nodes - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    tier2_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tier2_model.train()
    for epoch in range(30):
        logits = tier2_model(node_features, warmup_ei)
        loss = tier2_criterion(logits, node_labels.float())
        tier2_optimizer.zero_grad()
        loss.backward()
        tier2_optimizer.step()
    tier2_model.eval()
    print(f"  Tier 2 trained: 30 epochs", flush=True)

    # --- Tier 3: SubGNN+CSP subgraph classifier ---
    print("Training Tier 3 (SubGNN+CSP)...", flush=True)

    # Extract subgraphs from warmup period
    feat_dim = node_features.shape[1]
    fraud_nodes_list = list(fraud_set)
    benign_nodes_list = [n for n in range(num_nodes) if n not in fraud_set]

    n_sub = min(100, len(fraud_nodes_list))
    sampled_fraud = np.random.choice(fraud_nodes_list, n_sub, replace=False)
    sampled_benign = np.random.choice(benign_nodes_list, n_sub, replace=False)

    train_subs, train_labels = [], []
    for node_id, label in [(n, 1) for n in sampled_fraud] + [(n, 0) for n in sampled_benign]:
        try:
            subset, sub_ei, _, _ = k_hop_subgraph(
                int(node_id), 2, warmup_ei, relabel_nodes=True, num_nodes=num_nodes)
            if len(subset) < 3 or sub_ei.shape[1] < 2:
                continue
            if len(subset) > 300:
                subset = subset[:300]
                mask = (sub_ei[0] < 300) & (sub_ei[1] < 300)
                sub_ei = sub_ei[:, mask]
                if sub_ei.shape[1] < 2:
                    continue
            x = node_features[subset] if subset.max() < node_features.shape[0] else \
                torch.randn(len(subset), feat_dim)
            d = Data(x=x, edge_index=sub_ei)
            d.position_encoding = compute_position_encoding(d)
            train_subs.append(d)
            train_labels.append(label)
        except Exception:
            continue

    print(f"  Extracted {len(train_subs)} subgraphs ({sum(train_labels)} fraud)", flush=True)

    # Train Tier 3
    tier3_model = FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2)
    tier3_optimizer = torch.optim.Adam(tier3_model.parameters(), lr=1e-3, weight_decay=1e-4)
    tier3_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(tier3_optimizer, T_max=60)

    n_pos_t3 = sum(train_labels)
    n_neg_t3 = len(train_labels) - n_pos_t3
    n_total_t3 = len(train_labels)
    t3_weight = torch.tensor([n_total_t3/(2*max(n_neg_t3,1)), n_total_t3/(2*max(n_pos_t3,1))], dtype=torch.float32)
    tier3_criterion = nn.CrossEntropyLoss(weight=t3_weight)

    best_state = None
    best_loss = float('inf')

    tier3_model.train()
    for epoch in range(60):
        indices = list(range(len(train_subs)))
        np.random.shuffle(indices)
        epoch_loss = 0
        for i in range(0, len(indices), 32):
            batch_idx = indices[i:i+32]
            batch_data = [train_subs[j] for j in batch_idx]
            batch_labels_t = torch.tensor([train_labels[j] for j in batch_idx], dtype=torch.long)
            try:
                batch = Batch.from_data_list(batch_data)
                pe = batch.position_encoding if hasattr(batch, 'position_encoding') else None
                logits = tier3_model(batch.x, batch.edge_index, batch.batch, position_encoding=pe)
                loss = tier3_criterion(logits, batch_labels_t)
                tier3_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tier3_model.parameters(), 1.0)
                tier3_optimizer.step()
                epoch_loss += loss.item()
            except Exception:
                continue
        tier3_scheduler.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(tier3_model.state_dict())

    if best_state:
        tier3_model.load_state_dict(best_state)
    tier3_model.eval()
    print(f"  Tier 3 trained: 60 epochs", flush=True)

    # Determine Tier 1 threshold using Youden's J on warmup data
    tier1_probs = tier1_model.predict_proba(tier1_X)[:, 1]
    if len(np.unique(tier1_y)) > 1:
        fpr, tpr, thresholds = roc_curve(tier1_y, tier1_probs)
        j_scores = tpr - fpr
        tier1_threshold = float(thresholds[np.argmax(j_scores)])
    else:
        tier1_threshold = 0.3
    print(f"\nTier 1 threshold (Youden's J): {tier1_threshold:.4f}", flush=True)

    # Tier 2 threshold
    tier2_threshold = 0.5

    # ========================================================================
    # PHASE 2: Stream evaluation edges through 3-tier cascade
    # ========================================================================

    print(f"\nPhase 2: Streaming {eval_n} edges through 3-tier cascade...", flush=True)

    # Per-tier statistics
    tier1_latencies = []
    tier2_latencies = []
    tier3_latencies = []
    tier_distribution = {"tier1_safe": 0, "tier2_suspicious": 0,
                         "tier3_fraud": 0, "tier3_likely_fraud": 0}
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Incrementally grown edge index for Tier 2/3
    running_src = sorted_src[:warmup_n].tolist()
    running_dst = sorted_dst[:warmup_n].tolist()

    stream_start = time.perf_counter()

    for idx in range(warmup_n, num_edges):
        src_i = int(sorted_src[idx])
        dst_i = int(sorted_dst[idx])
        is_fraud_edge = (src_i in fraud_set or dst_i in fraud_set)

        # Update running graph
        running_src.append(src_i)
        running_dst.append(dst_i)

        # === TIER 1: XGBoost ===
        t0 = time.perf_counter()
        feat = get_edge_features(src_i, dst_i, patterns, node_features)
        tier1_score = float(tier1_model.predict_proba(feat.reshape(1, -1))[0][1])
        t1_latency = (time.perf_counter() - t0) * 1000
        tier1_latencies.append(t1_latency)

        if tier1_score < tier1_threshold:
            tier_distribution["tier1_safe"] += 1
            if is_fraud_edge:
                false_negatives += 1
            else:
                true_negatives += 1
            continue

        # === TIER 2: SAGEConv ===
        t0 = time.perf_counter()
        with torch.no_grad():
            # Use warmup graph + some recent edges for efficiency
            n_recent = min(len(running_src), warmup_n + 10000)
            recent_ei = torch.tensor([running_src[-n_recent:], running_dst[-n_recent:]], dtype=torch.long)
            try:
                logits = tier2_model(node_features, recent_ei)
                src_score = float(torch.sigmoid(logits[src_i])) if src_i < logits.shape[0] else 0.5
                dst_score = float(torch.sigmoid(logits[dst_i])) if dst_i < logits.shape[0] else 0.5
                tier2_score = max(src_score, dst_score)
            except Exception:
                tier2_score = 0.5
        t2_latency = (time.perf_counter() - t0) * 1000
        tier2_latencies.append(t2_latency)

        if tier2_score < tier2_threshold:
            tier_distribution["tier2_suspicious"] += 1
            if is_fraud_edge:
                false_negatives += 1
            else:
                true_negatives += 1
            continue

        # === TIER 3: SubGNN+CSP ===
        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                full_ei = torch.tensor([running_src[-n_recent:], running_dst[-n_recent:]], dtype=torch.long)
                current_num_nodes = max(max(running_src[-n_recent:]), max(running_dst[-n_recent:])) + 1
                subset, sub_ei, _, _ = k_hop_subgraph(
                    src_i, 2, full_ei, relabel_nodes=True, num_nodes=current_num_nodes)

                if len(subset) < 3 or sub_ei.shape[1] < 2:
                    tier3_fraud = False
                    tier3_conf = 0.0
                else:
                    if len(subset) > 300:
                        subset = subset[:300]
                        mask = (sub_ei[0] < 300) & (sub_ei[1] < 300)
                        sub_ei = sub_ei[:, mask]

                    x = node_features[subset] if subset.max() < node_features.shape[0] else \
                        torch.randn(len(subset), feat_dim)
                    d = Data(x=x, edge_index=sub_ei)
                    pe = compute_position_encoding(d)
                    batch_idx_t = torch.zeros(len(subset), dtype=torch.long)
                    logits = tier3_model(x, sub_ei, batch_idx_t, position_encoding=pe)
                    probs = F.softmax(logits, dim=1)
                    tier3_conf = float(probs[0, 1])
                    tier3_fraud = tier3_conf >= 0.5
            except Exception:
                tier3_fraud = False
                tier3_conf = 0.0

        t3_latency = (time.perf_counter() - t0) * 1000
        tier3_latencies.append(t3_latency)

        if tier3_fraud:
            tier_distribution["tier3_fraud"] += 1
            if is_fraud_edge:
                true_positives += 1
            else:
                false_positives += 1
        else:
            tier_distribution["tier3_likely_fraud"] += 1
            if is_fraud_edge:
                false_negatives += 1
            else:
                true_negatives += 1

        # Progress
        done = idx - warmup_n + 1
        if done % 5000 == 0:
            elapsed = time.perf_counter() - stream_start
            rate = done / elapsed
            print(f"  [{done}/{eval_n}] {rate:.0f} edges/s, "
                  f"T1={tier_distribution['tier1_safe']}, T2={tier_distribution['tier2_suspicious']}, "
                  f"T3F={tier_distribution['tier3_fraud']}, T3L={tier_distribution['tier3_likely_fraud']}",
                  flush=True)

    stream_elapsed = time.perf_counter() - stream_start
    throughput = eval_n / stream_elapsed

    # ========================================================================
    # COMPILE RESULTS
    # ========================================================================

    t1_p50 = float(np.percentile(tier1_latencies, 50)) if tier1_latencies else 0
    t1_p99 = float(np.percentile(tier1_latencies, 99)) if tier1_latencies else 0
    t2_p50 = float(np.percentile(tier2_latencies, 50)) if tier2_latencies else 0
    t2_p99 = float(np.percentile(tier2_latencies, 99)) if tier2_latencies else 0
    t3_p50 = float(np.percentile(tier3_latencies, 50)) if tier3_latencies else 0
    t3_p99 = float(np.percentile(tier3_latencies, 99)) if tier3_latencies else 0

    total_detections = tier_distribution["tier3_fraud"]
    precision = true_positives / max(total_detections, 1) * 100

    tier1_filter_rate = tier_distribution["tier1_safe"] / eval_n * 100
    tier12_filter_rate = (tier_distribution["tier1_safe"] + tier_distribution["tier2_suspicious"]) / eval_n * 100

    results = {
        "throughput": round(throughput, 1),
        "total_eval_edges": eval_n,
        "elapsed_seconds": round(stream_elapsed, 1),
        "tier_distribution": tier_distribution,
        "filter_rates": {
            "tier1": round(tier1_filter_rate, 2),
            "tier1_2": round(tier12_filter_rate, 2),
        },
        "latency": {
            "tier1_p50": round(t1_p50, 3),
            "tier1_p99": round(t1_p99, 3),
            "tier2_p50": round(t2_p50, 3),
            "tier2_p99": round(t2_p99, 3),
            "tier3_p50": round(t3_p50, 3),
            "tier3_p99": round(t3_p99, 3),
        },
        "latency_targets": {
            "tier1_met": t1_p50 < 5.0,
            "tier2_met": t2_p50 < 50.0,
            "tier3_met": t3_p50 < 500.0,
        },
        "detections": total_detections,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": round(precision, 1),
        "confusion": {
            "tp": true_positives,
            "fp": false_positives,
            "tn": true_negatives,
            "fn": false_negatives,
        },
        "tier1_threshold": round(tier1_threshold, 4),
        "counts": {
            "tier1_edges": len(tier1_latencies),
            "tier2_edges": len(tier2_latencies),
            "tier3_edges": len(tier3_latencies),
        },
    }

    print("\n" + "=" * 70)
    print("STREAMING SIMULATION RESULTS")
    print("=" * 70)
    print(f"Throughput:        {throughput:.1f} edges/sec")
    print(f"Total eval edges:  {eval_n}")
    print(f"Elapsed:           {stream_elapsed:.1f}s")
    print(f"\nTier Distribution:")
    print(f"  Tier 1 (safe):         {tier_distribution['tier1_safe']}")
    print(f"  Tier 2 (suspicious):   {tier_distribution['tier2_suspicious']}")
    print(f"  Tier 3 (fraud ring):   {tier_distribution['tier3_fraud']}")
    print(f"  Tier 3 (likely fraud): {tier_distribution['tier3_likely_fraud']}")
    print(f"\nFilter Rates:")
    print(f"  Tier 1:   {tier1_filter_rate:.1f}%")
    print(f"  Tier 1+2: {tier12_filter_rate:.1f}%")
    print(f"\nLatency (P50 / P99):")
    print(f"  Tier 1: {t1_p50:.3f}ms / {t1_p99:.3f}ms  (target <5ms)    {'✓' if t1_p50 < 5 else '✗'}")
    print(f"  Tier 2: {t2_p50:.3f}ms / {t2_p99:.3f}ms  (target <50ms)   {'✓' if t2_p50 < 50 else '✗'}")
    print(f"  Tier 3: {t3_p50:.3f}ms / {t3_p99:.3f}ms  (target <500ms)  {'✓' if t3_p50 < 500 else '✗'}")
    print(f"\nDetection:")
    print(f"  Fraud rings detected: {total_detections}")
    print(f"  True positives:       {true_positives}")
    print(f"  False positives:      {false_positives}")
    print(f"  Precision:            {precision:.1f}%")
    print(f"  TP+FN (total fraud):  {true_positives + false_negatives}")
    print("=" * 70, flush=True)

    with open(table_dir / "streaming_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {table_dir / 'streaming_results.json'}")

    return results


if __name__ == "__main__":
    run_streaming_simulation()
