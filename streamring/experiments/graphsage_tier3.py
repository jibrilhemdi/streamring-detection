"""
GraphSAGE as Tier 3 backbone — shows StreamRing architecture is backbone-agnostic.

Compares SubGNN+CSP vs GraphSAGE in the same 3-tier streaming cascade:
- Same Tier 1 (XGBoost) and Tier 2 (SAGEConv) for both
- Tier 3 swapped: SubGNN+CSP vs GraphSAGE subgraph classifier
- Measures accuracy and latency for both configurations
"""

import os, sys, json, time, copy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

import xgboost as xgb

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.reproducibility import set_seed
from src.gnn_models.subgnn_encoder import SubGNNEncoder, FraudRingClassifier

base = project_root
table_dir = base / "results" / "tables"

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
# GraphSAGE Subgraph Classifier (Tier 3 backbone)
# ============================================================================

class GraphSAGETier3(nn.Module):
    """GraphSAGE for subgraph-level fraud ring classification.

    Matches SubGNN+CSP interface: forward(x, edge_index, batch, position_encoding=None)
    """

    def __init__(self, input_dim, hidden_dim=128, embedding_dim=64,
                 num_classes=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_d, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.pool_project = nn.Linear(hidden_dim, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, x, edge_index, batch, position_encoding=None):
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = global_mean_pool(h, batch) + global_max_pool(h, batch)
        h = self.pool_project(h)
        return self.classifier(h)


# ============================================================================
# TIER 2: SAGEConv Node Classifier (shared)
# ============================================================================

class SAGENodeClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, 0.2, training=self.training)
        return self.conv2(h, edge_index).squeeze(-1)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def extract_subgraphs(warmup_ei, node_features, fraud_set, num_nodes, feat_dim):
    """Extract subgraphs for Tier 3 training."""
    fraud_nodes_list = list(fraud_set)
    benign_nodes_list = [n for n in range(num_nodes) if n not in fraud_set]

    n_sub = min(100, len(fraud_nodes_list))
    sampled_fraud = np.random.choice(fraud_nodes_list, n_sub, replace=False)
    sampled_benign = np.random.choice(benign_nodes_list, n_sub, replace=False)

    subs, labels = [], []
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
            subs.append(d)
            labels.append(label)
        except Exception:
            continue
    return subs, labels


def train_tier3_model(model, train_subs, train_labels, n_epochs=60, lr=1e-3):
    """Train a Tier 3 model (works for both SubGNN+CSP and GraphSAGE)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    n_total = len(train_labels)
    weight = torch.tensor([n_total/(2*max(n_neg,1)), n_total/(2*max(n_pos,1))], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_state = None
    best_loss = float('inf')

    model.train()
    for epoch in range(n_epochs):
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
                logits = model(batch.x, batch.edge_index, batch.batch, position_encoding=pe)
                loss = criterion(logits, batch_labels_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            except Exception:
                continue
        scheduler.step()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def evaluate_tier3_cascade(tier1_model, tier2_model, tier3_model, tier1_threshold,
                           sorted_src, sorted_dst, sorted_time, warmup_n, num_edges,
                           fraud_set, patterns, node_features, feat_dim, num_nodes,
                           model_name="model"):
    """Run streaming evaluation with a given Tier 3 model. Returns results dict."""

    def get_edge_features(src, dst, patterns_dict, nf):
        src_p = patterns_dict.get(int(src), np.zeros(12))
        dst_p = patterns_dict.get(int(dst), np.zeros(12))
        src_nf = nf[src].numpy() if src < nf.shape[0] else np.zeros(nf.shape[1])
        dst_nf = nf[dst].numpy() if dst < nf.shape[0] else np.zeros(nf.shape[1])
        return np.concatenate([src_p, dst_p, src_nf, dst_nf])

    eval_n = num_edges - warmup_n
    tier1_latencies, tier2_latencies, tier3_latencies = [], [], []
    tp, fp, tn, fn = 0, 0, 0, 0
    tier_dist = {"tier1_safe": 0, "tier2_suspicious": 0,
                 "tier3_fraud": 0, "tier3_likely_fraud": 0}

    running_src = sorted_src[:warmup_n].tolist()
    running_dst = sorted_dst[:warmup_n].tolist()

    stream_start = time.perf_counter()

    for idx in range(warmup_n, num_edges):
        src_i = int(sorted_src[idx])
        dst_i = int(sorted_dst[idx])
        is_fraud = (src_i in fraud_set or dst_i in fraud_set)

        running_src.append(src_i)
        running_dst.append(dst_i)

        # Tier 1
        t0 = time.perf_counter()
        feat = get_edge_features(src_i, dst_i, patterns, node_features)
        t1_score = float(tier1_model.predict_proba(feat.reshape(1, -1))[0][1])
        tier1_latencies.append((time.perf_counter() - t0) * 1000)

        if t1_score < tier1_threshold:
            tier_dist["tier1_safe"] += 1
            if is_fraud: fn += 1
            else: tn += 1
            continue

        # Tier 2
        t0 = time.perf_counter()
        with torch.no_grad():
            n_recent = min(len(running_src), warmup_n + 10000)
            recent_ei = torch.tensor([running_src[-n_recent:], running_dst[-n_recent:]], dtype=torch.long)
            try:
                logits = tier2_model(node_features, recent_ei)
                src_sc = float(torch.sigmoid(logits[src_i])) if src_i < logits.shape[0] else 0.5
                dst_sc = float(torch.sigmoid(logits[dst_i])) if dst_i < logits.shape[0] else 0.5
                t2_score = max(src_sc, dst_sc)
            except Exception:
                t2_score = 0.5
        tier2_latencies.append((time.perf_counter() - t0) * 1000)

        if t2_score < 0.5:
            tier_dist["tier2_suspicious"] += 1
            if is_fraud: fn += 1
            else: tn += 1
            continue

        # Tier 3
        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                full_ei = torch.tensor([running_src[-n_recent:], running_dst[-n_recent:]], dtype=torch.long)
                current_nn = max(max(running_src[-n_recent:]), max(running_dst[-n_recent:])) + 1
                subset, sub_ei, _, _ = k_hop_subgraph(
                    src_i, 2, full_ei, relabel_nodes=True, num_nodes=current_nn)

                if len(subset) < 3 or sub_ei.shape[1] < 2:
                    t3_fraud = False
                else:
                    if len(subset) > 300:
                        subset = subset[:300]
                        mask = (sub_ei[0] < 300) & (sub_ei[1] < 300)
                        sub_ei = sub_ei[:, mask]

                    x = node_features[subset] if subset.max() < node_features.shape[0] else \
                        torch.randn(len(subset), feat_dim)
                    d = Data(x=x, edge_index=sub_ei)
                    pe = compute_position_encoding(d)
                    batch_t = torch.zeros(len(subset), dtype=torch.long)
                    logits = tier3_model(x, sub_ei, batch_t, position_encoding=pe)
                    probs = F.softmax(logits, dim=1)
                    t3_fraud = float(probs[0, 1]) >= 0.5
            except Exception:
                t3_fraud = False

        tier3_latencies.append((time.perf_counter() - t0) * 1000)

        if t3_fraud:
            tier_dist["tier3_fraud"] += 1
            if is_fraud: tp += 1
            else: fp += 1
        else:
            tier_dist["tier3_likely_fraud"] += 1
            if is_fraud: fn += 1
            else: tn += 1

        done = idx - warmup_n + 1
        if done % 10000 == 0:
            elapsed = time.perf_counter() - stream_start
            print(f"  [{model_name}] {done}/{eval_n} ({done/elapsed:.0f} edges/s)", flush=True)

    elapsed = time.perf_counter() - stream_start
    throughput = eval_n / elapsed

    t3_p50 = float(np.percentile(tier3_latencies, 50)) if tier3_latencies else 0
    t3_p99 = float(np.percentile(tier3_latencies, 99)) if tier3_latencies else 0
    precision = tp / max(tier_dist["tier3_fraud"], 1) * 100

    return {
        "model": model_name,
        "throughput": round(throughput, 1),
        "tier3_latency_p50": round(t3_p50, 3),
        "tier3_latency_p99": round(t3_p99, 3),
        "tier3_meets_target": t3_p50 < 500.0,
        "detections": tier_dist["tier3_fraud"],
        "precision": round(precision, 1),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "tier_distribution": tier_dist,
        "filter_rate_tier1": round(tier_dist["tier1_safe"] / eval_n * 100, 2),
        "filter_rate_tier12": round((tier_dist["tier1_safe"] + tier_dist["tier2_suspicious"]) / eval_n * 100, 2),
    }


def main():
    print("=" * 70)
    print("GraphSAGE vs SubGNN+CSP as Tier 3 Backbone")
    print("=" * 70, flush=True)

    # Load data
    gd = torch.load(base / "data" / "graphs" / "dao_hack_graph.pt", weights_only=False)
    labels = torch.load(base / "data" / "processed" / "dao_hack_labels.pt", weights_only=False)
    patterns = torch.load(base / "data" / "processed" / "dao_hack_patterns.pt", weights_only=False)

    edge_index = gd["edge_index"]
    node_features = gd["node_features"]
    edge_time = gd["edge_time"]
    num_nodes = gd["num_nodes"]
    num_edges = edge_index.shape[1]
    feat_dim = node_features.shape[1]

    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    for nid, lbl in labels.items():
        if nid < num_nodes:
            node_labels[nid] = lbl
    fraud_set = set(n for n, l in labels.items() if l == 1 and n < num_nodes)

    time_order = edge_time.argsort()
    sorted_src = edge_index[0][time_order]
    sorted_dst = edge_index[1][time_order]
    sorted_time = edge_time[time_order]

    warmup_n = int(num_edges * 0.6)
    eval_n = num_edges - warmup_n
    warmup_ei = torch.stack([sorted_src[:warmup_n], sorted_dst[:warmup_n]])

    print(f"Graph: {num_nodes} nodes, {num_edges} edges, fraud: {len(fraud_set)}")
    print(f"Warmup: {warmup_n} edges, Eval: {eval_n} edges\n", flush=True)

    # ========================================================================
    # Train shared Tier 1 + Tier 2
    # ========================================================================
    print("Training shared Tier 1 (XGBoost)...", flush=True)
    def get_edge_features(src, dst, patterns_dict, nf):
        src_p = patterns_dict.get(int(src), np.zeros(12))
        dst_p = patterns_dict.get(int(dst), np.zeros(12))
        src_nf = nf[src].numpy() if src < nf.shape[0] else np.zeros(nf.shape[1])
        dst_nf = nf[dst].numpy() if dst < nf.shape[0] else np.zeros(nf.shape[1])
        return np.concatenate([src_p, dst_p, src_nf, dst_nf])

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
    scale_pos = (len(tier1_y) - tier1_y.sum()) / max(tier1_y.sum(), 1)

    tier1_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos, eval_metric="logloss",
        random_state=SEED, verbosity=0, n_jobs=1, tree_method="hist"
    )
    tier1_model.fit(tier1_X, tier1_y)

    tier1_probs = tier1_model.predict_proba(tier1_X)[:, 1]
    fpr, tpr, thresholds = roc_curve(tier1_y, tier1_probs)
    tier1_threshold = float(thresholds[np.argmax(tpr - fpr)])
    print(f"  Tier 1 threshold: {tier1_threshold:.4f}", flush=True)

    print("Training shared Tier 2 (SAGEConv)...", flush=True)
    tier2_model = SAGENodeClassifier(feat_dim, hidden=64)
    tier2_opt = torch.optim.Adam(tier2_model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_pos = node_labels.sum().item()
    pos_wt = torch.tensor([(num_nodes - n_pos) / max(n_pos, 1)], dtype=torch.float32)
    t2_crit = nn.BCEWithLogitsLoss(pos_weight=pos_wt)

    tier2_model.train()
    for epoch in range(30):
        logits = tier2_model(node_features, warmup_ei)
        loss = t2_crit(logits, node_labels.float())
        tier2_opt.zero_grad()
        loss.backward()
        tier2_opt.step()
    tier2_model.eval()
    print("  Tier 2 trained (30 epochs)", flush=True)

    # ========================================================================
    # Extract shared subgraphs
    # ========================================================================
    print("\nExtracting subgraphs for Tier 3 training...", flush=True)
    train_subs, train_labels = extract_subgraphs(
        warmup_ei, node_features, fraud_set, num_nodes, feat_dim)
    print(f"  {len(train_subs)} subgraphs ({sum(train_labels)} fraud)", flush=True)

    # ========================================================================
    # Train and evaluate: SubGNN+CSP
    # ========================================================================
    print("\n--- Training SubGNN+CSP (Tier 3) ---", flush=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    subgnn_model = FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2)
    subgnn_model = train_tier3_model(subgnn_model, train_subs, train_labels)
    print("  SubGNN+CSP trained", flush=True)

    print("\nStreaming with SubGNN+CSP...", flush=True)
    subgnn_results = evaluate_tier3_cascade(
        tier1_model, tier2_model, subgnn_model, tier1_threshold,
        sorted_src, sorted_dst, sorted_time, warmup_n, num_edges,
        fraud_set, patterns, node_features, feat_dim, num_nodes,
        model_name="SubGNN+CSP")

    # ========================================================================
    # Train and evaluate: GraphSAGE
    # ========================================================================
    print("\n--- Training GraphSAGE (Tier 3) ---", flush=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    sage_model = GraphSAGETier3(feat_dim, hidden_dim=128, embedding_dim=64,
                                num_classes=2, num_layers=2, dropout=0.2)
    sage_model = train_tier3_model(sage_model, train_subs, train_labels)
    print("  GraphSAGE trained", flush=True)

    print("\nStreaming with GraphSAGE...", flush=True)
    sage_results = evaluate_tier3_cascade(
        tier1_model, tier2_model, sage_model, tier1_threshold,
        sorted_src, sorted_dst, sorted_time, warmup_n, num_edges,
        fraud_set, patterns, node_features, feat_dim, num_nodes,
        model_name="GraphSAGE")

    # ========================================================================
    # COMPARE
    # ========================================================================
    print("\n" + "=" * 70)
    print("TIER 3 BACKBONE COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'SubGNN+CSP':>15} {'GraphSAGE':>15}")
    print("-" * 60)
    print(f"{'Throughput (edges/s)':<30} {subgnn_results['throughput']:>15} {sage_results['throughput']:>15}")
    print(f"{'Tier 3 Latency P50 (ms)':<30} {subgnn_results['tier3_latency_p50']:>15.3f} {sage_results['tier3_latency_p50']:>15.3f}")
    print(f"{'Tier 3 Latency P99 (ms)':<30} {subgnn_results['tier3_latency_p99']:>15.3f} {sage_results['tier3_latency_p99']:>15.3f}")
    print(f"{'Meets <500ms target':<30} {str(subgnn_results['tier3_meets_target']):>15} {str(sage_results['tier3_meets_target']):>15}")
    print(f"{'Detections':<30} {subgnn_results['detections']:>15} {sage_results['detections']:>15}")
    print(f"{'Precision (%)':<30} {subgnn_results['precision']:>15.1f} {sage_results['precision']:>15.1f}")
    print(f"{'TP':<30} {subgnn_results['confusion']['tp']:>15} {sage_results['confusion']['tp']:>15}")
    print(f"{'FP':<30} {subgnn_results['confusion']['fp']:>15} {sage_results['confusion']['fp']:>15}")
    print(f"{'Filter rate T1 (%)':<30} {subgnn_results['filter_rate_tier1']:>15.1f} {sage_results['filter_rate_tier1']:>15.1f}")
    print(f"{'Filter rate T1+2 (%)':<30} {subgnn_results['filter_rate_tier12']:>15.1f} {sage_results['filter_rate_tier12']:>15.1f}")
    print("=" * 70, flush=True)

    # Save
    combined = {
        "SubGNN+CSP": subgnn_results,
        "GraphSAGE": sage_results,
        "shared_config": {
            "tier1_threshold": tier1_threshold,
            "warmup_edges": warmup_n,
            "eval_edges": eval_n,
            "train_subgraphs": len(train_subs),
            "period": "dao_hack",
        }
    }
    with open(table_dir / "graphsage_tier3_results.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved to {table_dir / 'graphsage_tier3_results.json'}")


if __name__ == "__main__":
    main()
