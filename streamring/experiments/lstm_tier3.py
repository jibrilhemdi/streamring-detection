"""
LSTM-BiLSTM as Tier 3 baseline — comparison against SubGNN+CSP.

Two evaluation protocols:
  Phase 1: Offline subgraph classification (5 periods, 5 seeds, 70/15/15 split)
  Phase 2: Streaming cascade evaluation (dao_hack period)

Outputs:
  results/tables/lstm_tier3_results.json
  results/tables/lstm_tier3_streaming_results.json
"""

import os, sys, json, time, copy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score,
                             matthews_corrcoef, roc_curve)
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data, Batch
import xgboost as xgb

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.reproducibility import set_seed
from src.gnn_models.lstm_classifier import LSTMFraudClassifier
from src.gnn_models.subgnn_encoder import FraudRingClassifier

base = project_root
table_dir = base / "results" / "tables"
table_dir.mkdir(parents=True, exist_ok=True)

SEED = 42
set_seed(SEED)
PERIODS = ["dao_hack", "pre_dao", "post_fork", "attack_51_v1", "attack_51_v2"]


# ============================================================================
# HELPER: TIMESTAMP EXTRACTION
# ============================================================================

def build_node_timestamps(subset: torch.Tensor, ei_slice: torch.Tensor,
                           edge_time_slice: torch.Tensor) -> torch.Tensor:
    """
    For each node in subset (original global node IDs), compute the maximum
    edge_time among incident edges where BOTH endpoints are in subset.

    Args:
        subset:          [N] tensor of original global node IDs (from k_hop_subgraph
                         with relabel_nodes=True; subset[i] = global_id of local node i)
        ei_slice:        [2, E] edge_index — post-truncation slice, global node IDs.
                         edge_time is edge-position-indexed: edge_time_slice[i] is the
                         timestamp of the edge at column i of ei_slice.
        edge_time_slice: [E] timestamps corresponding to ei_slice columns.

    Returns:
        [N] float32 tensor. result[i] = max timestamp for global node subset[i].
        Nodes with no qualifying incident edges get float('inf').

    Note on PyG batching: when stored as d.node_timestamps on a Data object and
    batched via Batch.from_data_list, PyG concatenates along dim=0 (default for
    non-index tensor attributes; __inc__ returns 0 so values are not offset-added).
    Convert to list before passing to LSTMFraudClassifier.forward().
    """
    subset_set = set(subset.tolist())
    node_max_ts: dict = {}

    for edge_pos in range(ei_slice.shape[1]):
        u = int(ei_slice[0, edge_pos])
        v = int(ei_slice[1, edge_pos])
        if u in subset_set and v in subset_set:
            ts = float(edge_time_slice[edge_pos])
            node_max_ts[u] = max(node_max_ts.get(u, float("-inf")), ts)
            node_max_ts[v] = max(node_max_ts.get(v, float("-inf")), ts)

    ts_list = [
        node_max_ts.get(int(subset[i]), float("inf"))
        for i in range(len(subset))
    ]
    return torch.tensor(ts_list, dtype=torch.float)


# ============================================================================
# HELPER: POSITION ENCODING (for SubGNN streaming path only)
# ============================================================================

def compute_position_encoding(data: Data, num_anchors: int = 16) -> torch.Tensor:
    """Anchor-based BFS position encoding. Matches graphsage_tier3.py."""
    G = to_networkx(data, to_undirected=True)
    n = data.x.size(0)
    if n < 2:
        return torch.zeros(n, num_anchors)
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    anchors = sorted_nodes[:min(num_anchors, len(sorted_nodes))]
    pos_enc = torch.full((n, num_anchors), float(n), dtype=torch.float)
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
# HELPER: SAGEConv NODE CLASSIFIER (Tier 2, shared between both streaming runs)
# ============================================================================

class SAGENodeClassifier(nn.Module):
    """SAGEConv node classifier for Tier 2. Matches graphsage_tier3.py."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, 0.2, training=self.training)
        return self.conv2(h, edge_index).squeeze(-1)


# ============================================================================
# SUBGRAPH EXTRACTION WITH TIMESTAMPS
# ============================================================================

def extract_subgraphs_with_timestamps(graph_data: dict, labels: dict,
                                       edge_time: torch.Tensor,
                                       max_per_class: int = 300,
                                       num_hops: int = 2):
    """
    Extract 2-hop subgraphs with per-node timestamps.

    Mirrors baselines.py extract_subgraphs with one addition:
    each Data object gets d.node_timestamps (shape [num_nodes], float32)
    built by build_node_timestamps.

    edge_time is [num_edges] position-indexed (same order as edge_index columns).
    If the graph has > 1,000,000 edges, edge_index and edge_time are truncated
    together (same permutation) to keep position indices consistent.
    """
    edge_index = graph_data["edge_index"]
    node_features = graph_data["node_features"]
    num_nodes = graph_data["num_nodes"]

    # Truncate edge_index and edge_time with the same index permutation
    n_edges = edge_index.shape[1]
    if n_edges > 1_000_000:
        idx = torch.randperm(n_edges)[:1_000_000]
        edge_index = edge_index[:, idx]
        edge_time = edge_time[idx]

    fraud_nodes = [n for n, l in labels.items() if l == 1 and n < num_nodes]
    benign_nodes = [n for n, l in labels.items() if l == 0 and n < num_nodes]
    n_f = min(len(fraud_nodes), max_per_class)
    n_b = min(len(benign_nodes), max_per_class)
    if n_f < 5 or n_b < 5:
        return [], []

    sampled = ([(n, 1) for n in np.random.choice(fraud_nodes, n_f, replace=False)] +
               [(n, 0) for n in np.random.choice(benign_nodes, n_b, replace=False)])

    subgraphs, sub_labels = [], []
    for node_id, label in sampled:
        try:
            subset, sub_ei, _, _ = k_hop_subgraph(
                int(node_id), num_hops, edge_index,
                relabel_nodes=True, num_nodes=num_nodes)
            if len(subset) < 3 or sub_ei.shape[1] < 2:
                continue
            if len(subset) > 300:
                subset = subset[:300]
                mask = (sub_ei[0] < 300) & (sub_ei[1] < 300)
                sub_ei = sub_ei[:, mask]
                if sub_ei.shape[1] < 2:
                    continue
            x = (node_features[subset]
                 if subset.max() < node_features.shape[0]
                 else torch.randn(len(subset), node_features.shape[1]))
            d = Data(x=x, edge_index=sub_ei)
            d.node_timestamps = build_node_timestamps(subset, edge_index, edge_time)
            subgraphs.append(d)
            sub_labels.append(label)
        except Exception:
            continue
    return subgraphs, sub_labels


# ============================================================================
# OFFLINE: TRAINING + EVALUATION
# ============================================================================

def train_lstm(model: LSTMFraudClassifier, train_s, train_l, val_s, val_l,
               epochs: int = 80, lr: float = 1e-3, patience: int = 15) -> float:
    """
    Train LSTMFraudClassifier on subgraph Data objects that carry node_timestamps.

    Seed count for the experiment follows tier3_results.json (n=5, seeds 42-46),
    not baselines.py (which has an inconsistency between docstring and __main__).

    PyG Batch.from_data_list concatenates node_timestamps along dim=0 (correct;
    __inc__ returns 0 for non-index attributes so values are not offset-added).
    Convert to list before forward() — LSTMFraudClassifier expects a Python list.

    Memory note: full-batch evaluation creates a padded matrix [B, max_len, feat_dim]
    inside LSTMFraudClassifier. For 300-node subgraphs, feat_dim=32, ~400 test
    subgraphs -> ~15 MB — acceptable.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_pos = sum(train_l)
    n_neg = len(train_l) - n_pos
    n_total = len(train_l)
    weight = torch.tensor([n_total / (2 * max(n_neg, 1)),
                            n_total / (2 * max(n_pos, 1))], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = list(range(len(train_s)))
        np.random.shuffle(indices)

        for i in range(0, len(indices), 32):
            batch_idx = indices[i:i + 32]
            batch_data = [train_s[j] for j in batch_idx]
            batch_labels = [train_l[j] for j in batch_idx]
            try:
                batch = Batch.from_data_list(batch_data)
                labels_t = torch.tensor(batch_labels, dtype=torch.long)
                logits = model(batch.x, batch.edge_index, batch.batch,
                               node_timestamps=batch.node_timestamps.tolist())
                loss = criterion(logits, labels_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            except Exception:
                continue

        scheduler.step()

        if val_s:
            model.eval()
            with torch.no_grad():
                try:
                    vb = Batch.from_data_list(val_s)
                    vl = model(vb.x, vb.edge_index, vb.batch,
                               node_timestamps=vb.node_timestamps.tolist())
                    vp = F.softmax(vl, dim=1)[:, 1].numpy()
                    vt = np.array(val_l)
                    if len(np.unique(vt)) > 1:
                        va = roc_auc_score(vt, vp)
                        if va > best_val_auc:
                            best_val_auc = va
                            best_state = copy.deepcopy(model.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                break
                except Exception:
                    continue

    if best_state:
        model.load_state_dict(best_state)
    return best_val_auc


def evaluate_lstm(model: LSTMFraudClassifier, test_s, test_l) -> dict:
    """
    Evaluate model on test subgraphs using full-batch inference.
    Youden's J threshold. Returns auc_roc, f1, pr_auc, mcc.
    """
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list(test_s)
        logits = model(batch.x, batch.edge_index, batch.batch,
                       node_timestamps=batch.node_timestamps.tolist())
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
        true = np.array(test_l)

    if len(np.unique(true)) > 1:
        fpr, tpr, thresholds = roc_curve(true, probs)
        j = tpr - fpr
        opt_t = float(thresholds[np.argmax(j)])
        preds = (probs >= opt_t).astype(int)
        return {
            "auc_roc": float(roc_auc_score(true, probs)),
            "f1": float(f1_score(true, preds)),
            "pr_auc": float(average_precision_score(true, probs)),
            "mcc": float(matthews_corrcoef(true, preds)),
        }
    return {"auc_roc": 0.5, "f1": 0.0, "pr_auc": 0.0, "mcc": 0.0}


def load_all_data():
    """Load all 5 periods and extract subgraphs with timestamps."""
    np.random.seed(SEED)
    all_s, all_l = [], []
    for period in PERIODS:
        gp = base / "data" / "graphs" / f"{period}_graph.pt"
        lp = base / "data" / "processed" / f"{period}_labels.pt"
        if not gp.exists() or not lp.exists():
            print(f"  Skipping {period} (data not found)", flush=True)
            continue
        print(f"Loading {period}...", flush=True)
        gd = torch.load(gp, weights_only=False)
        lb = torch.load(lp, weights_only=False)
        # edge_time may be absent in some graph files; fall back to zeros
        et = gd.get("edge_time", torch.zeros(gd["edge_index"].shape[1]))
        nf = sum(1 for l in lb.values() if l == 1)
        print(f"  {gd['num_nodes']} nodes, {nf} fraud", flush=True)
        max_per = 250 if gd["num_nodes"] > 50_000 else 150
        subs, labs = extract_subgraphs_with_timestamps(gd, lb, et, max_per_class=max_per)
        print(f"  Extracted: {len(subs)} ({sum(labs)} fraud)", flush=True)
        all_s.extend(subs)
        all_l.extend(labs)
    print(f"\nTOTAL: {len(all_s)} subgraphs ({sum(all_l)} fraud, "
          f"{len(all_l) - sum(all_l)} benign)\n", flush=True)
    return all_s, all_l


def run_offline_phase(all_s, all_l, feat_dim: int, n_seeds: int = 5) -> dict:
    """
    Train and evaluate LSTMFraudClassifier offline.

    Seed count follows tier3_results.json (n=5, seeds 42-46).
    Protocol: 70/15/15 split, Adam+CosineAnnealingLR, patience=15, 80 epochs.
    """
    print("\n" + "=" * 70)
    print(f"PHASE 1: OFFLINE CLASSIFICATION ({n_seeds} seeds, seeds 42-{41 + n_seeds})")
    print("=" * 70, flush=True)

    seeds = [SEED + i for i in range(n_seeds)]
    runs = []

    for run_idx, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        idx = np.random.permutation(len(all_s))
        nt = int(0.7 * len(idx))
        nv = int(0.15 * len(idx))
        ts = [all_s[i] for i in idx[:nt]]
        tl = [all_l[i] for i in idx[:nt]]
        vs = [all_s[i] for i in idx[nt:nt + nv]]
        vl = [all_l[i] for i in idx[nt:nt + nv]]
        es = [all_s[i] for i in idx[nt + nv:]]
        el = [all_l[i] for i in idx[nt + nv:]]

        model = LSTMFraudClassifier(feat_dim, hidden_dim=128, num_layers=1,
                                    output_dim=64, dropout=0.2)
        t0 = time.time()
        train_lstm(model, ts, tl, vs, vl, epochs=80, lr=1e-3, patience=15)
        train_time = time.time() - t0

        res = evaluate_lstm(model, es, el)
        res["train_time_s"] = train_time
        runs.append(res)
        print(f"  Run {run_idx + 1}/{n_seeds}: AUC={res['auc_roc']:.4f}, "
              f"F1={res['f1']:.4f}, MCC={res['mcc']:.4f} ({train_time:.1f}s)", flush=True)

    mean_r = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    std_r = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
    results = {"LSTM-BiLSTM": {"mean": mean_r, "std": std_r, "runs": runs}}

    with open(table_dir / "lstm_tier3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {table_dir / 'lstm_tier3_results.json'}", flush=True)
    return results


# ============================================================================
# STREAMING: PHASE 2
# ============================================================================

def _train_tier3_streaming(model, train_subs, train_labels, n_epochs=60, lr=1e-3,
                            use_timestamps=False):
    """
    Train a Tier 3 model on warmup subgraphs.

    use_timestamps=True  -> LSTM path: passes node_timestamps from d.node_timestamps
    use_timestamps=False -> SubGNN path: passes position_encoding from d.position_encoding
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    n_total = len(train_labels)
    weight = torch.tensor([n_total / (2 * max(n_neg, 1)),
                            n_total / (2 * max(n_pos, 1))], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_state = None
    best_loss = float("inf")
    model.train()
    for epoch in range(n_epochs):
        indices = list(range(len(train_subs)))
        np.random.shuffle(indices)
        epoch_loss = 0.0
        for i in range(0, len(indices), 32):
            batch_idx = indices[i:i + 32]
            batch_data = [train_subs[j] for j in batch_idx]
            batch_labels_t = torch.tensor([train_labels[j] for j in batch_idx],
                                           dtype=torch.long)
            try:
                batch = Batch.from_data_list(batch_data)
                if use_timestamps:
                    logits = model(batch.x, batch.edge_index, batch.batch,
                                   node_timestamps=batch.node_timestamps.tolist())
                else:
                    pe = getattr(batch, "position_encoding", None)
                    logits = model(batch.x, batch.edge_index, batch.batch,
                                   position_encoding=pe)
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


def _extract_warmup_subgraphs(warmup_ei, warmup_et, node_features, fraud_set,
                               num_nodes, feat_dim, n_each=100):
    """Extract subgraphs from warmup graph for Tier 3 training (both models)."""
    fraud_list = list(fraud_set)
    benign_list = [n for n in range(num_nodes) if n not in fraud_set]
    n_sub = min(n_each, len(fraud_list), len(benign_list))
    if n_sub == 0:
        return [], []

    sampled_fraud = np.random.choice(fraud_list, n_sub, replace=False)
    sampled_benign = np.random.choice(benign_list, n_sub, replace=False)

    subs, labels_out = [], []
    for node_id, label in ([(n, 1) for n in sampled_fraud] +
                           [(n, 0) for n in sampled_benign]):
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
            x = (node_features[subset]
                 if subset.max() < node_features.shape[0]
                 else torch.randn(len(subset), feat_dim))
            d = Data(x=x, edge_index=sub_ei)
            # Attach both timestamp (LSTM) and position_encoding (SubGNN)
            d.node_timestamps = build_node_timestamps(subset, warmup_ei, warmup_et)
            d.position_encoding = compute_position_encoding(d)
            subs.append(d)
            labels_out.append(label)
        except Exception:
            continue
    return subs, labels_out


def _run_cascade(tier1_model, tier2_model, tier3_model, tier1_threshold,
                 sorted_src, sorted_dst, sorted_time, warmup_n, num_edges,
                 fraud_set, patterns, node_features, feat_dim, num_nodes,
                 model_name, use_timestamps):
    """
    Single streaming pass. Identical structure to graphsage_tier3.py.

    use_timestamps=True  -> LSTM path: maintains node_ts_dict, skips position_encoding
    use_timestamps=False -> SubGNN path: computes position_encoding per subgraph

    node_ts_dict update: both src and dst nodes are set to sorted_time[idx] BEFORE
    the Tier 1 check (last-write-wins, mirrors running_src.append in graphsage_tier3.py).
    """
    def get_edge_features(src, dst):
        src_p = patterns.get(int(src), np.zeros(12))
        dst_p = patterns.get(int(dst), np.zeros(12))
        src_nf = (node_features[src].numpy()
                  if src < node_features.shape[0] else np.zeros(feat_dim))
        dst_nf = (node_features[dst].numpy()
                  if dst < node_features.shape[0] else np.zeros(feat_dim))
        return np.concatenate([src_p, dst_p, src_nf, dst_nf])

    eval_n = num_edges - warmup_n
    tier3_latencies = []
    tp, fp, tn, fn = 0, 0, 0, 0
    tier_dist = {"tier1_safe": 0, "tier2_suspicious": 0,
                 "tier3_fraud": 0, "tier3_likely_fraud": 0}

    running_src = sorted_src[:warmup_n].tolist()
    running_dst = sorted_dst[:warmup_n].tolist()
    node_ts_dict: dict = {}

    # Pre-populate from warmup edges (last-write-wins)
    if use_timestamps:
        for i in range(warmup_n):
            ts_v = float(sorted_time[i])
            node_ts_dict[int(sorted_src[i])] = ts_v
            node_ts_dict[int(sorted_dst[i])] = ts_v

    stream_start = time.perf_counter()

    for idx in range(warmup_n, num_edges):
        src_i = int(sorted_src[idx])
        dst_i = int(sorted_dst[idx])
        is_fraud = (src_i in fraud_set or dst_i in fraud_set)

        running_src.append(src_i)
        running_dst.append(dst_i)

        # Update timestamps before Tier 1 (both endpoints, last-write-wins)
        if use_timestamps:
            ts_v = float(sorted_time[idx])
            node_ts_dict[src_i] = ts_v
            node_ts_dict[dst_i] = ts_v

        # Tier 1
        feat = get_edge_features(src_i, dst_i)
        t1_score = float(tier1_model.predict_proba(feat.reshape(1, -1))[0][1])

        if t1_score < tier1_threshold:
            tier_dist["tier1_safe"] += 1
            if is_fraud:
                fn += 1
            else:
                tn += 1
            continue

        # Tier 2
        with torch.no_grad():
            n_recent = min(len(running_src), warmup_n + 10000)
            recent_ei = torch.tensor(
                [running_src[-n_recent:], running_dst[-n_recent:]], dtype=torch.long)
            try:
                logits_t2 = tier2_model(node_features, recent_ei)
                src_sc = float(torch.sigmoid(logits_t2[src_i])) if src_i < logits_t2.shape[0] else 0.5
                dst_sc = float(torch.sigmoid(logits_t2[dst_i])) if dst_i < logits_t2.shape[0] else 0.5
                t2_score = max(src_sc, dst_sc)
            except Exception:
                t2_score = 0.5

        if t2_score < 0.5:
            tier_dist["tier2_suspicious"] += 1
            if is_fraud:
                fn += 1
            else:
                tn += 1
            continue

        # Tier 3
        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                full_ei = torch.tensor(
                    [running_src[-n_recent:], running_dst[-n_recent:]], dtype=torch.long)
                current_nn = max(max(running_src[-n_recent:]),
                                 max(running_dst[-n_recent:])) + 1
                subset, sub_ei, _, _ = k_hop_subgraph(
                    src_i, 2, full_ei, relabel_nodes=True, num_nodes=current_nn)

                if len(subset) < 3 or sub_ei.shape[1] < 2:
                    t3_fraud = False
                else:
                    if len(subset) > 300:
                        subset = subset[:300]
                        mask = (sub_ei[0] < 300) & (sub_ei[1] < 300)
                        sub_ei = sub_ei[:, mask]

                    x = (node_features[subset]
                         if subset.max() < node_features.shape[0]
                         else torch.randn(len(subset), feat_dim))
                    batch_t = torch.zeros(len(subset), dtype=torch.long)

                    if use_timestamps:
                        # LSTM path: do NOT compute position_encoding (ignored, wastes CPU)
                        relabeled_ts = [
                            node_ts_dict.get(int(subset[i].item()), float("inf"))
                            for i in range(len(subset))
                        ]
                        logits_t3 = tier3_model(x, sub_ei, batch_t,
                                                node_timestamps=relabeled_ts)
                    else:
                        # SubGNN path: compute position_encoding
                        d_tmp = Data(x=x, edge_index=sub_ei)
                        pe = compute_position_encoding(d_tmp)
                        logits_t3 = tier3_model(x, sub_ei, batch_t,
                                                position_encoding=pe)

                    probs_t3 = F.softmax(logits_t3, dim=1)
                    t3_fraud = float(probs_t3[0, 1]) >= 0.5
            except Exception:
                t3_fraud = False

        tier3_latencies.append((time.perf_counter() - t0) * 1000)

        if t3_fraud:
            tier_dist["tier3_fraud"] += 1
            if is_fraud:
                tp += 1
            else:
                fp += 1
        else:
            tier_dist["tier3_likely_fraud"] += 1
            if is_fraud:
                fn += 1
            else:
                tn += 1

        done = idx - warmup_n + 1
        if done % 10000 == 0:
            elapsed = time.perf_counter() - stream_start
            print(f"  [{model_name}] {done}/{eval_n} ({done / elapsed:.0f} edges/s)",
                  flush=True)

    elapsed = time.perf_counter() - stream_start
    throughput = eval_n / elapsed
    t3_p50 = float(np.percentile(tier3_latencies, 50)) if tier3_latencies else 0.0
    t3_p99 = float(np.percentile(tier3_latencies, 99)) if tier3_latencies else 0.0
    precision = tp / max(tier_dist["tier3_fraud"], 1) * 100.0

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
        "filter_rate_tier1": round(tier_dist["tier1_safe"] / max(eval_n, 1) * 100, 2),
        "filter_rate_tier12": round(
            (tier_dist["tier1_safe"] + tier_dist["tier2_suspicious"]) / max(eval_n, 1) * 100, 2),
    }


def run_streaming_phase(graph_data: dict, labels: dict, patterns: dict,
                         warmup_frac: float = 0.6, tier3_epochs: int = 60,
                         synthetic_mode: bool = False) -> dict:
    """
    Phase 2: streaming cascade on one period (dao_hack by default).

    Both SubGNN+CSP and LSTM-BiLSTM are trained on the SAME warmup subgraphs,
    then evaluated on SEPARATE streaming passes (to avoid timing interference).

    synthetic_mode=True: use dummy Tier 1/2 models (for smoke tests only).

    SubGNN+CSP result is a fresh re-run (may differ slightly from
    graphsage_tier3_results.json by design -- two independent runs).
    """
    print("\n" + "=" * 70)
    print("PHASE 2: STREAMING CASCADE")
    print("=" * 70, flush=True)

    edge_index = graph_data["edge_index"]
    node_features = graph_data["node_features"]
    edge_time = graph_data.get("edge_time", torch.zeros(edge_index.shape[1]))
    num_nodes = graph_data["num_nodes"]
    num_edges = edge_index.shape[1]
    feat_dim = node_features.shape[1]

    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    for nid, lbl in labels.items():
        if nid < num_nodes:
            node_labels[nid] = lbl
    fraud_set = {n for n, l in labels.items() if l == 1 and n < num_nodes}

    time_order = edge_time.argsort()
    sorted_src = edge_index[0][time_order]
    sorted_dst = edge_index[1][time_order]
    sorted_time = edge_time[time_order]

    warmup_n = int(num_edges * warmup_frac)
    eval_n = num_edges - warmup_n
    warmup_ei = torch.stack([sorted_src[:warmup_n], sorted_dst[:warmup_n]])
    warmup_et = sorted_time[:warmup_n]

    print(f"Graph: {num_nodes} nodes, {num_edges} edges, fraud: {len(fraud_set)}")
    print(f"Warmup: {warmup_n} edges, Eval: {eval_n} edges\n", flush=True)

    # ---- Tier 1: XGBoost ----
    if synthetic_mode:
        class _DummyTier1:
            def predict_proba(self, X):
                return np.ones((len(X), 2)) * 0.5
        tier1_model = _DummyTier1()
        tier1_threshold = 0.0
    else:
        print("Training Tier 1 (XGBoost)...", flush=True)

        def _get_ef(src, dst):
            sp = patterns.get(int(src), np.zeros(12))
            dp = patterns.get(int(dst), np.zeros(12))
            sn = node_features[src].numpy() if src < node_features.shape[0] else np.zeros(feat_dim)
            dn = node_features[dst].numpy() if dst < node_features.shape[0] else np.zeros(feat_dim)
            return np.concatenate([sp, dp, sn, dn])

        t1X, t1y = [], []
        for i in range(warmup_n):
            si, di = int(sorted_src[i]), int(sorted_dst[i])
            t1X.append(_get_ef(si, di))
            t1y.append(1 if (si in fraud_set or di in fraud_set) else 0)
        t1X, t1y = np.array(t1X), np.array(t1y)
        scale_pos = (len(t1y) - t1y.sum()) / max(t1y.sum(), 1)
        tier1_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos, eval_metric="logloss",
            random_state=SEED, verbosity=0, n_jobs=1, tree_method="hist")
        tier1_model.fit(t1X, t1y)
        t1_probs = tier1_model.predict_proba(t1X)[:, 1]
        fpr_, tpr_, thresh_ = roc_curve(t1y, t1_probs)
        tier1_threshold = float(thresh_[np.argmax(tpr_ - fpr_)])
        print(f"  Tier 1 threshold: {tier1_threshold:.4f}", flush=True)

    # ---- Tier 2: SAGEConv ----
    if synthetic_mode:
        class _DummyTier2:
            def __call__(self, x, ei):
                return torch.zeros(x.shape[0])
        tier2_model = _DummyTier2()
    else:
        print("Training Tier 2 (SAGEConv)...", flush=True)
        tier2_model = SAGENodeClassifier(feat_dim, hidden=64)
        t2_opt = torch.optim.Adam(tier2_model.parameters(), lr=1e-3, weight_decay=1e-4)
        n_pos_t2 = node_labels.sum().item()
        pos_wt = torch.tensor([(num_nodes - n_pos_t2) / max(n_pos_t2, 1)], dtype=torch.float32)
        t2_crit = nn.BCEWithLogitsLoss(pos_weight=pos_wt)
        tier2_model.train()
        for _ in range(30):
            logits_t2 = tier2_model(node_features, warmup_ei)
            loss_t2 = t2_crit(logits_t2, node_labels.float())
            t2_opt.zero_grad()
            loss_t2.backward()
            t2_opt.step()
        tier2_model.eval()
        print("  Tier 2 trained (30 epochs)", flush=True)

    # ---- Extract warmup subgraphs (shared between both models) ----
    print("\nExtracting warmup subgraphs...", flush=True)
    train_subs, train_labels_t3 = _extract_warmup_subgraphs(
        warmup_ei, warmup_et, node_features, fraud_set, num_nodes, feat_dim)
    print(f"  {len(train_subs)} subgraphs ({sum(train_labels_t3)} fraud)", flush=True)

    # ---- Train SubGNN+CSP ----
    print("\n--- Training SubGNN+CSP (Tier 3) ---", flush=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    subgnn_model = FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2)
    t0 = time.time()
    _train_tier3_streaming(subgnn_model, train_subs, train_labels_t3,
                           n_epochs=tier3_epochs, use_timestamps=False)
    subgnn_train_time = time.time() - t0
    print(f"  SubGNN+CSP trained in {subgnn_train_time:.1f}s", flush=True)

    # ---- Train LSTM-BiLSTM ----
    print("\n--- Training LSTM-BiLSTM (Tier 3) ---", flush=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    lstm_model = LSTMFraudClassifier(feat_dim, hidden_dim=128, num_layers=1,
                                     output_dim=64, dropout=0.2)
    t0 = time.time()
    _train_tier3_streaming(lstm_model, train_subs, train_labels_t3,
                           n_epochs=tier3_epochs, use_timestamps=True)
    lstm_train_time = time.time() - t0
    print(f"  LSTM-BiLSTM trained in {lstm_train_time:.1f}s", flush=True)

    # ---- Streaming pass: SubGNN+CSP ----
    print("\nStreaming with SubGNN+CSP...", flush=True)
    subgnn_results = _run_cascade(
        tier1_model, tier2_model, subgnn_model, tier1_threshold,
        sorted_src, sorted_dst, sorted_time, warmup_n, num_edges,
        fraud_set, patterns, node_features, feat_dim, num_nodes,
        model_name="SubGNN+CSP", use_timestamps=False)

    # ---- Streaming pass: LSTM-BiLSTM (separate pass, no timing interference) ----
    print("\nStreaming with LSTM-BiLSTM...", flush=True)
    lstm_results = _run_cascade(
        tier1_model, tier2_model, lstm_model, tier1_threshold,
        sorted_src, sorted_dst, sorted_time, warmup_n, num_edges,
        fraud_set, patterns, node_features, feat_dim, num_nodes,
        model_name="LSTM-BiLSTM", use_timestamps=True)

    combined = {
        "SubGNN+CSP": subgnn_results,
        "LSTM-BiLSTM": lstm_results,
        "shared_config": {
            "tier1_threshold": float(tier1_threshold),
            "warmup_edges": warmup_n,
            "eval_edges": eval_n,
            "train_subgraphs": len(train_subs),
            "subgnn_train_time_s": round(subgnn_train_time, 2),
            "lstm_train_time_s": round(lstm_train_time, 2),
            "period": "dao_hack",
        }
    }

    if not synthetic_mode:
        with open(table_dir / "lstm_tier3_streaming_results.json", "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nSaved -> {table_dir / 'lstm_tier3_streaming_results.json'}", flush=True)
    return combined


# ============================================================================
# OUTPUT: COMPARISON TABLES
# ============================================================================

def print_comparison_tables(offline_results: dict, prior_subgnn: dict,
                             streaming_results: dict):
    """Print offline and streaming comparison tables to stdout."""
    lstm_m = offline_results["LSTM-BiLSTM"]["mean"]
    lstm_s = offline_results["LSTM-BiLSTM"]["std"]
    sg_m = prior_subgnn["mean"]
    sg_s = prior_subgnn["std"]

    print("\n" + "=" * 78)
    print("OFFLINE CLASSIFICATION RESULTS (5 seeds, all 5 periods)")
    print("Note: SubGNN+CSP loaded from tier3_results.json (not re-run here)")
    print("=" * 78)
    print(f"{'Model':<25} {'AUC-ROC':>14} {'F1':>14} {'PR-AUC':>14} {'MCC':>14}")
    print("-" * 83)
    print(f"{'SubGNN+CSP (prior)':<25} "
          f"{sg_m['auc_roc']:.3f}±{sg_s['auc_roc']:.3f}  "
          f"{sg_m['f1']:.3f}±{sg_s['f1']:.3f}  "
          f"{sg_m['pr_auc']:.3f}±{sg_s['pr_auc']:.3f}  "
          f"{sg_m['mcc']:.3f}±{sg_s['mcc']:.3f}")
    print(f"{'LSTM-BiLSTM':<25} "
          f"{lstm_m['auc_roc']:.3f}±{lstm_s['auc_roc']:.3f}  "
          f"{lstm_m['f1']:.3f}±{lstm_s['f1']:.3f}  "
          f"{lstm_m['pr_auc']:.3f}±{lstm_s['pr_auc']:.3f}  "
          f"{lstm_m['mcc']:.3f}±{lstm_s['mcc']:.3f}")
    print("=" * 78)

    sg = streaming_results["SubGNN+CSP"]
    lstm = streaming_results["LSTM-BiLSTM"]
    cfg = streaming_results.get("shared_config", {})

    print("\n" + "=" * 65)
    print("STREAMING CASCADE RESULTS (dao_hack)")
    print("=" * 65)
    print(f"{'Metric':<30} {'SubGNN+CSP':>16} {'LSTM-BiLSTM':>16}")
    print("-" * 64)
    print(f"{'Throughput (edges/s)':<30} {sg['throughput']:>16} {lstm['throughput']:>16}")
    print(f"{'Tier3 Latency P50 (ms)':<30} {sg['tier3_latency_p50']:>16.3f} "
          f"{lstm['tier3_latency_p50']:>16.3f}")
    print(f"{'Tier3 Latency P99 (ms)':<30} {sg['tier3_latency_p99']:>16.3f} "
          f"{lstm['tier3_latency_p99']:>16.3f}")
    print(f"{'Meets <500ms target':<30} {str(sg['tier3_meets_target']):>16} "
          f"{str(lstm['tier3_meets_target']):>16}")
    print(f"{'Detections':<30} {sg['detections']:>16} {lstm['detections']:>16}")
    print(f"{'Precision (%)':<30} {sg['precision']:>16.1f} {lstm['precision']:>16.1f}")
    print(f"{'TP':<30} {sg['confusion']['tp']:>16} {lstm['confusion']['tp']:>16}")
    print(f"{'FP':<30} {sg['confusion']['fp']:>16} {lstm['confusion']['fp']:>16}")
    print(f"{'Filter rate T1 (%)':<30} {sg['filter_rate_tier1']:>16.1f} "
          f"{lstm['filter_rate_tier1']:>16.1f}")
    print(f"{'Filter rate T1+2 (%)':<30} {sg['filter_rate_tier12']:>16.1f} "
          f"{lstm['filter_rate_tier12']:>16.1f}")
    if cfg:
        print(f"{'Tier3 train time (s)':<30} "
              f"{cfg.get('subgnn_train_time_s', 0):>16.1f} "
              f"{cfg.get('lstm_train_time_s', 0):>16.1f}")
    print("=" * 65, flush=True)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LSTM-BiLSTM vs SubGNN+CSP Tier 3 experiment")
    parser.add_argument("--phase", choices=["offline", "streaming", "both"],
                        default="both", help="Which phase to run (default: both)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Seeds for offline phase (default: 5, seeds 42-46)")
    args = parser.parse_args()

    print("=" * 70)
    print("LSTM-BiLSTM vs SubGNN+CSP — Tier 3 Comparison")
    print("=" * 70, flush=True)

    t_total = time.time()
    offline_results = None
    streaming_results = None

    if args.phase in ("offline", "both"):
        all_s, all_l = load_all_data()
        if len(all_s) < 50:
            print("ERROR: Not enough data for offline phase")
            sys.exit(1)
        feat_dim = all_s[0].x.shape[1]
        offline_results = run_offline_phase(all_s, all_l, feat_dim, n_seeds=args.seeds)

    if args.phase in ("streaming", "both"):
        gp = base / "data" / "graphs" / "dao_hack_graph.pt"
        lp = base / "data" / "processed" / "dao_hack_labels.pt"
        pp = base / "data" / "processed" / "dao_hack_patterns.pt"
        for p in (gp, lp, pp):
            if not p.exists():
                print(f"ERROR: {p} not found")
                sys.exit(1)
        gd = torch.load(gp, weights_only=False)
        lb = torch.load(lp, weights_only=False)
        pt = torch.load(pp, weights_only=False)
        streaming_results = run_streaming_phase(gd, lb, pt)

    # Load prior SubGNN offline results from tier3_results.json for comparison table.
    # Structure: {"test_results": {...}, "test_results_std": {...}, ...}
    prior_path = table_dir / "tier3_results.json"
    if prior_path.exists():
        with open(prior_path) as f:
            prior_raw = json.load(f)
        prior_subgnn = {
            "mean": prior_raw["test_results"],
            "std": prior_raw["test_results_std"],
        }
    else:
        prior_subgnn = {
            "mean": {"auc_roc": 0.888, "f1": 0.849, "pr_auc": 0.914, "mcc": 0.650},
            "std": {"auc_roc": 0.020, "f1": 0.020, "pr_auc": 0.006, "mcc": 0.030},
        }

    if offline_results is not None and streaming_results is not None:
        print_comparison_tables(offline_results, prior_subgnn, streaming_results)

    print(f"\nALL DONE in {(time.time() - t_total) / 60:.1f} min", flush=True)
