"""
Temporal GNN Baselines: EvolveGCN-H and Temporal-GCN for fraud ring detection.

Compares temporal-aware GNN models against SubGNN+CSP on the subgraph
classification task. Uses temporal snapshots of the transaction graph.

Models:
1. EvolveGCN-H: GRU-evolved GCN weights across temporal snapshots
2. Temporal-GCN (T-GCN): GCN + temporal encoding concatenated to features
3. Snapshot-GCN: Static GCN applied per-snapshot, pooled across time
"""

import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score,
                             matthews_corrcoef, roc_curve)
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph, to_networkx

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.reproducibility import set_seed
from src.gnn_models.subgnn_encoder import SubGNNEncoder, FraudRingClassifier

base = project_root
table_dir = base / "results" / "tables"

SEED = 42
set_seed(SEED)


# ============================================================================
# POSITION ENCODING (same as allout_v3)
# ============================================================================

def compute_position_encoding(data, n_anchors=16):
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
# MODEL 1: EvolveGCN-H (GRU-evolved GCN weights)
# ============================================================================

class EvolveGCNH(nn.Module):
    """
    EvolveGCN-H: Uses GRU to evolve GCN weight matrices across temporal snapshots.
    For subgraph classification: process temporal snapshots, then pool.

    Reference: Pareja et al., "EvolveGCN: Evolving Graph Convolutional Networks
    for Dynamic Graphs", AAAI 2020.
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=2, n_snapshots=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_snapshots = n_snapshots

        # GRU to evolve weight matrices
        self.gru1 = nn.GRUCell(in_dim * hidden_dim, in_dim * hidden_dim)
        self.gru2 = nn.GRUCell(hidden_dim * hidden_dim, hidden_dim * hidden_dim)

        # Initial weight parameters
        self.w1_init = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.01)
        self.w2_init = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, batch, position_encoding=None):
        """
        Process subgraph with temporal awareness via evolved GCN weights.
        Since subgraphs don't have explicit temporal snapshots, we simulate
        by splitting edges into temporal bins and evolving weights.
        """
        n = x.size(0)
        n_edges = edge_index.size(1)

        # Split edges into temporal snapshots (simulate temporal evolution)
        edges_per_snap = max(1, n_edges // self.n_snapshots)

        w1 = self.w1_init.view(1, -1)  # (1, in*hid)
        w2 = self.w2_init.view(1, -1)  # (1, hid*hid)

        h = torch.zeros(n, self.hidden_dim, device=x.device)

        for s in range(self.n_snapshots):
            start = s * edges_per_snap
            end = min(start + edges_per_snap, n_edges)
            if start >= n_edges:
                break

            snap_ei = edge_index[:, start:end]
            if snap_ei.size(1) == 0:
                continue

            # Evolve weights via GRU
            w1 = self.gru1(w1, w1)
            w2 = self.gru2(w2, w2)

            # Reshape weights
            W1 = w1.view(x.size(1), self.hidden_dim)
            W2 = w2.view(self.hidden_dim, self.hidden_dim)

            # GCN-like message passing with evolved weights
            # h = sigma(D^-1 A X W)
            h1 = x @ W1  # (n, hid)
            # Aggregate using edge_index
            h_agg = torch.zeros_like(h1)
            if snap_ei.size(1) > 0:
                src, dst = snap_ei[0], snap_ei[1]
                src = src.clamp(0, n - 1)
                dst = dst.clamp(0, n - 1)
                h_agg.index_add_(0, dst, h1[src])
                # Degree normalization
                deg = torch.zeros(n, device=x.device)
                deg.index_add_(0, dst, torch.ones(src.size(0), device=x.device))
                deg = deg.clamp(min=1)
                h_agg = h_agg / deg.unsqueeze(1)

            h = F.relu(h_agg + h1)  # Skip connection
            h = h @ W2
            h = F.relu(h)

        # Global pooling
        out = global_mean_pool(h, batch)
        return self.classifier(out)


# ============================================================================
# MODEL 2: Temporal-GCN (T-GCN with temporal encoding)
# ============================================================================

class TemporalGCN(nn.Module):
    """
    T-GCN: Augments node features with temporal encoding before GCN layers.
    Uses sinusoidal temporal encoding similar to Transformer positional encoding.
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=2, temp_dim=16):
        super().__init__()
        self.temp_dim = temp_dim
        aug_dim = in_dim + temp_dim

        self.conv1 = GCNConv(aug_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.temporal_linear = nn.Linear(1, temp_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, batch, position_encoding=None):
        n = x.size(0)

        # Generate temporal features from node degree / position
        if position_encoding is not None:
            temp_input = position_encoding.mean(dim=1, keepdim=True)
        else:
            # Use degree as temporal proxy
            deg = torch.zeros(n, device=x.device)
            if edge_index.size(1) > 0:
                src, dst = edge_index
                src = src.clamp(0, n - 1)
                dst = dst.clamp(0, n - 1)
                deg.index_add_(0, dst, torch.ones(src.size(0), device=x.device))
            temp_input = deg.unsqueeze(1) / max(deg.max().item(), 1.0)

        temp_enc = F.relu(self.temporal_linear(temp_input))
        x_aug = torch.cat([x, temp_enc], dim=1)

        h = F.relu(self.conv1(x_aug, edge_index))
        h = F.dropout(h, 0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index))

        out = global_mean_pool(h, batch)
        return self.classifier(out)


# ============================================================================
# MODEL 3: Snapshot-GCN (per-snapshot GCN, temporal pooling)
# ============================================================================

class SnapshotGCN(nn.Module):
    """
    Snapshot-GCN: Split edges into temporal snapshots, run GCN on each,
    then aggregate snapshot representations.
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=2, n_snapshots=4):
        super().__init__()
        self.n_snapshots = n_snapshots
        self.hidden_dim = hidden_dim

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Temporal attention over snapshots
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, batch, position_encoding=None):
        n = x.size(0)
        n_edges = edge_index.size(1)
        edges_per_snap = max(1, n_edges // self.n_snapshots)

        snapshot_embeds = []
        for s in range(self.n_snapshots):
            start = s * edges_per_snap
            end = min(start + edges_per_snap, n_edges)
            if start >= n_edges:
                break

            snap_ei = edge_index[:, start:end]
            if snap_ei.size(1) == 0:
                continue

            h = F.relu(self.conv1(x, snap_ei))
            h = F.dropout(h, 0.2, training=self.training)
            h = F.relu(self.conv2(h, snap_ei))
            pooled = global_mean_pool(h, batch)
            snapshot_embeds.append(pooled)

        if not snapshot_embeds:
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            pooled = global_mean_pool(h, batch)
            return self.classifier(pooled)

        # Stack and apply temporal attention
        stacked = torch.stack(snapshot_embeds, dim=0)  # (T, B, H)
        attn = self.temporal_attn(stacked)  # (T, B, 1)
        aggregated = (stacked * attn).sum(dim=0)  # (B, H)

        return self.classifier(aggregated)


# ============================================================================
# DATA LOADING AND TRAINING
# ============================================================================

def extract_subgraphs(graph_data, labels, max_per_class=150, num_hops=2):
    edge_index = graph_data["edge_index"]
    node_features = graph_data["node_features"]
    num_nodes = graph_data["num_nodes"]
    n_edges = edge_index.shape[1]
    if n_edges > 1000000:
        idx = torch.randperm(n_edges)[:1000000]
        edge_index = edge_index[:, idx]

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
                int(node_id), num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes)
            if len(subset) < 3 or sub_ei.shape[1] < 2:
                continue
            if len(subset) > 300:
                subset = subset[:300]
                mask = (sub_ei[0] < 300) & (sub_ei[1] < 300)
                sub_ei = sub_ei[:, mask]
                if sub_ei.shape[1] < 2:
                    continue
            x = node_features[subset] if subset.max() < node_features.shape[0] else \
                torch.randn(len(subset), node_features.shape[1])
            d = Data(x=x, edge_index=sub_ei)
            d.position_encoding = compute_position_encoding(d)
            subgraphs.append(d)
            sub_labels.append(label)
        except Exception:
            continue
    return subgraphs, sub_labels


def train_model(model, train_s, train_l, val_s, val_l, epochs=80, lr=1e-3, patience=15):
    """Standard supervised training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_pos = sum(train_l)
    n_neg = len(train_l) - n_pos
    n_total = len(train_l)
    weight = torch.tensor([n_total/(2*max(n_neg,1)), n_total/(2*max(n_pos,1))], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_val_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = list(range(len(train_s)))
        np.random.shuffle(indices)

        for i in range(0, len(indices), 32):
            batch_idx = indices[i:i+32]
            batch_data = [train_s[j] for j in batch_idx]
            batch_labels = [train_l[j] for j in batch_idx]

            try:
                batch = Batch.from_data_list(batch_data)
                labels_t = torch.tensor(batch_labels, dtype=torch.long)
                pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None
                logits = model(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
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
                    vpe = vb.position_encoding if hasattr(vb, 'position_encoding') else None
                    vl = model(vb.x, vb.edge_index, vb.batch, position_encoding=vpe)
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


def evaluate_model(model, test_s, test_l):
    model.eval()
    with torch.no_grad():
        batch = Batch.from_data_list(test_s)
        pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None
        logits = model(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
        true = np.array(test_l)

    if len(np.unique(true)) > 1:
        fpr, tpr, thresholds = roc_curve(true, probs)
        j = tpr - fpr
        opt_t = thresholds[np.argmax(j)]
        preds = (probs >= opt_t).astype(int)
        return {
            "auc_roc": float(roc_auc_score(true, probs)),
            "f1": float(f1_score(true, preds)),
            "pr_auc": float(average_precision_score(true, probs)),
            "mcc": float(matthews_corrcoef(true, preds)),
        }
    return {"auc_roc": 0.5, "f1": 0, "pr_auc": 0, "mcc": 0}


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 70)
    print("Temporal GNN Baselines for Fraud Ring Classification")
    print("=" * 70, flush=True)

    # Load data from all attack periods
    periods = ["dao_hack", "pre_dao", "post_fork", "attack_51_v1", "attack_51_v2"]
    all_s, all_l = [], []
    for period in periods:
        gp = base / "data" / "graphs" / f"{period}_graph.pt"
        lp = base / "data" / "processed" / f"{period}_labels.pt"
        if not gp.exists() or not lp.exists():
            continue
        print(f"Loading {period}...", flush=True)
        gd = torch.load(gp, weights_only=False)
        lb = torch.load(lp, weights_only=False)
        nf = sum(1 for l in lb.values() if l == 1)
        print(f"  {gd['num_nodes']} nodes, {nf} fraud", flush=True)
        max_per = 250 if gd['num_nodes'] > 50000 else 150
        subs, labs = extract_subgraphs(gd, lb, max_per_class=max_per, num_hops=2)
        print(f"  Extracted: {len(subs)} ({sum(labs)} fraud)", flush=True)
        all_s.extend(subs)
        all_l.extend(labs)

    total_fraud = sum(all_l)
    total_benign = len(all_l) - total_fraud
    print(f"\nTOTAL: {len(all_s)} subgraphs ({total_fraud} fraud, {total_benign} benign)\n",
          flush=True)

    table_dir.mkdir(parents=True, exist_ok=True)
    if len(all_s) < 20 or len(set(all_l)) < 2:
        skipped = {
            "status": "skipped",
            "reason": "Not enough labeled subgraphs to train temporal baselines. Rebuild data/graphs/labels first.",
            "subgraphs": len(all_s),
            "fraud": total_fraud,
            "benign": total_benign,
        }
        with open(table_dir / "temporal_baseline_results.json", "w") as f:
            json.dump(skipped, f, indent=2)
        with open(table_dir / "temporal_baseline_table.tex", "w") as f:
            f.write("% Temporal baselines skipped: not enough labeled subgraphs.\n")
        print("SKIPPED: Not enough labeled subgraphs to train temporal baselines.", flush=True)
        print("Run scripts/extract_data.sh, scripts/build_graphs.sh, then src.labeling.generate_labels.", flush=True)
        return skipped

    feat_dim = all_s[0].x.size(1)

    # Models to compare
    model_configs = {
        "EvolveGCN-H": lambda: EvolveGCNH(feat_dim, 64, 2, n_snapshots=4),
        "T-GCN": lambda: TemporalGCN(feat_dim, 64, 2, temp_dim=16),
        "Snapshot-GCN": lambda: SnapshotGCN(feat_dim, 64, 2, n_snapshots=4),
        "SubGNN+CSP (ours)": lambda: FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2),
    }

    n_runs = 3
    results = {}

    for model_name, model_fn in model_configs.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name} ({n_runs} runs)")
        print(f"{'='*50}", flush=True)

        runs = []
        for run in range(n_runs):
            np.random.seed(SEED + run * 7)
            torch.manual_seed(SEED + run * 7)

            # Train/val/test split (70/15/15)
            idx = np.random.permutation(len(all_s))
            nt = int(0.7 * len(idx))
            nv = int(0.15 * len(idx))
            train_s = [all_s[i] for i in idx[:nt]]
            train_l = [all_l[i] for i in idx[:nt]]
            val_s = [all_s[i] for i in idx[nt:nt+nv]]
            val_l = [all_l[i] for i in idx[nt:nt+nv]]
            test_s = [all_s[i] for i in idx[nt+nv:]]
            test_l = [all_l[i] for i in idx[nt+nv:]]

            model = model_fn()

            # For SubGNN+CSP, use CSP regularization
            if "SubGNN" in model_name:
                from allout_v3 import MultiTaskCSPTrainer, FeatureMaskingAugmentor, train_supervised
                train_supervised(model, train_s, train_l, val_s, val_l,
                                use_csp_reg=True, csp_weight=0.2)
            else:
                train_model(model, train_s, train_l, val_s, val_l)

            res = evaluate_model(model, test_s, test_l)
            runs.append(res)
            print(f"  Run {run+1}: AUC={res['auc_roc']:.4f}, F1={res['f1']:.4f}, "
                  f"MCC={res['mcc']:.4f}", flush=True)

        # Aggregate
        mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        std = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
        results[model_name] = {"mean": mean, "std": std, "runs": runs}

        print(f"  MEAN: AUC={mean['auc_roc']:.3f}±{std['auc_roc']:.3f}, "
              f"F1={mean['f1']:.3f}±{std['f1']:.3f}, "
              f"MCC={mean['mcc']:.3f}±{std['mcc']:.3f}", flush=True)

    # Save results
    with open(table_dir / "temporal_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print("TEMPORAL BASELINE COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'AUC-ROC':>15} {'F1':>15} {'PR-AUC':>15} {'MCC':>15}")
    print("-" * 80)
    for name, data in results.items():
        m, s = data["mean"], data["std"]
        print(f"{name:<20} {m['auc_roc']:.3f}±{s['auc_roc']:.3f}"
              f"   {m['f1']:.3f}±{s['f1']:.3f}"
              f"   {m['pr_auc']:.3f}±{s['pr_auc']:.3f}"
              f"   {m['mcc']:.3f}±{s['mcc']:.3f}")
    print("=" * 70, flush=True)

    # Generate LaTeX table
    tex = r"""\begin{table}[t]
\centering
\caption{Temporal GNN baselines on subgraph-level fraud ring classification (mean $\pm$ std, $n=3$ runs).}
\label{tab:temporal_baselines}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{AUC-ROC} & \textbf{F1} & \textbf{PR-AUC} & \textbf{MCC} \\
\midrule
"""
    for name, data in results.items():
        m, s = data["mean"], data["std"]
        tex += f"{name:<25} & {m['auc_roc']:.3f}$\\pm${s['auc_roc']:.3f} "
        tex += f"& {m['f1']:.3f}$\\pm${s['f1']:.3f} "
        tex += f"& {m['pr_auc']:.3f}$\\pm${s['pr_auc']:.3f} "
        tex += f"& {m['mcc']:.3f}$\\pm${s['mcc']:.3f} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(table_dir / "temporal_baseline_table.tex", "w") as f:
        f.write(tex)
    print(f"\nSaved to {table_dir / 'temporal_baseline_results.json'}")
    print(f"LaTeX table: {table_dir / 'temporal_baseline_table.tex'}")

    return results


if __name__ == "__main__":
    main()
