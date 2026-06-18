"""
Baseline GNN models for comparison with StreamRing Tier 3 (SubGNN+CSP).

Models: GCN, GAT, GraphSAGE, GIN, EvolveGCN-H (static approximation)
All models follow the same subgraph classification protocol as allout_v3.py:
- 70/15/15 train/val/test split
- 3 runs with seeds 42, 43, 44
- CrossEntropyLoss with balanced class weights
- Youden's J threshold selection
- Metrics: AUC-ROC, F1, PR-AUC, MCC
"""

import os, sys, json, time, copy
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
from torch_geometric.nn import (GCNConv, GATConv, SAGEConv, GINConv,
                                global_mean_pool, global_max_pool)
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.reproducibility import set_seed

base = project_root
table_dir = base / "results" / "tables"
table_dir.mkdir(parents=True, exist_ok=True)

SEED = 42
set_seed(SEED)


# ============================================================================
# BASELINE MODELS
# ============================================================================

class BaselineGNN(nn.Module):
    """Base class for all baseline GNN subgraph classifiers."""

    def __init__(self, input_dim, hidden_dim=128, embedding_dim=64,
                 num_layers=2, dropout=0.2, num_classes=2):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._build_layers(input_dim, hidden_dim, num_layers)
        self.pool_project = nn.Linear(hidden_dim, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def _build_layers(self, input_dim, hidden_dim, num_layers):
        raise NotImplementedError

    def encode(self, x, edge_index, batch):
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = global_mean_pool(h, batch) + global_max_pool(h, batch)
        return self.pool_project(h)

    def forward(self, x, edge_index, batch, **kwargs):
        emb = self.encode(x, edge_index, batch)
        return self.classifier(emb)


class GCNBaseline(BaselineGNN):
    """Graph Convolutional Network (Kipf & Welling, ICLR 2017)."""

    def _build_layers(self, input_dim, hidden_dim, num_layers):
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_d, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))


class GATBaseline(BaselineGNN):
    """Graph Attention Network (Velickovic et al., ICLR 2018)."""

    def _build_layers(self, input_dim, hidden_dim, num_layers):
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            # 4 heads, concat in hidden layers, average in last
            heads = 4 if i < num_layers - 1 else 1
            out_d = hidden_dim // heads if i < num_layers - 1 else hidden_dim
            self.convs.append(GATConv(in_d, out_d, heads=heads, concat=True,
                                      dropout=self.dropout))
            self.norms.append(nn.BatchNorm1d(hidden_dim))


class GraphSAGEBaseline(BaselineGNN):
    """GraphSAGE (Hamilton et al., NeurIPS 2017)."""

    def _build_layers(self, input_dim, hidden_dim, num_layers):
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_d, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))


class GINBaseline(BaselineGNN):
    """Graph Isomorphism Network (Xu et al., ICLR 2019)."""

    def _build_layers(self, input_dim, hidden_dim, num_layers):
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(hidden_dim))


class MLP_Baseline(BaselineGNN):
    """MLP baseline (no message passing) - lower bound."""

    def _build_layers(self, input_dim, hidden_dim, num_layers):
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            # Use nn.Linear wrapped to match conv interface
            self.convs.append(_LinearConv(in_d, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def encode(self, x, edge_index, batch):
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h)  # no edge_index
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = global_mean_pool(h, batch) + global_max_pool(h, batch)
        return self.pool_project(h)


class _LinearConv(nn.Module):
    """Wrapper to make nn.Linear work like a conv layer."""
    def __init__(self, in_d, out_d):
        super().__init__()
        self.lin = nn.Linear(in_d, out_d)

    def forward(self, x, edge_index=None):
        return self.lin(x)


# ============================================================================
# SHARED DATA LOADING (from allout_v3.py)
# ============================================================================

def compute_position_encoding(data, num_anchors=16):
    """Compute anchor-based position encoding for a subgraph."""
    G = to_networkx(data, to_undirected=True)
    n = data.num_nodes
    if n == 0:
        return torch.zeros(0, num_anchors)
    degrees = dict(G.degree())
    k = min(num_anchors, n)
    anchor_nodes = sorted(degrees, key=degrees.get, reverse=True)[:k]
    pos_enc = torch.zeros(n, num_anchors)
    for i, anchor in enumerate(anchor_nodes):
        lengths = nx.single_source_shortest_path_length(G, anchor)
        for node, dist in lengths.items():
            if node < n:
                pos_enc[node, i] = dist
    max_dist = pos_enc.max()
    if max_dist > 0:
        pos_enc = 1.0 - (pos_enc / max_dist)
    pos_enc[pos_enc < 0] = 0
    return pos_enc


def extract_subgraphs(graph_data, labels, max_per_class=300, num_hops=2):
    from torch_geometric.utils import k_hop_subgraph
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


def load_all_data():
    # Use same 5 periods as allout_v3.py for consistency
    periods = ["dao_hack", "pre_dao", "post_fork", "attack_51_v1", "attack_51_v2"]
    np.random.seed(SEED)
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
        print(f"  Extracted: {len(subs)} ({sum(labs)} fraud, {len(labs)-sum(labs)} benign)", flush=True)
        all_s.extend(subs)
        all_l.extend(labs)
    print(f"\nTOTAL: {len(all_s)} subgraphs ({sum(all_l)} fraud, {len(all_l)-sum(all_l)} benign)\n", flush=True)
    return all_s, all_l


# ============================================================================
# TRAINING & EVALUATION (same protocol as allout_v3.py)
# ============================================================================

def train_model(model, train_s, train_l, val_s, val_l, epochs=80, lr=1e-3, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_pos = sum(train_l)
    n_neg = len(train_l) - n_pos
    n_total = len(train_l)
    weight = torch.tensor([n_total / (2 * max(n_neg, 1)), n_total / (2 * max(n_pos, 1))],
                          dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_val_auc = 0
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
                pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None
                logits = model(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
                loss = criterion(logits, labels_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            except Exception as e:
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


# def evaluate_model_with_latency(model, test_s, test_l):
#     """Return metrics and per‑subgraph inference latencies (ms)."""
#     model.eval()
#     latencies = []
#     probs = []
#     with torch.no_grad():
#         for sg in test_s:
#             t0 = time.perf_counter()
#             batch = Batch.from_data_list([sg])
#             pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None
#             logits = model(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
#             prob = F.softmax(logits, dim=1)[0, 1].item()
#             lat = (time.perf_counter() - t0) * 1000  # ms
#             latencies.append(lat)
#             probs.append(prob)
#     true = np.array(test_l)
#     # Youden's J threshold on full probabilities (could also use per‑run threshold)
#     if len(np.unique(true)) > 1:
#         fpr, tpr, thresholds = roc_curve(true, probs)
#         j = tpr - fpr
#         opt_t = thresholds[np.argmax(j)]
#         preds = (np.array(probs) >= opt_t).astype(int)
#         return {"auc_roc": roc_auc_score(true, probs),
#                 "f1": f1_score(true, preds),
#                 "pr_auc": average_precision_score(true, probs),
#                 "mcc": matthews_corrcoef(true, preds),
#                 "latency_p50": np.percentile(latencies, 50),
#                 "latency_p99": np.percentile(latencies, 99),
#                 "latency_mean": np.mean(latencies),
#                 "latency_std": np.std(latencies)}
#     else:
#         return {"auc_roc": 0.5, "f1": 0, "pr_auc": 0, "mcc": 0,
#                 "latency_p50": 0, "latency_p99": 0, "latency_mean": 0, "latency_std": 0}
    

def evaluate_model(model, test_s, test_l):
    model.eval()
    start_time = time.perf_counter()
    with torch.no_grad():
        batch = Batch.from_data_list(test_s)
        pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None
        logits = model(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
    
    total_time_ms = (time.perf_counter() - start_time) * 1000
    avg_time_per_graph_ms = total_time_ms / len(test_s)

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
            "latency_total_ms": total_time_ms,
            "latency_avg_per_graph_ms": avg_time_per_graph_ms,
        }
    return {"auc_roc": 0.5, "f1": 0, "pr_auc": 0, "mcc": 0,
            "latency_total_ms": total_time_ms,
            "latency_avg_per_graph_ms": avg_time_per_graph_ms,}


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

BASELINES = {
    "GCN": GCNBaseline,
    "GAT": GATBaseline,
    "GraphSAGE": GraphSAGEBaseline,
    "GIN": GINBaseline,
    "MLP": MLP_Baseline,
}


def run_baselines(all_s, all_l, feat_dim, n_runs=3):
    print("\n" + "=" * 70)
    print(f"BASELINE COMPARISON ({n_runs} runs each)")
    print("=" * 70, flush=True)

    # Also include SubGNN (our method) for direct comparison
    from src.gnn_models.subgnn_encoder import FraudRingClassifier

    all_results = {}

    for model_name, model_cls in BASELINES.items():
        print(f"\n--- {model_name} ---", flush=True)
        runs = []
        for run in range(n_runs):
            np.random.seed(SEED + run)
            torch.manual_seed(SEED + run)

            idx = np.random.permutation(len(all_s))
            nt = int(0.7 * len(idx))
            nv = int(0.15 * len(idx))
            ts = [all_s[i] for i in idx[:nt]]
            tl = [all_l[i] for i in idx[:nt]]
            vs = [all_s[i] for i in idx[nt:nt + nv]]
            vl = [all_l[i] for i in idx[nt:nt + nv]]
            es = [all_s[i] for i in idx[nt + nv:]]
            el = [all_l[i] for i in idx[nt + nv:]]

            model = model_cls(feat_dim, hidden_dim=128, embedding_dim=64,
                              num_layers=2, dropout=0.2)
            t0 = time.time()
            train_model(model, ts, tl, vs, vl, epochs=80, lr=1e-3)
            train_time = time.time() - t0

            res = evaluate_model(model, es, el)
            res["train_time_s"] = train_time
            # res = evaluate_model_with_latency(model, es, el)
            # res["train_time_s"] = train_time
            runs.append(res)
            print(f"  Run {run + 1}: AUC={res['auc_roc']:.4f}, F1={res['f1']:.4f}, "
                  f"MCC={res['mcc']:.4f} ({train_time:.1f}s)", flush=True)

        mean_r = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        std_r = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
        all_results[model_name] = {"mean": mean_r, "std": std_r, "runs": runs}
        print(f"  MEAN: AUC={mean_r['auc_roc']:.4f}±{std_r['auc_roc']:.4f}, "
              f"F1={mean_r['f1']:.4f}±{std_r['f1']:.4f}", flush=True)

    # Add SubGNN + CSP-reg (our method) for comparison
    print(f"\n--- SubGNN + CSP-reg (ours) ---", flush=True)
    runs = []
    for run in range(n_runs):
        np.random.seed(SEED + run)
        torch.manual_seed(SEED + run)

        idx = np.random.permutation(len(all_s))
        nt = int(0.7 * len(idx))
        nv = int(0.15 * len(idx))
        ts = [all_s[i] for i in idx[:nt]]
        tl = [all_l[i] for i in idx[:nt]]
        vs = [all_s[i] for i in idx[nt:nt + nv]]
        vl = [all_l[i] for i in idx[nt:nt + nv]]
        es = [all_s[i] for i in idx[nt + nv:]]
        el = [all_l[i] for i in idx[nt + nv:]]

        model = FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2)
        # Import training with CSP from allout_v3
        sys.path.insert(0, str(project_root / "experiments"))
        from allout_v3 import train_supervised as train_with_csp
        t0 = time.time()
        train_with_csp(model, ts, tl, vs, vl, epochs=80, lr=1e-3,
                       use_csp_reg=True, csp_weight=0.1)
        train_time = time.time() - t0

        res = evaluate_model(model, es, el)
        res["train_time_s"] = train_time
        # res = evaluate_model_with_latency(model, es, el)
        # res["train_time_s"] = train_time
        runs.append(res)
        print(f"  Run {run + 1}: AUC={res['auc_roc']:.4f}, F1={res['f1']:.4f}, "
              f"MCC={res['mcc']:.4f} ({train_time:.1f}s)", flush=True)

    mean_r = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    std_r = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
    all_results["SubGNN+CSP (ours)"] = {"mean": mean_r, "std": std_r, "runs": runs}
    print(f"  MEAN: AUC={mean_r['auc_roc']:.4f}±{std_r['auc_roc']:.4f}, "
          f"F1={mean_r['f1']:.4f}±{std_r['f1']:.4f}", flush=True)

    # Save results
    with open(table_dir / "baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Model':<22} {'AUC-ROC':>14} {'F1':>14} {'PR-AUC':>14} {'MCC':>14}")
    print("-" * 80)
    for name, r in all_results.items():
        m, s = r["mean"], r["std"]
        print(f"{name:<22} {m['auc_roc']:.3f}±{s['auc_roc']:.3f}  "
              f"{m['f1']:.3f}±{s['f1']:.3f}  "
              f"{m['pr_auc']:.3f}±{s['pr_auc']:.3f}  "
              f"{m['mcc']:.3f}±{s['mcc']:.3f}")
    print(f"{'=' * 70}\n")

    return all_results


def generate_baseline_latex(results):
    """Generate LaTeX table for paper."""
    tex = r"""\begin{table}[t]
\centering
\caption{Comparison with baseline GNN models on subgraph-level fraud ring classification (mean $\pm$ std, $n=3$ runs).}
\label{tab:baselines}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{AUC-ROC} & \textbf{F1} & \textbf{PR-AUC} & \textbf{MCC} \\
\midrule
"""
    # Order: baselines first, then ours (bold)
    baseline_order = ["MLP", "GCN", "GAT", "GraphSAGE", "GIN"]
    for name in baseline_order:
        if name not in results:
            continue
        m, s = results[name]["mean"], results[name]["std"]
        tex += f"{name:<20} & {m['auc_roc']:.3f}$\\pm${s['auc_roc']:.3f} & {m['f1']:.3f}$\\pm${s['f1']:.3f} & {m['pr_auc']:.3f}$\\pm${s['pr_auc']:.3f} & {m['mcc']:.3f}$\\pm${s['mcc']:.3f} \\\\\n"

    tex += r"\midrule" + "\n"
    if "SubGNN+CSP (ours)" in results:
        m, s = results["SubGNN+CSP (ours)"]["mean"], results["SubGNN+CSP (ours)"]["std"]
        tex += f"\\textbf{{SubGNN+CSP (ours)}} & \\textbf{{{m['auc_roc']:.3f}}}$\\pm${s['auc_roc']:.3f} & \\textbf{{{m['f1']:.3f}}}$\\pm${s['f1']:.3f} & \\textbf{{{m['pr_auc']:.3f}}}$\\pm${s['pr_auc']:.3f} & \\textbf{{{m['mcc']:.3f}}}$\\pm${s['mcc']:.3f} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex

# def generate_runtime_table(results):
#     tex = r"\begin{table}[t]\centering"
#     tex += r"\caption{Inference latency of static GNN baselines (ms)}\label{tab:runtime}"
#     tex += r"\begin{tabular}{lccc}\toprule"
#     tex += r"Model & P50 (ms) & P99 (ms) & Training time (s) \\ \midrule"
#     for name, data in results.items():
#         m = data["mean"]
#         tex += f"{name} & {m['latency_p50']:.2f} & {m['latency_p99']:.2f} & {m['train_time_s']:.1f} \\\\\n"
#     tex += r"\bottomrule\end{tabular}\end{table}"
#     with open(table_dir / "baseline_runtime.tex", "w") as f:
#         f.write(tex)

def generate_runtime_table(results):
    tex = r"\begin{table}[t]\centering"
    tex += r"\caption{Inference latency of static GNN baselines}\label{tab:runtime}"
    tex += r"\begin{tabular}{lcc}\toprule"
    tex += r"Model & Total time (ms) & Avg per graph (ms) \\ \midrule"
    for name, data in results.items():
        m = data["mean"]
        tex += f"{name} & {m['latency_total_ms']:.2f} & {m['latency_avg_per_graph_ms']:.2f} \\\\\n"
    tex += r"\bottomrule\end{tabular}\end{table}"
    with open(table_dir / "baseline_runtime.tex", "w") as f:
        f.write(tex)

if __name__ == "__main__":
    print("=" * 70)
    print("StreamRing Baseline Comparison")
    print("=" * 70, flush=True)

    all_s, all_l = load_all_data()
    if len(all_s) < 50:
        print("ERROR: Not enough data")
        sys.exit(1)
    feat_dim = all_s[0].x.shape[1]

    t0 = time.time()
    results = run_baselines(all_s, all_l, feat_dim, n_runs=5)
    generate_runtime_table(results)

    # Generate LaTeX table
    latex = generate_baseline_latex(results)
    with open(table_dir / "baseline_table.tex", "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {table_dir / 'baseline_table.tex'}")

    print(f"\nALL BASELINES COMPLETE in {(time.time() - t0) / 60:.1f} min")
