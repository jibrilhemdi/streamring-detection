"""
Augmentation Strategy Ablation for StreamRing CSP Regularizer.

Compares different augmentation strategies for contrastive subgraph pre-training:
1. Feature Masking (current default, mask_rate=0.2)
2. Edge Dropout (randomly remove edges)
3. Node Dropout (randomly remove nodes and their edges)
4. Subgraph Crop (sample random connected subgraph)
5. Feature Masking + Edge Dropout (combined)
6. Feature Masking + Node Dropout (combined)
7. No Augmentation (baseline: CSP with identity augmentor)
8. No CSP (supervised-only, no contrastive loss)

Each config: 3 runs with different seeds, reporting AUC, F1, PR-AUC, MCC.
"""

import os, sys
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import json, time, copy
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score,
                             matthews_corrcoef, roc_curve)
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, Batch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.reproducibility import set_seed
from src.gnn_models.subgnn_encoder import SubGNNEncoder, FraudRingClassifier

base = project_root
table_dir = base / "results" / "tables"
table_dir.mkdir(parents=True, exist_ok=True)


# ========== AUGMENTATION STRATEGIES ==========

class FeatureMaskingAugmentor:
    """Mask random node features with zeros."""
    def __init__(self, mask_rate=0.2):
        self.mask_rate = mask_rate

    def __call__(self, data):
        x = data.x.clone()
        mask = torch.rand(x.shape) < self.mask_rate
        x[mask] = 0
        return Data(x=x, edge_index=data.edge_index)


class EdgeDropoutAugmentor:
    """Randomly drop edges from the graph."""
    def __init__(self, drop_rate=0.1):
        self.drop_rate = drop_rate

    def __call__(self, data):
        ei = data.edge_index
        num_edges = ei.shape[1]
        if num_edges < 2:
            return Data(x=data.x.clone(), edge_index=ei.clone())
        keep_mask = torch.rand(num_edges) >= self.drop_rate
        # Ensure at least 1 edge remains
        if keep_mask.sum() < 1:
            keep_mask[0] = True
        new_ei = ei[:, keep_mask]
        return Data(x=data.x.clone(), edge_index=new_ei)


class NodeDropoutAugmentor:
    """Randomly drop nodes (and their edges)."""
    def __init__(self, drop_rate=0.1):
        self.drop_rate = drop_rate

    def __call__(self, data):
        num_nodes = data.x.shape[0]
        if num_nodes < 3:
            return Data(x=data.x.clone(), edge_index=data.edge_index.clone())

        keep_mask = torch.rand(num_nodes) >= self.drop_rate
        # Keep at least 2 nodes
        if keep_mask.sum() < 2:
            indices = torch.randperm(num_nodes)[:2]
            keep_mask[indices] = True

        keep_nodes = keep_mask.nonzero(as_tuple=True)[0]
        node_map = torch.full((num_nodes,), -1, dtype=torch.long)
        node_map[keep_nodes] = torch.arange(len(keep_nodes))

        # Filter edges
        ei = data.edge_index
        src_valid = keep_mask[ei[0]]
        dst_valid = keep_mask[ei[1]]
        edge_mask = src_valid & dst_valid
        new_ei = node_map[ei[:, edge_mask]]

        return Data(x=data.x[keep_nodes].clone(), edge_index=new_ei)


class SubgraphCropAugmentor:
    """Sample a random connected subgraph (BFS from random node)."""
    def __init__(self, crop_ratio=0.8):
        self.crop_ratio = crop_ratio

    def __call__(self, data):
        num_nodes = data.x.shape[0]
        target_size = max(2, int(num_nodes * self.crop_ratio))

        if num_nodes <= target_size:
            return Data(x=data.x.clone(), edge_index=data.edge_index.clone())

        # BFS from random start node
        start = np.random.randint(num_nodes)
        ei = data.edge_index.numpy()
        adj = defaultdict(set)
        for i in range(ei.shape[1]):
            adj[ei[0, i]].add(ei[1, i])
            adj[ei[1, i]].add(ei[0, i])

        visited = set()
        queue = [start]
        while queue and len(visited) < target_size:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            neighbors = list(adj[node])
            np.random.shuffle(neighbors)
            queue.extend(neighbors)

        if len(visited) < 2:
            return Data(x=data.x.clone(), edge_index=data.edge_index.clone())

        keep_nodes = sorted(visited)
        keep_set = set(keep_nodes)
        node_map = {old: new for new, old in enumerate(keep_nodes)}

        # Filter edges
        new_edges = []
        for i in range(ei.shape[1]):
            s, t = int(ei[0, i]), int(ei[1, i])
            if s in keep_set and t in keep_set:
                new_edges.append([node_map[s], node_map[t]])

        if not new_edges:
            return Data(x=data.x.clone(), edge_index=data.edge_index.clone())

        new_ei = torch.tensor(new_edges, dtype=torch.long).t()
        x_sub = data.x[keep_nodes].clone()
        return Data(x=x_sub, edge_index=new_ei)


class CombinedAugmentor:
    """Apply multiple augmentors sequentially."""
    def __init__(self, augmentors):
        self.augmentors = augmentors

    def __call__(self, data):
        result = data
        for aug in self.augmentors:
            result = aug(result)
        return result


class IdentityAugmentor:
    """No augmentation — returns a clone."""
    def __call__(self, data):
        return Data(x=data.x.clone(), edge_index=data.edge_index.clone())


# ========== DATA LOADING (reuse from allout_v3) ==========

def compute_position_encoding(data, num_anchors=16):
    """Compute anchor-based position encoding for a subgraph."""
    G = to_networkx(data, to_undirected=True)
    degrees = dict(G.degree())
    if not degrees:
        return torch.zeros(data.num_nodes, num_anchors)
    anchor_nodes = sorted(degrees, key=degrees.get, reverse=True)[:num_anchors]
    pos_enc = torch.zeros(data.num_nodes, num_anchors)
    for i, anchor in enumerate(anchor_nodes):
        try:
            lengths = nx.single_source_shortest_path_length(G, anchor)
            for node, dist in lengths.items():
                pos_enc[node, i] = dist
        except Exception:
            pass
    max_dist = pos_enc.max()
    if max_dist > 0:
        pos_enc = 1.0 - (pos_enc / max_dist)
    pos_enc[pos_enc < 0] = 0
    return pos_enc


def load_all_data():
    """Load all subgraphs from all periods."""
    from allout_v3 import load_all_data as _load
    return _load()


# ========== TRAINING ==========

class AugmentedCSPTrainer:
    """Joint supervised + contrastive training with configurable augmentation."""
    def __init__(self, model, augmentor, temperature=0.3, csp_weight=0.1):
        self.model = model
        self.augmentor = augmentor
        self.temperature = temperature
        self.csp_weight = csp_weight

    def contrastive_loss(self, z_i, z_j):
        bs = z_i.size(0)
        if bs < 2:
            return torch.tensor(0.0, requires_grad=True)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        labels = torch.cat([torch.arange(bs) + bs, torch.arange(bs)])
        mask = torch.eye(2*bs, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        return F.cross_entropy(sim, labels)

    def train_step(self, batch_data, batch_labels, optimizer, criterion):
        self.model.train()
        batch = Batch.from_data_list(batch_data)
        labels_t = torch.tensor(batch_labels, dtype=torch.long)

        pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None
        logits = self.model(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
        sup_loss = criterion(logits, labels_t)

        # Contrastive on augmented views
        aug_data = []
        for i, orig in enumerate(batch_data):
            aug = self.augmentor(orig)
            if hasattr(orig, 'position_encoding'):
                # Recompute position encoding if node structure changed
                if aug.x.shape[0] != orig.x.shape[0]:
                    aug.position_encoding = compute_position_encoding(aug, num_anchors=16)
                else:
                    aug.position_encoding = orig.position_encoding
            aug_data.append(aug)

        batch_aug = Batch.from_data_list(aug_data)
        pos_enc_aug = batch_aug.position_encoding if hasattr(batch_aug, 'position_encoding') else None
        z_orig = self.model.encode(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
        z_aug = self.model.encode(batch_aug.x, batch_aug.edge_index, batch_aug.batch, position_encoding=pos_enc_aug)
        csp_loss = self.contrastive_loss(z_orig, z_aug)

        total_loss = sup_loss + self.csp_weight * csp_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return float(total_loss), float(sup_loss), float(csp_loss)


def train_and_evaluate(all_s, all_l, feat_dim, augmentor, use_csp=True,
                       seed=42, epochs=80, batch_size=32, patience=15):
    """Train SubGNN with given augmentor and evaluate."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split: 70/15/15
    n = len(all_s)
    idx = np.random.permutation(n)
    tr, va, te = int(0.7*n), int(0.85*n), n
    train_idx, val_idx, test_idx = idx[:tr], idx[tr:va], idx[va:]

    train_s = [all_s[i] for i in train_idx]
    train_l = [all_l[i] for i in train_idx]
    val_s = [all_s[i] for i in val_idx]
    val_l = [all_l[i] for i in val_idx]
    test_s = [all_s[i] for i in test_idx]
    test_l = [all_l[i] for i in test_idx]

    # Model
    model = FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Balanced class weights
    n_pos = sum(train_l)
    n_neg = len(train_l) - n_pos
    w = torch.tensor([len(train_l)/(2*max(n_neg,1)), len(train_l)/(2*max(n_pos,1))])
    criterion = nn.CrossEntropyLoss(weight=w)

    if use_csp:
        trainer = AugmentedCSPTrainer(model, augmentor, temperature=0.3, csp_weight=0.1)

    best_val_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_s))
        epoch_loss = 0

        for start in range(0, len(train_s), batch_size):
            end = min(start + batch_size, len(train_s))
            batch_idx = perm[start:end]
            batch_data = [train_s[i] for i in batch_idx]
            batch_labels = [train_l[i] for i in batch_idx]

            if use_csp:
                loss, _, _ = trainer.train_step(batch_data, batch_labels, optimizer, criterion)
            else:
                batch = Batch.from_data_list(batch_data)
                labels_t = torch.tensor(batch_labels, dtype=torch.long)
                pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None
                logits = model(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
                loss = criterion(logits, labels_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss = float(loss)

            epoch_loss += loss

        scheduler.step()

        # Validation
        model.eval()
        val_scores, val_labels = [], []
        with torch.no_grad():
            for start in range(0, len(val_s), batch_size):
                end = min(start + batch_size, len(val_s))
                batch = Batch.from_data_list(val_s[start:end])
                pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None
                logits = model(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
                probs = F.softmax(logits, dim=1)[:, 1]
                val_scores.extend(probs.tolist())
                val_labels.extend(val_l[start:end])

        val_auc = roc_auc_score(val_labels, val_scores) if len(set(val_labels)) > 1 else 0.5
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_scores, test_labels = [], []
    with torch.no_grad():
        for start in range(0, len(test_s), batch_size):
            end = min(start + batch_size, len(test_s))
            batch = Batch.from_data_list(test_s[start:end])
            pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None
            logits = model(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
            probs = F.softmax(logits, dim=1)[:, 1]
            test_scores.extend(probs.tolist())
            test_labels.extend(test_l[start:end])

    y_true = np.array(test_labels)
    y_scores = np.array(test_scores)

    # Youden's J threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_idx = np.argmax(tpr - fpr)
    threshold = thresholds[j_idx]
    y_pred = (y_scores >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, y_scores)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def main():
    print("=" * 70)
    print("AUGMENTATION STRATEGY ABLATION")
    print("=" * 70)
    print(flush=True)

    print("Loading data...", flush=True)
    all_s, all_l = load_all_data()
    feat_dim = all_s[0].x.shape[1]

    # Define augmentation configs
    configs = {
        "Feature Masking (default)": (FeatureMaskingAugmentor(mask_rate=0.2), True),
        "Edge Dropout (10%)": (EdgeDropoutAugmentor(drop_rate=0.1), True),
        "Edge Dropout (20%)": (EdgeDropoutAugmentor(drop_rate=0.2), True),
        "Node Dropout (10%)": (NodeDropoutAugmentor(drop_rate=0.1), True),
        "Subgraph Crop (80%)": (SubgraphCropAugmentor(crop_ratio=0.8), True),
        "Feat Mask + Edge Drop": (CombinedAugmentor([
            FeatureMaskingAugmentor(mask_rate=0.2),
            EdgeDropoutAugmentor(drop_rate=0.1)
        ]), True),
        "Feat Mask + Node Drop": (CombinedAugmentor([
            FeatureMaskingAugmentor(mask_rate=0.2),
            NodeDropoutAugmentor(drop_rate=0.1)
        ]), True),
        "Identity (CSP, no aug)": (IdentityAugmentor(), True),
        "No CSP (supervised)": (None, False),
    }

    n_runs = 3
    seeds = [42, 123, 456]
    results = {}

    for config_name, (augmentor, use_csp) in configs.items():
        print(f"\n--- {config_name} ---", flush=True)
        runs = []
        for run_idx, seed in enumerate(seeds[:n_runs]):
            t0 = time.time()
            metrics = train_and_evaluate(
                all_s, all_l, feat_dim,
                augmentor=augmentor,
                use_csp=use_csp,
                seed=seed,
            )
            elapsed = time.time() - t0
            runs.append(metrics)
            print(f"  Run {run_idx+1}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
                  f"MCC={metrics['mcc']:.4f} ({elapsed:.1f}s)", flush=True)

        # Compute mean ± std
        mean_metrics = {}
        for key in ["auc", "f1", "pr_auc", "mcc"]:
            vals = [r[key] for r in runs]
            mean_metrics[f"{key}_mean"] = float(np.mean(vals))
            mean_metrics[f"{key}_std"] = float(np.std(vals))

        print(f"  MEAN: AUC={mean_metrics['auc_mean']:.4f}±{mean_metrics['auc_std']:.4f}, "
              f"F1={mean_metrics['f1_mean']:.4f}±{mean_metrics['f1_std']:.4f}", flush=True)

        results[config_name] = {
            "runs": runs,
            "mean": mean_metrics,
            "use_csp": use_csp,
            "augmentor": config_name,
        }

    # Save results
    out_path = table_dir / "augmentation_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Config':<30} {'AUC':>15} {'F1':>15} {'PR-AUC':>15} {'MCC':>15}")
    print("-" * 100)
    for name, data in results.items():
        m = data["mean"]
        print(f"{name:<30} {m['auc_mean']:.3f}±{m['auc_std']:.3f}  "
              f"{m['f1_mean']:.3f}±{m['f1_std']:.3f}  "
              f"{m['pr_auc_mean']:.3f}±{m['pr_auc_std']:.3f}  "
              f"{m['mcc_mean']:.3f}±{m['mcc_std']:.3f}")
    print("=" * 100)

    # Generate LaTeX table
    generate_latex_table(results)
    print("\nDone!", flush=True)


def generate_latex_table(results):
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Augmentation strategy ablation for CSP regularizer (mean $\pm$ std, $n=3$ runs). Best in \textbf{bold}.}",
        r"\label{tab:augmentation}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Augmentation} & \textbf{AUC-ROC} & \textbf{F1} & \textbf{PR-AUC} & \textbf{MCC} \\",
        r"\midrule",
    ]

    # Find best for bold
    best = {}
    for key in ["auc_mean", "f1_mean", "pr_auc_mean", "mcc_mean"]:
        best[key] = max(r["mean"][key] for r in results.values())

    for name, data in results.items():
        m = data["mean"]
        cells = []
        for key_base in ["auc", "f1", "pr_auc", "mcc"]:
            val = f"{m[f'{key_base}_mean']:.3f}$\\pm${m[f'{key_base}_std']:.3f}"
            if abs(m[f"{key_base}_mean"] - best[f"{key_base}_mean"]) < 0.001:
                val = r"\textbf{" + val + "}"
            cells.append(val)

        # Shorten name for LaTeX
        short_name = name.replace("(default)", "").replace("(CSP, no aug)", "(no aug)").strip()
        lines.append(f"{short_name} & {' & '.join(cells)} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])

    tex_path = base / "results" / "tables" / "augmentation_table.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"LaTeX table saved to {tex_path}")



set_seed(42)

if __name__ == "__main__":
    main()
