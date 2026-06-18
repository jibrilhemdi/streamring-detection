"""
StreamRing ALL-OUT v3: Correct experiments with validated approach.

Key findings from v2:
- Supervised SubGNN: AUC=0.88, F1=0.84 (excellent!)
- CSP pre-training HURTS (AUC=0.56) because edge dropout destroys fraud patterns
- Fix: Use CSP as multi-task regularizer (feature masking + joint training)

Experiments:
1. Ablation: Full(+CSP-reg) vs NoCSP vs Small vs Baseline
2. Label scarcity: +CSP-reg vs supervised-only
3. ROC curves + t-SNE
4. Scalability analysis
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
                             matthews_corrcoef, roc_curve, precision_recall_curve)
from torch_geometric.utils import k_hop_subgraph, subgraph, to_networkx
from torch_geometric.data import Data, Batch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.gnn_models.subgnn_encoder import SubGNNEncoder, FraudRingClassifier
from src.utils.reproducibility import set_runtime_threads, set_seed

base = project_root
table_dir = base / "results" / "tables"
table_dir.mkdir(parents=True, exist_ok=True)

SEED = 42
set_seed(SEED)
set_runtime_threads()


# ============================================================================
# MULTI-TASK CSP: Contrastive as regularizer during supervised training
# ============================================================================

class FeatureMaskingAugmentor:
    """Feature masking augmentation (preserves graph structure)."""
    def __init__(self, mask_rate=0.3):
        self.mask_rate = mask_rate

    def augment(self, data):
        x = data.x.clone()
        mask = torch.rand(x.shape) < self.mask_rate
        x[mask] = 0
        return Data(x=x, edge_index=data.edge_index)


class MultiTaskCSPTrainer:
    """Joint supervised + contrastive training."""
    def __init__(self, model, temperature=0.3, csp_weight=0.1):
        self.model = model
        self.augmentor = FeatureMaskingAugmentor(mask_rate=0.2)
        self.temperature = temperature
        self.csp_weight = csp_weight

    def contrastive_loss(self, z_i, z_j):
        bs = z_i.size(0)
        if bs < 2:
            return torch.tensor(0.0, requires_grad=True)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        reps = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(reps.unsqueeze(1), reps.unsqueeze(0), dim=2) / self.temperature
        labels = torch.cat([torch.arange(bs, 2*bs), torch.arange(bs)])
        mask = torch.eye(2*bs, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        return F.cross_entropy(sim, labels)

    def train_step(self, batch_data, batch_labels, optimizer, criterion):
        """One training step with joint supervised + contrastive loss."""
        self.model.train()
        batch = Batch.from_data_list(batch_data)
        labels_t = torch.tensor(batch_labels, dtype=torch.long)

        pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None

        # Supervised loss
        logits = self.model(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc)
        sup_loss = criterion(logits, labels_t)

        # Contrastive regularization via feature masking
        aug_data = [self.augmentor.augment(d) for d in batch_data]
        # Copy position_encoding to augmented data (feature masking doesn't change structure)
        for orig, aug in zip(batch_data, aug_data):
            if hasattr(orig, 'position_encoding'):
                aug.position_encoding = orig.position_encoding
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
        return sup_loss.item(), csp_loss.item()


# ============================================================================
# POSITION ENCODING
# ============================================================================

def compute_position_encoding(data, num_anchors=16):
    """Compute anchor-based position encoding for a subgraph.

    For each node, compute shortest-path distance to K anchor nodes
    (highest-degree nodes in the subgraph). Normalize to proximity scores.
    """
    G = to_networkx(data, to_undirected=True)
    n = data.num_nodes
    if n == 0:
        return torch.zeros(0, num_anchors)

    # Select anchors by degree (top-K highest degree nodes; node id breaks ties)
    degrees = dict(G.degree())
    k = min(num_anchors, n)
    anchor_nodes = sorted(degrees, key=lambda node: (-degrees[node], node))[:k]

    pos_enc = torch.zeros(n, num_anchors)
    for i, anchor in enumerate(anchor_nodes):
        lengths = nx.single_source_shortest_path_length(G, anchor)
        for node, dist in lengths.items():
            if node < n:
                pos_enc[node, i] = dist

    # Normalize: convert distance to proximity (1 = closest, 0 = farthest)
    max_dist = pos_enc.max()
    if max_dist > 0:
        pos_enc = 1.0 - (pos_enc / max_dist)
    pos_enc[pos_enc < 0] = 0
    return pos_enc


# ============================================================================
# SUBGRAPH EXTRACTION
# ============================================================================

def extract_subgraphs(graph_data, labels, max_per_class=300, num_hops=2):
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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_supervised(model, train_s, train_l, val_s, val_l, epochs=80, lr=1e-3,
                     patience=15, use_csp_reg=False, csp_weight=0.1):
    """Train with optional CSP regularization."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_pos = sum(train_l)
    n_neg = len(train_l) - n_pos
    n_total = len(train_l)
    weight = torch.tensor([n_total/(2*max(n_neg,1)), n_total/(2*max(n_pos,1))], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    csp_trainer = MultiTaskCSPTrainer(model, csp_weight=csp_weight) if use_csp_reg else None

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
                if use_csp_reg and csp_trainer:
                    csp_trainer.train_step(batch_data, batch_labels, optimizer, criterion)
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


def evaluate_model(model, test_s, test_l, return_probs=False):
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
        results = {
            "auc_roc": float(roc_auc_score(true, probs)),
            "f1": float(f1_score(true, preds)),
            "pr_auc": float(average_precision_score(true, probs)),
            "mcc": float(matthews_corrcoef(true, preds)),
        }
    else:
        results = {"auc_roc": 0.5, "f1": 0, "pr_auc": 0, "mcc": 0}

    if return_probs:
        return results, probs, true
    return results


# ============================================================================
# EXPERIMENT 1: ABLATION
# ============================================================================

def experiment_ablation(all_s, all_l, feat_dim, n_runs=3):
    print("\n" + "="*70)
    print(f"EXPERIMENT 1: ABLATION STUDY ({n_runs} runs)")
    print("="*70, flush=True)

    configs = [
        ("SubGNN + CSP-reg", True, 128, 2),
        ("SubGNN (supervised)", False, 128, 2),
        ("SubGNN-small + CSP-reg", True, 64, 2),
        ("SubGNN-small (baseline)", False, 64, 2),
    ]

    results = {}
    for cname, use_csp, hidden, layers in configs:
        print(f"\n--- {cname} ---", flush=True)
        runs = []
        for run in range(n_runs):
            np.random.seed(SEED + run)
            torch.manual_seed(SEED + run)

            idx = np.random.permutation(len(all_s))
            nt = int(0.7 * len(idx)); nv = int(0.15 * len(idx))
            ts = [all_s[i] for i in idx[:nt]]
            tl = [all_l[i] for i in idx[:nt]]
            vs = [all_s[i] for i in idx[nt:nt+nv]]
            vl = [all_l[i] for i in idx[nt:nt+nv]]
            es = [all_s[i] for i in idx[nt+nv:]]
            el = [all_l[i] for i in idx[nt+nv:]]

            model = FraudRingClassifier(feat_dim, hidden, 64, layers, dropout=0.2)
            train_supervised(model, ts, tl, vs, vl, epochs=80, lr=1e-3,
                           use_csp_reg=use_csp, csp_weight=0.1)
            res = evaluate_model(model, es, el)
            runs.append(res)
            print(f"  Run {run+1}: AUC={res['auc_roc']:.4f}, F1={res['f1']:.4f}, MCC={res['mcc']:.4f}", flush=True)

        mean_r = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        std_r = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
        results[cname] = {"mean": mean_r, "std": std_r, "runs": runs}
        print(f"  MEAN: AUC={mean_r['auc_roc']:.4f}±{std_r['auc_roc']:.4f}, "
              f"F1={mean_r['f1']:.4f}±{std_r['f1']:.4f}", flush=True)

    with open(table_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    return results


# ============================================================================
# EXPERIMENT 2: LABEL SCARCITY
# ============================================================================

def experiment_label_scarcity(all_s, all_l, feat_dim, n_runs=3):
    print("\n" + "="*70)
    print(f"EXPERIMENT 2: LABEL SCARCITY ({n_runs} runs)")
    print("="*70, flush=True)

    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    np.random.seed(SEED)
    idx = np.random.permutation(len(all_s))
    nt = int(0.7*len(idx)); nv = int(0.15*len(idx))
    train_s = [all_s[i] for i in idx[:nt]]
    train_l = [all_l[i] for i in idx[:nt]]
    val_s = [all_s[i] for i in idx[nt:nt+nv]]
    val_l = [all_l[i] for i in idx[nt:nt+nv]]
    test_s = [all_s[i] for i in idx[nt+nv:]]
    test_l = [all_l[i] for i in idx[nt+nv:]]

    results = {"fractions": fractions, "with_csp": [], "without_csp": []}

    for frac in fractions:
        n_labeled = max(8, int(len(train_s) * frac))
        print(f"\n--- {frac*100:.0f}% ({n_labeled} samples) ---", flush=True)
        csp_runs, nocsp_runs = [], []

        for run in range(n_runs):
            np.random.seed(SEED + run*7)
            torch.manual_seed(SEED + run*7)

            # Stratified sampling
            fi = [i for i, l in enumerate(train_l) if l == 1]
            bi = [i for i, l in enumerate(train_l) if l == 0]
            nf = max(2, n_labeled//2)
            nb = n_labeled - nf
            li = np.concatenate([
                np.random.choice(fi, min(nf, len(fi)), replace=False),
                np.random.choice(bi, min(nb, len(bi)), replace=False)
            ])
            sub_s = [train_s[i] for i in li]
            sub_l = [train_l[i] for i in li]

            # Without CSP
            m1 = FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2)
            train_supervised(m1, sub_s, sub_l, val_s, val_l, use_csp_reg=False)
            r1 = evaluate_model(m1, test_s, test_l)
            nocsp_runs.append(r1)

            # With CSP-reg
            m2 = FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2)
            train_supervised(m2, sub_s, sub_l, val_s, val_l, use_csp_reg=True, csp_weight=0.2)
            r2 = evaluate_model(m2, test_s, test_l)
            csp_runs.append(r2)

            print(f"  Run {run+1}: NoCSP AUC={r1['auc_roc']:.4f}, +CSP AUC={r2['auc_roc']:.4f}", flush=True)

        for runs, key in [(nocsp_runs, "without_csp"), (csp_runs, "with_csp")]:
            m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            s = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
            results[key].append({"mean": m, "std": s})

        gain = results["with_csp"][-1]["mean"]["auc_roc"] - results["without_csp"][-1]["mean"]["auc_roc"]
        print(f"  Δ AUC = {gain:+.4f}", flush=True)

    with open(table_dir / "label_scarcity_results.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    return results


# ============================================================================
# EXPERIMENT 3: ROC + t-SNE
# ============================================================================

def experiment_roc_tsne(all_s, all_l, feat_dim):
    print("\n" + "="*70)
    print("EXPERIMENT 3: ROC CURVES + t-SNE")
    print("="*70, flush=True)

    np.random.seed(SEED); torch.manual_seed(SEED)
    idx = np.random.permutation(len(all_s))
    nt = int(0.7*len(idx)); nv = int(0.15*len(idx))
    ts = [all_s[i] for i in idx[:nt]]
    tl = [all_l[i] for i in idx[:nt]]
    vs = [all_s[i] for i in idx[nt:nt+nv]]
    vl = [all_l[i] for i in idx[nt:nt+nv]]
    es = [all_s[i] for i in idx[nt+nv:]]
    el = [all_l[i] for i in idx[nt+nv:]]

    roc_data, tsne_data = {}, {}

    for name, use_csp in [("SubGNN+CSP-reg", True), ("SubGNN", False)]:
        print(f"\nTraining {name}...", flush=True)
        model = FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2)
        train_supervised(model, ts, tl, vs, vl, use_csp_reg=use_csp, csp_weight=0.1)
        res, probs, true = evaluate_model(model, es, el, return_probs=True)
        print(f"  {name}: AUC={res['auc_roc']:.4f}, F1={res['f1']:.4f}", flush=True)

        fpr, tpr, _ = roc_curve(true, probs)
        prec, rec, _ = precision_recall_curve(true, probs)
        roc_data[name] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "precision": prec.tolist(), "recall": rec.tolist(),
            "auc": float(roc_auc_score(true, probs)),
        }

        model.eval()
        with torch.no_grad():
            batch = Batch.from_data_list(es)
            pos_enc = batch.position_encoding if hasattr(batch, 'position_encoding') else None
            emb = model.encode(batch.x, batch.edge_index, batch.batch, position_encoding=pos_enc).numpy()
        tsne_data[name] = {"embeddings": emb.tolist(), "labels": el}

    with open(table_dir / "roc_data.json", "w") as f:
        json.dump(roc_data, f, sort_keys=True)
    with open(table_dir / "tsne_data.json", "w") as f:
        json.dump(tsne_data, f, sort_keys=True)
    print("ROC + t-SNE data saved", flush=True)
    return roc_data, tsne_data


# ============================================================================
# EXPERIMENT 4: SCALABILITY
# ============================================================================

def experiment_scalability(all_s, feat_dim):
    print("\n" + "="*70)
    print("EXPERIMENT 4: SCALABILITY")
    print("="*70, flush=True)

    model = FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2)
    model.eval()

    groups = defaultdict(list)
    for sg in all_s:
        n = sg.x.size(0)
        if n < 10: groups["<10"].append(sg)
        elif n < 50: groups["10-50"].append(sg)
        elif n < 100: groups["50-100"].append(sg)
        elif n < 200: groups["100-200"].append(sg)
        else: groups["200+"].append(sg)

    results = {}
    for sname, sgs in sorted(groups.items()):
        if len(sgs) < 3:
            continue
        latencies = []
        for sg in sgs[:30]:
            b = Batch.from_data_list([sg])
            pe = b.position_encoding if hasattr(b, 'position_encoding') else None
            with torch.no_grad():
                model(b.x, b.edge_index, b.batch, position_encoding=pe)  # warmup
            for _ in range(50):
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(b.x, b.edge_index, b.batch, position_encoding=pe)
                latencies.append((time.perf_counter()-t0)*1000)

        results[sname] = {
            "avg_nodes": float(np.mean([sg.x.size(0) for sg in sgs])),
            "n_subgraphs": len(sgs),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
        }
        print(f"  {sname}: P50={results[sname]['latency_p50_ms']:.3f}ms, "
              f"P99={results[sname]['latency_p99_ms']:.3f}ms", flush=True)

    with open(table_dir / "scalability_results.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    return results


# ============================================================================
# MAIN
# ============================================================================

def load_all_data():
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
        all_s.extend(subs); all_l.extend(labs)
    print(f"\nTOTAL: {len(all_s)} subgraphs ({sum(all_l)} fraud, {len(all_l)-sum(all_l)} benign)\n", flush=True)
    return all_s, all_l


if __name__ == "__main__":
    print("="*70)
    print("StreamRing ALL-OUT v3")
    print("="*70, flush=True)

    all_s, all_l = load_all_data()
    if len(all_s) < 50:
        print("ERROR: Not enough data"); sys.exit(1)
    feat_dim = all_s[0].x.shape[1]

    t0 = time.time()
    r1 = experiment_ablation(all_s, all_l, feat_dim, n_runs=5)
    r2 = experiment_label_scarcity(all_s, all_l, feat_dim, n_runs=5)
    r3, r4 = experiment_roc_tsne(all_s, all_l, feat_dim)
    r5 = experiment_scalability(all_s, feat_dim)

    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE in {(time.time()-t0)/60:.1f} min")
    print(f"{'='*70}", flush=True)
