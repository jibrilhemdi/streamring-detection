"""
Accuracy@Latency curves for StreamRing paper.

Shows how detection accuracy varies with latency budget across tiers.
Uses TRAINED models for each tier with real latency measurements.

Key fix: Previous version used untrained models (AUC=0.237/0.467).
Now trains XGBoost (Tier 1) and SubGNN+CSP (Tier 3) on train split,
evaluates accuracy on test split with real latency measurements.
"""

import os, sys, json, time, copy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from torch_geometric.data import Data, Batch

import xgboost as xgb

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.reproducibility import set_seed
from src.gnn_models.subgnn_encoder import FraudRingClassifier

base = project_root
fig_dir = base / "results" / "figures"
table_dir = base / "results" / "tables"

SEED = 42
set_seed(SEED)


def compute_position_encoding(data, n_anchors=16):
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


def extract_tier1_features(sg):
    """Extract tabular features from subgraph for XGBoost Tier 1."""
    x = sg.x
    n = x.size(0)
    e = sg.edge_index.size(1)

    # Per-feature statistics across nodes
    feat_mean = x.mean(dim=0).numpy()  # (feat_dim,)
    feat_std = x.std(dim=0).numpy()
    feat_max = x.max(dim=0).values.numpy()
    feat_min = x.min(dim=0).values.numpy()

    # Graph-level features
    graph_feats = np.array([
        n, e,
        e / max(n, 1),            # edge density
        float(x.mean()),          # global feature mean
        float(x.std()),           # global feature std
    ])

    return np.concatenate([feat_mean, feat_std, feat_max, feat_min, graph_feats])


def main():
    print("=" * 60)
    print("Accuracy@Latency — Fixed with Trained Models")
    print("=" * 60, flush=True)

    # Load data (same as allout_v3)
    from allout_v3 import load_all_data
    all_s, all_l = load_all_data()
    feat_dim = all_s[0].x.shape[1]

    # Train/test split (70/30)
    idx = np.random.permutation(len(all_s))
    nt = int(0.7 * len(idx))
    train_s = [all_s[i] for i in idx[:nt]]
    train_l = [all_l[i] for i in idx[:nt]]
    test_s = [all_s[i] for i in idx[nt:]]
    test_l = [all_l[i] for i in idx[nt:]]

    print(f"Train: {len(train_s)} ({sum(train_l)} fraud), Test: {len(test_s)} ({sum(test_l)} fraud)")

    # ================================================================
    # TRAIN TIER 1: XGBoost on subgraph-level tabular features
    # ================================================================
    print("\nTraining Tier 1 (XGBoost)...", flush=True)
    tier1_X_train = np.array([extract_tier1_features(sg) for sg in train_s])
    tier1_y_train = np.array(train_l)

    n_pos = tier1_y_train.sum()
    n_neg = len(tier1_y_train) - n_pos
    tier1_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="logloss", random_state=SEED, verbosity=0,
        n_jobs=1, tree_method="hist"
    )
    tier1_model.fit(tier1_X_train, tier1_y_train)
    print(f"  Tier 1 trained on {len(train_s)} subgraphs", flush=True)

    # ================================================================
    # TRAIN TIER 3: SubGNN+CSP on subgraph classification
    # ================================================================
    print("Training Tier 3 (SubGNN+CSP)...", flush=True)

    # Use 80% train / 20% val from training set
    nv = int(0.2 * len(train_s))
    val_s = train_s[-nv:]
    val_l = train_l[-nv:]
    actual_train_s = train_s[:-nv]
    actual_train_l = train_l[:-nv]

    tier3_model = FraudRingClassifier(feat_dim, 128, 64, 2, dropout=0.2)
    optimizer = torch.optim.Adam(tier3_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    n_p = sum(actual_train_l)
    n_n = len(actual_train_l) - n_p
    n_t = len(actual_train_l)
    weight = torch.tensor([n_t/(2*max(n_n,1)), n_t/(2*max(n_p,1))], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_val_auc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(80):
        tier3_model.train()
        indices = list(range(len(actual_train_s)))
        np.random.shuffle(indices)
        for i in range(0, len(indices), 32):
            bi = indices[i:i+32]
            bd = [actual_train_s[j] for j in bi]
            bl = torch.tensor([actual_train_l[j] for j in bi], dtype=torch.long)
            try:
                batch = Batch.from_data_list(bd)
                pe = batch.position_encoding if hasattr(batch, 'position_encoding') else None
                logits = tier3_model(batch.x, batch.edge_index, batch.batch, position_encoding=pe)
                loss = criterion(logits, bl)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tier3_model.parameters(), 1.0)
                optimizer.step()
            except Exception:
                continue
        scheduler.step()

        # Validation
        tier3_model.eval()
        with torch.no_grad():
            try:
                vb = Batch.from_data_list(val_s)
                vpe = vb.position_encoding if hasattr(vb, 'position_encoding') else None
                vl = tier3_model(vb.x, vb.edge_index, vb.batch, position_encoding=vpe)
                vp = F.softmax(vl, dim=1)[:, 1].numpy()
                vt = np.array(val_l)
                if len(np.unique(vt)) > 1:
                    va = roc_auc_score(vt, vp)
                    if va > best_val_auc:
                        best_val_auc = va
                        best_state = copy.deepcopy(tier3_model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= 15:
                            break
            except Exception:
                continue

    if best_state:
        tier3_model.load_state_dict(best_state)
    tier3_model.eval()
    print(f"  Tier 3 trained: best val AUC={best_val_auc:.4f}", flush=True)

    # ================================================================
    # EVALUATE ON TEST SET — Measure latency + accuracy per tier
    # ================================================================
    print(f"\nEvaluating on {len(test_s)} test subgraphs...", flush=True)

    tier1_X_test = np.array([extract_tier1_features(sg) for sg in test_s])
    test_labels = np.array(test_l)

    # --- Tier 1: XGBoost predictions + latency ---
    tier1_latencies = []
    tier1_scores = []
    for i in range(len(test_s)):
        t0 = time.perf_counter()
        score = float(tier1_model.predict_proba(tier1_X_test[i:i+1])[0, 1])
        lat = (time.perf_counter() - t0) * 1000
        tier1_latencies.append(lat)
        tier1_scores.append(score)

    tier1_scores = np.array(tier1_scores)
    tier1_latencies = np.array(tier1_latencies)

    # --- Tier 3: SubGNN predictions + latency ---
    # Warmup
    b = Batch.from_data_list([test_s[0]])
    pe = b.position_encoding if hasattr(b, 'position_encoding') else None
    with torch.no_grad():
        tier3_model(b.x, b.edge_index, b.batch, position_encoding=pe)

    tier3_latencies = []
    tier3_scores = []
    for i, sg in enumerate(test_s):
        b = Batch.from_data_list([sg])
        pe = b.position_encoding if hasattr(b, 'position_encoding') else None
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = tier3_model(b.x, b.edge_index, b.batch, position_encoding=pe)
            prob = F.softmax(logits, dim=1)[0, 1].item()
        lat = (time.perf_counter() - t0) * 1000
        tier3_latencies.append(lat)
        tier3_scores.append(prob)

    tier3_scores = np.array(tier3_scores)
    tier3_latencies = np.array(tier3_latencies)

    # ================================================================
    # COMPUTE ACCURACY@LATENCY
    # ================================================================

    # Tier 1 AUC + threshold
    t1_auc = float(roc_auc_score(test_labels, tier1_scores))
    fpr1, tpr1, th1 = roc_curve(test_labels, tier1_scores)
    t1_opt = th1[np.argmax(tpr1 - fpr1)]
    t1_preds = (tier1_scores >= t1_opt).astype(int)
    t1_f1 = float(f1_score(test_labels, t1_preds))

    # Tier 3 AUC + threshold
    t3_auc = float(roc_auc_score(test_labels, tier3_scores))
    fpr3, tpr3, th3 = roc_curve(test_labels, tier3_scores)
    t3_opt = th3[np.argmax(tpr3 - fpr3)]
    t3_preds = (tier3_scores >= t3_opt).astype(int)
    t3_f1 = float(f1_score(test_labels, t3_preds))

    print(f"\nTier 1: AUC={t1_auc:.4f}, F1={t1_f1:.4f}, P50={np.percentile(tier1_latencies, 50):.3f}ms")
    print(f"Tier 3: AUC={t3_auc:.4f}, F1={t3_f1:.4f}, P50={np.percentile(tier3_latencies, 50):.3f}ms")

    # Use streaming pipeline end-to-end latencies (not microbenchmarks)
    # Microbenchmarks only measure forward pass; streaming includes subgraph extraction
    streaming_path = table_dir / "streaming_results.json"
    if streaming_path.exists():
        with open(streaming_path) as f:
            streaming = json.load(f)
        t1_p50 = streaming["latency"]["tier1_p50"]
        t3_p50 = streaming["latency"]["tier3_p50"]
        t1_p99 = streaming["latency"]["tier1_p99"]
        t3_p99 = streaming["latency"]["tier3_p99"]
        print(f"  Using streaming pipeline latencies: Tier1 P50={t1_p50:.3f}ms, Tier3 P50={t3_p50:.3f}ms")
    else:
        t1_p50 = float(np.percentile(tier1_latencies, 50))
        t3_p50 = float(np.percentile(tier3_latencies, 50))
        t1_p99 = float(np.percentile(tier1_latencies, 99))
        t3_p99 = float(np.percentile(tier3_latencies, 99))
        print(f"  Warning: streaming_results.json not found, using microbenchmark latencies")

    # Accuracy/F1 at different latency budgets
    budgets = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]

    tier1_accs, tier3_accs, cascade_accs = [], [], []
    tier1_f1s, tier3_f1s, cascade_f1s = [], [], []

    for budget in budgets:
        # Tier 1: available if budget >= Tier 1 P50
        if t1_p50 <= budget:
            tier1_accs.append(t1_auc)
            tier1_f1s.append(t1_f1)
        else:
            tier1_accs.append(0.0)
            tier1_f1s.append(0.0)

        # Tier 3: available if budget >= Tier 3 P50
        if t3_p50 <= budget:
            tier3_accs.append(t3_auc)
            tier3_f1s.append(t3_f1)
        else:
            tier3_accs.append(0.0)
            tier3_f1s.append(0.0)

        # Cascade: best available tier within budget (max of available tiers)
        available_aucs = []
        available_f1s = []
        if t1_p50 <= budget:
            available_aucs.append(t1_auc)
            available_f1s.append(t1_f1)
        if t3_p50 <= budget:
            available_aucs.append(t3_auc)
            available_f1s.append(t3_f1)
        if available_aucs:
            cascade_accs.append(max(available_aucs))
            cascade_f1s.append(max(available_f1s))
        else:
            cascade_accs.append(0.0)
            cascade_f1s.append(0.0)

    # ================================================================
    # SAVE RESULTS
    # ================================================================

    data = {
        "budgets": budgets,
        "tier1_auc_at_budget": tier1_accs,
        "tier3_auc_at_budget": tier3_accs,
        "cascade_auc_at_budget": cascade_accs,
        "tier1_f1_at_budget": tier1_f1s,
        "tier3_f1_at_budget": tier3_f1s,
        "cascade_f1_at_budget": cascade_f1s,
        "tier1_p50_ms": t1_p50,
        "tier1_p99_ms": t1_p99,
        "tier3_p50_ms": t3_p50,
        "tier3_p99_ms": t3_p99,
        "tier1_auc": t1_auc,
        "tier3_auc": t3_auc,
        "tier1_f1": t1_f1,
        "tier3_f1": t3_f1,
        "n_test": len(test_s),
        "n_fraud_test": int(sum(test_l)),
    }

    with open(table_dir / "accuracy_at_latency.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {table_dir / 'accuracy_at_latency.json'}")

    # ================================================================
    # PLOT
    # ================================================================

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(budgets, tier1_accs, 'o-', color='#3498DB',
            label=f'Tier 1 XGBoost (AUC={t1_auc:.3f})', linewidth=2, markersize=5)
    ax.plot(budgets, tier3_accs, 's-', color='#E74C3C',
            label=f'Tier 3 SubGNN+CSP (AUC={t3_auc:.3f})', linewidth=2, markersize=5)
    ax.plot(budgets, cascade_accs, 'D-', color='#27AE60',
            label='StreamRing Cascade', linewidth=2.5, markersize=6)

    # Latency markers
    ax.axvline(x=t1_p50, color='#3498DB', linestyle=':', alpha=0.5,
               label=f'Tier 1 P50={t1_p50:.2f}ms')
    ax.axvline(x=t3_p50, color='#E74C3C', linestyle=':', alpha=0.5,
               label=f'Tier 3 P50={t3_p50:.2f}ms')

    # Budget targets
    ax.axvline(x=5.0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=50.0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=500.0, color='gray', linestyle='--', alpha=0.3)
    ax.text(5.0, 0.05, "5ms\n(T1 target)", ha='center', fontsize=7, color='gray')
    ax.text(50.0, 0.05, "50ms\n(T2 target)", ha='center', fontsize=7, color='gray')
    ax.text(500.0, 0.05, "500ms\n(T3 target)", ha='center', fontsize=7, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel("Latency Budget (ms)", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Detection Quality @ Latency Budget: StreamRing Cascading", fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.005, 600)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(fig_dir / f"fig15_accuracy_at_latency.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved fig15_accuracy_at_latency to {fig_dir}")
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
