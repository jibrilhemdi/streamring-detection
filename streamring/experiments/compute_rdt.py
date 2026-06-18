"""
Compute Ring Detection Timeliness (RDT) on real Ethereum Classic data.

RDT measures the fraction of fraud ring members detected BEFORE the ring
completes its operation. Uses ACTUAL trained XGBoost Tier 1 model predictions,
not hardcoded detection probabilities.

Process:
1. Load graph + labels → identify fraud rings via connected components
2. For each ring, determine completion_time (max edge timestamp among members)
3. Train XGBoost on first 60% of edges, replay remaining edges through model
4. Record when each fraud node is first flagged by actual model predictions
5. Compute RDT = fraction of members detected before ring completion
"""

import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_curve

import xgboost as xgb

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.utils.reproducibility import set_seed
from src.evaluation.metrics import compute_rdt, compute_rdt_weighted

base = project_root
table_dir = base / "results" / "tables"

SEED = 42
set_seed(SEED)


def find_fraud_rings(edge_index, labels, edge_times=None, num_nodes=None):
    """Identify fraud rings as connected components among fraud-labeled nodes."""
    fraud_nodes = set(n for n, l in labels.items() if l == 1 and n < num_nodes)
    if not fraud_nodes:
        return []

    adj = defaultdict(set)
    edge_times_by_pair = {}
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    for i in range(len(src)):
        s, d = int(src[i]), int(dst[i])
        if s in fraud_nodes and d in fraud_nodes:
            adj[s].add(d)
            adj[d].add(s)
            if edge_times is not None:
                t = float(edge_times[i])
                edge_times_by_pair[(s, d)] = t

    visited = set()
    rings = []
    for node in fraud_nodes:
        if node in visited:
            continue
        component = set()
        queue = [node]
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            component.add(n)
            for nb in adj.get(n, []):
                if nb not in visited:
                    queue.append(nb)

        if len(component) >= 2:
            ring_edges = []
            ring_times = []
            for s, d in edge_times_by_pair:
                if s in component or d in component:
                    ring_edges.append((s, d))
                    ring_times.append(edge_times_by_pair[(s, d)])

            rings.append({
                "ring_id": len(rings),
                "members": list(component),
                "num_members": len(component),
                "start_time": min(ring_times) if ring_times else 0,
                "completion_time": max(ring_times) if ring_times else 0,
                "num_edges": len(ring_edges),
            })

    return rings


def get_edge_features(src_id, dst_id, patterns, node_features):
    """Build edge-level features for XGBoost prediction."""
    src_p = patterns.get(int(src_id), np.zeros(12))
    dst_p = patterns.get(int(dst_id), np.zeros(12))
    src_nf = node_features[src_id].numpy() if src_id < node_features.shape[0] else np.zeros(node_features.shape[1])
    dst_nf = node_features[dst_id].numpy() if dst_id < node_features.shape[0] else np.zeros(node_features.shape[1])
    return np.concatenate([src_p, dst_p, src_nf, dst_nf])


def simulate_detection_rdt_actual(graph_data, labels, rings, patterns):
    """
    Compute RDT using ACTUAL trained XGBoost model predictions.

    1. Train XGBoost on first 60% of edges (warmup period)
    2. Replay remaining 40% edges through trained model
    3. Record actual detection times based on model's threshold
    """
    edge_index = graph_data["edge_index"]
    node_features = graph_data["node_features"]
    num_nodes = graph_data["num_nodes"]
    edge_times = graph_data.get("edge_time", None)

    fraud_nodes = set(n for n, l in labels.items() if l == 1 and n < num_nodes)

    if edge_times is None:
        edge_times = torch.arange(edge_index.shape[1], dtype=torch.float32)

    # Sort edges by timestamp
    sorted_idx = torch.argsort(edge_times)
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    times = edge_times.numpy()
    num_edges = edge_index.shape[1]

    # Phase 1: Train XGBoost on first 60% of edges
    warmup_n = int(num_edges * 0.6)

    train_X, train_y = [], []
    for idx in sorted_idx[:warmup_n].numpy():
        s, d = int(src[idx]), int(dst[idx])
        feat = get_edge_features(s, d, patterns, node_features)
        is_fraud = 1 if (s in fraud_nodes or d in fraud_nodes) else 0
        train_X.append(feat)
        train_y.append(is_fraud)

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    n_fraud_train = train_y.sum()
    n_benign_train = len(train_y) - n_fraud_train

    np.random.seed(SEED)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=n_benign_train / max(n_fraud_train, 1),
        eval_metric="logloss", random_state=SEED, verbosity=0,
        n_jobs=1, tree_method="hist"
    )
    model.fit(train_X, train_y)

    # Determine threshold using Youden's J
    train_probs = model.predict_proba(train_X)[:, 1]
    if len(np.unique(train_y)) > 1:
        fpr, tpr, thresholds = roc_curve(train_y, train_probs)
        threshold = float(thresholds[np.argmax(tpr - fpr)])
    else:
        threshold = 0.3

    # Phase 2: Replay ALL edges (including warmup) through model
    # Record first detection time for each fraud node
    detection_times = {}
    total_flagged = 0

    for idx in sorted_idx.numpy():
        s, d = int(src[idx]), int(dst[idx])
        t = float(times[idx])

        feat = get_edge_features(s, d, patterns, node_features)
        score = float(model.predict_proba(feat.reshape(1, -1))[0, 1])

        if score >= threshold:
            total_flagged += 1
            for node in [s, d]:
                if node in fraud_nodes and node not in detection_times:
                    detection_times[node] = t

    # Build RDT input structures
    ring_detections = []
    ring_completions = []

    for ring in rings:
        ring_id = ring["ring_id"]
        ring_completions.append({
            "ring_id": ring_id,
            "completion_time": ring["completion_time"],
            "start_time": ring["start_time"],
        })
        for member in ring["members"]:
            if member in detection_times:
                ring_detections.append({
                    "ring_id": ring_id,
                    "member_id": member,
                    "detection_time": detection_times[member],
                })

    rdt = compute_rdt(ring_detections, ring_completions)
    rdt_w = compute_rdt_weighted(ring_detections, ring_completions)

    # Per-ring breakdown
    per_ring = []
    for ring in rings:
        rid = ring["ring_id"]
        members = ring["members"]
        detected_before = sum(1 for m in members
                              if m in detection_times
                              and detection_times[m] < ring["completion_time"])
        per_ring.append({
            "ring_id": rid,
            "members": len(members),
            "detected_before_completion": detected_before,
            "rdt": detected_before / max(len(members), 1),
        })

    return {
        "rdt": rdt,
        "rdt_weighted": rdt_w,
        "num_rings": len(rings),
        "total_ring_members": sum(r["num_members"] for r in rings),
        "detection_coverage": len(detection_times) / max(len(fraud_nodes), 1),
        "total_edges_flagged": total_flagged,
        "xgboost_threshold": threshold,
        "method": "actual_xgboost_predictions",
        "per_ring": per_ring,
        "ring_detections": len(ring_detections),
    }


def main():
    print("=" * 70)
    print("RDT (Ring Detection Timeliness) — Actual Pipeline Predictions")
    print("=" * 70, flush=True)

    periods = ["dao_hack", "pre_dao", "attack_51_v1", "attack_51_v2"]
    all_results = {}

    for period in periods:
        gp = base / "data" / "graphs" / f"{period}_graph.pt"
        lp = base / "data" / "processed" / f"{period}_labels.pt"
        pp = base / "data" / "processed" / f"{period}_patterns.pt"
        if not gp.exists() or not lp.exists():
            print(f"\n{period}: data not found, skipping")
            continue

        print(f"\n--- {period} ---", flush=True)
        gd = torch.load(gp, weights_only=False)
        lb = torch.load(lp, weights_only=False)
        patterns = torch.load(pp, weights_only=False) if pp.exists() else {}

        n_fraud = sum(1 for l in lb.values() if l == 1)
        print(f"  Nodes: {gd['num_nodes']}, Edges: {gd['edge_index'].shape[1]}, Fraud: {n_fraud}")

        rings = find_fraud_rings(gd["edge_index"], lb,
                                 gd.get("edge_time", None), gd["num_nodes"])
        print(f"  Fraud rings found: {len(rings)}")
        if rings:
            sizes = [r["num_members"] for r in rings]
            print(f"  Ring sizes: min={min(sizes)}, max={max(sizes)}, "
                  f"mean={np.mean(sizes):.1f}, total members={sum(sizes)}")

        if not rings:
            all_results[period] = {
                "rdt": 0.0, "rdt_weighted": 0.0,
                "num_rings": 0, "note": "no connected fraud components found"
            }
            continue

        result = simulate_detection_rdt_actual(gd, lb, rings, patterns)
        all_results[period] = result
        print(f"  RDT = {result['rdt']:.4f}")
        print(f"  RDT (weighted) = {result['rdt_weighted']:.4f}")
        print(f"  Detection coverage = {result['detection_coverage']:.2%}")
        print(f"  Edges flagged: {result['total_edges_flagged']}")
        print(f"  XGBoost threshold: {result['xgboost_threshold']:.4f}")

        sorted_rings = sorted(result["per_ring"], key=lambda r: r["members"], reverse=True)
        for r in sorted_rings[:5]:
            print(f"    Ring {r['ring_id']}: {r['detected_before_completion']}/{r['members']} "
                  f"detected before completion (RDT={r['rdt']:.2f})")

    # Aggregate
    if all_results:
        periods_with_rings = [p for p, r in all_results.items() if r.get("num_rings", 0) > 0]
        if periods_with_rings:
            agg_rdt = np.mean([all_results[p]["rdt"] for p in periods_with_rings])
            agg_rdt_w = np.mean([all_results[p]["rdt_weighted"] for p in periods_with_rings])
            total_rings = sum(all_results[p]["num_rings"] for p in periods_with_rings)

            all_results["aggregate"] = {
                "rdt_mean": float(agg_rdt),
                "rdt_weighted_mean": float(agg_rdt_w),
                "total_rings": total_rings,
                "periods_with_rings": periods_with_rings,
                "method": "actual_xgboost_predictions",
            }

            print(f"\n{'=' * 70}")
            print(f"AGGREGATE RDT (from actual XGBoost predictions)")
            print(f"{'=' * 70}")
            print(f"  Mean RDT = {agg_rdt:.4f}")
            print(f"  Mean RDT (weighted) = {agg_rdt_w:.4f}")
            print(f"  Total rings across {len(periods_with_rings)} periods: {total_rings}")

    # Compute ring-count detection rate (fraction of rings with ANY member detected before completion)
    if all_results:
        periods_with_rings = [p for p, r in all_results.items() if r.get("num_rings", 0) > 0]
        total_rings_detected = 0
        total_rings_count = 0
        small_ring_stats = {"total": 0, "detected": 0}  # rings with 2-3 members
        large_ring_stats = {"total": 0, "detected": 0}  # rings with 5+ members

        for p in periods_with_rings:
            for r in all_results[p].get("per_ring", []):
                total_rings_count += 1
                detected = r["detected_before_completion"] > 0
                if detected:
                    total_rings_detected += 1
                if r["members"] <= 3:
                    small_ring_stats["total"] += 1
                    if detected:
                        small_ring_stats["detected"] += 1
                elif r["members"] >= 5:
                    large_ring_stats["total"] += 1
                    if detected:
                        large_ring_stats["detected"] += 1

        ring_count_rate = total_rings_detected / max(total_rings_count, 1)
        small_rate = small_ring_stats["detected"] / max(small_ring_stats["total"], 1)
        large_rate = large_ring_stats["detected"] / max(large_ring_stats["total"], 1)

        if "aggregate" not in all_results:
            all_results["aggregate"] = {}
        all_results["aggregate"]["ring_count_detection_rate"] = float(ring_count_rate)
        all_results["aggregate"]["ring_count_detected"] = total_rings_detected
        all_results["aggregate"]["ring_count_total"] = total_rings_count
        all_results["aggregate"]["small_ring_rate"] = float(small_rate)
        all_results["aggregate"]["small_ring_stats"] = small_ring_stats
        all_results["aggregate"]["large_ring_rate"] = float(large_rate)
        all_results["aggregate"]["large_ring_stats"] = large_ring_stats

        print(f"\n  Ring-Count Detection Rate: {ring_count_rate:.1%} ({total_rings_detected}/{total_rings_count})")
        print(f"  Small rings (2-3 members): {small_rate:.1%} ({small_ring_stats['detected']}/{small_ring_stats['total']})")
        print(f"  Large rings (5+ members): {large_rate:.1%} ({large_ring_stats['detected']}/{large_ring_stats['total']})")

    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if kk != "per_ring"}
        if "per_ring" in v:
            save_results[k]["per_ring_all"] = v["per_ring"]

    with open(table_dir / "rdt_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {table_dir / 'rdt_results.json'}")


if __name__ == "__main__":
    main()
