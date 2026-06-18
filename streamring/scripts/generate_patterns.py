import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.subgraph_matching.ifasi_index import IFASIIndex, TemporalEdge


PERIODS = ["dao_hack", "pre_dao", "post_fork", "attack_51_v1",
           "attack_51_v2", "normal_ops"]


def generate_patterns_for_period(base: Path, period: str, force: bool = False) -> Path:
    graph_path = base / "data" / "graphs" / f"{period}_graph.pt"
    out_path = base / "data" / "processed" / f"{period}_patterns.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"  {period}: patterns already exist at {out_path}")
        return out_path

    if not graph_path.exists():
        print(f"  {period}: graph not found, skipping")
        return out_path

    print(f"\n{'=' * 60}")
    print(f"Generating pattern features: {period}")
    print(f"{'=' * 60}")

    graph = torch.load(graph_path, weights_only=False)
    edge_index = graph["edge_index"].numpy()
    edge_time = graph["edge_time"].numpy()
    edge_attr = graph.get("edge_attr")
    if edge_attr is None:
        edge_values = np.zeros(edge_index.shape[1], dtype=np.float32)
    else:
        edge_values = np.asarray(edge_attr)[:, 0].astype(np.float32)

    index = IFASIIndex(max_temporal_gap=3600, window_size=21600)
    all_nodes = sorted(set(edge_index[0].tolist() + edge_index[1].tolist()))
    node_latest_time = {}

    print(f"  Edges: {edge_index.shape[1]:,}")
    print(f"  Nodes with incident edges: {len(all_nodes):,}")
    print("  Inserting edges into IFASI index...")
    start = time.perf_counter()

    for pos, edge_pos in enumerate(edge_time.argsort(), start=1):
        src = int(edge_index[0, edge_pos])
        dst = int(edge_index[1, edge_pos])
        timestamp = float(edge_time[edge_pos])
        value = float(edge_values[edge_pos])
        index.insert_edge(TemporalEdge(src, dst, timestamp, value))
        node_latest_time[src] = max(node_latest_time.get(src, timestamp), timestamp)
        node_latest_time[dst] = max(node_latest_time.get(dst, timestamp), timestamp)

        if pos % 25000 == 0:
            elapsed = time.perf_counter() - start
            print(f"    inserted {pos:,}/{edge_index.shape[1]:,} edges ({elapsed:.1f}s)")

    print("  Computing pattern feature vectors...")
    pattern_features = {}
    for pos, node in enumerate(all_nodes, start=1):
        pattern_features[node] = index.get_pattern_features(
            node, node_latest_time.get(node, 0.0)
        )
        if pos % 2500 == 0:
            elapsed = time.perf_counter() - start
            print(f"    computed {pos:,}/{len(all_nodes):,} nodes ({elapsed:.1f}s)")

    torch.save(pattern_features, str(out_path))
    elapsed = time.perf_counter() - start
    print(f"  Saved {len(pattern_features):,} pattern vectors to {out_path}")
    print(f"  Total time: {elapsed:.1f}s")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate IFASI pattern features")
    parser.add_argument("--period", default="all")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    periods = PERIODS if args.period == "all" else [args.period]

    for period in periods:
        generate_patterns_for_period(base, period, force=args.force)


if __name__ == "__main__":
    main()
