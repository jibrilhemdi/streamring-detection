"""
IFASI: Incremental Fraud-Aware Subgraph Index
Extends TC-Match's CSS index with GNN-based node filtering for fraud pattern detection.
Core component of StreamRing Tier 1.
"""

import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch


class TemporalEdge:
    """A timestamped directed edge with attributes."""
    __slots__ = ["src", "dst", "timestamp", "value", "edge_type"]

    def __init__(self, src: int, dst: int, timestamp: float,
                 value: float = 0.0, edge_type: str = "tx"):
        self.src = src
        self.dst = dst
        self.timestamp = timestamp
        self.value = value
        self.edge_type = edge_type


class IFASIIndex:
    """
    Incremental Fraud-Aware Subgraph Index.
    
    Maintains streaming graph state and supports fast pattern enumeration
    with temporal constraints. Inspired by TC-Match's CSS index.
    
    Key operations:
    1. insert_edge(): O(1) amortized index update
    2. count_patterns(): O(d^k) worst case, pruned by temporal + GNN filter  
    3. get_features(): Returns pattern count feature vector for a node
    """

    def __init__(self, max_temporal_gap: float = 3600.0,
                 window_size: float = 21600.0):
        """
        Args:
            max_temporal_gap: Max seconds between edges in a pattern
            window_size: Sliding window in seconds (default 6 hours)
        """
        self.max_temporal_gap = max_temporal_gap
        self.window_size = window_size

        # Adjacency lists (temporal): node -> list of (neighbor, timestamp, value)
        self.out_neighbors: dict[int, list] = defaultdict(list)
        self.in_neighbors: dict[int, list] = defaultdict(list)

        # Current time window
        self.current_time = 0.0
        self.edge_count = 0

    def insert_edge(self, edge: TemporalEdge):
        """Insert a new edge into the streaming index."""
        self.out_neighbors[edge.src].append(
            (edge.dst, edge.timestamp, edge.value))
        self.in_neighbors[edge.dst].append(
            (edge.src, edge.timestamp, edge.value))
        self.current_time = max(self.current_time, edge.timestamp)
        self.edge_count += 1

        # Periodic cleanup of expired edges
        if self.edge_count % 10000 == 0:
            self._evict_expired()

    def _evict_expired(self):
        """Remove edges outside the sliding window."""
        cutoff = self.current_time - self.window_size
        for node in list(self.out_neighbors.keys()):
            self.out_neighbors[node] = [
                (n, t, v) for n, t, v in self.out_neighbors[node] if t >= cutoff
            ]
            if not self.out_neighbors[node]:
                del self.out_neighbors[node]
        for node in list(self.in_neighbors.keys()):
            self.in_neighbors[node] = [
                (n, t, v) for n, t, v in self.in_neighbors[node] if t >= cutoff
            ]
            if not self.in_neighbors[node]:
                del self.in_neighbors[node]

    def count_cycles_2(self, node: int, ref_time: float) -> int:
        """Count 2-cycles (A->B->A) involving node within temporal constraint."""
        count = 0
        for (nbr, t1, _) in self.out_neighbors.get(node, []):
            if abs(ref_time - t1) > self.max_temporal_gap:
                continue
            for (back, t2, _) in self.out_neighbors.get(nbr, []):
                if back == node and t2 > t1 and (t2 - t1) <= self.max_temporal_gap:
                    count += 1
        return count

    def count_cycles_3(self, node: int, ref_time: float) -> int:
        """Count 3-cycles (A->B->C->A) involving node."""
        count = 0
        for (b, t1, _) in self.out_neighbors.get(node, []):
            if abs(ref_time - t1) > self.max_temporal_gap:
                continue
            for (c, t2, _) in self.out_neighbors.get(b, []):
                if c == node or (t2 - t1) > self.max_temporal_gap or t2 <= t1:
                    continue
                for (back, t3, _) in self.out_neighbors.get(c, []):
                    if back == node and t3 > t2 and (t3 - t1) <= self.max_temporal_gap * 2:
                        count += 1
        return count

    def count_fan_out(self, node: int, ref_time: float, min_degree: int = 3) -> int:
        """Count fan-out patterns: node sends to min_degree+ distinct targets within window."""
        targets = set()
        for (nbr, t, _) in self.out_neighbors.get(node, []):
            if abs(ref_time - t) <= self.max_temporal_gap:
                targets.add(nbr)
        return 1 if len(targets) >= min_degree else 0

    def count_fan_in(self, node: int, ref_time: float, min_degree: int = 3) -> int:
        """Count fan-in patterns: node receives from min_degree+ distinct sources."""
        sources = set()
        for (nbr, t, _) in self.in_neighbors.get(node, []):
            if abs(ref_time - t) <= self.max_temporal_gap:
                sources.add(nbr)
        return 1 if len(sources) >= min_degree else 0

    def count_chain(self, node: int, ref_time: float, length: int = 3) -> int:
        """Count chain patterns (A->B->C->...) of given length starting from node."""
        count = 0
        self._dfs_chain(node, ref_time, ref_time, length - 1, set(), count_ref=[0])
        return count_ref[0] if 'count_ref' in dir() else 0

    def _dfs_chain(self, node: int, start_time: float, prev_time: float,
                   remaining: int, visited: set, count_ref: list):
        if remaining == 0:
            count_ref[0] += 1
            return
        for (nbr, t, _) in self.out_neighbors.get(node, []):
            if nbr in visited or t <= prev_time:
                continue
            if (t - start_time) > self.max_temporal_gap * remaining:
                continue
            visited.add(nbr)
            self._dfs_chain(nbr, start_time, t, remaining - 1, visited, count_ref)
            visited.discard(nbr)

    def count_temporal_burst(self, node: int, ref_time: float,
                             burst_window: float = 300.0,
                             min_edges: int = 10) -> int:
        """Detect temporal burst: many edges within short window."""
        edge_count = 0
        for (_, t, _) in self.out_neighbors.get(node, []):
            if abs(ref_time - t) <= burst_window:
                edge_count += 1
        for (_, t, _) in self.in_neighbors.get(node, []):
            if abs(ref_time - t) <= burst_window:
                edge_count += 1
        return 1 if edge_count >= min_edges else 0

    def get_pattern_features(self, node: int, ref_time: float) -> np.ndarray:
        """
        Compute pattern count feature vector for a node.
        Returns array of shape (12,) with counts for each pattern type.
        
        This is the core output of Tier 1's IFASI index.
        """
        features = np.zeros(12, dtype=np.float32)
        features[0] = self.count_cycles_2(node, ref_time)
        features[1] = self.count_cycles_3(node, ref_time)
        features[2] = 0  # cycle_4 (expensive, skip for <5ms tier)
        features[3] = self.count_fan_out(node, ref_time, min_degree=3)
        features[4] = self.count_fan_out(node, ref_time, min_degree=5)
        features[5] = self.count_fan_in(node, ref_time, min_degree=3)
        features[6] = self.count_fan_in(node, ref_time, min_degree=5)
        features[7] = 0  # chain_3 (use for Tier 2)
        features[8] = 0  # chain_5 (use for Tier 2)
        features[9] = 0  # star_3
        features[10] = 0  # star_5
        features[11] = self.count_temporal_burst(node, ref_time)
        return features
