"""
StreamRing: Full 3-Tier Cascading Fraud Ring Detection Pipeline.
Integrates Tier 1 (Pattern Matcher), Tier 2 (Incremental GNN), and Tier 3 (Contrastive Subgraph).
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from torch_geometric.utils import k_hop_subgraph

from ..subgraph_matching.ifasi_index import IFASIIndex, TemporalEdge
from ..gnn_models.tgn_backbone import TGNBackbone
from ..gnn_models.subgnn_encoder import FraudRingClassifier
from ..gnn_models.lstm_classifier import LSTMFraudClassifier


class DetectionResult(Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    LIKELY_FRAUD = "likely_fraud"
    FRAUD_RING = "fraud_ring"


@dataclass
class StreamRingOutput:
    """Output of StreamRing detection for a single transaction."""
    result: DetectionResult
    confidence: float
    tier_reached: int  # 1, 2, or 3
    latency_ms: float
    ring_members: Optional[list] = None  # Set of addresses in detected ring


class StreamRingPipeline:
    """
    3-Tier Cascading Real-Time Fraud Ring Detection.
    
    Tier 1 (<5ms):  Pattern matching + XGBoost → filters >90% of transactions
    Tier 2 (<50ms): Incremental TGN + NeuroMatch → scored GNN inference
    Tier 3 (<500ms): Full subgraph extraction + contrastive classification
    """

    def __init__(self, tier1_model: xgb.XGBClassifier,
                 tier2_model,  # SAGEConv-based node classifier or TGNBackbone
                 tier3_model: Union[FraudRingClassifier, LSTMFraudClassifier],
                 ifasi_index: IFASIIndex,
                 tier1_threshold: float = 0.3,
                 tier2_threshold: float = 0.5,
                 tier3_threshold: float = 0.5,
                 device: str = "cpu",
                 tier3_variant: str = "subgnn"):
        if tier3_variant not in ("subgnn", "lstm"):
            raise ValueError(f"tier3_variant must be 'subgnn' or 'lstm', got {tier3_variant!r}")
        self.tier1_model = tier1_model
        self.tier2_model = tier2_model
        self.tier3_model = tier3_model
        self.ifasi = ifasi_index
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.tier3_threshold = tier3_threshold
        self.device = device
        self.tier3_variant = tier3_variant
        self._node_timestamps: dict = {}

        # Incremental graph state for Tier 2/3
        self._edge_index = [[], []]  # COO format, grown incrementally
        self._node_features = {}  # node_id -> feature vector
        self._num_nodes = 0

        # Statistics
        self.stats = {"tier1_count": 0, "tier2_count": 0, "tier3_count": 0,
                      "total_latency": 0.0, "total_count": 0}

    def process_transaction(self, src: int, dst: int, timestamp: float,
                            value: float = 0.0, edge_type: str = "tx",
                            node_features: dict = None) -> StreamRingOutput:
        """
        Process a single streaming transaction through the 3-tier cascade.
        
        Args:
            src: Source node (address) integer ID
            dst: Destination node integer ID
            timestamp: Unix timestamp of transaction
            value: Transaction value (in native currency)
            edge_type: "tx", "trace", or "token_transfer"
            node_features: Optional dict with precomputed features
            
        Returns:
            StreamRingOutput with detection result, confidence, latency
        """
        start_time = time.perf_counter()

        # Update incremental graph state for Tier 2/3
        self._update_graph_state(src, dst, timestamp, node_features)

        # === TIER 1: Pattern Matching (<5ms target) ===
        edge = TemporalEdge(src, dst, timestamp, value, edge_type)
        self.ifasi.insert_edge(edge)

        # Get pattern features for both source and destination
        src_patterns = self.ifasi.get_pattern_features(src, timestamp)
        dst_patterns = self.ifasi.get_pattern_features(dst, timestamp)
        features = np.concatenate([src_patterns, dst_patterns,
                                    [value, timestamp % 86400]])  # 26 features

        tier1_score = self.tier1_model.predict_proba(
            features.reshape(1, -1))[0][1]

        self.stats["tier1_count"] += 1

        if tier1_score < self.tier1_threshold:
            latency = (time.perf_counter() - start_time) * 1000
            self._update_stats(latency)
            return StreamRingOutput(
                result=DetectionResult.SAFE,
                confidence=1.0 - tier1_score,
                tier_reached=1,
                latency_ms=latency,
            )

        # === TIER 2: Incremental GNN (<50ms target) ===
        self.stats["tier2_count"] += 1
        tier2_score = self._run_tier2(src, dst, timestamp, value)

        if tier2_score < self.tier2_threshold:
            latency = (time.perf_counter() - start_time) * 1000
            self._update_stats(latency)
            return StreamRingOutput(
                result=DetectionResult.SUSPICIOUS,
                confidence=tier2_score,
                tier_reached=2,
                latency_ms=latency,
            )

        # === TIER 3: Contrastive Subgraph Classification (<500ms target) ===
        self.stats["tier3_count"] += 1
        tier3_result = self._run_tier3(src, dst, timestamp)

        latency = (time.perf_counter() - start_time) * 1000
        self._update_stats(latency)

        if tier3_result["is_fraud"]:
            return StreamRingOutput(
                result=DetectionResult.FRAUD_RING,
                confidence=tier3_result["confidence"],
                tier_reached=3,
                latency_ms=latency,
                ring_members=tier3_result.get("ring_members"),
            )
        else:
            return StreamRingOutput(
                result=DetectionResult.LIKELY_FRAUD,
                confidence=tier3_result["confidence"],
                tier_reached=3,
                latency_ms=latency,
            )

    def _update_graph_state(self, src: int, dst: int, timestamp: float,
                            node_features: dict = None):
        """Incrementally update the graph state with a new edge."""
        self._edge_index[0].append(src)
        self._edge_index[1].append(dst)
        self._num_nodes = max(self._num_nodes, src + 1, dst + 1)
        # Last-write-wins: stores the most recent timestamp per node.
        # Used by _run_tier3_lstm to sort subgraph nodes into temporal sequence.
        # See design spec §6 for the rationale and acknowledged limitation.
        self._node_timestamps[src] = timestamp
        self._node_timestamps[dst] = timestamp
        if node_features:
            for nid, feat in node_features.items():
                self._node_features[nid] = feat

    def _get_edge_index_tensor(self):
        """Get current graph as edge_index tensor."""
        if not self._edge_index[0]:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)
        return torch.tensor(self._edge_index, dtype=torch.long, device=self.device)

    def _run_tier2(self, src: int, dst: int, timestamp: float,
                   value: float) -> float:
        """
        Run Tier 2 GNN node-level inference.

        Uses SAGEConv-based temporal node classifier on the incrementally
        built graph. Scores both src and dst, returns max.
        """
        if self.tier2_model is None:
            return 0.5

        self.tier2_model.eval()
        with torch.no_grad():
            edge_index = self._get_edge_index_tensor()
            if edge_index.shape[1] < 2:
                return 0.5

            # Build node feature matrix from stored features
            feat_dim = next(iter(self._node_features.values())).shape[-1] \
                if self._node_features else 32
            x = torch.zeros(self._num_nodes, feat_dim, device=self.device)
            for nid, feat in self._node_features.items():
                if nid < self._num_nodes:
                    x[nid] = torch.as_tensor(feat, device=self.device)

            # Forward pass — handles both SAGEConv classifier and TGNBackbone
            try:
                logits = self.tier2_model(x, edge_index)
                probs = torch.sigmoid(logits).squeeze()
                # Score = max of src and dst fraud probability
                src_score = float(probs[src]) if src < len(probs) else 0.5
                dst_score = float(probs[dst]) if dst < len(probs) else 0.5
                return max(src_score, dst_score)
            except Exception:
                return 0.5

    def _extract_subgraph(self, src: int) -> Optional[dict]:
        """
        Extract 2-hop subgraph around src.

        Returns None if subgraph is too small (<3 nodes or <2 edges).
        Applies 300-node truncation. Builds node feature matrix x from
        self._node_features. Returns a single-subgraph batch (batch_idx=0).

        Returns dict keys:
            subset    — Tensor of original global node IDs [N]
            sub_ei    — relabeled edge_index [2, E]
            x         — node feature matrix [N, feat_dim]
            batch_idx — zeros tensor [N]
        """
        edge_index = self._get_edge_index_tensor()
        if edge_index.shape[1] < 2:
            return None

        try:
            subset, sub_ei, _, _ = k_hop_subgraph(
                src, num_hops=2, edge_index=edge_index,
                relabel_nodes=True, num_nodes=self._num_nodes)

            if len(subset) < 3 or sub_ei.shape[1] < 2:
                return None

            if len(subset) > 300:
                subset = subset[:300]
                # sub_ei uses relabeled indices 0..N-1 in the same order as subset,
                # so masking by relabeled index < 300 correctly drops edges to removed nodes.
                mask = (sub_ei[0] < 300) & (sub_ei[1] < 300)
                sub_ei = sub_ei[:, mask]

            feat_dim = next(iter(self._node_features.values())).shape[-1] \
                if self._node_features else 32
            x = torch.zeros(len(subset), feat_dim, device=self.device)
            for i, nid in enumerate(subset.tolist()):
                if nid in self._node_features:
                    x[i] = torch.as_tensor(
                        self._node_features[nid], device=self.device)

            batch_idx = torch.zeros(len(subset), dtype=torch.long,
                                    device=self.device)
            return {"subset": subset, "sub_ei": sub_ei, "x": x,
                    "batch_idx": batch_idx}
        except Exception:
            return None

    def _run_tier3(self, src: int, dst: int, timestamp: float) -> dict:
        """Dispatch to the configured Tier 3 variant."""
        if self.tier3_variant == "lstm":
            return self._run_tier3_lstm(src, dst, timestamp)
        return self._run_tier3_subgnn(src, dst, timestamp)

    def _run_tier3_subgnn(self, src: int, dst: int, timestamp: float) -> dict:
        """
        Tier 3 SubGNN: subgraph-level fraud ring classification.
        position_encoding is not passed; default None is used intentionally.
        """
        if self.tier3_model is None:
            return {"is_fraud": False, "confidence": 0.0, "ring_members": None}

        self.tier3_model.eval()
        with torch.no_grad():
            sg = self._extract_subgraph(src)
            if sg is None:
                return {"is_fraud": False, "confidence": 0.0, "ring_members": None}

            try:
                logits = self.tier3_model(sg["x"], sg["sub_ei"], sg["batch_idx"])
                probs = F.softmax(logits, dim=1)
                fraud_prob = float(probs[0, 1])
                return {
                    "is_fraud": fraud_prob >= self.tier3_threshold,
                    "confidence": fraud_prob,
                    "ring_members": sg["subset"].tolist()
                        if fraud_prob >= self.tier3_threshold else None,
                }
            except Exception:
                return {"is_fraud": False, "confidence": 0.0, "ring_members": None}

    def _run_tier3_lstm(self, src: int, dst: int, timestamp: float) -> dict:
        """
        Tier 3 LSTM: temporal sequence classification over same subgraph.
        Builds relabeled_ts to map global node IDs to timestamps for the LSTM.
        node_timestamps is the only extra kwarg; position_encoding is omitted.
        """
        if self.tier3_model is None:
            return {"is_fraud": False, "confidence": 0.0, "ring_members": None}

        self.tier3_model.eval()
        with torch.no_grad():
            sg = self._extract_subgraph(src)
            if sg is None:
                return {"is_fraud": False, "confidence": 0.0, "ring_members": None}

            try:
                relabeled_ts = [
                    self._node_timestamps.get(int(sg["subset"][i].item()), float("inf"))
                    for i in range(len(sg["subset"]))
                ]
                logits = self.tier3_model(
                    sg["x"], sg["sub_ei"], sg["batch_idx"],
                    node_timestamps=relabeled_ts,
                )
                probs = F.softmax(logits, dim=1)
                fraud_prob = float(probs[0, 1])
                return {
                    "is_fraud": fraud_prob >= self.tier3_threshold,
                    "confidence": fraud_prob,
                    "ring_members": sg["subset"].tolist()
                        if fraud_prob >= self.tier3_threshold else None,
                }
            except Exception:
                return {"is_fraud": False, "confidence": 0.0, "ring_members": None}

    def _update_stats(self, latency_ms: float):
        self.stats["total_latency"] += latency_ms
        self.stats["total_count"] += 1

    def get_statistics(self) -> dict:
        """Get pipeline processing statistics."""
        total = self.stats["total_count"]
        if total == 0:
            return self.stats
        return {
            **self.stats,
            "avg_latency_ms": self.stats["total_latency"] / total,
            "tier1_filter_rate": 1.0 - self.stats["tier2_count"] / max(total, 1),
            "tier2_filter_rate": 1.0 - self.stats["tier3_count"] / max(self.stats["tier2_count"], 1),
        }
