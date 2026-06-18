"""
Fraud pattern template definitions and subgraph matching for Tier 1.
Defines structural patterns (cycles, fans, chains, stars) and enumeration logic.
"""

from dataclasses import dataclass, field
from enum import Enum


class PatternType(Enum):
    CYCLE_2 = "cycle_2"          # A -> B -> A (wash trading)
    CYCLE_3 = "cycle_3"          # A -> B -> C -> A (circular laundering)
    CYCLE_4 = "cycle_4"          # 4-node cycle
    FAN_OUT_3 = "fan_out_3"      # A -> {B, C, D} (fund distribution)
    FAN_OUT_5 = "fan_out_5"      # A -> {B, C, D, E, F}
    FAN_IN_3 = "fan_in_3"        # {B, C, D} -> A (consolidation)
    FAN_IN_5 = "fan_in_5"        # 5+ sources -> single target
    CHAIN_3 = "chain_3"          # A -> B -> C (peeling chain, 3 hops)
    CHAIN_5 = "chain_5"          # A -> B -> C -> D -> E (5 hops)
    STAR_3 = "star_3"            # Hub -> {S1,S2,S3} -> Hub
    STAR_5 = "star_5"            # Hub -> {S1,...,S5} -> Hub
    TEMPORAL_BURST = "burst"     # Many edges in short time window


@dataclass
class FraudPattern:
    """Definition of a fraud pattern template."""
    pattern_type: PatternType
    min_nodes: int
    max_nodes: int
    min_edges: int
    description: str
    temporal_constraint_seconds: float = 3600.0  # Max time span for pattern
    
    # Structural constraints
    requires_cycle: bool = False
    min_fan_degree: int = 0


# Pre-defined fraud pattern library
FRAUD_PATTERNS = {
    PatternType.CYCLE_2: FraudPattern(
        pattern_type=PatternType.CYCLE_2,
        min_nodes=2, max_nodes=2, min_edges=2,
        description="Bidirectional transfer (wash trading indicator)",
        requires_cycle=True,
        temporal_constraint_seconds=3600,
    ),
    PatternType.CYCLE_3: FraudPattern(
        pattern_type=PatternType.CYCLE_3,
        min_nodes=3, max_nodes=3, min_edges=3,
        description="3-node circular money flow (laundering)",
        requires_cycle=True,
        temporal_constraint_seconds=7200,
    ),
    PatternType.CYCLE_4: FraudPattern(
        pattern_type=PatternType.CYCLE_4,
        min_nodes=4, max_nodes=4, min_edges=4,
        description="4-node circular money flow",
        requires_cycle=True,
        temporal_constraint_seconds=14400,
    ),
    PatternType.FAN_OUT_3: FraudPattern(
        pattern_type=PatternType.FAN_OUT_3,
        min_nodes=4, max_nodes=10, min_edges=3,
        description="Single source distributing to 3+ targets",
        min_fan_degree=3,
        temporal_constraint_seconds=3600,
    ),
    PatternType.FAN_OUT_5: FraudPattern(
        pattern_type=PatternType.FAN_OUT_5,
        min_nodes=6, max_nodes=20, min_edges=5,
        description="Single source distributing to 5+ targets",
        min_fan_degree=5,
        temporal_constraint_seconds=7200,
    ),
    PatternType.FAN_IN_3: FraudPattern(
        pattern_type=PatternType.FAN_IN_3,
        min_nodes=4, max_nodes=10, min_edges=3,
        description="3+ sources consolidating to single target",
        min_fan_degree=3,
        temporal_constraint_seconds=3600,
    ),
    PatternType.FAN_IN_5: FraudPattern(
        pattern_type=PatternType.FAN_IN_5,
        min_nodes=6, max_nodes=20, min_edges=5,
        description="5+ sources consolidating to single target",
        min_fan_degree=5,
        temporal_constraint_seconds=7200,
    ),
    PatternType.CHAIN_3: FraudPattern(
        pattern_type=PatternType.CHAIN_3,
        min_nodes=3, max_nodes=3, min_edges=2,
        description="3-hop sequential transfer chain",
        temporal_constraint_seconds=3600,
    ),
    PatternType.CHAIN_5: FraudPattern(
        pattern_type=PatternType.CHAIN_5,
        min_nodes=5, max_nodes=5, min_edges=4,
        description="5-hop peeling chain",
        temporal_constraint_seconds=14400,
    ),
    PatternType.STAR_3: FraudPattern(
        pattern_type=PatternType.STAR_3,
        min_nodes=4, max_nodes=4, min_edges=6,
        description="Hub distributes then collects from 3 spokes",
        requires_cycle=True,
        min_fan_degree=3,
        temporal_constraint_seconds=7200,
    ),
    PatternType.STAR_5: FraudPattern(
        pattern_type=PatternType.STAR_5,
        min_nodes=6, max_nodes=6, min_edges=10,
        description="Hub distributes then collects from 5 spokes",
        requires_cycle=True,
        min_fan_degree=5,
        temporal_constraint_seconds=14400,
    ),
    PatternType.TEMPORAL_BURST: FraudPattern(
        pattern_type=PatternType.TEMPORAL_BURST,
        min_nodes=2, max_nodes=50, min_edges=10,
        description="Abnormally high transaction frequency in short window",
        temporal_constraint_seconds=300,  # 5 minutes
    ),
}
