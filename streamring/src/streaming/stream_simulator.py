"""
Stream Simulator: Replays Ethereum Classic transactions in timestamp order
to simulate real-time streaming for evaluation.
"""

import time
from collections import deque
from typing import Callable, Optional

import pandas as pd
import numpy as np


class StreamSimulator:
    """
    Replays blockchain transactions as a stream, ordered by timestamp.
    Supports configurable replay speed and callback-based processing.
    """

    def __init__(self, replay_speed: float = 1.0,
                 warmup_transactions: int = 1000):
        """
        Args:
            replay_speed: Multiplier for replay speed.
                         1.0 = real-time, 10.0 = 10× faster, 0 = as fast as possible
            warmup_transactions: Number of initial transactions to process
                                before starting evaluation (builds graph state)
        """
        self.replay_speed = replay_speed
        self.warmup_transactions = warmup_transactions
        self.results = []
        self.latencies = []

    def load_transactions(self, tx_path: str, traces_path: str = None,
                          tt_path: str = None) -> pd.DataFrame:
        """Load and merge all transaction types, sorted by timestamp."""
        dfs = []

        tx_df = pd.read_csv(tx_path)
        tx_df["edge_type"] = "transaction"
        tx_df["timestamp"] = pd.to_numeric(tx_df["block_timestamp"], errors="coerce")
        dfs.append(tx_df[["from_address", "to_address", "value",
                          "timestamp", "edge_type"]].dropna())

        if traces_path:
            tr_df = pd.read_csv(traces_path)
            tr_df["edge_type"] = "trace"
            # Traces don't always have timestamps; use block_number as proxy
            if "block_timestamp" in tr_df.columns:
                tr_df["timestamp"] = pd.to_numeric(tr_df["block_timestamp"], errors="coerce")
            else:
                tr_df["timestamp"] = tr_df["block_number"].astype(float) * 14  # ~14s per block
            dfs.append(tr_df[["from_address", "to_address", "value",
                              "timestamp", "edge_type"]].dropna())

        if tt_path:
            tt_df = pd.read_csv(tt_path)
            tt_df["edge_type"] = "token_transfer"
            tt_df["timestamp"] = tt_df["block_number"].astype(float) * 14
            dfs.append(tt_df[["from_address", "to_address", "value",
                              "timestamp", "edge_type"]].dropna())

        combined = pd.concat(dfs, ignore_index=True)
        combined["value"] = pd.to_numeric(combined["value"], errors="coerce").fillna(0)
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        print(f"Loaded {len(combined)} events for streaming simulation")
        print(f"Time range: {combined['timestamp'].min():.0f} - "
              f"{combined['timestamp'].max():.0f}")
        return combined

    def run(self, events: pd.DataFrame,
            process_fn: Callable,
            addr_to_id: dict,
            labels: Optional[dict] = None) -> dict:
        """
        Run streaming simulation.
        
        Args:
            events: DataFrame with from_address, to_address, value, timestamp, edge_type
            process_fn: Callable(src_id, dst_id, timestamp, value, edge_type) -> result
            addr_to_id: Mapping from address string to integer ID
            labels: Optional dict mapping (src_addr, dst_addr, timestamp) -> is_fraud
            
        Returns:
            Dict with simulation results and statistics
        """
        self.results = []
        self.latencies = []
        prev_sim_time = None
        processed = 0
        is_warmup = True

        print(f"\nStarting stream simulation...")
        print(f"Warmup: {self.warmup_transactions} transactions")
        print(f"Replay speed: {'max' if self.replay_speed == 0 else f'{self.replay_speed}×'}")

        for idx, row in events.iterrows():
            src_addr = row["from_address"]
            dst_addr = row["to_address"]
            src_id = addr_to_id.get(src_addr)
            dst_id = addr_to_id.get(dst_addr)

            if src_id is None or dst_id is None:
                continue

            sim_time = row["timestamp"]
            value = float(row["value"])
            edge_type = row["edge_type"]

            # Simulate real-time delay
            if self.replay_speed > 0 and prev_sim_time is not None and not is_warmup:
                delay = (sim_time - prev_sim_time) / self.replay_speed
                if delay > 0 and delay < 10:  # Cap at 10 seconds real time
                    time.sleep(delay)
            prev_sim_time = sim_time

            # Process the transaction
            start = time.perf_counter()
            result = process_fn(src_id, dst_id, sim_time, value, edge_type)
            latency_ms = (time.perf_counter() - start) * 1000

            processed += 1
            if processed == self.warmup_transactions:
                is_warmup = False
                print(f"Warmup complete. Starting evaluation...")

            if not is_warmup:
                self.latencies.append(latency_ms)
                if labels is not None:
                    label_key = (src_addr, dst_addr, sim_time)
                    actual = labels.get(label_key, 0)
                    self.results.append({
                        "predicted": result,
                        "actual": actual,
                        "latency_ms": latency_ms,
                        "timestamp": sim_time,
                    })

            if processed % 10000 == 0:
                phase = "warmup" if is_warmup else "eval"
                avg_lat = np.mean(self.latencies[-1000:]) if self.latencies else 0
                print(f"  [{phase}] Processed {processed}, "
                      f"avg latency (last 1K): {avg_lat:.2f}ms")

        return self._compile_results()

    def _compile_results(self) -> dict:
        """Compile simulation results into summary statistics."""
        latencies = np.array(self.latencies) if self.latencies else np.array([0])
        return {
            "total_processed": len(self.latencies),
            "latency_p50": np.percentile(latencies, 50),
            "latency_p95": np.percentile(latencies, 95),
            "latency_p99": np.percentile(latencies, 99),
            "latency_mean": np.mean(latencies),
            "latency_std": np.std(latencies),
            "throughput_per_sec": len(self.latencies) / max(sum(self.latencies) / 1000, 0.001),
            "results": self.results,
        }
