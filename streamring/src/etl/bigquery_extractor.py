"""
BigQuery Extractor for Ethereum Classic blockchain data.
Extracts blocks, transactions, traces, and token_transfers from
Google BigQuery public dataset: bigquery-public-data.crypto_ethereum_classic
"""

import os
import argparse
import yaml
from pathlib import Path

from google.cloud import bigquery

DATASET = "bigquery-public-data.crypto_ethereum_classic"


def get_client(project: str = None):
    """Create BigQuery client. Requires ADC, GOOGLE_APPLICATION_CREDENTIALS, or a service account."""
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    return bigquery.Client(project=project)


def _dataset_name(config: dict) -> str:
    dataset = config.get("data", {}).get("bigquery_dataset", DATASET)
    return dataset if "." in dataset else f"{config.get('data', {}).get('bigquery_project', 'bigquery-public-data')}.{dataset}"


def extract_transactions(client: bigquery.Client, start_block: int, end_block: int, output_path: str, dataset: str):
    """Extract transactions for a block range."""
    query = f"""
    SELECT 
        `hash`, nonce, block_hash, block_number, transaction_index,
        from_address, to_address, value, gas, gas_price,
        input, block_timestamp
    FROM `{dataset}.transactions`
    WHERE block_number BETWEEN @start_block AND @end_block
    ORDER BY block_number, transaction_index
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_block", "INT64", start_block),
            bigquery.ScalarQueryParameter("end_block", "INT64", end_block),
        ]
    )
    print(f"Extracting transactions for blocks {start_block}-{end_block}...")
    df = client.query(query, job_config=job_config).to_dataframe()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} transactions to {output_path}")
    return df


def extract_traces(client: bigquery.Client, start_block: int, end_block: int, output_path: str, dataset: str):
    """Extract internal transaction traces for a block range."""
    query = f"""
    SELECT 
        block_number, transaction_hash, transaction_index,
        from_address, to_address, value, 
        trace_type, call_type, reward_type,
        gas, gas_used, subtraces, trace_address, error, status
    FROM `{dataset}.traces`
    WHERE block_number BETWEEN @start_block AND @end_block
    ORDER BY block_number, transaction_index
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_block", "INT64", start_block),
            bigquery.ScalarQueryParameter("end_block", "INT64", end_block),
        ]
    )
    print(f"Extracting traces for blocks {start_block}-{end_block}...")
    df = client.query(query, job_config=job_config).to_dataframe()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} traces to {output_path}")
    return df


def extract_blocks(client: bigquery.Client, start_block: int, end_block: int, output_path: str, dataset: str):
    """Extract block metadata for a block range."""
    query = f"""
    SELECT 
        number, `hash`, parent_hash, nonce, miner,
        difficulty, total_difficulty, size,
        gas_limit, gas_used, `timestamp`, transaction_count
    FROM `{dataset}.blocks`
    WHERE number BETWEEN @start_block AND @end_block
    ORDER BY number
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_block", "INT64", start_block),
            bigquery.ScalarQueryParameter("end_block", "INT64", end_block),
        ]
    )
    print(f"Extracting blocks {start_block}-{end_block}...")
    df = client.query(query, job_config=job_config).to_dataframe()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} blocks to {output_path}")
    return df


def extract_token_transfers(client: bigquery.Client, start_block: int, end_block: int, output_path: str, dataset: str):
    """Extract ERC-20/721 token transfers for a block range."""
    query = f"""
    SELECT 
        token_address, from_address, to_address, value,
        transaction_hash, log_index, block_number
    FROM `{dataset}.token_transfers`
    WHERE block_number BETWEEN @start_block AND @end_block
    ORDER BY block_number, log_index
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_block", "INT64", start_block),
            bigquery.ScalarQueryParameter("end_block", "INT64", end_block),
        ]
    )
    print(f"Extracting token transfers for blocks {start_block}-{end_block}...")
    df = client.query(query, job_config=job_config).to_dataframe()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} token transfers to {output_path}")
    return df


def extract_period(client: bigquery.Client, period_name: str, config: dict, base_dir: str, dataset: str):
    """Extract all data types for a named time period."""
    period = config["data"]["periods"][period_name]
    start = period["start_block"]
    end = period["end_block"]
    out_dir = os.path.join(base_dir, config["data"]["raw_dir"], period_name)

    print(f"\n{'='*60}")
    print(f"Extracting period: {period_name}")
    print(f"Description: {period['description']}")
    print(f"Blocks: {start} - {end}")
    print(f"BigQuery dataset: {dataset}")
    print(f"{'='*60}")

    extract_blocks(client, start, end, os.path.join(out_dir, "blocks.csv"), dataset)
    extract_transactions(client, start, end, os.path.join(out_dir, "transactions.csv"), dataset)
    extract_traces(client, start, end, os.path.join(out_dir, "traces.csv"), dataset)
    extract_token_transfers(client, start, end, os.path.join(out_dir, "token_transfers.csv"), dataset)


def main():
    parser = argparse.ArgumentParser(description="Extract Ethereum Classic data from BigQuery")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--period", default="all", help="Period name or 'all'")
    parser.add_argument("--base-dir", default=".", help="Base directory for output")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = _dataset_name(config)
    project = config.get("data", {}).get("billing_project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    client = get_client(project)

    if args.period == "all":
        for period_name in config["data"]["periods"]:
            extract_period(client, period_name, config, args.base_dir, dataset)
    else:
        extract_period(client, args.period, config, args.base_dir, dataset)

    print("\nExtraction complete!")


if __name__ == "__main__":
    main()
