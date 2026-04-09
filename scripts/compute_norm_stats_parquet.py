"""Compute normalization statistics by reading parquet files directly from HuggingFace.

Skips image decoding entirely — only reads 'state' and 'actions' columns.
Produces the same norm_stats.json format as the official compute_norm_stats.py.

Usage:
    uv run scripts/compute_norm_stats_parquet.py --config-name pi05_libero
    # Or with a local parquet cache:
    uv run scripts/compute_norm_stats_parquet.py --config-name pi05_libero --local-parquet-dir /path/to/parquets
"""

import argparse
import pathlib

import numpy as np
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def download_parquet_files(repo_id: str, cache_dir: str | None = None) -> list[str]:
    """List and download all parquet files from the HF dataset (without images)."""
    api = HfApi()
    all_files = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    parquet_paths = [f.path for f in all_files if f.path.endswith(".parquet")]
    print(f"Found {len(parquet_paths)} parquet files in {repo_id}")

    local_paths = []
    for i, p in enumerate(parquet_paths):
        local = hf_hub_download(repo_id, p, repo_type="dataset", cache_dir=cache_dir)
        local_paths.append(local)
        if (i + 1) % 100 == 0:
            print(f"  Downloaded {i + 1}/{len(parquet_paths)}")
    print(f"  Downloaded {len(local_paths)} parquet files")
    return local_paths


def read_columns_from_parquets(paths: list[str], columns: list[str]) -> dict[str, np.ndarray]:
    """Read only the specified columns from parquet files, skipping images."""
    arrays: dict[str, list] = {c: [] for c in columns}
    for path in paths:
        table = pq.read_table(path, columns=columns)
        for col in columns:
            col_data = table[col].to_pylist()
            arrays[col].extend(col_data)
    return {c: np.array(arrays[c], dtype=np.float32) for c in columns}


def main():
    parser = argparse.ArgumentParser(description="Compute norm stats from parquet files (no image decoding)")
    parser.add_argument("--config-name", required=True, help="Training config name, e.g. pi05_libero")
    parser.add_argument("--local-parquet-dir", default=None, help="Local dir with pre-downloaded parquet files")
    parser.add_argument("--repo-id", default=None, help="Override HuggingFace repo_id")
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    data_factory = config.data
    repo_id = args.repo_id or getattr(data_factory, "repo_id", None) or getattr(data_factory, "base_config", None) and getattr(data_factory.base_config, "repo_id", None)
    if repo_id is None:
        repo_id = "physical-intelligence/libero"
    print(f"Config: {args.config_name}, repo_id: {repo_id}")

    if args.local_parquet_dir:
        parquet_dir = pathlib.Path(args.local_parquet_dir)
        local_paths = sorted(str(p) for p in parquet_dir.rglob("*.parquet"))
        print(f"Using {len(local_paths)} local parquet files from {parquet_dir}")
    else:
        local_paths = download_parquet_files(repo_id)

    columns = ["state", "actions"]
    print(f"Reading columns {columns} from {len(local_paths)} parquet files...")
    data = read_columns_from_parquets(local_paths, columns)

    print(f"  state shape:   {data['state'].shape}")
    print(f"  actions shape: {data['actions'].shape}")

    norm_stats = {}
    for key in columns:
        arr = data[key]
        stats = normalize.RunningStats()
        batch_size = 10000
        for i in range(0, len(arr), batch_size):
            stats.update(arr[i : i + batch_size])
        norm_stats[key] = stats.get_statistics()

    output_path = config.assets_dirs / repo_id
    print(f"\nWriting norm stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    for key in columns:
        s = norm_stats[key]
        print(f"\n{key}:")
        print(f"  mean: {s.mean.tolist()}")
        print(f"  std:  {s.std.tolist()}")
        print(f"  q01:  {s.q01.tolist()}")
        print(f"  q99:  {s.q99.tolist()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
