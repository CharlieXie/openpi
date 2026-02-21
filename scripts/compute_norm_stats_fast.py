"""Fast norm stats computation by reading parquet columns directly (skipping images).

Equivalent to compute_norm_stats.py but skips image loading/decoding.
RunningStats.update() does reshape(-1, last_dim), so per-row actions == flattened sequences.
"""
import pathlib
import multiprocessing
import numpy as np
import pyarrow.parquet as pq
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config


def _read_parquet_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read only state and actions columns from a parquet file."""
    table = pq.read_table(path, columns=["state", "actions"])
    state = np.stack(table["state"].to_pylist()).astype(np.float32)
    actions = np.stack(table["actions"].to_pylist()).astype(np.float32)
    return state, actions


def main(
    config_name: str,
    num_workers: int = 80,
    data_dir: str | None = None,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Data config must have a repo_id")

    # Find parquet files
    if data_dir is None:
        import os
        hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
        data_dir = pathlib.Path(hf_home).expanduser() / "lerobot" / repo_id
    else:
        data_dir = pathlib.Path(data_dir)

    parquet_files = sorted(data_dir.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}/data/")

    print(f"Found {len(parquet_files)} parquet files in {data_dir}")
    print(f"Reading state & actions columns with {num_workers} workers (skipping images)...")

    # Read files in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(pool.imap_unordered(
            _read_parquet_file,
            [str(f) for f in parquet_files],
            chunksize=max(1, len(parquet_files) // (num_workers * 4)),
        ))

    print(f"Aggregating statistics from {len(results)} files...")

    # Compute running stats
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()

    for state_arr, action_arr in results:
        state_stats.update(state_arr)
        action_stats.update(action_arr)

    total_frames = sum(s.shape[0] for s, _ in results)
    print(f"Total frames processed: {total_frames}")

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print("Done!")

    # Print summary
    print(f"\n  state  mean: {norm_stats['state'].mean}")
    print(f"  state  std:  {norm_stats['state'].std}")
    print(f"  actions mean: {norm_stats['actions'].mean}")
    print(f"  actions std:  {norm_stats['actions'].std}")


if __name__ == "__main__":
    tyro.cli(main)
