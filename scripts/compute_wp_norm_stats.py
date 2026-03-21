"""Compute normalization statistics (q01, q99, mean, std, min, max) from waypoint-filtered RLDS.

Action: applies normalize_gripper *before* computing stats so that the
gripper dimension reflects the training convention {0, 1} rather than
the raw RLDS values {-1, +1}.

Proprio (state): only computes stats for the continuous dimensions
(e.g. dims 0-5 for LIBERO), excluding gripper qpos dimensions.  This
matches the VLM tokenizer which quantises only continuous dims and
encodes gripper as a separate binary token.

Usage:
  python scripts/compute_wp_norm_stats.py \
    --rlds_dir /workspace/data/libero/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0 \
    --robot_type libero \
    --output_dir /workspace/data/libero/libero_object_wp_001/norm_stats
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds_dir", type=str, required=True)
    parser.add_argument("--robot_type", type=str, default="libero")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    from openpi.waypoint.robot_config import get_robot_config

    rc = get_robot_config(args.robot_type)

    print(f"Loading RLDS from {args.rlds_dir} ...")
    builder = tfds.builder_from_directory(args.rlds_dir)
    dataset = builder.as_dataset(split="train")

    all_actions = []
    all_proprios = []
    n_episodes = 0
    n_steps = 0

    for episode in dataset:
        n_episodes += 1
        ep_actions = []
        for step in episode["steps"]:
            action = step["action"].numpy().astype(np.float32)
            state = step["observation"]["state"].numpy().astype(np.float32)
            ep_actions.append(action)
            # Only keep continuous proprio dims (exclude gripper)
            all_proprios.append(state[:rc.continuous_proprio_dim])
            n_steps += 1

        # Apply normalize_gripper to actions (converts gripper from raw to {0,1})
        ep_actions = np.stack(ep_actions)
        ep_actions = rc.normalize_gripper(ep_actions)
        for a in ep_actions:
            all_actions.append(a)

        if n_episodes % 50 == 0:
            print(f"  Processed {n_episodes} episodes, {n_steps} steps ...")

    print(f"Total: {n_episodes} episodes, {n_steps} steps")

    all_actions = np.array(all_actions)
    all_proprios = np.array(all_proprios)

    def compute_stats(arr):
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "q01": np.percentile(arr, 1, axis=0).tolist(),
            "q99": np.percentile(arr, 99, axis=0).tolist(),
        }

    stats = {
        "action": compute_stats(all_actions),
        "proprio": compute_stats(all_proprios),
        "num_transitions": n_steps,
        "num_trajectories": n_episodes,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_statistics.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {out_path}")
    print(f"Action dims: {all_actions.shape[1]} (gripper preprocessed via normalize_gripper)")
    print(f"Proprio dims: {all_proprios.shape[1]} (continuous only, gripper excluded)")
    print(f"Action q01: {stats['action']['q01']}")
    print(f"Action q99: {stats['action']['q99']}")
    print(f"Proprio q01: {stats['proprio']['q01']}")
    print(f"Proprio q99: {stats['proprio']['q99']}")


if __name__ == "__main__":
    main()
