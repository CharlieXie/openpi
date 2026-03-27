"""Benchmark data loading with/without image augmentation.

Usage:
  .venv/bin/python scripts/benchmark_dataload.py --config configs/waypoint_joint_calvin.yaml
"""

import argparse
import logging
import time

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def benchmark(cfg, num_batches=20):
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    from openpi.waypoint.ae_dataset import WaypointAECollator, WaypointAEDataset
    from openpi.waypoint.normalize import load_dataset_statistics
    from openpi.waypoint.robot_config import get_robot_config
    from openpi.waypoint.tokenizer import WaypointTokenizer
    from openpi.waypoint.vlm_dataset import WaypointVLMCollator, WaypointVLMDataset

    rc = get_robot_config(cfg["robot_type"])
    stats = load_dataset_statistics(cfg["dataset_statistics_path"])

    wp_tokenizer = WaypointTokenizer(
        proprio_dim=rc.continuous_proprio_dim,
        num_waypoints=cfg.get("num_waypoints", 7),
        max_token_len=cfg.get("vlm_max_token_len", 256),
        use_gripper_token=True,
    )

    image_aug = cfg.get("image_aug", False)
    image_aug_cfg = cfg.get("image_aug_cfg", None)

    log.info(f"=== Benchmark: image_aug={image_aug} ===")

    # --- VLM DataLoader ---
    vlm_ds = WaypointVLMDataset(
        wp_rlds_dir=cfg["wp_rlds_dir"],
        robot_config=rc,
        dataset_statistics=stats,
        wp_tokenizer=wp_tokenizer,
        norm_type=cfg.get("norm_type", "q99"),
        num_waypoints=cfg.get("num_waypoints", 7),
        stride=cfg.get("stride", 4),
        shuffle_buffer_size=cfg.get("vlm_shuffle_buffer_size", 5000),
        image_aug=image_aug,
        image_aug_cfg=image_aug_cfg if image_aug else None,
    )
    vlm_loader = torch.utils.data.DataLoader(
        vlm_ds, batch_size=cfg.get("vlm_batch_size", 64),
        num_workers=0, collate_fn=WaypointVLMCollator(),
    )

    # --- AE DataLoader ---
    ae_ds = WaypointAEDataset(
        original_rlds_dir=cfg["original_rlds_dir"],
        wp_indices_path=cfg["wp_indices_path"],
        robot_config=rc,
        dataset_statistics=stats,
        norm_type=cfg.get("norm_type", "q99"),
        max_duration=cfg.get("max_duration", 32),
        horizon_steps=cfg.get("horizon_steps", 32),
        model_action_dim=cfg.get("model_action_dim", 32),
        model_proprio_dim=cfg.get("model_proprio_dim", 32),
        shuffle_buffer_size=cfg.get("ae_shuffle_buffer_size", 1000),
        image_aug=image_aug,
        image_aug_cfg=image_aug_cfg if image_aug else None,
    )
    ae_loader = torch.utils.data.DataLoader(
        ae_ds, batch_size=cfg.get("ae_batch_size", 64),
        num_workers=0, collate_fn=WaypointAECollator(),
    )

    # --- Benchmark VLM loading ---
    log.info(f"Timing VLM DataLoader ({num_batches} batches, bs={cfg.get('vlm_batch_size', 64)})...")
    vlm_iter = iter(vlm_loader)
    vlm_times = []
    for i in range(num_batches):
        t0 = time.perf_counter()
        batch = next(vlm_iter)
        dt = time.perf_counter() - t0
        vlm_times.append(dt)
        if i < 3 or i == num_batches - 1:
            log.info(f"  VLM batch {i}: {dt:.3f}s")

    # --- Benchmark AE loading ---
    log.info(f"Timing AE DataLoader ({num_batches} batches, bs={cfg.get('ae_batch_size', 64)})...")
    ae_iter = iter(ae_loader)
    ae_times = []
    for i in range(num_batches):
        t0 = time.perf_counter()
        batch = next(ae_iter)
        dt = time.perf_counter() - t0
        ae_times.append(dt)
        if i < 3 or i == num_batches - 1:
            log.info(f"  AE  batch {i}: {dt:.3f}s")

    # --- Benchmark combined (simulating one training step) ---
    log.info(f"Timing combined VLM+AE loading ({num_batches} steps)...")
    vlm_iter2 = iter(vlm_loader)
    ae_iter2 = iter(ae_loader)
    combined_times = []
    for i in range(num_batches):
        t0 = time.perf_counter()
        vb = next(vlm_iter2)
        ab = next(ae_iter2)
        dt = time.perf_counter() - t0
        combined_times.append(dt)
        if i < 3 or i == num_batches - 1:
            log.info(f"  Combined step {i}: {dt:.3f}s")

    # Skip first 2 batches (warmup)
    skip = min(2, len(vlm_times) - 1)
    vlm_avg = np.mean(vlm_times[skip:])
    ae_avg = np.mean(ae_times[skip:])
    combined_avg = np.mean(combined_times[skip:])

    log.info(f"\n{'='*60}")
    log.info(f"Results (image_aug={image_aug}, skip first {skip}):")
    log.info(f"  VLM avg:      {vlm_avg:.3f}s/batch")
    log.info(f"  AE  avg:      {ae_avg:.3f}s/batch")
    log.info(f"  Combined avg: {combined_avg:.3f}s/step")
    log.info(f"  VLM total:    {sum(vlm_times):.1f}s")
    log.info(f"  AE  total:    {sum(ae_times):.1f}s")
    log.info(f"{'='*60}")

    return {
        "vlm_avg": vlm_avg,
        "ae_avg": ae_avg,
        "combined_avg": combined_avg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=15)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Test 1: With augmentation as configured
    result_aug = benchmark(cfg, num_batches=args.num_batches)

    # Test 2: Without augmentation
    cfg_no_aug = dict(cfg)
    cfg_no_aug["image_aug"] = False
    result_no_aug = benchmark(cfg_no_aug, num_batches=args.num_batches)

    # Summary
    log.info(f"\n{'='*60}")
    log.info("COMPARISON SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  With aug:    VLM={result_aug['vlm_avg']:.3f}s  AE={result_aug['ae_avg']:.3f}s  Combined={result_aug['combined_avg']:.3f}s")
    log.info(f"  Without aug: VLM={result_no_aug['vlm_avg']:.3f}s  AE={result_no_aug['ae_avg']:.3f}s  Combined={result_no_aug['combined_avg']:.3f}s")
    diff = result_aug["combined_avg"] - result_no_aug["combined_avg"]
    log.info(f"  Augmentation overhead per step: {diff:.3f}s ({diff/result_no_aug['combined_avg']*100:.1f}%)")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
