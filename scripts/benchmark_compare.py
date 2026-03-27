"""Head-to-head LIBERO vs CALVIN data loading benchmark.

Profiles: RLDS read, image processing, augmentation, collation.

Usage:
  .venv/bin/python scripts/benchmark_compare.py
"""

import logging
import time
import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def benchmark_dataset(name, cfg, num_batches=10):
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

    log.info(f"\n{'='*60}")
    log.info(f"Benchmarking: {name} (image_aug={image_aug})")
    log.info(f"{'='*60}")

    # VLM loader
    vlm_ds = WaypointVLMDataset(
        wp_rlds_dir=cfg["wp_rlds_dir"],
        robot_config=rc, dataset_statistics=stats, wp_tokenizer=wp_tokenizer,
        norm_type=cfg.get("norm_type", "q99"),
        num_waypoints=cfg.get("num_waypoints", 7),
        stride=cfg.get("stride", 4),
        shuffle_buffer_size=cfg.get("vlm_shuffle_buffer_size", 5000),
        image_aug=image_aug,
        image_aug_cfg=image_aug_cfg if image_aug else None,
    )
    vlm_loader = torch.utils.data.DataLoader(
        vlm_ds, batch_size=cfg.get("vlm_batch_size", 192),
        num_workers=0, collate_fn=WaypointVLMCollator(),
    )

    # AE loader
    ae_ds = WaypointAEDataset(
        original_rlds_dir=cfg["original_rlds_dir"],
        wp_indices_path=cfg["wp_indices_path"],
        robot_config=rc, dataset_statistics=stats,
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
        ae_ds, batch_size=cfg.get("ae_batch_size", 192),
        num_workers=0, collate_fn=WaypointAECollator(),
    )

    # Benchmark combined
    vlm_iter = iter(vlm_loader)
    ae_iter = iter(ae_loader)

    vlm_times = []
    ae_times = []
    combined_times = []

    for i in range(num_batches):
        t0 = time.perf_counter()
        vb = next(vlm_iter)
        t1 = time.perf_counter()
        ab = next(ae_iter)
        t2 = time.perf_counter()

        vlm_t = t1 - t0
        ae_t = t2 - t1
        total_t = t2 - t0
        vlm_times.append(vlm_t)
        ae_times.append(ae_t)
        combined_times.append(total_t)
        log.info(f"  [{name}] step {i}: VLM={vlm_t:.2f}s  AE={ae_t:.2f}s  total={total_t:.2f}s")

    skip = 2
    r = {
        "vlm_avg": np.mean(vlm_times[skip:]),
        "ae_avg": np.mean(ae_times[skip:]),
        "combined_avg": np.mean(combined_times[skip:]),
    }
    log.info(f"  [{name}] Average (skip {skip}): VLM={r['vlm_avg']:.2f}s  AE={r['ae_avg']:.2f}s  Combined={r['combined_avg']:.2f}s")
    return r


def main():
    # CALVIN config (current)
    calvin_cfg = {
        "robot_type": "calvin",
        "original_rlds_dir": "/workspace/data/calvin_abc_rlds",
        "wp_indices_path": "/workspace/data/calvin_abc_wp_0_02/waypoint_indices.json",
        "wp_rlds_dir": "/workspace/data/calvin_abc_wp_0_02/calvin_abc_wp/1.0.0",
        "dataset_statistics_path": "/workspace/data/dataset_statistics.json",
        "vlm_batch_size": 192,
        "ae_batch_size": 192,
        "num_waypoints": 7,
        "vlm_max_token_len": 256,
        "stride": 1,
        "norm_type": "q99",
        "max_duration": 32,
        "horizon_steps": 32,
        "model_action_dim": 32,
        "model_proprio_dim": 32,
        "vlm_shuffle_buffer_size": 20000,
        "ae_shuffle_buffer_size": 5000,
        "image_aug": True,
        "image_aug_cfg": {
            "random_resized_crop_scale": [0.9, 1.0],
            "brightness": 0.2,
            "contrast": [0.8, 1.2],
            "saturation": [0.8, 1.2],
            "hue": 0.05,
        },
    }

    # LIBERO config (same batch size, same aug, same model dims)
    libero_cfg = {
        "robot_type": "libero",
        "original_rlds_dir": "/workspace/data/libero/whole/modified_libero_rlds/libero_all_no_noops/1.0.0",
        "wp_indices_path": "/workspace/data/libero/whole/libero_object_wp_001/waypoint_indices.json",
        "wp_rlds_dir": "/workspace/data/libero/whole/libero_object_wp_001/waypoint_filtered_rlds__libero/1.0.0",
        "dataset_statistics_path": "/workspace/data/libero/whole/dataset_statistics.json",
        "vlm_batch_size": 192,
        "ae_batch_size": 192,
        "num_waypoints": 7,
        "vlm_max_token_len": 256,
        "stride": 1,
        "norm_type": "q99",
        "max_duration": 32,
        "horizon_steps": 32,
        "model_action_dim": 32,
        "model_proprio_dim": 32,
        "vlm_shuffle_buffer_size": 5000,
        "ae_shuffle_buffer_size": 1000,
        "image_aug": True,
        "image_aug_cfg": {
            "random_resized_crop_scale": [0.9, 0.9],
            "brightness": 0.2,
            "contrast": [0.8, 1.2],
            "saturation": [0.8, 1.2],
            "hue": 0.05,
        },
    }

    # LIBERO with CALVIN's shuffle buffer sizes (to isolate shuffle buffer effect)
    libero_big_buf_cfg = dict(libero_cfg)
    libero_big_buf_cfg["vlm_shuffle_buffer_size"] = 20000
    libero_big_buf_cfg["ae_shuffle_buffer_size"] = 5000

    # CALVIN with LIBERO's shuffle buffer sizes (to isolate shuffle buffer effect)
    calvin_small_buf_cfg = dict(calvin_cfg)
    calvin_small_buf_cfg["vlm_shuffle_buffer_size"] = 5000
    calvin_small_buf_cfg["ae_shuffle_buffer_size"] = 1000

    # CALVIN without augmentation
    calvin_no_aug_cfg = dict(calvin_cfg)
    calvin_no_aug_cfg["image_aug"] = False

    N = 10

    results = {}

    # Test 1: LIBERO with its normal config
    results["LIBERO (buf=5k/1k, aug=on)"] = benchmark_dataset("LIBERO (buf=5k/1k, aug=on)", libero_cfg, N)

    # Test 2: CALVIN with its current config
    results["CALVIN (buf=20k/5k, aug=on)"] = benchmark_dataset("CALVIN (buf=20k/5k, aug=on)", calvin_cfg, N)

    # Test 3: CALVIN with small buffers (same as LIBERO)
    results["CALVIN (buf=5k/1k, aug=on)"] = benchmark_dataset("CALVIN (buf=5k/1k, aug=on)", calvin_small_buf_cfg, N)

    # Test 4: CALVIN without augmentation
    results["CALVIN (buf=20k/5k, aug=off)"] = benchmark_dataset("CALVIN (buf=20k/5k, aug=off)", calvin_no_aug_cfg, N)

    # Test 5: LIBERO with big buffers (same as CALVIN)
    results["LIBERO (buf=20k/5k, aug=on)"] = benchmark_dataset("LIBERO (buf=20k/5k, aug=on)", libero_big_buf_cfg, N)

    # Summary
    log.info(f"\n{'='*70}")
    log.info("FINAL COMPARISON (data loading only, no GPU)")
    log.info(f"{'='*70}")
    log.info(f"{'Config':<35} {'VLM':>8} {'AE':>8} {'Total':>8}")
    log.info(f"{'-'*70}")
    for label, r in results.items():
        log.info(f"{label:<35} {r['vlm_avg']:>7.2f}s {r['ae_avg']:>7.2f}s {r['combined_avg']:>7.2f}s")
    log.info(f"{'-'*70}")

    # Analysis
    base_calvin = results["CALVIN (buf=20k/5k, aug=on)"]["combined_avg"]
    base_libero = results["LIBERO (buf=5k/1k, aug=on)"]["combined_avg"]
    log.info(f"\nDifference (CALVIN - LIBERO): {base_calvin - base_libero:.2f}s/step")

    if "CALVIN (buf=5k/1k, aug=on)" in results:
        small_buf = results["CALVIN (buf=5k/1k, aug=on)"]["combined_avg"]
        log.info(f"Shuffle buffer effect on CALVIN: {base_calvin - small_buf:.2f}s (20k/5k -> 5k/1k)")

    if "CALVIN (buf=20k/5k, aug=off)" in results:
        no_aug = results["CALVIN (buf=20k/5k, aug=off)"]["combined_avg"]
        log.info(f"Augmentation effect on CALVIN: {base_calvin - no_aug:.2f}s")

    if "LIBERO (buf=20k/5k, aug=on)" in results:
        big_buf = results["LIBERO (buf=20k/5k, aug=on)"]["combined_avg"]
        log.info(f"Shuffle buffer effect on LIBERO: {big_buf - base_libero:.2f}s (5k/1k -> 20k/5k)")


if __name__ == "__main__":
    main()
