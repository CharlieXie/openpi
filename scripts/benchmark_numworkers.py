"""Benchmark: test if num_workers>0 works with tf.data IterableDataset, and measure speedup.

Usage:
  .venv/bin/python scripts/benchmark_numworkers.py --config configs/waypoint_joint_calvin.yaml
"""

import argparse
import logging
import time
import traceback

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _make_loaders(cfg, num_workers=0, prefetch_factor=None):
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

    vlm_ds = WaypointVLMDataset(
        wp_rlds_dir=cfg["wp_rlds_dir"],
        robot_config=rc, dataset_statistics=stats, wp_tokenizer=wp_tokenizer,
        norm_type=cfg.get("norm_type", "q99"),
        num_waypoints=cfg.get("num_waypoints", 7),
        stride=cfg.get("stride", 4),
        shuffle_buffer_size=cfg.get("vlm_shuffle_buffer_size", 5000),
        image_aug=cfg.get("image_aug", False),
        image_aug_cfg=cfg.get("image_aug_cfg", None),
    )
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
        image_aug=cfg.get("image_aug", False),
        image_aug_cfg=cfg.get("image_aug_cfg", None),
    )

    loader_kwargs = dict(num_workers=num_workers)
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor or 2
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["multiprocessing_context"] = "fork"

    vlm_loader = torch.utils.data.DataLoader(
        vlm_ds, batch_size=cfg.get("vlm_batch_size", 64),
        collate_fn=WaypointVLMCollator(), **loader_kwargs,
    )
    ae_loader = torch.utils.data.DataLoader(
        ae_ds, batch_size=cfg.get("ae_batch_size", 64),
        collate_fn=WaypointAECollator(), **loader_kwargs,
    )
    return vlm_loader, ae_loader


def benchmark_loaders(cfg, num_workers, num_batches=10):
    log.info(f"\n{'='*60}")
    log.info(f"Testing num_workers={num_workers}")
    log.info(f"{'='*60}")
    try:
        vlm_loader, ae_loader = _make_loaders(cfg, num_workers=num_workers)
    except Exception as e:
        log.error(f"Failed to create loaders: {e}")
        return None

    vlm_iter = iter(vlm_loader)
    ae_iter = iter(ae_loader)

    times = []
    for i in range(num_batches):
        t0 = time.perf_counter()
        try:
            vb = next(vlm_iter)
            ab = next(ae_iter)
        except Exception as e:
            log.error(f"Step {i} failed: {e}")
            traceback.print_exc()
            return None
        dt = time.perf_counter() - t0
        times.append(dt)
        log.info(f"  Step {i}: {dt:.3f}s")

    skip = min(2, len(times) - 1)
    avg = np.mean(times[skip:])
    log.info(f"  Average (skip {skip}): {avg:.3f}s/step")
    return avg


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = {}
    for nw in [0, 1, 2]:
        avg = benchmark_loaders(cfg, num_workers=nw, num_batches=args.num_batches)
        if avg is not None:
            results[nw] = avg

    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    for nw, avg in results.items():
        log.info(f"  num_workers={nw}: {avg:.3f}s/step")
    if 0 in results and len(results) > 1:
        baseline = results[0]
        for nw, avg in results.items():
            if nw > 0:
                log.info(f"  Speedup (nw={nw} vs nw=0): {baseline/avg:.2f}x ({baseline - avg:.1f}s saved)")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
