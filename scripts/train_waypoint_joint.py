"""
Joint training script for Waypoint VLA (VLM + Action Expert sharing backbone).

Usage:
  # Single GPU
  python scripts/train_waypoint_joint.py --config configs/waypoint_joint_libero.yaml

  # Multi-GPU DDP
  torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_waypoint_joint.py --config configs/waypoint_joint_libero.yaml
"""

import argparse
import contextlib
import logging
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import wandb
import yaml

# Reuse infrastructure from train_waypoint.py
from train_waypoint import (
    cleanup_ddp,
    cosine_lr,
    count_params,
    init_logging,
    init_wandb,
    log_gpu_info,
    log_gpu_memory,
    save_checkpoint,
    setup_ddp,
)


class _PrefetchIter:
    """Background-thread prefetcher for a DataLoader.

    While the main thread runs GPU forward/backward, a daemon thread
    pre-loads the next batch via the DataLoader iterator, hiding most
    of the CPU data-loading latency behind GPU compute.
    """

    def __init__(self, loader, prefetch_count=2):
        self._loader = loader
        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_count)
        self._stop = threading.Event()
        self._exception: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        it = iter(self._loader)
        while not self._stop.is_set():
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self._loader)
                batch = next(it)
            except Exception as exc:
                self._exception = exc
                return
            self._queue.put(batch)

    def __next__(self):
        if self._exception is not None:
            raise self._exception
        return self._queue.get()

    def stop(self):
        self._stop.set()


def train_joint(cfg, device, use_ddp, is_main):
    from openpi.waypoint.ae_dataset import WaypointAECollator, WaypointAEDataset
    from openpi.waypoint.joint_model import PI0WaypointJoint
    from openpi.waypoint.normalize import load_dataset_statistics
    from openpi.waypoint.robot_config import get_robot_config
    from openpi.waypoint.tokenizer import WaypointTokenizer
    from openpi.waypoint.vlm_dataset import WaypointVLMCollator, WaypointVLMDataset
    import openpi.models.pi0_config as pi0_config

    rc = get_robot_config(cfg["robot_type"])
    stats = load_dataset_statistics(cfg["dataset_statistics_path"])

    # --- Datasets ---
    wp_tokenizer = WaypointTokenizer(
        proprio_dim=rc.continuous_proprio_dim,
        num_waypoints=cfg.get("num_waypoints", 7),
        max_token_len=cfg.get("vlm_max_token_len", 256),
        use_gripper_token=True,
    )

    vlm_dataset = WaypointVLMDataset(
        wp_rlds_dir=cfg["wp_rlds_dir"],
        robot_config=rc,
        dataset_statistics=stats,
        wp_tokenizer=wp_tokenizer,
        norm_type=cfg.get("norm_type", "q99"),
        num_waypoints=cfg.get("num_waypoints", 7),
        stride=cfg.get("stride", 4),
        shuffle_buffer_size=cfg.get("vlm_shuffle_buffer_size", 5000),
        image_aug=cfg.get("image_aug", False),
        image_aug_cfg=cfg.get("image_aug_cfg", None),
        episode_shuffle_buffer=cfg.get("episode_shuffle_buffer_size", 0),
    )
    vlm_collator = WaypointVLMCollator()
    vlm_loader = torch.utils.data.DataLoader(
        vlm_dataset, batch_size=cfg.get("vlm_batch_size", 64),
        num_workers=0, collate_fn=vlm_collator,
    )

    ae_dataset = WaypointAEDataset(
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
        episode_shuffle_buffer=cfg.get("episode_shuffle_buffer_size", 0),
        image_aug=cfg.get("image_aug", False),
        image_aug_cfg=cfg.get("image_aug_cfg", None),
    )
    ae_collator = WaypointAECollator()
    ae_loader = torch.utils.data.DataLoader(
        ae_dataset, batch_size=cfg.get("ae_batch_size", 64),
        num_workers=0, collate_fn=ae_collator,
    )

    # --- Model ---
    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=cfg.get("model_action_dim", 32),
        action_horizon=cfg.get("horizon_steps", 32),
        max_token_len=cfg.get("ae_max_token_len", 64),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        dtype=cfg.get("precision", "bfloat16"),
    )

    model = PI0WaypointJoint(
        config=model_cfg,
        vlm_max_token_len=cfg.get("vlm_max_token_len", 256),
        gradient_strategy=cfg.get("gradient_strategy", "none"),
        gradient_scale=cfg.get("gradient_scale", 0.1),
    ).to(device)
    model.gradient_checkpointing_enable()
    if is_main:
        log_gpu_memory(device, prefix="[After model init] ")

    # --- Load pretrained weights ---
    if cfg.get("pretrained_weight_path"):
        weight_path = os.path.join(cfg["pretrained_weight_path"], "model.safetensors")
        logging.info(f"Loading pretrained weights from {weight_path}")
        PI0WaypointJoint.load_pretrained_weights(model, weight_path, device)
        if is_main:
            log_gpu_memory(device, prefix="[After weight load] ")

    # --- LoRA ---
    if cfg.get("lora_enabled", False):
        import openpi.models_pytorch.lora_pytorch as lora_utils
        lora_cfg = lora_utils.LoRATrainingConfig(
            enabled=True,
            attn_rank=cfg.get("lora_rank", 16),
            ffn_rank=cfg.get("lora_rank", 16),
            attn_alpha=cfg.get("lora_alpha", 16.0),
            ffn_alpha=cfg.get("lora_alpha", 16.0),
            apply_to="all",
            train_non_lora_layers=True,
            train_vision_encoder=cfg.get("train_vision_encoder", True),
        )
        frozen, trainable = lora_utils.apply_lora_to_pi0_pytorch(model, lora_cfg)
        logging.info(f"LoRA applied: {trainable:,} trainable, {frozen:,} frozen")

    # --- DDP ---
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    # --- Optimizer ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.get("peak_lr", 3e-5), weight_decay=1e-4)
    if is_main:
        logging.info(f"Optimizer: AdamW  |  weight_decay=1e-4  |  {len(trainable_params)} param groups")
        log_gpu_memory(device, prefix="[After optimizer init] ")

    # --- Checkpoint directory ---
    save_dir = Path(cfg.get("checkpoint_dir", "checkpoints/{exp_name}").format(
        exp_name=cfg.get("exp_name", "waypoint_joint"),
    ))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_interval = cfg.get("save_interval", 500)

    global_step = 0
    # NOTE: resume not fully implemented yet (need to save/load optimizer state)
    # if cfg.get("resume", False):
    #     global_step = load_latest_checkpoint(model, optimizer, save_dir, device)

    # --- LR schedule ---
    warmup = cfg.get("warmup_steps", 100)
    peak_lr = cfg.get("peak_lr", 3e-5)
    decay_steps = cfg.get("num_train_steps", 5000)
    end_lr = cfg.get("end_lr", 1e-7)
    num_steps = cfg.get("num_train_steps", 5000)
    log_interval = cfg.get("log_interval", 5)
    ae_loss_weight = cfg.get("ae_loss_weight", 1.0)
    gradient_strategy = cfg.get("gradient_strategy", "none")
    gradient_scale = cfg.get("gradient_scale", 0.1)

    if is_main:
        init_lr = peak_lr / (warmup + 1)
        logging.info(f"LR schedule: init={init_lr:.2e} -> peak={peak_lr:.2e} (warmup {warmup} steps) -> end={end_lr:.2e} (cosine over {decay_steps} steps)")
        logging.info(f"Training: {global_step} -> {num_steps} steps  |  save every {save_interval}  |  log every {log_interval}")
        logging.info(f"Gradient strategy: {gradient_strategy}  |  AE loss weight: {ae_loss_weight}  |  Gradient scale: {gradient_scale}")

    resuming = cfg.get("resume", False)
    init_wandb(cfg, "joint", is_main, resuming=resuming)

    if is_main:
        total_p, trainable_p = count_params(model)
        logging.info(f"Model: {total_p/1e6:.1f}M total, {trainable_p/1e6:.1f}M trainable")
        if wandb.run and getattr(wandb.run, "mode", None) != "disabled":
            wandb.run.summary["total_params"] = total_p
            wandb.run.summary["trainable_params"] = trainable_p
            wandb.run.summary["gradient_strategy"] = gradient_strategy
            wandb.run.summary["ae_loss_weight"] = ae_loss_weight
            wandb.run.summary["gradient_scale"] = gradient_scale

    # --- Training loop ---
    model.train()

    pbar = tqdm.tqdm(total=num_steps, initial=global_step, desc="Joint Training") if is_main else None
    start_time = time.time()
    infos = []

    class _Obs:
        """Adapter to convert AE batch dict into the observation object expected by preprocessing."""
        def __init__(self, b):
            self.images = b["images"]
            self.image_masks = b["image_masks"]
            self.state = b["start_proprio"]
            self.tokenized_prompt = b["prompt_tokens"]
            self.tokenized_prompt_mask = b["prompt_masks"]
            self.token_ar_mask = None
            self.token_loss_mask = None

    use_prefetch = cfg.get("prefetch", True)
    if use_prefetch:
        vlm_iter = _PrefetchIter(vlm_loader, prefetch_count=2)
        ae_iter = _PrefetchIter(ae_loader, prefetch_count=2)
        if is_main:
            logging.info("Data prefetching enabled (threaded, prefetch_count=2)")
    else:
        vlm_iter = iter(vlm_loader)
        ae_iter = iter(ae_loader)

    for global_step in range(global_step, num_steps):
        # LR update
        for pg in optimizer.param_groups:
            pg["lr"] = cosine_lr(global_step, warmup, peak_lr, decay_steps, end_lr)

        # --- Get batches ---
        if use_prefetch:
            vlm_batch = next(vlm_iter)
            ae_batch = next(ae_iter)
        else:
            try:
                vlm_batch = next(vlm_iter)
            except StopIteration:
                vlm_iter = iter(vlm_loader)
                vlm_batch = next(vlm_iter)
            try:
                ae_batch = next(ae_iter)
            except StopIteration:
                ae_iter = iter(ae_loader)
                ae_batch = next(ae_iter)

        # Move VLM batch to device
        vlm_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in vlm_batch.items()}
        for k in list(vlm_batch.get("images", {}).keys()):
            vlm_batch["images"][k] = vlm_batch["images"][k].to(device)
        for k in list(vlm_batch.get("image_masks", {}).keys()):
            vlm_batch["image_masks"][k] = vlm_batch["image_masks"][k].to(device)

        # Move AE batch to device
        ae_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in ae_batch.items()}
        for k in list(ae_batch.get("images", {}).keys()):
            ae_batch["images"][k] = ae_batch["images"][k].to(device)
        for k in list(ae_batch.get("image_masks", {}).keys()):
            ae_batch["image_masks"][k] = ae_batch["image_masks"][k].to(device)

        # --- VLM forward + backward (skip DDP sync) ---
        no_sync_ctx = model.no_sync if use_ddp else contextlib.nullcontext
        with no_sync_ctx():
            vlm_loss = model(mode="vlm", batch=vlm_batch)
            vlm_loss.backward()

        # --- AE forward + backward (triggers DDP sync) ---
        ae_obs = _Obs(ae_batch)
        ae_loss = model(
            mode="ae",
            observation=ae_obs,
            start_proprio=ae_batch["start_proprio"],
            end_proprio=ae_batch["end_proprio"],
            actions=ae_batch["actions"],
            duration=ae_batch["duration"],
            action_pad_mask=ae_batch["action_pad_mask"],
            action_dim_mask=ae_batch["action_dim_mask"],
        )
        (ae_loss_weight * ae_loss).backward()

        # --- Optimizer step ---
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # --- Logging ---
        if is_main:
            infos.append({
                "train/vlm_loss": vlm_loss.item(),
                "train/ae_loss": ae_loss.item(),
                "train/total_loss": vlm_loss.item() + ae_loss_weight * ae_loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": float(grad_norm),
            })

        if is_main and global_step % log_interval == 0 and infos:
            avg = {k: float(np.mean([i[k] for i in infos])) for k in infos[0]}
            elapsed = time.time() - start_time
            steps_per_sec = log_interval / max(elapsed, 1e-6)
            eta_sec = (num_steps - global_step) / max(steps_per_sec, 1e-6)
            eta_min = eta_sec / 60
            gpu_alloc = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
            logging.info(
                f"[Joint] step={global_step}/{num_steps} "
                f"vlm={avg['train/vlm_loss']:.4f} "
                f"ae={avg['train/ae_loss']:.4f} "
                f"lr={avg['train/lr']:.2e} "
                f"gnorm={avg['train/grad_norm']:.2f} "
                f"sps={steps_per_sec:.1f} "
                f"gpu={gpu_alloc:.1f}GB "
                f"eta={eta_min:.0f}min"
            )
            avg["train/steps_per_sec"] = steps_per_sec
            avg["train/gpu_mem_gb"] = gpu_alloc
            wandb.log(avg, step=global_step)
            infos = []
            start_time = time.time()

        step_for_save = global_step + 1
        save_checkpoint(model, optimizer, step_for_save, save_dir, is_main, save_interval)
        if pbar:
            pbar.update(1)
            pbar.set_postfix(
                vlm=f"{vlm_loss.item():.4f}",
                ae=f"{ae_loss.item():.4f}",
                step=step_for_save,
            )

    if pbar:
        pbar.close()
    if use_prefetch:
        vlm_iter.stop()
        ae_iter.stop()
    wandb.finish()


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    logging.info(f"Config file: {os.path.abspath(args.config)}")
    logging.info("Training mode: joint (VLM + AE)")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.resume:
        cfg["resume"] = True

    use_ddp, local_rank, device = setup_ddp()
    is_main = not use_ddp or dist.get_rank() == 0

    if is_main:
        log_gpu_info(device)

    train_joint(cfg, device, use_ddp, is_main)
    cleanup_ddp()


if __name__ == "__main__":
    main()
