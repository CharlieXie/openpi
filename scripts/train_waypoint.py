"""
Unified training script for Waypoint VLA (VLM + Action Expert).

Usage:
  # Action Expert training
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint.py \
    --mode ae --config configs/waypoint_ae_libero.yaml

  # VLM waypoint training
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_waypoint.py \
    --mode vlm --config configs/waypoint_vlm_libero.yaml
"""

import argparse
import dataclasses
import gc
import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb
import yaml


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E"}

    class Fmt(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = Fmt(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, step, save_dir, is_main, save_interval):
    if not is_main or step % save_interval != 0:
        return
    final_dir = save_dir / f"{step}"
    tmp_dir = save_dir / f"tmp_{step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.save_model(model_to_save, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save({"global_step": step, "timestamp": time.time()}, tmp_dir / "metadata.pt")

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    logging.info(f"Saved checkpoint step {step} -> {final_dir}")


def load_latest_checkpoint(model, optimizer, save_dir, device):
    steps = [int(d.name) for d in save_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not steps:
        return 0
    latest = max(steps)
    ckpt = save_dir / str(latest)
    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.load_model(model_to_load, ckpt / "model.safetensors", device=str(device))
    optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location=device, weights_only=False))
    meta = torch.load(ckpt / "metadata.pt", map_location=device, weights_only=False)
    logging.info(f"Resumed from step {meta['global_step']}")
    return meta["global_step"]


def cosine_lr(step, warmup, peak_lr, decay_steps, end_lr):
    if step < warmup:
        init_lr = peak_lr / (warmup + 1)
        return init_lr + (peak_lr - init_lr) * step / warmup
    progress = min(1.0, (step - warmup) / max(1, decay_steps - warmup))
    cos = 0.5 * (1 + np.cos(np.pi * progress))
    return end_lr + (peak_lr - end_lr) * cos


# ---------------------------------------------------------------------------
# Action Expert training
# ---------------------------------------------------------------------------

def train_ae(cfg, device, use_ddp, is_main):
    from openpi.waypoint.ae_dataset import WaypointAEDataset, WaypointAECollator
    from openpi.waypoint.ae_model import PI0WaypointAE
    from openpi.waypoint.normalize import load_dataset_statistics
    from openpi.waypoint.robot_config import get_robot_config
    import openpi.models.pi0_config as pi0_config

    rc = get_robot_config(cfg["robot_type"])
    stats = load_dataset_statistics(cfg["dataset_statistics_path"])

    dataset = WaypointAEDataset(
        original_rlds_dir=cfg["original_rlds_dir"],
        wp_indices_path=cfg["wp_indices_path"],
        robot_config=rc,
        dataset_statistics=stats,
        norm_type=cfg.get("norm_type", "q99"),
        max_duration=cfg.get("max_duration", 32),
        horizon_steps=cfg.get("horizon_steps", 32),
        model_action_dim=cfg.get("model_action_dim", 32),
        model_proprio_dim=cfg.get("model_proprio_dim", 32),
        shuffle_buffer_size=cfg.get("shuffle_buffer_size", 10000),
    )
    collator = WaypointAECollator()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.get("batch_size", 32),
        num_workers=0, collate_fn=collator,
    )

    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=cfg.get("model_action_dim", 32),
        action_horizon=cfg.get("horizon_steps", 32),
        max_token_len=cfg.get("max_token_len", 64),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        dtype=cfg.get("precision", "bfloat16"),
    )

    model = PI0WaypointAE(model_cfg).to(device)
    model.gradient_checkpointing_enable()

    if cfg.get("pretrained_weight_path"):
        logging.info(f"Loading pretrained weights from {cfg['pretrained_weight_path']}")
        weight_path = os.path.join(cfg["pretrained_weight_path"], "model.safetensors")
        safetensors.torch.load_model(model, weight_path, strict=False)

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

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.get("peak_lr", 5e-5), weight_decay=1e-4)

    save_dir = Path(cfg.get("checkpoint_dir", "checkpoints/waypoint_ae"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_interval = cfg.get("save_interval", 1000)

    global_step = 0
    if cfg.get("resume", False):
        global_step = load_latest_checkpoint(model, optimizer, save_dir, device)

    warmup = cfg.get("warmup_steps", 500)
    peak_lr = cfg.get("peak_lr", 5e-5)
    decay_steps = cfg.get("num_train_steps", 30000)
    end_lr = cfg.get("end_lr", 1e-6)
    num_steps = cfg.get("num_train_steps", 30000)
    log_interval = cfg.get("log_interval", 50)

    if is_main and cfg.get("wandb_enabled", True):
        wandb.init(name=cfg.get("exp_name", "waypoint_ae"), project="waypoint_vla")

    model.train()
    pbar = tqdm.tqdm(total=num_steps, initial=global_step, desc="AE Training") if is_main else None
    start_time = time.time()
    infos = []

    for batch in loader:
        if global_step >= num_steps:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for k in batch.get("images", {}):
            batch["images"][k] = batch["images"][k].to(device)
        for k in batch.get("image_masks", {}):
            batch["image_masks"][k] = batch["image_masks"][k].to(device)

        for pg in optimizer.param_groups:
            pg["lr"] = cosine_lr(global_step, warmup, peak_lr, decay_steps, end_lr)

        class _Obs:
            def __init__(self, b):
                self.images = b["images"]
                self.image_masks = b["image_masks"]
                self.state = batch["start_proprio"]
                self.tokenized_prompt = b["prompt_tokens"]
                self.tokenized_prompt_mask = b["prompt_masks"]
                self.token_ar_mask = None
                self.token_loss_mask = None
        obs = _Obs(batch)

        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        loss = raw_model(
            observation=obs,
            start_proprio=batch["start_proprio"],
            end_proprio=batch["end_proprio"],
            actions=batch["actions"],
            duration=batch["duration"],
            action_pad_mask=batch["action_pad_mask"],
            action_dim_mask=batch["action_dim_mask"],
        )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main:
            infos.append({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm)})

        if is_main and global_step % log_interval == 0 and infos:
            avg = {k: np.mean([i[k] for i in infos]) for k in infos[0]}
            elapsed = time.time() - start_time
            logging.info(f"step={global_step} loss={avg['loss']:.4f} lr={avg['lr']:.2e} gnorm={avg['grad_norm']:.2f} t={elapsed:.1f}s")
            if cfg.get("wandb_enabled", True):
                wandb.log({**avg, "step": global_step}, step=global_step)
            infos = []
            start_time = time.time()

        global_step += 1
        save_checkpoint(model, optimizer, global_step, save_dir, is_main, save_interval)
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    if pbar:
        pbar.close()
    if is_main and cfg.get("wandb_enabled", True):
        wandb.finish()


# ---------------------------------------------------------------------------
# VLM waypoint training
# ---------------------------------------------------------------------------

def train_vlm(cfg, device, use_ddp, is_main):
    from openpi.waypoint.vlm_dataset import WaypointVLMDataset, WaypointVLMCollator
    from openpi.waypoint.vlm_model import PI0WaypointVLM
    from openpi.waypoint.normalize import load_dataset_statistics
    from openpi.waypoint.robot_config import get_robot_config
    from openpi.waypoint.tokenizer import WaypointTokenizer
    import openpi.models.pi0_config as pi0_config

    rc = get_robot_config(cfg["robot_type"])
    stats = load_dataset_statistics(cfg["dataset_statistics_path"])

    wp_tokenizer = WaypointTokenizer(
        proprio_dim=rc.actual_proprio_dim,
        num_waypoints=cfg.get("num_waypoints", 7),
        max_token_len=cfg.get("max_token_len", 256),
    )

    dataset = WaypointVLMDataset(
        wp_rlds_dir=cfg["wp_rlds_dir"],
        robot_config=rc,
        dataset_statistics=stats,
        wp_tokenizer=wp_tokenizer,
        norm_type=cfg.get("norm_type", "q99"),
        num_waypoints=cfg.get("num_waypoints", 7),
        stride=cfg.get("stride", 4),
        shuffle_buffer_size=cfg.get("shuffle_buffer_size", 5000),
    )
    collator = WaypointVLMCollator()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.get("batch_size", 32),
        num_workers=0, collate_fn=collator,
    )

    model_cfg = pi0_config.Pi0Config(
        pi05=False,
        max_token_len=cfg.get("max_token_len", 256),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        dtype=cfg.get("precision", "bfloat16"),
    )

    model = PI0WaypointVLM(model_cfg).to(device)
    model.gradient_checkpointing_enable()

    if cfg.get("pretrained_weight_path"):
        logging.info(f"Loading pretrained PaliGemma weights from {cfg['pretrained_weight_path']}")
        import openpi.models_pytorch.pi0_pytorch as _pi0
        full_model = _pi0.PI0Pytorch(pi0_config.Pi0Config(
            pi05=True, dtype=cfg.get("precision", "bfloat16"),
            paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        ))
        weight_path = os.path.join(cfg["pretrained_weight_path"], "model.safetensors")
        safetensors.torch.load_model(full_model, weight_path, strict=False)

        pg_state = {}
        for name, param in full_model.paligemma_with_expert.paligemma.named_parameters():
            pg_state[f"paligemma.{name}"] = param.data
        missing, unexpected = model.load_state_dict(pg_state, strict=False)
        logging.info(f"PaliGemma weights loaded: {len(pg_state)} params, {len(missing)} missing, {len(unexpected)} unexpected")
        del full_model, pg_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cfg.get("lora_enabled", False):
        import openpi.models_pytorch.lora_pytorch as lora_utils
        lora_cfg = lora_utils.LoRATrainingConfig(
            enabled=True,
            attn_rank=cfg.get("lora_rank", 16),
            ffn_rank=cfg.get("lora_rank", 16),
            apply_to="paligemma_only",
            train_non_lora_layers=False,
            train_vision_encoder=cfg.get("train_vision_encoder", True),
        )
        frozen, trainable = lora_utils.apply_lora_to_pi0_pytorch(model, lora_cfg)
        logging.info(f"LoRA applied: {trainable:,} trainable, {frozen:,} frozen")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.get("peak_lr", 5e-5), weight_decay=1e-4)

    save_dir = Path(cfg.get("checkpoint_dir", "checkpoints/waypoint_vlm"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_interval = cfg.get("save_interval", 1000)

    global_step = 0
    if cfg.get("resume", False):
        global_step = load_latest_checkpoint(model, optimizer, save_dir, device)

    warmup = cfg.get("warmup_steps", 500)
    peak_lr = cfg.get("peak_lr", 5e-5)
    decay_steps = cfg.get("num_train_steps", 30000)
    end_lr = cfg.get("end_lr", 1e-6)
    num_steps = cfg.get("num_train_steps", 30000)
    log_interval = cfg.get("log_interval", 50)

    if is_main and cfg.get("wandb_enabled", True):
        wandb.init(name=cfg.get("exp_name", "waypoint_vlm"), project="waypoint_vla")

    model.train()
    pbar = tqdm.tqdm(total=num_steps, initial=global_step, desc="VLM Training") if is_main else None
    start_time = time.time()
    infos = []

    for batch in loader:
        if global_step >= num_steps:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for k in batch.get("images", {}):
            batch["images"][k] = batch["images"][k].to(device)
        for k in batch.get("image_masks", {}):
            batch["image_masks"][k] = batch["image_masks"][k].to(device)

        for pg in optimizer.param_groups:
            pg["lr"] = cosine_lr(global_step, warmup, peak_lr, decay_steps, end_lr)

        raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        loss = raw_model(batch)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if is_main:
            infos.append({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm)})

        if is_main and global_step % log_interval == 0 and infos:
            avg = {k: np.mean([i[k] for i in infos]) for k in infos[0]}
            elapsed = time.time() - start_time
            logging.info(f"step={global_step} loss={avg['loss']:.4f} lr={avg['lr']:.2e} gnorm={avg['grad_norm']:.2f} t={elapsed:.1f}s")
            if cfg.get("wandb_enabled", True):
                wandb.log({**avg, "step": global_step}, step=global_step)
            infos = []
            start_time = time.time()

        global_step += 1
        save_checkpoint(model, optimizer, global_step, save_dir, is_main, save_interval)
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    if pbar:
        pbar.close()
    if is_main and cfg.get("wandb_enabled", True):
        wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ae", "vlm"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.resume:
        cfg["resume"] = True

    use_ddp, local_rank, device = setup_ddp()
    is_main = not use_ddp or dist.get_rank() == 0

    if args.mode == "ae":
        train_ae(cfg, device, use_ddp, is_main)
    else:
        train_vlm(cfg, device, use_ddp, is_main)

    cleanup_ddp()


if __name__ == "__main__":
    main()
