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
        # Use gloo — nccl has compatibility issues in this environment
        # (matches the behaviour of the existing train_pytorch.py)
        backend = "gloo"
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

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def init_wandb(cfg, mode, is_main, resuming=False):
    if not is_main or not cfg.get("wandb_enabled", True):
        wandb.init(mode="disabled")
        return
    run_name = cfg.get("exp_name", f"waypoint_{mode}")
    project = cfg.get("wandb_project", "waypoint_vla")
    exp_name = cfg.get("exp_name", f"waypoint_{mode}")
    run_id_file = Path(cfg.get("checkpoint_dir", "checkpoints").format(exp_name=exp_name)) / "wandb_run_id.txt"
    if resuming and run_id_file.exists():
        run_id = run_id_file.read_text().strip()
        wandb.init(id=run_id, resume="must", project=project)
    else:
        wandb.init(
            name=run_name,
            project=project,
            config={k: v for k, v in cfg.items() if not isinstance(v, (dict, list))},
            tags=[mode, cfg.get("robot_type", ""), cfg.get("paligemma_variant", "")],
        )
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(wandb.run.id)


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
    if is_main:
        log_gpu_memory(device, prefix="[After model init] ")

    if cfg.get("pretrained_weight_path"):
        logging.info(f"Loading pretrained weights from {cfg['pretrained_weight_path']}")
        weight_path = os.path.join(cfg["pretrained_weight_path"], "model.safetensors")
        # time_mlp_in is intentionally resized from [W,W] → [2W,W] to accept
        # cat(time_emb, dur_emb). Skip it during weight loading and keep random init.
        state_dict = safetensors.torch.load_file(weight_path, device=str(device))
        own_state = model.state_dict()
        loaded, skipped = 0, 0
        for name, param in state_dict.items():
            if name not in own_state:
                skipped += 1
                continue
            if own_state[name].shape != param.shape:
                logging.info(f"  Skipping {name}: shape {param.shape} != {own_state[name].shape}")
                skipped += 1
                continue
            own_state[name].copy_(param)
            loaded += 1
        logging.info(f"Loaded {loaded} weight tensors, skipped {skipped}")
        if is_main:
            log_gpu_memory(device, prefix="[After weight load] ")

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
    if is_main:
        logging.info(f"Optimizer: AdamW  |  weight_decay=1e-4  |  {len(trainable_params)} param groups")
        log_gpu_memory(device, prefix="[After optimizer init] ")

    save_dir = Path(cfg.get("checkpoint_dir", "checkpoints/{exp_name}").format(exp_name=cfg.get("exp_name", "waypoint_ae")))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_interval = cfg.get("save_interval", 1000)

    global_step = 0
    if cfg.get("resume", False):
        global_step = load_latest_checkpoint(model, optimizer, save_dir, device)
        if is_main:
            log_gpu_memory(device, prefix="[After resume] ")

    warmup = cfg.get("warmup_steps", 500)
    peak_lr = cfg.get("peak_lr", 5e-5)
    decay_steps = cfg.get("num_train_steps", 30000)
    end_lr = cfg.get("end_lr", 1e-6)
    num_steps = cfg.get("num_train_steps", 30000)
    log_interval = cfg.get("log_interval", 50)

    if is_main:
        init_lr = peak_lr / (warmup + 1)
        logging.info(f"LR schedule: init={init_lr:.2e} -> peak={peak_lr:.2e} (warmup {warmup} steps) -> end={end_lr:.2e} (cosine over {decay_steps} steps)")
        logging.info(f"Training: {global_step} -> {num_steps} steps  |  save every {save_interval}  |  log every {log_interval}")

    resuming = cfg.get("resume", False)
    init_wandb(cfg, "ae", is_main, resuming=resuming)

    if is_main:
        total_p, trainable_p = count_params(model)
        logging.info(f"Model: {total_p/1e6:.1f}M total, {trainable_p/1e6:.1f}M trainable")
        logging.info(f"Dataset: {dataset.total_pairs:,} sample pairs")
        if wandb.run and wandb.run.mode != "disabled":
            wandb.run.summary["total_params"] = total_p
            wandb.run.summary["trainable_params"] = trainable_p
            wandb.run.summary["dataset_pairs"] = dataset.total_pairs

    model.train()
    pbar = tqdm.tqdm(total=num_steps, initial=global_step, desc="AE Training") if is_main else None
    start_time = time.time()
    infos = []

    for batch in loader:
        if global_step >= num_steps:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for k in list(batch.get("images", {}).keys()):
            batch["images"][k] = batch["images"][k].to(device)
        for k in list(batch.get("image_masks", {}).keys()):
            batch["image_masks"][k] = batch["image_masks"][k].to(device)

        for pg in optimizer.param_groups:
            pg["lr"] = cosine_lr(global_step, warmup, peak_lr, decay_steps, end_lr)

        class _Obs:
            def __init__(self, b):
                self.images = b["images"]
                self.image_masks = b["image_masks"]
                self.state = b["start_proprio"]
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
            infos.append({
                "train/loss": loss.item(),
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
                f"[AE] step={global_step}/{num_steps} "
                f"loss={avg['train/loss']:.4f} "
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

        global_step += 1
        save_checkpoint(model, optimizer, global_step, save_dir, is_main, save_interval)
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

    if pbar:
        pbar.close()
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
    if is_main:
        log_gpu_memory(device, prefix="[After model init] ")

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
        if is_main:
            log_gpu_memory(device, prefix="[After weight load] ")

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
    if is_main:
        logging.info(f"Optimizer: AdamW  |  weight_decay=1e-4  |  {len(trainable_params)} param groups")
        log_gpu_memory(device, prefix="[After optimizer init] ")

    save_dir = Path(cfg.get("checkpoint_dir", "checkpoints/{exp_name}").format(exp_name=cfg.get("exp_name", "waypoint_vlm")))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_interval = cfg.get("save_interval", 1000)

    global_step = 0
    if cfg.get("resume", False):
        global_step = load_latest_checkpoint(model, optimizer, save_dir, device)
        if is_main:
            log_gpu_memory(device, prefix="[After resume] ")

    warmup = cfg.get("warmup_steps", 500)
    peak_lr = cfg.get("peak_lr", 5e-5)
    decay_steps = cfg.get("num_train_steps", 30000)
    end_lr = cfg.get("end_lr", 1e-6)
    num_steps = cfg.get("num_train_steps", 30000)
    log_interval = cfg.get("log_interval", 50)

    if is_main:
        init_lr = peak_lr / (warmup + 1)
        logging.info(f"LR schedule: init={init_lr:.2e} -> peak={peak_lr:.2e} (warmup {warmup} steps) -> end={end_lr:.2e} (cosine over {decay_steps} steps)")
        logging.info(f"Training: {global_step} -> {num_steps} steps  |  save every {save_interval}  |  log every {log_interval}")

    resuming = cfg.get("resume", False)
    init_wandb(cfg, "vlm", is_main, resuming=resuming)

    if is_main:
        total_p, trainable_p = count_params(model)
        logging.info(f"Model: {total_p/1e6:.1f}M total, {trainable_p/1e6:.1f}M trainable")
        if wandb.run and wandb.run.mode != "disabled":
            wandb.run.summary["total_params"] = total_p
            wandb.run.summary["trainable_params"] = trainable_p

    model.train()
    pbar = tqdm.tqdm(total=num_steps, initial=global_step, desc="VLM Training") if is_main else None
    start_time = time.time()
    infos = []

    for batch in loader:
        if global_step >= num_steps:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for k in list(batch.get("images", {}).keys()):
            batch["images"][k] = batch["images"][k].to(device)
        for k in list(batch.get("image_masks", {}).keys()):
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
            infos.append({
                "train/loss": loss.item(),
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
                f"[VLM] step={global_step}/{num_steps} "
                f"loss={avg['train/loss']:.4f} "
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

        global_step += 1
        save_checkpoint(model, optimizer, global_step, save_dir, is_main, save_interval)
        if pbar:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

    if pbar:
        pbar.close()
    wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def log_gpu_info(device):
    if not torch.cuda.is_available():
        logging.info("GPU: not available (running on CPU)")
        return
    idx = device.index if device.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    total_mem = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    logging.info(f"GPU {idx}: {name}  |  Total memory: {total_mem:.1f} GB")


def log_gpu_memory(device, prefix=""):
    if not torch.cuda.is_available():
        return
    idx = device.index if device.index is not None else 0
    alloc = torch.cuda.memory_allocated(idx) / 1024**3
    reserved = torch.cuda.memory_reserved(idx) / 1024**3
    total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    logging.info(f"{prefix}GPU mem: {alloc:.1f} GB alloc / {reserved:.1f} GB reserved / {total:.1f} GB total  ({total - alloc:.1f} GB free)")


def log_training_config(cfg, mode, device, use_ddp):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    batch_size = cfg.get("batch_size", 32)
    effective_batch = batch_size * world_size
    num_steps = cfg.get("num_train_steps", 30000)
    total_samples = effective_batch * num_steps

    sep = "=" * 72
    logging.info(sep)
    logging.info(f"  WAYPOINT {mode.upper()} TRAINING — Configuration Summary")
    logging.info(sep)

    logging.info(f"  Mode:                  {mode}")
    logging.info(f"  Robot type:            {cfg.get('robot_type', 'N/A')}")
    logging.info(f"  Device:                {device}")
    logging.info(f"  DDP:                   {use_ddp}  (world_size={world_size})")
    logging.info(f"  Precision:             {cfg.get('precision', 'N/A')}")
    logging.info("")

    logging.info("  --- Data ---")
    if mode == "ae":
        logging.info(f"  RLDS dir:              {cfg.get('original_rlds_dir', 'N/A')}")
        logging.info(f"  WP indices:            {cfg.get('wp_indices_path', 'N/A')}")
    elif mode == "vlm":
        logging.info(f"  WP RLDS dir:           {cfg.get('wp_rlds_dir', 'N/A')}")
    logging.info(f"  Dataset stats:         {cfg.get('dataset_statistics_path', 'N/A')}")
    logging.info(f"  Norm type:             {cfg.get('norm_type', 'N/A')}")
    logging.info(f"  Shuffle buffer:        {cfg.get('shuffle_buffer_size', 'N/A')}")
    logging.info("")

    logging.info("  --- Model ---")
    logging.info(f"  PaliGemma variant:     {cfg.get('paligemma_variant', 'N/A')}")
    if mode == "ae":
        logging.info(f"  Action expert variant: {cfg.get('action_expert_variant', 'N/A')}")
    logging.info(f"  Action dim:            {cfg.get('model_action_dim', 'N/A')}")
    logging.info(f"  Proprio dim:           {cfg.get('model_proprio_dim', 'N/A')}")
    logging.info(f"  Horizon steps:         {cfg.get('horizon_steps', 'N/A')}")
    logging.info(f"  Max token len:         {cfg.get('max_token_len', 'N/A')}")
    logging.info(f"  Pretrained weights:    {cfg.get('pretrained_weight_path', 'NONE')}")
    logging.info(f"  LoRA enabled:          {cfg.get('lora_enabled', False)}")
    if cfg.get("lora_enabled", False):
        logging.info(f"  LoRA rank:             {cfg.get('lora_rank', 16)}")
        logging.info(f"  LoRA alpha:            {cfg.get('lora_alpha', 16.0)}")
    logging.info("")

    logging.info("  --- Training ---")
    logging.info(f"  Batch size (per GPU):  {batch_size}")
    logging.info(f"  Effective batch size:  {effective_batch}  ({batch_size} x {world_size} GPUs)")
    logging.info(f"  Num train steps:       {num_steps}")
    logging.info(f"  Total samples:         {total_samples:,}  (~{total_samples/1e6:.2f}M)")
    logging.info(f"  Warmup steps:          {cfg.get('warmup_steps', 500)}")
    logging.info(f"  Peak LR:               {cfg.get('peak_lr', 5e-5)}")
    logging.info(f"  End LR:                {cfg.get('end_lr', 1e-6)}")
    logging.info(f"  LR schedule:           cosine decay")
    logging.info("")

    logging.info("  --- Checkpointing ---")
    logging.info(f"  Checkpoint dir:        {cfg.get('checkpoint_dir', 'N/A')}")
    logging.info(f"  Save interval:         {cfg.get('save_interval', 1000)} steps")
    logging.info(f"  Log interval:          {cfg.get('log_interval', 50)} steps")
    logging.info(f"  Resume:                {cfg.get('resume', False)}")
    logging.info(f"  W&B enabled:           {cfg.get('wandb_enabled', True)}")
    logging.info(f"  Experiment name:       {cfg.get('exp_name', 'N/A')}")
    logging.info(sep)


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ae", "vlm"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    logging.info(f"Config file: {os.path.abspath(args.config)}")
    logging.info(f"Training mode: {args.mode}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.resume:
        cfg["resume"] = True

    use_ddp, local_rank, device = setup_ddp()
    is_main = not use_ddp or dist.get_rank() == 0

    if is_main:
        log_gpu_info(device)
        log_training_config(cfg, args.mode, device, use_ddp)

    if args.mode == "ae":
        train_ae(cfg, device, use_ddp, is_main)
    else:
        train_vlm(cfg, device, use_ddp, is_main)

    cleanup_ddp()


if __name__ == "__main__":
    main()
