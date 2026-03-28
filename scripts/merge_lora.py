"""Merge LoRA weights into a base model and export a standard checkpoint.

Usage:
  # Merge LoRA into the base weights used for training
  python scripts/merge_lora.py \
      --base /workspace/models/pi05_base_pytorch/model.safetensors \
      --lora checkpoints/exp/500/lora.safetensors \
      --config configs/waypoint_joint_libero.yaml \
      --output checkpoints/exp/500/model_merged.safetensors

  # If base is a joint-model checkpoint directory (contains model.safetensors)
  python scripts/merge_lora.py \
      --base /path/to/joint/checkpoint \
      --lora checkpoints/exp/500/lora.safetensors \
      --config configs/waypoint_joint_libero.yaml \
      --output checkpoints/exp/500/model_merged.safetensors

After merging, the output checkpoint can be loaded by eval scripts directly:
  joint_checkpoint: checkpoints/exp/500   # eval loads model_merged.safetensors
"""

import argparse
import logging
import os
import sys
import time

import safetensors.torch
import torch
import yaml


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint into base model")
    parser.add_argument("--base", required=True, help="Base model weights (.safetensors file or checkpoint dir)")
    parser.add_argument("--lora", required=True, help="LoRA checkpoint (.safetensors)")
    parser.add_argument("--config", required=True, help="Training config YAML (for model architecture)")
    parser.add_argument("--output", required=True, help="Output merged model path (.safetensors)")
    parser.add_argument("--device", default="cpu", help="Device for merge computation (default: cpu)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve base weight path
    base_path = args.base
    if os.path.isdir(base_path):
        base_path = os.path.join(base_path, "model.safetensors")
    if not os.path.exists(base_path):
        logging.error(f"Base weights not found: {base_path}")
        sys.exit(1)
    if not os.path.exists(args.lora):
        logging.error(f"LoRA checkpoint not found: {args.lora}")
        sys.exit(1)

    device = args.device
    logging.info(f"Base weights: {base_path}")
    logging.info(f"LoRA checkpoint: {args.lora}")
    logging.info(f"Device: {device}")

    # --- Build model (empty weights) ---
    logging.info("Building model architecture...")
    t0 = time.time()

    import openpi.models.gemma as _gemma
    import openpi.models.pi0_config as pi0_config
    from openpi.waypoint.joint_model import PI0WaypointJoint

    model_cfg = pi0_config.Pi05Config(
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        action_dim=cfg.get("model_action_dim", 32),
        action_horizon=cfg.get("horizon_steps", 32),
        dtype=cfg.get("precision", "bfloat16"),
    )

    model = PI0WaypointJoint(
        config=model_cfg,
        vlm_max_token_len=cfg.get("vlm_max_token_len", 256),
        gradient_strategy="none",
    )
    logging.info(f"Model built: {time.time() - t0:.1f}s")

    # --- Load base weights ---
    logging.info("Loading base weights...")
    t0 = time.time()
    PI0WaypointJoint.load_pretrained_weights(model, base_path, device)
    logging.info(f"Base weights loaded: {time.time() - t0:.1f}s")

    # --- Apply LoRA structure ---
    logging.info("Injecting LoRA structure...")
    import openpi.models_pytorch.lora_pytorch as lora_utils

    vision_mode = cfg.get("vision_encoder_mode", None)
    if vision_mode is None:
        vision_mode = "full" if cfg.get("train_vision_encoder", False) else "freeze"

    lora_kwargs = dict(
        enabled=True,
        rank=cfg.get("lora_rank", 16),
        alpha=cfg.get("lora_alpha", 16.0),
        dropout=cfg.get("lora_dropout", 0.0),
        use_rslora=cfg.get("lora_use_rslora", False),
        init_lora_weights=cfg.get("lora_init", True),
        vision_encoder_mode=vision_mode,
        apply_to=cfg.get("lora_apply_to", "all"),
    )
    if "lora_modules_to_skip" in cfg:
        lora_kwargs["modules_to_not_lora"] = cfg["lora_modules_to_skip"]
    if "lora_trainable_modules" in cfg:
        lora_kwargs["trainable_non_lora_modules"] = cfg["lora_trainable_modules"]
    lora_cfg = lora_utils.LoRATrainingConfig(**lora_kwargs)
    lora_utils.apply_lora_to_model(model, lora_cfg)

    # --- Load LoRA weights ---
    logging.info("Loading LoRA weights...")
    t0 = time.time()
    lora_utils.load_lora_checkpoint(model, args.lora)
    logging.info(f"LoRA weights loaded: {time.time() - t0:.1f}s")

    # --- Merge ---
    logging.info("Merging LoRA into base weights...")
    t0 = time.time()
    count = lora_utils.merge_lora_weights(model)
    logging.info(f"Merged {count} layers: {time.time() - t0:.1f}s")

    # --- Save ---
    logging.info(f"Saving merged model to {args.output} ...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    safetensors.torch.save_model(model, args.output)
    file_size = os.path.getsize(args.output) / (1024**3)
    logging.info(f"Saved: {args.output} ({file_size:.2f} GB, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
