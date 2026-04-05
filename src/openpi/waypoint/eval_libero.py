"""Two-stage LIBERO evaluation for Waypoint VLA.

Pipeline:
  1. VLM predicts M=7 waypoints autoregressively.
  2. Action Expert fills actions between each waypoint pair via flow matching.
  3. Execute actions in LIBERO environment.
  4. Replan after exhausting all waypoints.

Usage:
  python -m openpi.waypoint.eval_libero --config configs/eval_waypoint_libero.yaml
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import argparse
import contextlib
import json
import logging
import math
import pathlib
import time
from pathlib import Path

import imageio
import numpy as np
import safetensors.torch
import torch
import yaml
from PIL import Image

try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
    torch.serialization.add_safe_globals([np.ndarray])
    torch.serialization.add_safe_globals([np.dtype])
except Exception:
    pass

from openpi.waypoint.normalize import (
    NormalizationHelper,
    load_dataset_statistics,
    pad_to_dim,
    unnormalize_q99,
    unnormalize_normal,
)
from openpi.waypoint.robot_config import get_robot_config
from openpi.waypoint.tokenizer import WaypointTokenizer

logger = logging.getLogger(__name__)

_image_save_dir = None
_image_frame_idx = 0

MAX_STEPS_MAP = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# def quat2axisangle(quat):
#     """Convert quaternion to axis-angle."""
#     from transforms3d.quaternions import quat2axangle
#     axis, angle = quat2axangle(quat)
#     return (axis * angle).astype(np.float32)

def quat2axisangle(quat):
    """Convert quaternion (x,y,z,w) to axis-angle, matching robosuite convention."""
    import math
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)
    
def get_proprio_from_obs(obs):
    """Extract 8d LIBERO proprio: [EEF_pos(3), EEF_axisangle(3), gripper_qpos(2)]."""
    eef_pos = obs["robot0_eef_pos"]
    eef_rot = quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]
    return np.concatenate([eef_pos, eef_rot, gripper]).astype(np.float32)


def center_crop_and_resize(img_array: np.ndarray, crop_scale: float, target_size: int = 224) -> np.ndarray:
    """Center crop with area ratio ``crop_scale``, then resize back to ``target_size``.

    Inference-time deterministic equivalent of training's
    ``RandomResizedCrop(target_size, scale=(lo, hi), ratio=(1, 1))``.
    Set ``crop_scale`` to the midpoint of ``(lo, hi)`` for a representative crop.
    """
    from PIL import Image as PILImage
    h, w = img_array.shape[:2]
    side_ratio = math.sqrt(crop_scale)
    crop_h = int(round(h * side_ratio))
    crop_w = int(round(w * side_ratio))
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    cropped = img_array[start_h:start_h + crop_h, start_w:start_w + crop_w]
    if cropped.shape[0] != target_size or cropped.shape[1] != target_size:
        cropped = np.array(
            PILImage.fromarray(cropped).resize((target_size, target_size), PILImage.BILINEAR),
            dtype=np.uint8,
        )
    return cropped


def get_libero_images(env, obs, size=224, center_crop_scale=None):
    """Extract camera images from LIBERO observation.

    If ``center_crop_scale`` is not None, a center crop with the given area
    ratio is applied after resize — matching the inference-time equivalent of
    training's ``RandomResizedCrop`` augmentation.
    """
    global _image_frame_idx
    from PIL import Image as PILImage
    images = {}
    agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
    if agentview is not None:
        img = PILImage.fromarray(agentview[::-1, ::-1])
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        if center_crop_scale is not None:
            arr = center_crop_and_resize(arr, center_crop_scale, size)
        images["base_0_rgb"] = arr

    wrist = obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand_rgb"))
    if wrist is not None:
        img = PILImage.fromarray(wrist[::-1, ::-1])
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        if center_crop_scale is not None:
            arr = center_crop_and_resize(arr, center_crop_scale, size)
        images["left_wrist_0_rgb"] = arr

    return images


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def skip_init_weights():
    """Bypass random weight initialization when loading from checkpoint.

    Saves ~30-60s for large models (3B+ params) by replacing torch.nn.init
    functions with no-ops. Safe when all weights are overwritten by a checkpoint.
    """
    saved = {}
    _noop = lambda x, *a, **kw: x
    for name in (
        "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_", "xavier_normal_",
        "uniform_", "normal_", "zeros_", "ones_", "constant_", "orthogonal_",
        "trunc_normal_",
    ):
        if hasattr(torch.nn.init, name):
            saved[name] = getattr(torch.nn.init, name)
            setattr(torch.nn.init, name, _noop)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch.nn.init, name, fn)


def _apply_sdpa(gemma_model, label: str):
    """Enable SDPA (Scaled Dot-Product Attention) on a GemmaModel.

    Falls back to eager if SDPA is unavailable or fails.
    """
    try:
        gemma_model.config._attn_implementation = "sdpa"
        logger.info(f"{label}: enabled SDPA attention")
    except Exception as e:
        logger.warning(f"{label}: failed to enable SDPA: {e}")


def _apply_compile(parent_module, attr_name: str, label: str):
    """Apply torch.compile to a sub-module, replacing the attribute in-place."""
    try:
        original = getattr(parent_module, attr_name)
        compiled = torch.compile(original)
        setattr(parent_module, attr_name, compiled)
        logger.info(f"{label}: enabled torch.compile")
    except Exception as e:
        logger.warning(f"{label}: torch.compile failed: {e}")


def load_vlm(cfg, device):
    """Load VLM waypoint predictor from checkpoint.

    Handles checkpoints saved from PI0WaypointAE structure by remapping
    paligemma_with_expert.paligemma.* -> paligemma.* keys.
    """
    from openpi.waypoint.vlm_model import PI0WaypointVLM
    import openpi.models.pi0_config as pi0_config

    t0 = time.time()
    vlm_precision = cfg.get("vlm_precision", "bfloat16")
    model_cfg = pi0_config.Pi0Config(
        pi05=False,
        max_token_len=cfg.get("max_token_len", 256),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        dtype=vlm_precision,
    )
    logger.info(f"VLM precision: {vlm_precision}")
    with skip_init_weights():
        model = PI0WaypointVLM(model_cfg)
    logger.info(f"VLM model init: {time.time() - t0:.1f}s")

    ckpt_path = cfg["vlm_checkpoint"]
    ckpt_file = os.path.join(ckpt_path, "model.safetensors")
    logger.info(f"Loading VLM from {ckpt_path}")

    # Read only the safetensors header (~91KB) to detect key format,
    # instead of loading the entire checkpoint (~11GB) just to inspect keys.
    with open(ckpt_file, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header_keys = list(json.loads(f.read(header_size)).keys())

    has_pg_direct = any(k.startswith("paligemma.") for k in header_keys)
    has_pg_nested = any(k.startswith("paligemma_with_expert.paligemma.") for k in header_keys)

    t0 = time.time()
    if has_pg_direct:
        safetensors.torch.load_model(model, ckpt_file)
    elif has_pg_nested:
        PREFIX = "paligemma_with_expert.paligemma."
        state_dict = safetensors.torch.load_file(ckpt_file, device="cpu")
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith(PREFIX):
                new_key = "paligemma." + k[len(PREFIX):]
                remapped[new_key] = v
        own_state = model.state_dict()
        loaded, skipped = 0, 0
        for k, v in remapped.items():
            if k in own_state and own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(own_state, strict=False)
        del state_dict
        logger.info(f"VLM: loaded {loaded} params, skipped {skipped} (remapped from AE checkpoint)")
    else:
        raise ValueError(f"Cannot find PaliGemma weights in checkpoint: {ckpt_file}")
    logger.info(f"VLM weight load: {time.time() - t0:.1f}s")

    t0 = time.time()
    model = model.to(device).eval()
    logger.info(f"VLM to {device}: {time.time() - t0:.1f}s")

    _apply_sdpa(model.paligemma.model.language_model, "VLM language_model")

    if cfg.get("torch_compile", False):
        _apply_compile(model.paligemma.model, "language_model", "VLM language_model")

    return model


def load_ae(cfg, device):
    """Load Action Expert from checkpoint."""
    from openpi.waypoint.ae_model import PI0WaypointAE
    import openpi.models.pi0_config as pi0_config

    t0 = time.time()
    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=cfg.get("model_action_dim", 32),
        action_horizon=cfg.get("horizon_steps", 32),
        max_token_len=cfg.get("max_token_len", 64),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        dtype=cfg.get("precision", "bfloat16"),
    )
    with skip_init_weights():
        model = PI0WaypointAE(model_cfg)
    logger.info(f"AE model init: {time.time() - t0:.1f}s")

    ckpt_path = cfg["ae_checkpoint"]
    logger.info(f"Loading Action Expert from {ckpt_path}")
    t0 = time.time()
    safetensors.torch.load_model(model, os.path.join(ckpt_path, "model.safetensors"))
    logger.info(f"AE weight load: {time.time() - t0:.1f}s")

    t0 = time.time()
    model = model.to(device).eval()
    logger.info(f"AE to {device}: {time.time() - t0:.1f}s")

    _apply_sdpa(model.paligemma_with_expert.paligemma.model.language_model, "AE paligemma")
    _apply_sdpa(model.paligemma_with_expert.gemma_expert.model, "AE gemma_expert")

    if cfg.get("torch_compile", False):
        _apply_compile(
            model.paligemma_with_expert.gemma_expert, "model", "AE gemma_expert",
        )

    return model


def load_joint(cfg, device):
    """Load joint VLM+AE model from a single checkpoint.

    The joint model shares a single PaliGemma backbone between VLM and AE,
    saving ~50% VRAM compared to loading two separate models.
    """
    from openpi.waypoint.joint_model import PI0WaypointJoint
    import openpi.models.pi0_config as pi0_config

    t0 = time.time()
    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=cfg.get("model_action_dim", 32),
        action_horizon=cfg.get("horizon_steps", 32),
        max_token_len=cfg.get("ae_max_token_len", cfg.get("max_token_len", 64)),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        dtype=cfg.get("precision", "bfloat16"),
    )
    with skip_init_weights():
        model = PI0WaypointJoint(
            config=model_cfg,
            vlm_max_token_len=cfg.get("vlm_max_token_len", cfg.get("max_token_len", 256)),
            gradient_strategy="none",
        )
    logger.info(f"Joint model init: {time.time() - t0:.1f}s")

    ckpt_path = cfg["joint_checkpoint"]
    ckpt_file = os.path.join(ckpt_path, "model.safetensors")
    logger.info(f"Loading joint model from {ckpt_path}")
    t0 = time.time()
    PI0WaypointJoint.load_pretrained_weights(model, ckpt_file, "cpu")
    logger.info(f"Joint weight load: {time.time() - t0:.1f}s")

    t0 = time.time()
    model = model.to(device).eval()
    logger.info(f"Joint model to {device}: {time.time() - t0:.1f}s")

    _apply_sdpa(model.paligemma_with_expert.paligemma.model.language_model, "Joint paligemma")
    _apply_sdpa(model.paligemma_with_expert.gemma_expert.model, "Joint gemma_expert")

    if cfg.get("torch_compile", False):
        _apply_compile(
            model.paligemma_with_expert.gemma_expert, "model", "Joint gemma_expert",
        )

    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_waypoints(vlm, images, instruction, wp_tokenizer, state_continuous_norm, gripper_binary, device):
    """VLM autoregressive waypoint prediction.

    Args:
        state_continuous_norm: normalized continuous proprio (e.g. 6D for LIBERO).
        gripper_binary: 0=close, 1=open.
    """
    img_tensors = {}
    img_masks = {}
    for key, arr in images.items():
        t = torch.from_numpy(arr).float() / 127.5 - 1.0
        img_tensors[key] = t.unsqueeze(0).to(device)
        img_masks[key] = torch.ones(1, dtype=torch.bool, device=device)

    for model_key in ["base_0_rgb", "left_wrist_0_rgb"]:
        if model_key not in img_tensors:
            img_tensors[model_key] = torch.zeros(1, 224, 224, 3, device=device)
            img_masks[model_key] = torch.zeros(1, dtype=torch.bool, device=device)

    from openpi.waypoint.tokenizer import PROPRIO_N_BINS

    proprio_dim = wp_tokenizer.proprio_dim
    state_for_prompt = state_continuous_norm[:proprio_dim]
    discretized = np.digitize(np.clip(state_for_prompt, -1, 1), np.linspace(-1, 1, PROPRIO_N_BINS + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized.astype(int)))

    if wp_tokenizer.use_gripper_token:
        grip_str = "open" if gripper_binary else "closed"
        prompt_text = f"Task: {instruction.strip().replace('_', ' ').lower()}, State: {state_str}, Gripper: {grip_str};\n"
    else:
        prompt_text = f"Task: {instruction.strip().replace('_', ' ').lower()}, State: {state_str};\n"

    prompt_tokens_list = wp_tokenizer._pg_tokenizer.encode(prompt_text, add_bos=True)
    prompt_tokens = torch.tensor([prompt_tokens_list], dtype=torch.long, device=device)
    prompt_mask = torch.ones_like(prompt_tokens, dtype=torch.bool)

    waypoints = vlm.generate_waypoints(
        images=img_tensors,
        image_masks=img_masks,
        prompt_tokens=prompt_tokens,
        prompt_mask=prompt_mask,
        wp_tokenizer=wp_tokenizer,
    )
    return waypoints[0]


def load_pg_tokenizer():
    """Load PaliGemma SentencePiece tokenizer (cached, call once)."""
    import sentencepiece
    import openpi.shared.download as download

    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        return sentencepiece.SentencePieceProcessor(model_proto=f.read())


@torch.no_grad()
def predict_actions(ae_model, images, instruction, start_wp, end_wp, duration, device, pg_tok):
    """Action Expert flow matching inference."""
    img_tensors = {}
    img_masks = {}
    for key, arr in images.items():
        t = torch.from_numpy(arr).float() / 127.5 - 1.0  # (H, W, C)
        t = t.permute(2, 0, 1)  # -> (C, H, W)
        img_tensors[key] = t.unsqueeze(0).to(device)  # (1, C, H, W)
        img_masks[key] = torch.ones(1, dtype=torch.bool, device=device)

    for model_key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        if model_key not in img_tensors:
            img_tensors[model_key] = torch.zeros(1, 3, 224, 224, device=device)
            img_masks[model_key] = torch.zeros(1, dtype=torch.bool, device=device)

    text = f"Task: {instruction.strip().replace('_', ' ').lower()}, \n"
    tids = pg_tok.encode(text, add_bos=True)
    max_len = 64
    tids_len = len(tids)
    if tids_len < max_len:
        tids = tids + [0] * (max_len - tids_len)
        mask = [True] * tids_len + [False] * (max_len - tids_len)
    else:
        tids = tids[:max_len]
        mask = [True] * max_len

    prompt_tokens = torch.tensor([tids[:max_len]], dtype=torch.long, device=device)
    prompt_mask = torch.tensor([mask[:max_len]], dtype=torch.bool, device=device)

    class _Obs:
        def __init__(self):
            self.images = img_tensors
            self.image_masks = img_masks
            self.state = torch.zeros(1, 32, device=device)
            self.tokenized_prompt = prompt_tokens
            self.tokenized_prompt_mask = prompt_mask
            self.token_ar_mask = None
            self.token_loss_mask = None
    obs = _Obs()

    start_t = torch.from_numpy(start_wp).float().unsqueeze(0).to(device)
    end_t = torch.from_numpy(end_wp).float().unsqueeze(0).to(device)
    dur_t = torch.tensor([float(duration)], dtype=torch.float32, device=device)

    actions = ae_model.sample_actions(obs, start_t, end_t, dur_t)
    return actions.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _fmt_array(a, n=4):
    """Format first n elements of an array for logging."""
    flat = np.asarray(a).flatten()
    vals = " ".join(f"{v:.4f}" for v in flat[:n])
    if len(flat) > n:
        vals += " ..."
    return f"[{vals}]"


def run_episode(
    vlm, ae_model, wp_tokenizer, norm_helper,
    env, initial_state, task_desc, cfg, device, pg_tok,
):
    """Run one LIBERO episode with the two-stage waypoint pipeline.

    Returns:
        success: bool
        replay_images: list of uint8 numpy images for video recording
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    rc = get_robot_config("libero")
    model_proprio_dim = cfg.get("model_proprio_dim", 32)
    actual_action_dim = rc.actual_action_dim
    max_steps = MAX_STEPS_MAP.get(cfg.get("task_suite", "libero_object"), 280)
    num_steps_wait = cfg.get("num_steps_wait", 10)
    crop_scale = cfg.get("center_crop_scale") if cfg.get("center_crop", False) else None

    t = 0
    done = False
    reward = 0.0
    replay_images = []
    replan_count = 0

    dummy_action = np.zeros(7)
    while t < num_steps_wait:
        obs, _, done, _ = env.step(dummy_action)
        t += 1
        if done:
            return True, replay_images

    while t < max_steps + num_steps_wait and not done:
        agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
        if agentview is not None:
            head_frame = np.ascontiguousarray(agentview[::-1, ::-1])
            wrist_raw = obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand_rgb"))
            if wrist_raw is not None:
                wrist_frame = np.ascontiguousarray(wrist_raw[::-1, ::-1])
                if wrist_frame.shape[0] != head_frame.shape[0]:
                    from PIL import Image as PILImage
                    wrist_frame = np.array(
                        PILImage.fromarray(wrist_frame).resize(
                            (wrist_frame.shape[1], head_frame.shape[0]), PILImage.BILINEAR
                        )
                    )
                replay_images.append(np.concatenate([head_frame, wrist_frame], axis=1))
            else:
                replay_images.append(head_frame)

        images = get_libero_images(env, obs, center_crop_scale=crop_scale)
        proprio_raw = get_proprio_from_obs(obs)
        # Split into continuous + binary gripper
        continuous_raw, gripper_binary = rc.split_proprio(proprio_raw)
        continuous_norm = norm_helper.normalize_proprio(continuous_raw)

        t_vlm = time.time()
        waypoints = predict_waypoints(
            vlm, images, task_desc, wp_tokenizer,
            continuous_norm, gripper_binary, device,
        )
        vlm_ms = (time.time() - t_vlm) * 1000

        replan_count += 1
        if not waypoints:
            logger.info(f"  [replan {replan_count}] VLM returned empty waypoints ({vlm_ms:.0f}ms), stopping")
            break

        valid_wps = [(p, d) for p, d in waypoints if d > 0]
        durations = [d for _, d in valid_wps]
        logger.info(
            f"  [replan {replan_count}] VLM: {len(waypoints)} waypoints, "
            f"{len(valid_wps)} valid, durations={durations}, vlm_time={vlm_ms:.0f}ms"
        )
        for wi, (pv, dur) in enumerate(waypoints):
            logger.info(f"    wp[{wi}]: proprio={_fmt_array(pv, 6)}, duration={dur}")

        # Build 7D start_wp: [continuous_norm(6D), gripper_binary(1D)]
        start_wp_7d = np.concatenate([continuous_norm, [float(gripper_binary)]])
        start_wp = pad_to_dim(start_wp_7d, model_proprio_dim)
        steps_this_cycle = 0

        max_dur = cfg.get("horizon_steps", 32)
        for wp_idx, (proprio_values, duration) in enumerate(waypoints):
            # proprio_values is 7D: [6D continuous_norm, 1D gripper_binary]
            if duration == 0:
                break
            if done:
                break
            if duration < 0:
                logger.warning(f"    wp[{wp_idx}]: negative duration {duration}, skipping")
                continue
            if duration > max_dur:
                logger.warning(f"    wp[{wp_idx}]: duration {duration} exceeds max {max_dur}, clamping")
                duration = max_dur

            end_wp = pad_to_dim(proprio_values, model_proprio_dim)

            fresh_images = get_libero_images(env, obs, center_crop_scale=crop_scale)
            t_ae = time.time()
            actions_norm = predict_actions(
                ae_model, fresh_images, task_desc, start_wp, end_wp, duration, device, pg_tok,
            )
            ae_ms = (time.time() - t_ae) * 1000

            num_execute = min(int(duration), actions_norm.shape[0])
            logger.info(
                f"    ae[{wp_idx}]: shape={actions_norm.shape}, execute={num_execute}, "
                f"range=[{actions_norm.min():.3f}, {actions_norm.max():.3f}], ae_time={ae_ms:.0f}ms"
            )

            for step_i in range(num_execute):
                action_raw = norm_helper.unnormalize_actions(actions_norm[step_i, :actual_action_dim])

                gripper = action_raw[-1]
                gripper = gripper * 2.0 - 1.0
                gripper = np.sign(gripper)
                gripper = -gripper
                action_raw[-1] = gripper

                obs, reward, done, info = env.step(action_raw)
                t += 1
                steps_this_cycle += 1

                agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
                if agentview is not None:
                    head_frame = np.ascontiguousarray(agentview[::-1, ::-1])
                    wrist_raw = obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand_rgb"))
                    if wrist_raw is not None:
                        wrist_frame = np.ascontiguousarray(wrist_raw[::-1, ::-1])
                        if wrist_frame.shape[0] != head_frame.shape[0]:
                            from PIL import Image as PILImage
                            wrist_frame = np.array(
                                PILImage.fromarray(wrist_frame).resize(
                                    (wrist_frame.shape[1], head_frame.shape[0]), PILImage.BILINEAR
                                )
                            )
                        replay_images.append(np.concatenate([head_frame, wrist_frame], axis=1))
                    else:
                        replay_images.append(head_frame)

                if done:
                    break

            # Use actual robot observation instead of VLM-predicted end_wp
            # to prevent error accumulation across waypoint pairs.
            actual_proprio = get_proprio_from_obs(obs)
            actual_cont, actual_grip = rc.split_proprio(actual_proprio)
            actual_cont_norm = norm_helper.normalize_proprio(actual_cont)
            start_wp_7d = np.concatenate([actual_cont_norm, [float(actual_grip)]])
            start_wp = pad_to_dim(start_wp_7d, model_proprio_dim)

        if steps_this_cycle == 0 and not done:
            logger.warning(f"  [replan {replan_count}] no actions executed, advancing with no-op")
            obs, reward, done, info = env.step(np.zeros(actual_action_dim))
            t += 1

    logger.info(f"  episode done: steps={t}, replans={replan_count}, success={done and reward > 0}")
    return done and reward > 0, replay_images


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    t_total = time.time()
    use_joint = "joint_checkpoint" in cfg
    if use_joint:
        joint_model = load_joint(cfg, device)
        vlm = joint_model      # generate_waypoints() lives on joint model
        ae_model = joint_model  # sample_actions() lives on joint model
        logger.info(f"Joint model loaded (shared backbone): {time.time() - t_total:.1f}s")
    else:
        vlm = load_vlm(cfg, device)
        ae_model = load_ae(cfg, device)
        logger.info(f"Total model loading (separate VLM+AE): {time.time() - t_total:.1f}s")

    t0 = time.time()
    pg_tok = load_pg_tokenizer()
    logger.info(f"PaliGemma tokenizer loaded: {time.time() - t0:.1f}s")

    rc = get_robot_config("libero")
    stats = load_dataset_statistics(cfg["dataset_statistics_path"])
    norm_helper = NormalizationHelper(stats, cfg.get("norm_type", "q99"))
    if rc.action_norm_mask is not None:
        norm_helper.action_norm_mask = rc.action_norm_mask

    wp_tokenizer = WaypointTokenizer(
        proprio_dim=rc.continuous_proprio_dim,
        num_waypoints=cfg.get("num_waypoints", 7),
        max_token_len=cfg.get("max_token_len", 256),
        use_gripper_token=True,
    )

    if cfg.get("center_crop", False):
        logger.info(
            f"Center crop enabled: area_scale={cfg.get('center_crop_scale', 0.95)}, "
            f"side_ratio={math.sqrt(cfg.get('center_crop_scale', 0.95)):.4f}"
        )

    video_out_path = pathlib.Path(cfg.get("video_out_path", "data/libero/videos"))
    video_out_path.mkdir(parents=True, exist_ok=True)

    global _image_save_dir, _image_frame_idx
    _image_save_dir = pathlib.Path("image")
    _image_save_dir.mkdir(parents=True, exist_ok=True)
    _image_frame_idx = 0
    logger.info(f"Inference images will be saved to: {_image_save_dir.resolve()}")

    from libero.libero import benchmark

    task_suite_name = cfg.get("task_suite", "libero_object")
    bm = benchmark.get_benchmark_dict()[task_suite_name]()
    num_tasks = bm.n_tasks
    num_trials = cfg.get("num_trials_per_task", 3)

    task_start = cfg.get("task_start", 0)
    task_end = cfg.get("task_end", num_tasks)
    task_end = min(task_end, num_tasks)
    logger.info(f"Evaluating tasks [{task_start}, {task_end}) out of {num_tasks} total")

    results = {}
    total_success = 0
    total_episodes = 0

    for task_idx in range(task_start, task_end):
        task = bm.get_task(task_idx)
        task_name = task.name
        task_desc = task.language
        initial_states = bm.get_task_init_states(task_idx)

        env_args = {
            "bddl_file_name": bm.get_task_bddl_file_path(task_idx),
            "camera_heights": 256,
            "camera_widths": 256,
        }

        from libero.libero.envs import OffScreenRenderEnv
        t0 = time.time()
        env = OffScreenRenderEnv(**env_args)
        env.seed(7)
        logger.info(f"Env init for task {task_idx}: {time.time() - t0:.1f}s")

        successes = 0
        for trial in range(min(num_trials, len(initial_states))):
            t_ep = time.time()
            logger.info(f"Task {task_idx}/{num_tasks}: {task_name} trial {trial}")
            success, replay_images = run_episode(
                vlm, ae_model, wp_tokenizer, norm_helper,
                env, initial_states[trial], task_desc, cfg, device, pg_tok,
            )
            ep_secs = time.time() - t_ep
            if success:
                successes += 1
                total_success += 1
            total_episodes += 1

            suffix = "success" if success else "failure"
            task_segment = task_desc.replace(" ", "_")
            if replay_images:
                video_file = video_out_path / f"rollout_{task_segment}_t{trial}_{suffix}.mp4"
                imageio.mimwrite(
                    str(video_file),
                    [np.asarray(x) for x in replay_images],
                    fps=20,
                )
                logger.info(f"  -> {suffix.upper()} ({ep_secs:.1f}s, video: {video_file})")
            else:
                logger.info(f"  -> {suffix.upper()} ({ep_secs:.1f}s)")

            trials_done = trial + 1
            task_sr = successes / trials_done
            overall_sr = total_success / max(total_episodes, 1)
            logger.info(
                f"  [成功率] 当前任务: {successes}/{trials_done} = {task_sr:.2%} | "
                f"整体: {total_success}/{total_episodes} = {overall_sr:.2%}"
            )

        results[task_name] = {
            "success_rate": successes / min(num_trials, len(initial_states)),
            "successes": successes,
            "trials": min(num_trials, len(initial_states)),
        }
        env.close()

    overall_rate = total_success / max(total_episodes, 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"Overall success rate: {overall_rate:.2%} ({total_success}/{total_episodes})")
    for name, r in results.items():
        logger.info(f"  {name}: {r['success_rate']:.2%} ({r['successes']}/{r['trials']})")
    logger.info(f"Total eval time: {time.time() - t_total:.1f}s")

    return results


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--task-start", type=int, default=None, help="First task index (inclusive)")
    parser.add_argument("--task-end", type=int, default=None, help="Last task index (exclusive)")
    parser.add_argument("--results-file", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.task_start is not None:
        cfg["task_start"] = args.task_start
    if args.task_end is not None:
        cfg["task_end"] = args.task_end

    results = evaluate(cfg)

    if args.results_file:
        with open(args.results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
