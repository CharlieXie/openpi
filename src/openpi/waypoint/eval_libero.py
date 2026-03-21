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
import pathlib
import time
from pathlib import Path

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

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


def get_libero_images(env, obs, size=224):
    """Extract camera images from LIBERO observation."""
    global _image_frame_idx
    from PIL import Image as PILImage
    images = {}
    agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
    if agentview is not None:
        img = PILImage.fromarray(agentview[::-1, ::-1])
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        images["base_0_rgb"] = np.array(img, dtype=np.uint8)

    wrist = obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand_rgb"))
    if wrist is not None:
        img = PILImage.fromarray(wrist[::-1, ::-1])
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        images["left_wrist_0_rgb"] = np.array(img, dtype=np.uint8)

    # if _image_save_dir is not None and images:
    #     frames = []
    #     if "base_0_rgb" in images:
    #         frames.append(images["base_0_rgb"])
    #     if "left_wrist_0_rgb" in images:
    #         frames.append(images["left_wrist_0_rgb"])
    #     if frames:
    #         combined = np.concatenate(frames, axis=1)
    #         save_path = pathlib.Path(_image_save_dir) / f"frame_{_image_frame_idx:06d}.png"
    #         PILImage.fromarray(combined).save(str(save_path))
    #         _image_frame_idx += 1

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
def predict_waypoints(vlm, images, instruction, wp_tokenizer, state_norm, device):
    """VLM autoregressive waypoint prediction."""
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

    proprio_dim = wp_tokenizer.proprio_dim
    state_for_prompt = state_norm[:proprio_dim]
    prompt_text = f"Task: {instruction.strip().replace('_', ' ').lower()}, State: "
    discretized = np.digitize(np.clip(state_for_prompt, -1, 1), np.linspace(-1, 1, 257)[:-1]) - 1
    prompt_text += " ".join(map(str, discretized.astype(int))) + ";\n"

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

        images = get_libero_images(env, obs)
        proprio_raw = get_proprio_from_obs(obs)
        proprio_norm = norm_helper.normalize_proprio(proprio_raw)
        state_padded = pad_to_dim(proprio_norm, model_proprio_dim)

        t_vlm = time.time()
        waypoints = predict_waypoints(vlm, images, task_desc, wp_tokenizer, state_padded, device)
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

        start_wp = state_padded.copy()
        steps_this_cycle = 0

        max_dur = cfg.get("horizon_steps", 32)
        for wp_idx, (proprio_values, duration) in enumerate(waypoints):
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

            fresh_images = get_libero_images(env, obs)
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

            start_wp = end_wp.copy()

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
        proprio_dim=rc.actual_proprio_dim,
        num_waypoints=cfg.get("num_waypoints", 7),
        max_token_len=cfg.get("max_token_len", 256),
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

    results = {}
    total_success = 0
    total_episodes = 0

    for task_idx in range(num_tasks):
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
        env.seed(0)
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
                    fps=10,
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


# ---------------------------------------------------------------------------
# Parallel evaluation infrastructure
# ---------------------------------------------------------------------------

def _record_frame(obs, replay_images):
    """Extract agentview + wrist camera frame and append to replay buffer."""
    agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
    if agentview is None:
        return
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


class _InferenceRequest:
    """Pending GPU inference request submitted by an episode thread."""
    __slots__ = ("type", "data", "result", "event")

    def __init__(self, req_type: str, data: dict):
        self.type = req_type
        self.data = data
        self.result = None
        self.event = threading.Event()


class InferenceServer:
    """Dedicated GPU thread that collects, batches, and serves inference requests.

    Episode threads submit VLM/AE requests via ``submit_vlm`` / ``submit_ae``
    (which block until the result is ready). The server thread collects requests
    from a shared queue, waits briefly for more same-type requests to arrive
    (opportunistic batching), then runs batched inference and unblocks callers.

    This ensures:
      - Only one thread ever touches CUDA (no contention).
      - Multiple episode threads can run env.step() in parallel while the
        server is idle or processing another episode's request.
    """

    def __init__(
        self,
        vlm,
        ae_model,
        wp_tokenizer,
        device,
        pg_tok,
        max_batch: int = 8,
        max_wait_ms: float = 8,
    ):
        self.vlm = vlm
        self.ae_model = ae_model
        self.wp_tokenizer = wp_tokenizer
        self.device = device
        self.pg_tok = pg_tok
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self._queue: Queue = Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._serve_loop, daemon=True, name="inference-server",
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)

    # -- public API (called from episode threads, blocks until done) --------

    def submit_vlm(self, images: dict, instruction: str, state_padded: np.ndarray):
        req = _InferenceRequest("vlm", {
            "images": images,
            "instruction": instruction,
            "state_padded": state_padded,
        })
        self._queue.put(req)
        req.event.wait()
        return req.result

    def submit_ae(
        self, images: dict, instruction: str,
        start_wp: np.ndarray, end_wp: np.ndarray, duration: float,
    ):
        req = _InferenceRequest("ae", {
            "images": images,
            "instruction": instruction,
            "start_wp": start_wp,
            "end_wp": end_wp,
            "duration": duration,
        })
        self._queue.put(req)
        req.event.wait()
        return req.result

    # -- server loop (single dedicated GPU thread) -------------------------

    def _serve_loop(self):
        while self._running:
            try:
                first = self._queue.get(timeout=0.5)
            except Empty:
                continue

            batch = [first]
            deadline = time.time() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    req = self._queue.get(timeout=max(0.001, remaining))
                    if req.type == first.type:
                        batch.append(req)
                    else:
                        self._queue.put(req)
                        break
                except Empty:
                    break

            try:
                if first.type == "vlm":
                    self._run_vlm_batch(batch)
                else:
                    self._run_ae_batch(batch)
            except Exception as exc:
                logger.error(
                    "Inference server error (%s, batch=%d): %s",
                    first.type, len(batch), exc, exc_info=True,
                )
                for req in batch:
                    req.result = [] if req.type == "vlm" else None
                    req.event.set()

    # -- batched VLM -------------------------------------------------------

    @torch.no_grad()
    def _run_vlm_batch(self, requests: list):
        B = len(requests)
        device = self.device
        wp_tok = self.wp_tokenizer

        img_tensors: dict[str, torch.Tensor] = {}
        img_masks: dict[str, torch.Tensor] = {}
        for key in ["base_0_rgb", "left_wrist_0_rgb"]:
            imgs, masks = [], []
            for req in requests:
                raw = req.data["images"]
                if key in raw:
                    t = torch.from_numpy(raw[key]).float() / 127.5 - 1.0
                    imgs.append(t.unsqueeze(0))
                    masks.append(torch.ones(1, dtype=torch.bool))
                else:
                    imgs.append(torch.zeros(1, 224, 224, 3))
                    masks.append(torch.zeros(1, dtype=torch.bool))
            img_tensors[key] = torch.cat(imgs, dim=0).to(device)
            img_masks[key] = torch.cat(masks, dim=0).to(device)

        token_lists = []
        for req in requests:
            d = req.data
            proprio_dim = wp_tok.proprio_dim
            state_for_prompt = d["state_padded"][:proprio_dim]
            prompt = f"Task: {d['instruction'].strip().replace('_', ' ').lower()}, State: "
            disc = np.digitize(
                np.clip(state_for_prompt, -1, 1),
                np.linspace(-1, 1, 257)[:-1],
            ) - 1
            prompt += " ".join(map(str, disc.astype(int))) + ";\n"
            token_lists.append(wp_tok._pg_tokenizer.encode(prompt, add_bos=True))

        max_len = max(len(tl) for tl in token_lists)
        padded_ids, padded_mask = [], []
        for tl in token_lists:
            pad = max_len - len(tl)
            padded_ids.append(tl + [0] * pad)
            padded_mask.append([True] * len(tl) + [False] * pad)

        prompt_tokens = torch.tensor(padded_ids, dtype=torch.long, device=device)
        prompt_mask = torch.tensor(padded_mask, dtype=torch.bool, device=device)

        t0 = time.time()
        all_waypoints = self.vlm.generate_waypoints(
            images=img_tensors,
            image_masks=img_masks,
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
            wp_tokenizer=wp_tok,
        )
        elapsed = (time.time() - t0) * 1000
        logger.info(f"  [server] VLM batch={B}, time={elapsed:.0f}ms")

        for i, req in enumerate(requests):
            req.result = all_waypoints[i]
            req.event.set()

    # -- batched AE (direct model calls, bypasses preprocess_observation) ---

    @torch.no_grad()
    def _run_ae_batch(self, requests: list):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        B = len(requests)
        device = self.device
        model = self.ae_model

        camera_order = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        images_list: list[torch.Tensor] = []
        img_masks_list: list[torch.Tensor] = []
        for key in camera_order:
            has_any = any(key in r.data["images"] for r in requests)
            if not has_any:
                continue
            imgs, masks = [], []
            for req in requests:
                raw = req.data["images"]
                if key in raw:
                    t = torch.from_numpy(raw[key]).float() / 127.5 - 1.0
                    t = t.permute(2, 0, 1)
                    imgs.append(t.unsqueeze(0))
                    masks.append(torch.ones(1, dtype=torch.bool))
                else:
                    imgs.append(torch.zeros(1, 3, 224, 224))
                    masks.append(torch.zeros(1, dtype=torch.bool))
            images_list.append(torch.cat(imgs, dim=0).to(device))
            img_masks_list.append(torch.cat(masks, dim=0).to(device))

        AE_MAX_TOK = 64
        tids_batch, mask_batch = [], []
        for req in requests:
            text = f"Task: {req.data['instruction'].strip().replace('_', ' ').lower()}, \n"
            tids = self.pg_tok.encode(text, add_bos=True)
            n = len(tids)
            if n < AE_MAX_TOK:
                mask_batch.append([True] * n + [False] * (AE_MAX_TOK - n))
                tids = tids + [0] * (AE_MAX_TOK - n)
            else:
                tids = tids[:AE_MAX_TOK]
                mask_batch.append([True] * AE_MAX_TOK)
            tids_batch.append(tids)

        lang_tokens = torch.tensor(tids_batch, dtype=torch.long, device=device)
        lang_masks = torch.tensor(mask_batch, dtype=torch.bool, device=device)

        start_proprio = torch.stack(
            [torch.from_numpy(r.data["start_wp"]).float() for r in requests],
        ).to(device)
        end_proprio = torch.stack(
            [torch.from_numpy(r.data["end_wp"]).float() for r in requests],
        ).to(device)
        duration = torch.tensor(
            [float(r.data["duration"]) for r in requests],
            dtype=torch.float32, device=device,
        )

        # --- embed prefix (images + language) via model helper ---
        prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
            images_list, img_masks_list, lang_tokens, lang_masks,
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
        prefix_att_4d = prefix_att_2d[:, None, :, :]
        prefix_att_4d = torch.where(prefix_att_4d, 0.0, -2.3819763e38)

        model_dtype = (
            model.paligemma_with_expert.paligemma
            .language_model.layers[0].self_attn.q_proj.weight.dtype
        )
        if prefix_embs.dtype != model_dtype:
            prefix_embs = prefix_embs.to(model_dtype)
        if prefix_att_4d.dtype != model_dtype:
            prefix_att_4d = prefix_att_4d.to(model_dtype)

        _, past_kv = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # --- iterative denoising ---
        num_steps = 10
        noise = torch.randn(B, model.action_horizon, model.action_dim, device=device)
        dt = -1.0 / num_steps
        x_t = noise
        t_val = torch.tensor(1.0, device=device)

        t0 = time.time()
        while t_val >= -dt / 2:
            expanded_t = t_val.expand(B)
            suffix_embs, suffix_pad, suffix_att, adarms_cond = model.embed_suffix(
                start_proprio, end_proprio, x_t, expanded_t, duration,
            )

            suffix_len = suffix_pad.shape[1]
            prefix_len = prefix_pad.shape[1]

            prefix_2d = prefix_pad[:, None, :].expand(B, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
            full_att = torch.cat([prefix_2d, suffix_att_2d], dim=2)
            full_att_4d = full_att[:, None, :, :]
            full_att_4d = torch.where(full_att_4d, 0.0, -2.3819763e38)
            if full_att_4d.dtype != model_dtype:
                full_att_4d = full_att_4d.to(model_dtype)

            prefix_offsets = torch.sum(prefix_pad, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1

            if suffix_embs.dtype != model_dtype:
                suffix_embs = suffix_embs.to(model_dtype)

            outputs, _ = model.paligemma_with_expert.forward(
                attention_mask=full_att_4d,
                position_ids=position_ids,
                past_key_values=past_kv,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = outputs[1][:, -model.action_horizon:].float()
            v_t = model.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t
            t_val = t_val + dt

        elapsed = (time.time() - t0) * 1000
        if B > 1:
            logger.debug(f"  [server] AE batch={B}, time={elapsed:.0f}ms")

        for i, req in enumerate(requests):
            req.result = x_t[i].cpu().numpy()
            req.event.set()


# ---------------------------------------------------------------------------
# Episode runner for parallel mode
# ---------------------------------------------------------------------------

def run_episode_with_server(
    server: InferenceServer,
    wp_tokenizer,
    norm_helper,
    env,
    initial_state,
    task_desc: str,
    cfg: dict,
    pg_tok,
    ep_label: str = "",
):
    """Run one LIBERO episode, delegating all GPU work to *server*.

    Semantically identical to ``run_episode`` but uses ``server.submit_vlm``
    and ``server.submit_ae`` instead of calling models directly.  This lets
    multiple episode threads run env.step() in parallel while the server
    thread batches and processes GPU requests.
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    rc = get_robot_config("libero")
    model_proprio_dim = cfg.get("model_proprio_dim", 32)
    actual_action_dim = rc.actual_action_dim
    max_steps = MAX_STEPS_MAP.get(cfg.get("task_suite", "libero_object"), 280)
    num_steps_wait = cfg.get("num_steps_wait", 10)

    t = 0
    done = False
    reward = 0.0
    replay_images: list[np.ndarray] = []
    replan_count = 0

    dummy_action = np.zeros(7)
    while t < num_steps_wait:
        obs, _, done, _ = env.step(dummy_action)
        t += 1
        if done:
            return True, replay_images

    while t < max_steps + num_steps_wait and not done:
        _record_frame(obs, replay_images)

        images = get_libero_images(env, obs)
        proprio_raw = get_proprio_from_obs(obs)
        proprio_norm = norm_helper.normalize_proprio(proprio_raw)
        state_padded = pad_to_dim(proprio_norm, model_proprio_dim)

        t_vlm = time.time()
        waypoints = server.submit_vlm(images, task_desc, state_padded)
        vlm_ms = (time.time() - t_vlm) * 1000

        replan_count += 1
        if not waypoints:
            logger.info(f"  {ep_label}[replan {replan_count}] VLM empty ({vlm_ms:.0f}ms)")
            break

        valid_wps = [(p, d) for p, d in waypoints if d > 0]
        durations = [d for _, d in valid_wps]
        logger.info(
            f"  {ep_label}[replan {replan_count}] VLM: {len(waypoints)} wps, "
            f"{len(valid_wps)} valid, durations={durations}, vlm={vlm_ms:.0f}ms"
        )

        start_wp = state_padded.copy()
        steps_this_cycle = 0
        max_dur = cfg.get("horizon_steps", 32)

        for wp_idx, (proprio_values, duration) in enumerate(waypoints):
            if duration <= 0 or done:
                break
            if duration > max_dur:
                duration = max_dur

            end_wp = pad_to_dim(proprio_values, model_proprio_dim)
            fresh_images = get_libero_images(env, obs)

            t_ae = time.time()
            actions_norm = server.submit_ae(
                fresh_images, task_desc, start_wp, end_wp, float(duration),
            )
            ae_ms = (time.time() - t_ae) * 1000

            if actions_norm is None:
                logger.warning(f"  {ep_label}ae[{wp_idx}]: server returned None")
                break

            num_execute = min(int(duration), actions_norm.shape[0])
            logger.info(
                f"  {ep_label}  ae[{wp_idx}]: exec={num_execute}, ae={ae_ms:.0f}ms"
            )

            for step_i in range(num_execute):
                action_raw = norm_helper.unnormalize_actions(
                    actions_norm[step_i, :actual_action_dim],
                )
                gripper = action_raw[-1]
                gripper = -np.sign(gripper * 2.0 - 1.0)
                action_raw[-1] = gripper

                obs, reward, done, info = env.step(action_raw)
                t += 1
                steps_this_cycle += 1
                _record_frame(obs, replay_images)

                if done:
                    break

            start_wp = end_wp.copy()

        if steps_this_cycle == 0 and not done:
            obs, reward, done, info = env.step(np.zeros(actual_action_dim))
            t += 1

    success = done and reward > 0
    logger.info(
        f"  {ep_label}episode done: steps={t}, replans={replan_count}, success={success}"
    )
    return success, replay_images


# ---------------------------------------------------------------------------
# Parallel evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_parallel(cfg, num_parallel: int = 4):
    """Run LIBERO evaluation with *num_parallel* episodes in flight.

    Architecture:
        - 1 InferenceServer thread  (GPU, batches VLM/AE requests)
        - N episode worker threads  (CPU, run env.step in parallel)

    MuJoCo's C core releases the GIL, so env.step() from different threads
    truly runs in parallel.  All CUDA operations are serialized through the
    single inference server thread (no contention).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | parallel episodes: {num_parallel}")

    t_total = time.time()

    use_joint = "joint_checkpoint" in cfg
    if use_joint:
        joint_model = load_joint(cfg, device)
        vlm = joint_model
        ae_model = joint_model
        logger.info(f"Joint model loaded: {time.time() - t_total:.1f}s")
    else:
        vlm = load_vlm(cfg, device)
        ae_model = load_ae(cfg, device)
        logger.info(f"Separate VLM+AE loaded: {time.time() - t_total:.1f}s")

    pg_tok = load_pg_tokenizer()
    rc = get_robot_config("libero")
    stats = load_dataset_statistics(cfg["dataset_statistics_path"])
    norm_helper = NormalizationHelper(stats, cfg.get("norm_type", "q99"))
    if rc.action_norm_mask is not None:
        norm_helper.action_norm_mask = rc.action_norm_mask

    wp_tokenizer = WaypointTokenizer(
        proprio_dim=rc.actual_proprio_dim,
        num_waypoints=cfg.get("num_waypoints", 7),
        max_token_len=cfg.get("max_token_len", 256),
    )

    video_out_path = pathlib.Path(cfg.get("video_out_path", "data/libero/videos"))
    video_out_path.mkdir(parents=True, exist_ok=True)

    server = InferenceServer(
        vlm=vlm,
        ae_model=ae_model,
        wp_tokenizer=wp_tokenizer,
        device=device,
        pg_tok=pg_tok,
        max_batch=num_parallel,
        max_wait_ms=8,
    )
    server.start()

    from libero.libero import benchmark

    task_suite_name = cfg.get("task_suite", "libero_object")
    bm = benchmark.get_benchmark_dict()[task_suite_name]()
    num_tasks = bm.n_tasks
    num_trials = cfg.get("num_trials_per_task", 3)

    episode_specs = []
    for task_idx in range(num_tasks):
        task = bm.get_task(task_idx)
        initial_states = bm.get_task_init_states(task_idx)
        for trial in range(min(num_trials, len(initial_states))):
            episode_specs.append({
                "task_idx": task_idx,
                "trial": trial,
                "task_name": task.name,
                "task_desc": task.language,
                "initial_state": initial_states[trial],
                "bddl_file": bm.get_task_bddl_file_path(task_idx),
            })

    logger.info(
        f"Total episodes: {len(episode_specs)} "
        f"({num_tasks} tasks x {num_trials} trials), parallel={num_parallel}"
    )

    results_lock = threading.Lock()
    results_by_task: dict[str, dict] = {}
    counters = {"success": 0, "done": 0}

    env_create_lock = threading.Lock()

    def _run_one(spec):
        from libero.libero.envs import OffScreenRenderEnv

        tidx, trial = spec["task_idx"], spec["trial"]
        label = f"T{tidx}t{trial} "

        env_args = {
            "bddl_file_name": spec["bddl_file"],
            "camera_heights": 256,
            "camera_widths": 256,
        }
        with env_create_lock:
            env = OffScreenRenderEnv(**env_args)
            env.seed(0)

        t_ep = time.time()
        try:
            success, replay_images = run_episode_with_server(
                server, wp_tokenizer, norm_helper,
                env, spec["initial_state"], spec["task_desc"],
                cfg, pg_tok, ep_label=label,
            )
        except Exception as exc:
            logger.error(f"{label}episode crashed: {exc}", exc_info=True)
            success = False
            replay_images = []
        finally:
            env.close()

        ep_secs = time.time() - t_ep

        suffix = "success" if success else "failure"
        task_segment = spec["task_desc"].replace(" ", "_")
        if replay_images:
            vf = video_out_path / f"rollout_{task_segment}_t{trial}_{suffix}.mp4"
            imageio.mimwrite(str(vf), [np.asarray(x) for x in replay_images], fps=10)

        with results_lock:
            counters["done"] += 1
            if success:
                counters["success"] += 1
            tn = spec["task_name"]
            if tn not in results_by_task:
                results_by_task[tn] = {"successes": 0, "trials": 0}
            results_by_task[tn]["trials"] += 1
            if success:
                results_by_task[tn]["successes"] += 1
            sr = counters["success"] / max(counters["done"], 1)

        logger.info(
            f"  {label}{'SUCCESS' if success else 'FAILURE'} ({ep_secs:.1f}s) | "
            f"overall: {counters['success']}/{counters['done']} = {sr:.2%}"
        )
        return spec, success

    try:
        with ThreadPoolExecutor(max_workers=num_parallel) as pool:
            futures = [pool.submit(_run_one, s) for s in episode_specs]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    logger.error(f"Episode thread error: {exc}", exc_info=True)
    finally:
        server.stop()

    overall_rate = counters["success"] / max(counters["done"], 1)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Overall success rate: {overall_rate:.2%} ({counters['success']}/{counters['done']})")
    for name, r in results_by_task.items():
        sr = r["successes"] / max(r["trials"], 1)
        logger.info(f"  {name}: {sr:.2%} ({r['successes']}/{r['trials']})")
    logger.info(f"Total eval time: {time.time() - t_total:.1f}s")

    return results_by_task


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--num_parallel", type=int, default=1,
        help="Number of episodes to run in parallel (1 = original serial mode).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.num_parallel > 1:
        evaluate_parallel(cfg, num_parallel=args.num_parallel)
    else:
        evaluate(cfg)


if __name__ == "__main__":
    main()
