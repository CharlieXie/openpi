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
import enum
import json
import logging
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
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
# Batch episode support
# ---------------------------------------------------------------------------

class EpState(enum.Enum):
    NEED_VLM = "need_vlm"
    NEED_AE = "need_ae"
    EXECUTING = "executing"
    DONE = "done"


class BatchEpisode:
    """Per-episode state for batch evaluation."""

    def __init__(self, env, trial_idx, task_desc):
        self.env = env
        self.trial_idx = trial_idx
        self.task_desc = task_desc
        # Episode tracking
        self.state = EpState.NEED_VLM
        self.obs = None
        self.t = 0
        self.done = False
        self.reward = 0.0
        # VLM outputs
        self.waypoints = []
        self.wp_idx = 0
        self.start_wp = None
        self.current_end_wp = None
        self.current_duration = 0
        self.steps_this_cycle = 0
        # AE outputs
        self.action_buffer = None
        self.action_idx = 0
        self.num_execute = 0
        # Recording
        self.replay_images = []
        self.replan_count = 0
        self.success = False
        # Cached normalized proprio (set by VLM phase)
        self.state_padded = None


def _collect_replay_frame(obs):
    """Extract concatenated head+wrist replay frame from observation."""
    agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
    if agentview is None:
        return None
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
        return np.concatenate([head_frame, wrist_frame], axis=1)
    return head_frame


def _advance_to_valid_wp(ep):
    """Advance ep.wp_idx to the next waypoint with duration > 0.

    Returns True if a valid waypoint was found.
    Returns False on duration == 0 (end signal) or list exhausted.
    Skips waypoints with negative duration (same as original run_episode).
    """
    while ep.wp_idx < len(ep.waypoints):
        _, dur = ep.waypoints[ep.wp_idx]
        if dur == 0:
            return False
        if dur < 0:
            logger.warning(f"    [ep{ep.trial_idx}] wp[{ep.wp_idx}]: negative duration {dur}, skipping")
            ep.wp_idx += 1
            continue
        return True
    return False


@torch.no_grad()
def _batch_vlm_inference(vlm, episodes, wp_tokenizer, norm_helper, cfg, device, max_steps_total):
    """Batch VLM waypoint prediction for multiple episodes."""
    model_proprio_dim = cfg.get("model_proprio_dim", 32)
    max_dur = cfg.get("horizon_steps", 32)
    actual_action_dim = get_robot_config("libero").actual_action_dim
    B = len(episodes)

    # Collect replay frame before VLM (matches original top-of-loop behavior)
    for ep in episodes:
        frame = _collect_replay_frame(ep.obs)
        if frame is not None:
            ep.replay_images.append(frame)

    # Build batched images
    all_images = [get_libero_images(ep.env, ep.obs) for ep in episodes]

    img_tensors = {}
    img_masks = {}
    for model_key in ["base_0_rgb", "left_wrist_0_rgb"]:
        tensors = []
        masks = []
        for images in all_images:
            if model_key in images:
                t = torch.from_numpy(images[model_key]).float() / 127.5 - 1.0
                tensors.append(t)
                masks.append(True)
            else:
                tensors.append(torch.zeros(224, 224, 3))
                masks.append(False)
        img_tensors[model_key] = torch.stack(tensors).to(device)
        img_masks[model_key] = torch.tensor(masks, dtype=torch.bool, device=device)

    # Build per-episode prompt tokens (different proprio → different tokens)
    prompt_tokens_list = []
    for ep in episodes:
        proprio_raw = get_proprio_from_obs(ep.obs)
        proprio_norm = norm_helper.normalize_proprio(proprio_raw)
        state_padded = pad_to_dim(proprio_norm, model_proprio_dim)
        ep.state_padded = state_padded

        proprio_dim = wp_tokenizer.proprio_dim
        state_for_prompt = state_padded[:proprio_dim]
        prompt_text = f"Task: {ep.task_desc.strip().replace('_', ' ').lower()}, State: "
        discretized = np.digitize(np.clip(state_for_prompt, -1, 1), np.linspace(-1, 1, 257)[:-1]) - 1
        prompt_text += " ".join(map(str, discretized.astype(int))) + ";\n"

        tokens = wp_tokenizer._pg_tokenizer.encode(prompt_text, add_bos=True)
        prompt_tokens_list.append(tokens)

    # Pad tokens to max length
    max_tok_len = max(len(t) for t in prompt_tokens_list)
    padded_tokens = []
    padded_masks = []
    for tokens in prompt_tokens_list:
        pad_len = max_tok_len - len(tokens)
        padded_tokens.append(tokens + [0] * pad_len)
        padded_masks.append([True] * len(tokens) + [False] * pad_len)

    prompt_tokens = torch.tensor(padded_tokens, dtype=torch.long, device=device)
    prompt_mask = torch.tensor(padded_masks, dtype=torch.bool, device=device)

    t_vlm = time.time()
    waypoints_batch = vlm.generate_waypoints(
        images=img_tensors,
        image_masks=img_masks,
        prompt_tokens=prompt_tokens,
        prompt_mask=prompt_mask,
        wp_tokenizer=wp_tokenizer,
    )
    vlm_ms = (time.time() - t_vlm) * 1000

    # Distribute results
    for ep, waypoints in zip(episodes, waypoints_batch):
        ep.replan_count += 1
        ep.waypoints = waypoints
        ep.wp_idx = 0
        ep.start_wp = ep.state_padded.copy()
        ep.steps_this_cycle = 0

        if not waypoints:
            logger.info(
                f"  [ep{ep.trial_idx} replan {ep.replan_count}] VLM empty ({vlm_ms:.0f}ms), stopping"
            )
            ep.state = EpState.DONE
            continue

        valid_wps = [(p, d) for p, d in waypoints if d > 0]
        durations = [d for _, d in valid_wps]
        logger.info(
            f"  [ep{ep.trial_idx} replan {ep.replan_count}] VLM: {len(waypoints)} wps, "
            f"{len(valid_wps)} valid, durations={durations}, vlm_time={vlm_ms / B:.0f}ms"
        )
        for wi, (pv, dur) in enumerate(waypoints):
            logger.info(f"    [ep{ep.trial_idx}] wp[{wi}]: proprio={_fmt_array(pv, 6)}, duration={dur}")

        if not _advance_to_valid_wp(ep):
            # No valid waypoints — do no-op
            logger.warning(f"  [ep{ep.trial_idx} replan {ep.replan_count}] no valid wps, advancing with no-op")
            ep.obs, ep.reward, ep.done, _ = ep.env.step(np.zeros(actual_action_dim))
            ep.t += 1
            if ep.done:
                ep.success = ep.reward > 0
                ep.state = EpState.DONE
            elif ep.t >= max_steps_total:
                ep.state = EpState.DONE
            else:
                ep.state = EpState.NEED_VLM
        else:
            ep.state = EpState.NEED_AE


@torch.no_grad()
def _batch_ae_inference(ae_model, episodes, cfg, device, pg_tok):
    """Batch AE action prediction for multiple episodes."""
    model_proprio_dim = cfg.get("model_proprio_dim", 32)
    max_dur = cfg.get("horizon_steps", 32)
    B = len(episodes)

    # Prepare per-episode data
    all_images = []
    start_wps = []
    end_wps = []
    durations = []

    for ep in episodes:
        wp_proprio, wp_duration = ep.waypoints[ep.wp_idx]
        duration = wp_duration
        if duration > max_dur:
            logger.warning(
                f"    [ep{ep.trial_idx}] wp[{ep.wp_idx}]: duration {duration} exceeds max {max_dur}, clamping"
            )
            duration = max_dur

        end_wp = pad_to_dim(wp_proprio, model_proprio_dim)
        ep.current_end_wp = end_wp
        ep.current_duration = duration

        fresh_images = get_libero_images(ep.env, ep.obs)
        all_images.append(fresh_images)
        start_wps.append(ep.start_wp)
        end_wps.append(end_wp)
        durations.append(float(duration))

    # Build batched image tensors (CHW for AE)
    img_tensors = {}
    img_masks = {}
    for model_key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        tensors = []
        masks = []
        for images in all_images:
            if model_key in images:
                t = torch.from_numpy(images[model_key]).float() / 127.5 - 1.0
                t = t.permute(2, 0, 1)  # HWC -> CHW
                tensors.append(t)
                masks.append(True)
            else:
                tensors.append(torch.zeros(3, 224, 224))
                masks.append(False)
        img_tensors[model_key] = torch.stack(tensors).to(device)
        img_masks[model_key] = torch.tensor(masks, dtype=torch.bool, device=device)

    # Build prompt tokens (same instruction for all episodes in same task)
    task_desc = episodes[0].task_desc
    text = f"Task: {task_desc.strip().replace('_', ' ').lower()}, \n"
    tids = pg_tok.encode(text, add_bos=True)
    ae_max_len = 64
    tids_len = len(tids)
    if tids_len < ae_max_len:
        tids_padded = tids + [0] * (ae_max_len - tids_len)
        tok_mask = [True] * tids_len + [False] * (ae_max_len - tids_len)
    else:
        tids_padded = tids[:ae_max_len]
        tok_mask = [True] * ae_max_len

    prompt_tokens = torch.tensor([tids_padded] * B, dtype=torch.long, device=device)
    prompt_mask = torch.tensor([tok_mask] * B, dtype=torch.bool, device=device)

    class _Obs:
        def __init__(self):
            self.images = img_tensors
            self.image_masks = img_masks
            self.state = torch.zeros(B, 32, device=device)
            self.tokenized_prompt = prompt_tokens
            self.tokenized_prompt_mask = prompt_mask
            self.token_ar_mask = None
            self.token_loss_mask = None

    obs = _Obs()

    start_t = torch.from_numpy(np.stack(start_wps)).float().to(device)
    end_t = torch.from_numpy(np.stack(end_wps)).float().to(device)
    dur_t = torch.tensor(durations, dtype=torch.float32, device=device)

    t_ae = time.time()
    actions_batch = ae_model.sample_actions(obs, start_t, end_t, dur_t)
    ae_ms = (time.time() - t_ae) * 1000

    actions_np = actions_batch.cpu().numpy()

    for i, ep in enumerate(episodes):
        ep.action_buffer = actions_np[i]
        ep.action_idx = 0
        ep.num_execute = min(int(ep.current_duration), ep.action_buffer.shape[0])
        ep.state = EpState.EXECUTING

        logger.info(
            f"    [ep{ep.trial_idx}] ae[{ep.wp_idx}]: shape={ep.action_buffer.shape}, "
            f"execute={ep.num_execute}, "
            f"range=[{ep.action_buffer.min():.3f}, {ep.action_buffer.max():.3f}], "
            f"ae_time={ae_ms / B:.0f}ms"
        )


def _drain_episode_actions(ep, norm_helper, actual_action_dim, max_steps_total):
    """Execute all remaining actions in ep's action buffer, then transition."""
    while ep.action_idx < ep.num_execute:
        action_raw = norm_helper.unnormalize_actions(
            ep.action_buffer[ep.action_idx, :actual_action_dim]
        )

        # Gripper processing (matches original run_episode)
        gripper = action_raw[-1]
        gripper = gripper * 2.0 - 1.0
        gripper = np.sign(gripper)
        gripper = -gripper
        action_raw[-1] = gripper

        ep.obs, ep.reward, ep.done, _ = ep.env.step(action_raw)
        ep.t += 1
        ep.action_idx += 1
        ep.steps_this_cycle += 1

        # Collect replay frame
        frame = _collect_replay_frame(ep.obs)
        if frame is not None:
            ep.replay_images.append(frame)

        if ep.done:
            ep.success = ep.reward > 0
            ep.state = EpState.DONE
            return

        if ep.t >= max_steps_total:
            ep.state = EpState.DONE
            return

    # Action buffer exhausted — advance to next waypoint
    ep.start_wp = ep.current_end_wp.copy()
    ep.wp_idx += 1

    if _advance_to_valid_wp(ep):
        ep.state = EpState.NEED_AE
    else:
        # No more valid waypoints in this replan cycle
        if ep.steps_this_cycle == 0:
            # No actions were executed at all — do no-op to prevent stuck
            logger.warning(f"  [ep{ep.trial_idx} replan {ep.replan_count}] no actions executed, advancing with no-op")
            ep.obs, ep.reward, ep.done, _ = ep.env.step(np.zeros(actual_action_dim))
            ep.t += 1
            if ep.done:
                ep.success = ep.reward > 0
                ep.state = EpState.DONE
                return

        if ep.t >= max_steps_total:
            ep.state = EpState.DONE
        else:
            ep.state = EpState.NEED_VLM


def run_batch_episodes(
    vlm, ae_model, wp_tokenizer, norm_helper,
    envs, initial_states, task_desc, cfg, device, pg_tok,
):
    """Run multiple episodes in batch with dynamic scheduling.

    Returns:
        List of (success, replay_images) tuples, one per episode.
    """
    rc = get_robot_config("libero")
    actual_action_dim = rc.actual_action_dim
    max_steps = MAX_STEPS_MAP.get(cfg.get("task_suite", "libero_object"), 280)
    num_steps_wait = cfg.get("num_steps_wait", 10)
    max_steps_total = max_steps + num_steps_wait

    # Initialize episodes
    episodes = []
    for i, (env, init_state) in enumerate(zip(envs, initial_states)):
        ep = BatchEpisode(env, i, task_desc)
        env.reset()
        ep.obs = env.set_init_state(init_state)
        episodes.append(ep)

    # Wait steps (per episode, parallel threads — CPU only)
    def _wait_episode(ep, n_wait):
        dummy = np.zeros(7)
        while ep.t < n_wait:
            ep.obs, _, ep.done, _ = ep.env.step(dummy)
            ep.t += 1
            if ep.done:
                ep.state = EpState.DONE
                ep.success = True
                break

    with ThreadPoolExecutor(max_workers=len(episodes)) as wait_pool:
        list(wait_pool.map(lambda ep: _wait_episode(ep, num_steps_wait), episodes))

    # Main scheduling loop
    with ThreadPoolExecutor(max_workers=len(episodes)) as executor:
        while any(ep.state != EpState.DONE for ep in episodes):
            # Phase 1: Batch VLM for all episodes needing replan
            vlm_eps = [ep for ep in episodes if ep.state == EpState.NEED_VLM]
            if vlm_eps:
                _batch_vlm_inference(vlm, vlm_eps, wp_tokenizer, norm_helper, cfg, device, max_steps_total)

            # Phase 2: Batch AE for all episodes needing action prediction
            ae_eps = [ep for ep in episodes if ep.state == EpState.NEED_AE]
            if ae_eps:
                _batch_ae_inference(ae_model, ae_eps, cfg, device, pg_tok)

            # Phase 3: Drain all EXECUTING episodes in parallel threads
            executing_eps = [ep for ep in episodes if ep.state == EpState.EXECUTING]
            if executing_eps:
                futures = {
                    executor.submit(
                        _drain_episode_actions, ep, norm_helper, actual_action_dim, max_steps_total
                    ): ep
                    for ep in executing_eps
                }
                for future in futures:
                    future.result()  # wait for all to complete

            # Safety: break if no episode was actionable (should not happen)
            if not vlm_eps and not ae_eps and not executing_eps:
                logger.warning("Batch loop: no actionable episodes, breaking")
                break

    for ep in episodes:
        logger.info(
            f"  ep{ep.trial_idx} done: steps={ep.t}, replans={ep.replan_count}, success={ep.success}"
        )

    return [(ep.success, ep.replay_images) for ep in episodes]


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
    batch_size = cfg.get("batch_size", 1)

    for task_idx in range(num_tasks):
        task = bm.get_task(task_idx)
        task_name = task.name
        task_desc = task.language
        initial_states = bm.get_task_init_states(task_idx)
        actual_num_trials = min(num_trials, len(initial_states))

        env_args = {
            "bddl_file_name": bm.get_task_bddl_file_path(task_idx),
            "camera_heights": 256,
            "camera_widths": 256,
        }

        from libero.libero.envs import OffScreenRenderEnv
        successes = 0

        if batch_size > 1:
            # --- Batch mode ---
            t0 = time.time()
            n_envs = min(batch_size, actual_num_trials)
            envs = [OffScreenRenderEnv(**env_args) for _ in range(n_envs)]
            for e in envs:
                e.seed(0)
            logger.info(f"Env init for task {task_idx} ({n_envs} envs): {time.time() - t0:.1f}s")

            for batch_start in range(0, actual_num_trials, batch_size):
                batch_end = min(batch_start + batch_size, actual_num_trials)
                cur_batch_size = batch_end - batch_start
                batch_envs = envs[:cur_batch_size]
                batch_init_states = [initial_states[t] for t in range(batch_start, batch_end)]

                t_batch = time.time()
                trial_ids = list(range(batch_start, batch_end))
                logger.info(f"Task {task_idx}/{num_tasks}: {task_name} trials {trial_ids}")

                batch_results = run_batch_episodes(
                    vlm, ae_model, wp_tokenizer, norm_helper,
                    batch_envs, batch_init_states, task_desc, cfg, device, pg_tok,
                )
                batch_secs = time.time() - t_batch

                for offset, (success, replay_images) in enumerate(batch_results):
                    trial = batch_start + offset
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
                        logger.info(f"  trial {trial} -> {suffix.upper()} (video: {video_file})")
                    else:
                        logger.info(f"  trial {trial} -> {suffix.upper()}")

                logger.info(f"  batch [{batch_start}:{batch_end}] done ({batch_secs:.1f}s)")

                overall_sr = total_success / max(total_episodes, 1)
                logger.info(
                    f"  [成功率] 当前任务: {successes}/{batch_end} = {successes / batch_end:.2%} | "
                    f"整体: {total_success}/{total_episodes} = {overall_sr:.2%}"
                )

            for e in envs:
                e.close()
        else:
            # --- Serial mode (original, unchanged) ---
            t0 = time.time()
            env = OffScreenRenderEnv(**env_args)
            env.seed(0)
            logger.info(f"Env init for task {task_idx}: {time.time() - t0:.1f}s")

            for trial in range(actual_num_trials):
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

            env.close()

        results[task_name] = {
            "success_rate": successes / actual_num_trials,
            "successes": successes,
            "trials": actual_num_trials,
        }

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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)


if __name__ == "__main__":
    main()
