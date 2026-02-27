"""Two-stage LIBERO evaluation for Waypoint VLA — parallel state-machine variant.

Pipeline (per episode):
  1. VLM predicts M=7 waypoints autoregressively.
  2. Action Expert fills actions between each waypoint pair via flow matching.
  3. Execute actions in LIBERO environment.
  4. Replan after exhausting all waypoints.

Parallelism:
  Multiple tasks run concurrently.  AE inference requests from different
  episodes are dynamically batched into a single GPU forward pass, giving
  near-linear speed-up on the action-expert stage.

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
from enum import Enum, auto
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
    """Action Expert flow matching inference (single episode, kept for debugging)."""
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
# Parallel evaluation — helpers
# ---------------------------------------------------------------------------

def _fmt_array(a, n=4):
    """Format first n elements of an array for logging."""
    flat = np.asarray(a).flatten()
    vals = " ".join(f"{v:.4f}" for v in flat[:n])
    if len(flat) > n:
        vals += " ..."
    return f"[{vals}]"


def _record_replay_frame(obs, replay_images):
    """Capture agentview + optional wrist frame for video replay."""
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


@torch.no_grad()
def _batched_predict_actions(ae_model, requests, device, pg_tok):
    """Batched Action Expert inference across multiple concurrent episodes.

    Args:
        requests: list of dicts, each containing:
            images (dict[str, ndarray]), instruction (str),
            start_wp (ndarray), end_wp (ndarray), duration (float)
    Returns:
        list of (horizon, action_dim) numpy arrays, one per request.
    """
    N = len(requests)
    if N == 0:
        return []

    max_len = 64

    img_tensors = {}
    img_masks = {}
    for model_key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        imgs = []
        masks = []
        for req in requests:
            if model_key in req["images"]:
                arr = req["images"][model_key]
                t = torch.from_numpy(arr).float() / 127.5 - 1.0
                t = t.permute(2, 0, 1)
                imgs.append(t)
                masks.append(True)
            else:
                imgs.append(torch.zeros(3, 224, 224))
                masks.append(False)
        img_tensors[model_key] = torch.stack(imgs).to(device)
        img_masks[model_key] = torch.tensor(masks, dtype=torch.bool, device=device)

    all_tids = []
    all_tmask = []
    for req in requests:
        text = f"Task: {req['instruction'].strip().replace('_', ' ').lower()}, \n"
        tids = pg_tok.encode(text, add_bos=True)
        tids_len = len(tids)
        if tids_len < max_len:
            tids = tids + [0] * (max_len - tids_len)
            tmask = [True] * tids_len + [False] * (max_len - tids_len)
        else:
            tids = tids[:max_len]
            tmask = [True] * max_len
        all_tids.append(tids)
        all_tmask.append(tmask)

    prompt_tokens = torch.tensor(all_tids, dtype=torch.long, device=device)
    prompt_mask = torch.tensor(all_tmask, dtype=torch.bool, device=device)

    class _BatchObs:
        __slots__ = (
            "images", "image_masks", "state",
            "tokenized_prompt", "tokenized_prompt_mask",
            "token_ar_mask", "token_loss_mask",
        )
        def __init__(self):
            self.images = img_tensors
            self.image_masks = img_masks
            self.state = torch.zeros(N, 32, device=device)
            self.tokenized_prompt = prompt_tokens
            self.tokenized_prompt_mask = prompt_mask
            self.token_ar_mask = None
            self.token_loss_mask = None

    start_t = torch.stack([torch.from_numpy(r["start_wp"]).float() for r in requests]).to(device)
    end_t = torch.stack([torch.from_numpy(r["end_wp"]).float() for r in requests]).to(device)
    dur_t = torch.tensor([float(r["duration"]) for r in requests], dtype=torch.float32, device=device)

    actions = ae_model.sample_actions(_BatchObs(), start_t, end_t, dur_t)
    return [actions[i].cpu().numpy() for i in range(N)]


# ---------------------------------------------------------------------------
# Episode state machine
# ---------------------------------------------------------------------------

class _EpState(Enum):
    WAIT = auto()
    NEED_VLM = auto()
    NEED_AE = auto()
    EXECUTING = auto()
    DONE = auto()


class _Slot:
    """One evaluation slot: a task environment running trials sequentially."""

    def __init__(self, task_idx, task_name, task_desc, env,
                 initial_states, num_trials, cfg):
        self.task_idx = task_idx
        self.task_name = task_name
        self.task_desc = task_desc
        self.env = env
        self.initial_states = initial_states
        self.num_trials = num_trials

        self.max_steps = MAX_STEPS_MAP.get(cfg.get("task_suite", "libero_object"), 280)
        self.num_steps_wait = cfg.get("num_steps_wait", 10)
        self.max_dur = cfg.get("horizon_steps", 32)
        self.model_proprio_dim = cfg.get("model_proprio_dim", 32)
        rc = get_robot_config("libero")
        self.actual_action_dim = rc.actual_action_dim

        self.current_trial = -1
        self.successes = 0
        self.trial_results = []

        self.state = _EpState.DONE
        self.obs = None
        self.t = 0
        self.ep_done = False
        self.reward = 0.0
        self.replay_images = []
        self.replan_count = 0
        self.ep_start_time = 0.0

        self.waypoints = None
        self.wp_idx = 0
        self.start_wp = None
        self.end_wp = None
        self.steps_this_cycle = 0

        self.actions_norm = None
        self.action_step = 0
        self.num_execute = 0

    @property
    def all_done(self):
        return self.current_trial >= self.num_trials

    def init_trial(self):
        """Reset env and start the next trial."""
        self.current_trial += 1
        self.state = _EpState.WAIT
        self.obs = None
        self.t = 0
        self.ep_done = False
        self.reward = 0.0
        self.replay_images = []
        self.replan_count = 0
        self.ep_start_time = time.time()
        self.waypoints = None
        self.wp_idx = 0
        self.start_wp = None
        self.end_wp = None
        self.steps_this_cycle = 0
        self.actions_norm = None
        self.action_step = 0
        self.num_execute = 0

        self.env.reset()
        self.obs = self.env.set_init_state(self.initial_states[self.current_trial])
        logger.info(
            f"Task {self.task_idx}: {self.task_name} "
            f"trial {self.current_trial}/{self.num_trials}"
        )

    def _advance_to_next_valid_wp(self):
        """Skip to next waypoint with positive duration, or transition out."""
        while self.wp_idx < len(self.waypoints):
            _, dur = self.waypoints[self.wp_idx]
            if dur == 0:
                break
            if dur < 0:
                logger.warning(
                    f"    [T{self.task_idx} t{self.current_trial} "
                    f"wp{self.wp_idx}]: negative duration {dur}, skipping"
                )
                self.wp_idx += 1
                continue
            self.state = _EpState.NEED_AE
            return

        if self.steps_this_cycle == 0 and not self.ep_done:
            logger.warning(
                f"  [T{self.task_idx} t{self.current_trial} "
                f"replan {self.replan_count}] "
                f"no actions executed, advancing with no-op"
            )
            self.obs, self.reward, done, _ = self.env.step(
                np.zeros(self.actual_action_dim)
            )
            self.t += 1
            if done:
                self.ep_done = True

        max_t = self.max_steps + self.num_steps_wait
        if self.ep_done or self.t >= max_t:
            self.state = _EpState.DONE
        else:
            self.state = _EpState.NEED_VLM

    def finish_trial(self, video_out_path):
        """Record trial result and optionally save video."""
        success = self.ep_done and self.reward > 0
        if success:
            self.successes += 1
        ep_secs = time.time() - self.ep_start_time

        suffix = "success" if success else "failure"
        task_segment = self.task_desc.replace(" ", "_")
        if self.replay_images:
            vf = video_out_path / f"rollout_{task_segment}_t{self.current_trial}_{suffix}.mp4"
            imageio.mimwrite(str(vf), [np.asarray(x) for x in self.replay_images], fps=10)
            logger.info(f"  -> {suffix.upper()} ({ep_secs:.1f}s, video: {vf})")
        else:
            logger.info(f"  -> {suffix.upper()} ({ep_secs:.1f}s)")

        self.trial_results.append(success)
        return success


# ---------------------------------------------------------------------------
# Main evaluation — parallel state-machine loop
# ---------------------------------------------------------------------------

def evaluate(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    t_total = time.time()
    vlm = load_vlm(cfg, device)
    ae_model = load_ae(cfg, device)
    logger.info(f"Total model loading: {time.time() - t_total:.1f}s")

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
    from libero.libero.envs import OffScreenRenderEnv

    task_suite_name = cfg.get("task_suite", "libero_object")
    bm = benchmark.get_benchmark_dict()[task_suite_name]()
    num_tasks = bm.n_tasks
    num_trials = cfg.get("num_trials_per_task", 3)
    num_parallel = cfg.get("num_parallel", num_tasks)

    logger.info(
        f"Parallel eval: {num_tasks} tasks × {num_trials} trials, "
        f"num_parallel={num_parallel}"
    )

    # -- Pre-compute task info -----------------------------------------------
    all_task_infos = []
    for task_idx in range(num_tasks):
        task = bm.get_task(task_idx)
        all_task_infos.append({
            "task_idx": task_idx,
            "task_name": task.name,
            "task_desc": task.language,
            "initial_states": bm.get_task_init_states(task_idx),
            "bddl_file": bm.get_task_bddl_file_path(task_idx),
        })

    # -- Create active slots -------------------------------------------------
    task_queue = list(range(num_tasks))
    slots: list[_Slot] = []
    results = {}
    total_success = 0
    total_episodes = 0

    def _make_slot(task_idx):
        info = all_task_infos[task_idx]
        n_trials = min(num_trials, len(info["initial_states"]))
        env_args = {
            "bddl_file_name": info["bddl_file"],
            "camera_heights": 256,
            "camera_widths": 256,
        }
        t0 = time.time()
        env = OffScreenRenderEnv(**env_args)
        env.seed(7)
        logger.info(f"Env init for task {task_idx}: {time.time() - t0:.1f}s")
        slot = _Slot(
            task_idx=info["task_idx"],
            task_name=info["task_name"],
            task_desc=info["task_desc"],
            env=env,
            initial_states=info["initial_states"],
            num_trials=n_trials,
            cfg=cfg,
        )
        slot.init_trial()
        return slot

    while len(slots) < num_parallel and task_queue:
        slots.append(_make_slot(task_queue.pop(0)))

    # -- State-machine tick loop ---------------------------------------------
    model_proprio_dim = cfg.get("model_proprio_dim", 32)
    tick = 0

    while slots:
        tick += 1

        # ---- 1. WAIT: step env with dummy action --------------------------
        for s in slots:
            if s.state != _EpState.WAIT:
                continue
            s.obs, _, done, _ = s.env.step(np.zeros(7))
            s.t += 1
            if done:
                s.ep_done = True
                s.reward = 1.0
                s.state = _EpState.DONE
            elif s.t >= s.num_steps_wait:
                s.state = _EpState.NEED_VLM

        # ---- 2. EXECUTING: step one action ---------------------------------
        for s in slots:
            if s.state != _EpState.EXECUTING:
                continue

            a_raw = norm_helper.unnormalize_actions(
                s.actions_norm[s.action_step, :s.actual_action_dim]
            )
            gripper = a_raw[-1]
            gripper = gripper * 2.0 - 1.0
            gripper = np.sign(gripper)
            gripper = -gripper
            a_raw[-1] = gripper

            s.obs, s.reward, done, _ = s.env.step(a_raw)
            s.t += 1
            s.steps_this_cycle += 1
            s.action_step += 1
            _record_replay_frame(s.obs, s.replay_images)

            if done:
                s.ep_done = True
                s.state = _EpState.DONE
                continue

            max_t = s.max_steps + s.num_steps_wait
            if s.t >= max_t:
                s.state = _EpState.DONE
                continue

            if s.action_step >= s.num_execute:
                s.start_wp = s.end_wp.copy()
                s.wp_idx += 1
                s._advance_to_next_valid_wp()

        # ---- 3. NEED_VLM: predict waypoints (serial) ----------------------
        for s in slots:
            if s.state != _EpState.NEED_VLM:
                continue

            max_t = s.max_steps + s.num_steps_wait
            if s.t >= max_t:
                s.state = _EpState.DONE
                continue

            _record_replay_frame(s.obs, s.replay_images)

            images = get_libero_images(s.env, s.obs)
            proprio_raw = get_proprio_from_obs(s.obs)
            proprio_norm = norm_helper.normalize_proprio(proprio_raw)
            state_padded = pad_to_dim(proprio_norm, model_proprio_dim)

            t_vlm = time.time()
            waypoints = predict_waypoints(
                vlm, images, s.task_desc, wp_tokenizer, state_padded, device,
            )
            vlm_ms = (time.time() - t_vlm) * 1000

            s.replan_count += 1
            if not waypoints:
                logger.info(
                    f"  [T{s.task_idx} t{s.current_trial} replan {s.replan_count}] "
                    f"VLM returned empty waypoints ({vlm_ms:.0f}ms), stopping"
                )
                s.state = _EpState.DONE
                continue

            valid_wps = [(p, d) for p, d in waypoints if d > 0]
            durations = [d for _, d in valid_wps]
            logger.info(
                f"  [T{s.task_idx} t{s.current_trial} replan {s.replan_count}] "
                f"VLM: {len(waypoints)} wp, {len(valid_wps)} valid, "
                f"durations={durations}, vlm_time={vlm_ms:.0f}ms"
            )
            for wi, (pv, dur) in enumerate(waypoints):
                logger.info(f"    wp[{wi}]: proprio={_fmt_array(pv, 6)}, duration={dur}")

            s.start_wp = state_padded.copy()
            s.steps_this_cycle = 0
            s.waypoints = waypoints
            s.wp_idx = 0
            s._advance_to_next_valid_wp()

        # ---- 4. NEED_AE: batched action expert inference -------------------
        ae_slots = [s for s in slots if s.state == _EpState.NEED_AE]
        if ae_slots:
            requests = []
            for s in ae_slots:
                wp_proprio, wp_dur = s.waypoints[s.wp_idx]
                duration = int(wp_dur)
                if duration > s.max_dur:
                    logger.warning(
                        f"    [T{s.task_idx} t{s.current_trial} wp{s.wp_idx}]: "
                        f"duration {duration} exceeds max {s.max_dur}, clamping"
                    )
                    duration = s.max_dur
                s.end_wp = pad_to_dim(wp_proprio, model_proprio_dim)
                fresh_images = get_libero_images(s.env, s.obs)
                requests.append({
                    "images": fresh_images,
                    "instruction": s.task_desc,
                    "start_wp": s.start_wp,
                    "end_wp": s.end_wp,
                    "duration": duration,
                })

            t_ae = time.time()
            all_actions = _batched_predict_actions(ae_model, requests, device, pg_tok)
            ae_ms = (time.time() - t_ae) * 1000

            for s, actions_norm, req in zip(ae_slots, all_actions, requests):
                dur = int(req["duration"])
                s.actions_norm = actions_norm
                s.action_step = 0
                s.num_execute = min(dur, actions_norm.shape[0])
                s.state = _EpState.EXECUTING
                logger.info(
                    f"    [T{s.task_idx} t{s.current_trial} ae wp{s.wp_idx}]: "
                    f"shape={actions_norm.shape}, execute={s.num_execute}, "
                    f"range=[{actions_norm.min():.3f}, {actions_norm.max():.3f}], "
                    f"ae_batch={len(ae_slots)}, ae_time={ae_ms:.0f}ms"
                )

        # ---- 5. Handle DONE episodes ---------------------------------------
        finished = [s for s in slots if s.state == _EpState.DONE]
        for s in finished:
            logger.info(
                f"  [T{s.task_idx} t{s.current_trial}] episode done: "
                f"steps={s.t}, replans={s.replan_count}, "
                f"success={s.ep_done and s.reward > 0}"
            )

            success = s.finish_trial(video_out_path)
            total_episodes += 1
            if success:
                total_success += 1

            trials_done = s.current_trial + 1
            task_sr = s.successes / trials_done
            overall_sr = total_success / max(total_episodes, 1)
            logger.info(
                f"  [成功率] 任务{s.task_idx}: {s.successes}/{trials_done} = {task_sr:.2%} | "
                f"整体: {total_success}/{total_episodes} = {overall_sr:.2%}"
            )

            if not s.all_done:
                s.init_trial()
            else:
                results[s.task_name] = {
                    "success_rate": s.successes / s.num_trials,
                    "successes": s.successes,
                    "trials": s.num_trials,
                }
                s.env.close()
                logger.info(
                    f"  Task {s.task_idx} ({s.task_name}) complete: "
                    f"{s.successes}/{s.num_trials} = "
                    f"{s.successes / s.num_trials:.2%}"
                )

        slots = [s for s in slots if not s.all_done]

        while len(slots) < num_parallel and task_queue:
            slots.append(_make_slot(task_queue.pop(0)))

    # -- Summary -------------------------------------------------------------
    overall_rate = total_success / max(total_episodes, 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"Overall success rate: {overall_rate:.2%} ({total_success}/{total_episodes})")
    for name, r in results.items():
        logger.info(f"  {name}: {r['success_rate']:.2%} ({r['successes']}/{r['trials']})")
    logger.info(f"Total eval time: {time.time() - t_total:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
