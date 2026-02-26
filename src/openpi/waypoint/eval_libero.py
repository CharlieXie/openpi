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
import json
import logging
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


def get_libero_images(env, obs, size=224, save_dir=None, step=None):
    """Extract camera images from LIBERO observation."""
    from PIL import Image as PILImage
    images = {}
    agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
    if agentview is not None:
        img = PILImage.fromarray(agentview[::-1, ::-1])
        # img = PILImage.fromarray(agentview[::-1])

        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            suffix = f"_{step:05d}" if step is not None else ""
            img.save(os.path.join(save_dir, f"agentview{suffix}.png"))
        images["base_0_rgb"] = np.array(img, dtype=np.uint8)

    wrist = obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand_rgb"))
    if wrist is not None:
        # img = PILImage.fromarray(wrist[::-1])
        img = PILImage.fromarray(wrist[::-1, ::-1])
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        images["left_wrist_0_rgb"] = np.array(img, dtype=np.uint8)

    return images


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _fix_weight_tying(state_dict):
    """Duplicate tied weights so both keys exist in the state dict.

    HuggingFace PaliGemma ties embed_tokens and lm_head weights, so the
    checkpoint may only store one copy. We need both for load_state_dict.
    Works with arbitrary prefixes (e.g. paligemma_with_expert.paligemma.).
    """
    for key in list(state_dict.keys()):
        if key.endswith(".lm_head.weight"):
            prefix = key[: -len("lm_head.weight")]
            embed_key = prefix + "model.language_model.embed_tokens.weight"
            if embed_key not in state_dict:
                state_dict[embed_key] = state_dict[key]
        elif key.endswith(".model.language_model.embed_tokens.weight"):
            prefix = key[: -len("model.language_model.embed_tokens.weight")]
            lm_head_key = prefix + "lm_head.weight"
            if lm_head_key not in state_dict:
                state_dict[lm_head_key] = state_dict[key]


def _materialize_meta_tensors(module):
    """Replace remaining meta tensors after load_state_dict(assign=True).

    Handles two categories:
    1. Parameters not in checkpoint (should not happen if _fix_weight_tying
       was applied, but kept as safety net).
    2. Non-persistent buffers (position_ids, inv_freq, etc.) that are computed
       in __init__ but not saved to checkpoint — these need correct values,
       not garbage.
    """
    for mod in module.modules():
        for name, param in list(mod.named_parameters(recurse=False)):
            if param.is_meta:
                setattr(mod, name, torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype),
                    requires_grad=param.requires_grad,
                ))
        for name in list(mod._buffers.keys()):
            buf = mod._buffers[name]
            if buf is None or not buf.is_meta:
                continue
            if name == "position_ids":
                mod._buffers[name] = torch.arange(buf.shape[-1]).expand(buf.shape)
            elif name == "inv_freq":
                dim = buf.shape[0] * 2
                base = getattr(mod, "base", 10000.0)
                if hasattr(mod, "config") and hasattr(mod.config, "rope_theta"):
                    base = mod.config.rope_theta
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                mod._buffers[name] = inv_freq
            else:
                mod._buffers[name] = torch.zeros(buf.shape, dtype=buf.dtype)


def load_vlm(cfg, device):
    """Load VLM waypoint predictor from checkpoint.

    Uses torch.device('meta') to construct the model skeleton without allocating
    real memory (~0.5s vs ~100s with CPU allocation), then loads weights directly
    via load_state_dict(assign=True) which replaces meta tensors in-place.

    Handles checkpoints saved from PI0WaypointAE structure by remapping
    paligemma_with_expert.paligemma.* -> paligemma.* keys.
    """
    from openpi.waypoint.vlm_model import PI0WaypointVLM
    import openpi.models.pi0_config as pi0_config

    t0 = time.time()
    model_cfg = pi0_config.Pi0Config(
        pi05=False,
        max_token_len=cfg.get("max_token_len", 256),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        dtype="float32",
    )
    with torch.device("meta"):
        model = PI0WaypointVLM(model_cfg)
    logger.info(f"VLM model init: {time.time() - t0:.1f}s")

    ckpt_path = cfg["vlm_checkpoint"]
    ckpt_file = os.path.join(ckpt_path, "model.safetensors")
    logger.info(f"Loading VLM from {ckpt_path}")

    with open(ckpt_file, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header_keys = list(json.loads(f.read(header_size)).keys())

    has_pg_direct = any(k.startswith("paligemma.") for k in header_keys)
    has_pg_nested = any(k.startswith("paligemma_with_expert.paligemma.") for k in header_keys)

    t0 = time.time()
    state_dict = safetensors.torch.load_file(ckpt_file, device="cpu")
    if has_pg_direct:
        _fix_weight_tying(state_dict)
        model.load_state_dict(state_dict, assign=True, strict=False)
    elif has_pg_nested:
        PREFIX = "paligemma_with_expert.paligemma."
        remapped = {
            "paligemma." + k[len(PREFIX):]: v
            for k, v in state_dict.items()
            if k.startswith(PREFIX)
        }
        _fix_weight_tying(remapped)
        model.load_state_dict(remapped, assign=True, strict=False)
        logger.info(f"VLM: loaded {len(remapped)} params (remapped from AE checkpoint)")
    else:
        raise ValueError(f"Cannot find PaliGemma weights in checkpoint: {ckpt_file}")
    del state_dict
    _materialize_meta_tensors(model)
    logger.info(f"VLM weight load: {time.time() - t0:.1f}s")

    t0 = time.time()
    model = model.to(device).eval()
    logger.info(f"VLM to {device}: {time.time() - t0:.1f}s")
    return model


def load_ae(cfg, device):
    """Load Action Expert from checkpoint.

    Uses torch.device('meta') for fast skeleton construction, then loads
    weights via load_state_dict(assign=True).
    """
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
    with torch.device("meta"):
        model = PI0WaypointAE(model_cfg)
    logger.info(f"AE model init: {time.time() - t0:.1f}s")

    ckpt_path = cfg["ae_checkpoint"]
    logger.info(f"Loading Action Expert from {ckpt_path}")
    t0 = time.time()
    state_dict = safetensors.torch.load_file(
        os.path.join(ckpt_path, "model.safetensors"), device="cpu"
    )
    _fix_weight_tying(state_dict)
    model.load_state_dict(state_dict, assign=True, strict=False)
    del state_dict
    _materialize_meta_tensors(model)
    logger.info(f"AE weight load: {time.time() - t0:.1f}s")

    t0 = time.time()
    model = model.to(device).eval()
    logger.info(f"AE to {device}: {time.time() - t0:.1f}s")
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

    for model_key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
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
            replay_images.append(np.ascontiguousarray(agentview[::-1, ::-1]))

        images = get_libero_images(env, obs, save_dir="/workspace/openpi/data/libero/images", step=t)
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

            fresh_images = get_libero_images(env, obs, save_dir="/workspace/openpi/data/libero/images", step=t)
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
                    replay_images.append(np.ascontiguousarray(agentview[::-1, ::-1]))

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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)


if __name__ == "__main__":
    main()
