"""Two-stage LIBERO evaluation for Waypoint VLA.

Pipeline:
  1. VLM predicts M=7 waypoints autoregressively.
  2. Action Expert fills actions between each waypoint pair via flow matching.
  3. Execute actions in LIBERO environment.
  4. Replan after exhausting all waypoints.

Usage:
  python -m openpi.waypoint.eval_libero --config configs/eval_waypoint_libero.yaml
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import yaml
from PIL import Image

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


def quat2axisangle(quat):
    """Convert quaternion to axis-angle."""
    from transforms3d.quaternions import quat2axangle
    axis, angle = quat2axangle(quat)
    return (axis * angle).astype(np.float32)


def get_proprio_from_obs(obs):
    """Extract 8d LIBERO proprio: [EEF_pos(3), EEF_axisangle(3), gripper_qpos(2)]."""
    eef_pos = obs["robot0_eef_pos"]
    eef_rot = quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]
    return np.concatenate([eef_pos, eef_rot, gripper]).astype(np.float32)


def get_libero_images(env, obs, size=224):
    """Extract camera images from LIBERO observation."""
    from PIL import Image as PILImage
    images = {}
    agentview = obs.get("agentview_image", obs.get("agentview_rgb"))
    if agentview is not None:
        img = PILImage.fromarray(agentview[::-1])
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        images["base_0_rgb"] = np.array(img, dtype=np.uint8)

    wrist = obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand_rgb"))
    if wrist is not None:
        img = PILImage.fromarray(wrist[::-1])
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        images["left_wrist_0_rgb"] = np.array(img, dtype=np.uint8)

    return images


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_vlm(cfg, device):
    """Load VLM waypoint predictor from checkpoint."""
    from openpi.waypoint.vlm_model import PI0WaypointVLM
    import openpi.models.pi0_config as pi0_config

    model_cfg = pi0_config.Pi0Config(
        pi05=False,
        max_token_len=cfg.get("max_token_len", 256),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        dtype="float32",
    )
    model = PI0WaypointVLM(model_cfg)

    ckpt_path = cfg["vlm_checkpoint"]
    logger.info(f"Loading VLM from {ckpt_path}")
    safetensors.torch.load_model(model, os.path.join(ckpt_path, "model.safetensors"))

    return model.to(device).eval()


def load_ae(cfg, device):
    """Load Action Expert from checkpoint."""
    from openpi.waypoint.ae_model import PI0WaypointAE
    import openpi.models.pi0_config as pi0_config

    model_cfg = pi0_config.Pi0Config(
        pi05=True,
        action_dim=cfg.get("model_action_dim", 32),
        action_horizon=cfg.get("horizon_steps", 32),
        max_token_len=cfg.get("max_token_len", 64),
        paligemma_variant=cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=cfg.get("action_expert_variant", "gemma_300m"),
        dtype="float32",
    )
    model = PI0WaypointAE(model_cfg)

    ckpt_path = cfg["ae_checkpoint"]
    logger.info(f"Loading Action Expert from {ckpt_path}")
    safetensors.torch.load_model(model, os.path.join(ckpt_path, "model.safetensors"))

    return model.to(device).eval()


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

    prompt_text = f"Task: {instruction.strip().replace('_', ' ').lower()}, State: "
    discretized = np.digitize(np.clip(state_norm, -1, 1), np.linspace(-1, 1, 257)[:-1]) - 1
    prompt_text += " ".join(map(str, discretized.astype(int))) + ";\n"
    prompt_text += "Action: "

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


@torch.no_grad()
def predict_actions(ae_model, images, instruction, start_wp, end_wp, duration, device):
    """Action Expert flow matching inference."""
    import sentencepiece
    import openpi.shared.download as download

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

    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        pg_tok = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    text = f"Task: {instruction.strip().replace('_', ' ').lower()}, \n"
    tids = pg_tok.encode(text, add_bos=True)
    max_len = 64
    if len(tids) < max_len:
        tids = tids + [0] * (max_len - len(tids))
        mask = [True] * len(tids[:max_len])
        mask[len(pg_tok.encode(text, add_bos=True)):] = [False] * (max_len - len(pg_tok.encode(text, add_bos=True)))
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

def run_episode(
    vlm, ae_model, wp_tokenizer, norm_helper,
    env, initial_state, task_desc, cfg, device,
):
    """Run one LIBERO episode with the two-stage waypoint pipeline."""
    env.reset()
    obs = env.set_init_state(initial_state)

    rc = get_robot_config("libero")
    model_proprio_dim = cfg.get("model_proprio_dim", 32)
    actual_action_dim = rc.actual_action_dim
    max_steps = MAX_STEPS_MAP.get(cfg.get("task_suite", "libero_object"), 280)
    num_steps_wait = cfg.get("num_steps_wait", 10)

    t = 0
    done = False
    replay = []

    dummy_action = np.zeros(7)
    while t < num_steps_wait:
        obs, _, done, _ = env.step(dummy_action)
        t += 1
        if done:
            return True, replay

    while t < max_steps + num_steps_wait and not done:
        images = get_libero_images(env, obs)
        proprio_raw = get_proprio_from_obs(obs)
        proprio_norm = norm_helper.normalize_proprio(proprio_raw)
        state_padded = pad_to_dim(proprio_norm, model_proprio_dim)

        waypoints = predict_waypoints(vlm, images, task_desc, wp_tokenizer, state_padded, device)

        if not waypoints:
            break

        start_wp = state_padded.copy()

        for proprio_values, duration in waypoints:
            if duration == 0:
                break
            if done:
                break

            end_wp = pad_to_dim(proprio_values, model_proprio_dim)

            fresh_images = get_libero_images(env, obs)
            actions_norm = predict_actions(ae_model, fresh_images, task_desc, start_wp, end_wp, duration, device)

            num_execute = min(int(duration), actions_norm.shape[0])
            for step_i in range(num_execute):
                action_raw = norm_helper.unnormalize_actions(actions_norm[step_i, :actual_action_dim])

                gripper = action_raw[-1]
                gripper = gripper * 2.0 - 1.0
                gripper = np.sign(gripper)
                gripper = -gripper
                action_raw[-1] = gripper

                obs, reward, done, info = env.step(action_raw)
                t += 1
                replay.append({"obs": obs, "action": action_raw})

                if done:
                    break

            start_wp = end_wp.copy()

    return done and reward > 0, replay


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vlm = load_vlm(cfg, device)
    ae_model = load_ae(cfg, device)

    rc = get_robot_config("libero")
    stats = load_dataset_statistics(cfg["dataset_statistics_path"])
    norm_helper = NormalizationHelper(stats, cfg.get("norm_type", "q99"))

    wp_tokenizer = WaypointTokenizer(
        proprio_dim=rc.actual_proprio_dim,
        num_waypoints=cfg.get("num_waypoints", 7),
        max_token_len=cfg.get("max_token_len", 256),
    )

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
            "bddl_file_name": os.path.join(
                bm.get_task_bddl_file_path(task_idx),
            ),
            "camera_heights": 256,
            "camera_widths": 256,
        }

        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)

        successes = 0
        for trial in range(min(num_trials, len(initial_states))):
            logger.info(f"Task {task_idx}/{num_tasks}: {task_name} trial {trial}")
            success, replay = run_episode(
                vlm, ae_model, wp_tokenizer, norm_helper,
                env, initial_states[trial], task_desc, cfg, device,
            )
            if success:
                successes += 1
                total_success += 1
            total_episodes += 1
            logger.info(f"  -> {'SUCCESS' if success else 'FAIL'}")

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
