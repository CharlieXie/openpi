"""Two-stage CALVIN evaluation for Waypoint VLA.

Pipeline:
  1. VLM predicts M=7 waypoints autoregressively.
  2. Action Expert fills actions between each waypoint pair via flow matching.
  3. Execute actions in CALVIN environment.
  4. Replan after exhausting all waypoints.

Uses CALVIN's chain-task evaluation protocol:
  - 1000 evaluation sequences, each with 5 subtasks in a row.
  - Report avg_seq_len and chain success rates (1/5 through 5/5).

Usage:
  python -m openpi.waypoint.eval_calvin --config configs/eval_waypoint_joint_calvin.yaml
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import argparse
import copy
import json
import logging
import math
import pathlib
import time
from collections import Counter
from pathlib import Path

import imageio
import numpy as np
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
)
from openpi.waypoint.robot_config import get_robot_config
from openpi.waypoint.tokenizer import WaypointTokenizer

# Reuse model loading utilities from eval_libero
from openpi.waypoint.eval_libero import (
    load_joint,
    load_vlm,
    load_ae,
    load_pg_tokenizer,
    predict_waypoints,
    predict_actions,
    center_crop_and_resize,
)

logger = logging.getLogger(__name__)

CALVIN_ROOT = os.environ.get("CALVIN_ROOT", "calvin")


# ---------------------------------------------------------------------------
# CALVIN observation helpers
# ---------------------------------------------------------------------------

def get_proprio_from_obs_calvin(obs):
    """Extract proprio from CALVIN environment observation.

    CALVIN env returns obs["robot_obs"] as a 15D vector:
      [0:3]  TCP position (x, y, z)
      [3:6]  TCP orientation (euler x, y, z)
      [6]    Gripper width (meters, 0~0.08)
      [7:14] 7 joint angles (radians)
      [14]   Gripper action (-1/+1, +1=open)

    We return the full 15D; RobotConfig.split_proprio() handles
    extracting continuous_proprio (dims 0:6) and binarizing gripper (dim 6).
    """
    robot_obs = obs["robot_obs"]
    return np.array(robot_obs, dtype=np.float32)


def get_calvin_images(obs, size=224, center_crop_scale=None):
    """Extract camera images from CALVIN observation.

    CALVIN env returns:
      obs["rgb_obs"]["rgb_static"]  — (200, 200, 3) uint8, static camera
      obs["rgb_obs"]["rgb_gripper"] — (84, 84, 3) uint8, wrist camera
    """
    from PIL import Image as PILImage
    images = {}

    static = obs["rgb_obs"]["rgb_static"]
    if static is not None:
        img = PILImage.fromarray(static)
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        if center_crop_scale is not None:
            arr = center_crop_and_resize(arr, center_crop_scale, size)
        images["base_0_rgb"] = arr

    gripper_img = obs["rgb_obs"]["rgb_gripper"]
    if gripper_img is not None:
        img = PILImage.fromarray(gripper_img)
        if img.size != (size, size):
            img = img.resize((size, size), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        if center_crop_scale is not None:
            arr = center_crop_and_resize(arr, center_crop_scale, size)
        images["left_wrist_0_rgb"] = arr

    return images


# ---------------------------------------------------------------------------
# CALVIN environment setup
# ---------------------------------------------------------------------------

def make_calvin_env(dataset_path, device):
    """Create CALVIN environment using CalvinEnvWrapperRaw."""
    val_folder = Path(dataset_path) / "validation"
    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["rel_actions"],
        "language": ["language"],
    }

    # Import CalvinEnvWrapperRaw — available from calvin_env package
    # Try the VLA-Adapter path first, then fall back to direct import
    try:
        from calvin_env_wrapper import CalvinEnvWrapperRaw
    except ImportError:
        from calvin_env.envs.play_table_env import get_env
        import gym

        class CalvinEnvWrapperRaw(gym.Wrapper):
            def __init__(self, abs_datasets_dir, obs_space, dev, **kwargs):
                env = get_env(abs_datasets_dir, show_gui=False, obs_space=obs_space, **kwargs)
                super().__init__(env)
                self.observation_space_keys = obs_space
                self.device = dev
                self.relative_actions = "rel_actions" in obs_space["actions"]

            def step(self, action):
                if self.relative_actions:
                    assert len(action) == 7
                o, r, d, i = self.env.step(action)
                return o, r, d, i

            def reset(self, robot_obs=None, scene_obs=None, **kwargs):
                if scene_obs is not None or robot_obs is not None:
                    return self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
                return self.env.reset()

            def get_info(self):
                return self.env.get_info()

            def get_obs(self):
                return self.env.get_obs()

    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _fmt_array(a, n=4):
    """Format first n elements of an array for logging."""
    flat = np.asarray(a).flatten()
    vals = " ".join(f"{v:.4f}" for v in flat[:n])
    if len(flat) > n:
        vals += " ..."
    return f"[{vals}]"


# ---------------------------------------------------------------------------
# Episode runner (single subtask)
# ---------------------------------------------------------------------------

def run_calvin_subtask(
    vlm, ae_model, wp_tokenizer, norm_helper, rc,
    env, task_desc, cfg, device, pg_tok,
    ep_len=360,
    task_oracle=None, start_info=None, subtask=None,
):
    """Run one CALVIN subtask with the two-stage waypoint pipeline.

    Returns:
        success: bool
        replay_images: list of (static, gripper) image pairs for video
    """
    model_proprio_dim = cfg.get("model_proprio_dim", 32)
    actual_action_dim = rc.actual_action_dim
    crop_scale = cfg.get("center_crop_scale") if cfg.get("center_crop", False) else None

    t = 0
    done = False
    replay_images = []
    replan_count = 0

    obs = env.get_obs()

    while t < ep_len and not done:
        # Collect replay frames
        static_frame = obs["rgb_obs"]["rgb_static"]
        gripper_frame = obs["rgb_obs"]["rgb_gripper"]
        if static_frame is not None:
            replay_images.append(copy.deepcopy(static_frame))

        images = get_calvin_images(obs, center_crop_scale=crop_scale)
        proprio_raw = get_proprio_from_obs_calvin(obs)
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

        # Build 7D start_wp: [continuous_norm(6D), gripper_binary(1D)]
        start_wp_7d = np.concatenate([continuous_norm, [float(gripper_binary)]])
        start_wp = pad_to_dim(start_wp_7d, model_proprio_dim)
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

            fresh_images = get_calvin_images(obs, center_crop_scale=crop_scale)
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

                # CALVIN gripper post-processing:
                # Model output is in [0,1] (0=close, 1=open).
                # CALVIN env expects [-1,+1] with +1=open, -1=close.
                # So: x*2-1 -> sign (NO negate, unlike LIBERO).
                gripper = action_raw[-1]
                gripper = gripper * 2.0 - 1.0
                gripper = np.sign(gripper)
                action_raw[-1] = gripper

                obs, reward, done, info = env.step(action_raw.tolist())
                t += 1
                steps_this_cycle += 1

                # Collect replay frames
                static_frame = obs["rgb_obs"]["rgb_static"]
                if static_frame is not None:
                    replay_images.append(copy.deepcopy(static_frame))

                # Check task completion via oracle
                if task_oracle is not None and start_info is not None and subtask is not None:
                    current_info = env.get_info()
                    current_task_info = task_oracle.get_task_info_for_set(
                        start_info, current_info, {subtask}
                    )
                    if len(current_task_info) > 0:
                        logger.info(f"  subtask '{subtask}' succeeded at step {t}")
                        return True, replay_images

                if done:
                    break

            start_wp = end_wp.copy()

        if steps_this_cycle == 0 and not done:
            logger.warning(f"  [replan {replan_count}] no actions executed, advancing with no-op")
            obs, reward, done, info = env.step(np.zeros(actual_action_dim).tolist())
            t += 1

    logger.info(f"  subtask done: steps={t}, replans={replan_count}, success=False")
    return False, replay_images


# ---------------------------------------------------------------------------
# Sequence evaluation (chain of 5 subtasks)
# ---------------------------------------------------------------------------

def evaluate_sequence(
    env, vlm, ae_model, wp_tokenizer, norm_helper, rc,
    task_oracle, initial_state, eval_sequence, val_annotations,
    cfg, device, pg_tok,
    eval_dir=None, sequence_i=0,
    ep_len=360,
):
    """Evaluate a chain of 5 subtasks. Returns number of consecutive successes."""
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    for subtask_i, subtask in enumerate(eval_sequence):
        lang_annotation = val_annotations[subtask][0]
        start_info = env.get_info()

        success, replay_images = run_calvin_subtask(
            vlm, ae_model, wp_tokenizer, norm_helper, rc,
            env, lang_annotation, cfg, device, pg_tok,
            ep_len=ep_len,
            task_oracle=task_oracle,
            start_info=start_info,
            subtask=subtask,
        )

        # Save video
        if eval_dir is not None and replay_images:
            suffix = "succ" if success else "fail"
            video_file = os.path.join(
                eval_dir, f"{sequence_i}-{subtask_i}-{subtask}-static-{suffix}.mp4"
            )
            try:
                imageio.mimwrite(
                    video_file,
                    [np.asarray(x) for x in replay_images],
                    fps=20,
                )
            except Exception as e:
                logger.warning(f"Failed to save video: {e}")

        if success:
            success_counter += 1
        else:
            return success_counter

    return success_counter


# ---------------------------------------------------------------------------
# CALVIN chain-task metrics
# ---------------------------------------------------------------------------

def count_success(results):
    """Compute chain success rates from list of success counts (0-5)."""
    n = len(results)
    if n == 0:
        return []
    success_rates = []
    for i in range(1, 6):
        success_rates.append(sum(r >= i for r in results) / n)
    return success_rates


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    t_total = time.time()

    # Load model
    use_joint = "joint_checkpoint" in cfg
    if use_joint:
        joint_model = load_joint(cfg, device)
        vlm = joint_model
        ae_model = joint_model
        logger.info(f"Joint model loaded (shared backbone): {time.time() - t_total:.1f}s")
    else:
        vlm = load_vlm(cfg, device)
        ae_model = load_ae(cfg, device)
        logger.info(f"Total model loading (separate VLM+AE): {time.time() - t_total:.1f}s")

    t0 = time.time()
    pg_tok = load_pg_tokenizer()
    logger.info(f"PaliGemma tokenizer loaded: {time.time() - t0:.1f}s")

    rc = get_robot_config("calvin")
    stats = load_dataset_statistics(cfg["dataset_statistics_path"])
    norm_helper = NormalizationHelper(stats, cfg.get("norm_type", "q99"))
    if rc.action_norm_mask is not None:
        norm_helper.action_norm_mask = rc.action_norm_mask

    wp_tokenizer = WaypointTokenizer(
        proprio_dim=rc.continuous_proprio_dim,
        num_waypoints=cfg.get("num_waypoints", 7),
        max_token_len=cfg.get("vlm_max_token_len", cfg.get("max_token_len", 256)),
        use_gripper_token=True,
    )

    if cfg.get("center_crop", False):
        logger.info(
            f"Center crop enabled: area_scale={cfg.get('center_crop_scale', 0.95)}, "
            f"side_ratio={math.sqrt(cfg.get('center_crop_scale', 0.95)):.4f}"
        )

    # Set up CALVIN environment
    calvin_dataset_path = cfg.get("calvin_dataset_path", os.path.join(CALVIN_ROOT, "dataset/task_ABC_D"))
    logger.info(f"CALVIN dataset path: {calvin_dataset_path}")

    t0 = time.time()
    env = make_calvin_env(calvin_dataset_path, device)
    logger.info(f"CALVIN env initialized: {time.time() - t0:.1f}s")

    # Load task oracle and annotations
    import hydra
    from omegaconf import OmegaConf

    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # Get evaluation sequences
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    from calvin_agent.evaluation.utils import count_success as calvin_count_success

    num_sequences = cfg.get("num_sequences", 1000)
    ep_len = cfg.get("ep_len", 360)
    eval_sequences = get_sequences(num_sequences)

    # Set up output directory
    eval_dir = cfg.get("video_out_path", "data/calvin/videos_wp")
    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Videos will be saved to: {eval_dir}")

    # Run evaluation
    results = []
    for seq_i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        t_seq = time.time()
        logger.info(f"Sequence {seq_i}/{num_sequences}: {' -> '.join(eval_sequence)}")

        result = evaluate_sequence(
            env, vlm, ae_model, wp_tokenizer, norm_helper, rc,
            task_oracle, initial_state, eval_sequence, val_annotations,
            cfg, device, pg_tok,
            eval_dir=eval_dir, sequence_i=seq_i,
            ep_len=ep_len,
        )
        results.append(result)
        seq_secs = time.time() - t_seq

        # Log running metrics
        success_rates = count_success(results)
        avg_seq_len = np.mean(results)
        sr_str = " | ".join(f"{i+1}/5: {sr:.1%}" for i, sr in enumerate(success_rates))
        logger.info(
            f"  -> {result}/5 subtasks ({seq_secs:.1f}s) | "
            f"avg_seq_len: {avg_seq_len:.2f} | {sr_str}"
        )

    # Final results
    avg_seq_len = np.mean(results)
    chain_sr = count_success(results)

    logger.info(f"\n{'='*60}")
    logger.info(f"CALVIN Evaluation Results ({num_sequences} sequences)")
    logger.info(f"Average successful sequence length: {avg_seq_len:.3f}")
    logger.info("Chain success rates:")
    for i, sr in enumerate(chain_sr):
        logger.info(f"  {i+1}/5: {sr:.1%}")
    logger.info(f"Total eval time: {time.time() - t_total:.1f}s")

    # Save results to JSON
    results_path = os.path.join(eval_dir, "eval_results.json")
    results_data = {
        "avg_seq_len": float(avg_seq_len),
        "chain_sr": {str(i+1): float(sr) for i, sr in enumerate(chain_sr)},
        "num_sequences": num_sequences,
        "per_sequence_results": results,
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return results_data


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
