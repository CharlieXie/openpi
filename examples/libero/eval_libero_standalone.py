"""Standalone LIBERO evaluation script.

Loads the pi0.5 model directly and runs LIBERO simulation in a single process,
without the client-server websocket architecture.

Usage:
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=osmesa uv run python examples/libero/eval_libero_standalone.py \
        --task-suite-name libero_10 --num-trials-per-task 3
"""

import collections
import dataclasses
import functools
import logging
import math
import pathlib
import time

import imageio
import numpy as np
import torch
import tqdm
import tyro

# LIBERO init states were saved with older torch; allow numpy globals for unpickling.
_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import download as _download

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Model config name registered in openpi
    config_name: str = "pi05_libero"
    # Checkpoint path (GCS URI or local path). None = use default for config.
    checkpoint_dir: str | None = None

    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 3
    num_steps_wait: int = 10
    replan_steps: int = 5
    resize_size: int = 224

    video_out_path: str = "data/libero/videos"
    seed: int = 7


DEFAULT_CHECKPOINTS: dict[str, str] = {
    "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
    "pi0_libero": "gs://openpi-assets/checkpoints/pi0_base",
    "pi0_fast_libero": "gs://openpi-assets/checkpoints/pi0_fast_base",
}


def _get_max_steps(suite_name: str) -> int:
    table = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if suite_name not in table:
        raise ValueError(f"Unknown task suite: {suite_name}")
    return table[suite_name]


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray(img)
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    pil = pil.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(pil, (paste_x, paste_y))
    return np.array(canvas)


def main(args: Args) -> None:
    np.random.seed(args.seed)

    # --- Load model ---
    logging.info("Loading model config: %s", args.config_name)
    config = _config.get_config(args.config_name)

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        ckpt_dir = DEFAULT_CHECKPOINTS.get(args.config_name)
        if ckpt_dir is None:
            raise ValueError(
                f"No default checkpoint for config '{args.config_name}'. "
                "Please specify --checkpoint-dir."
            )

    logging.info("Checkpoint: %s", ckpt_dir)
    policy = _policy_config.create_trained_policy(config, ckpt_dir)
    logging.info("Model loaded successfully.")

    # --- Set up LIBERO benchmark ---
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = _get_max_steps(args.task_suite_name)
    logging.info(
        "Task suite: %s  |  tasks: %d  |  trials/task: %d  |  max_steps: %d",
        args.task_suite_name, num_tasks, args.num_trials_per_task, max_steps,
    )

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # --- Evaluation loop ---
    total_episodes, total_successes = 0, 0
    per_task_results: dict[str, dict] = {}

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_task), desc=f"  T{task_id}", leave=False
        ):
            logging.info("Task: %s  |  Episode %d/%d", task_description, episode_idx + 1, args.num_trials_per_task)

            env.reset()
            action_plan: collections.deque = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False
            ep_start = time.time()

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = _resize_with_pad(img, args.resize_size, args.resize_size)
                    wrist_img = _resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    replay_images.append(img.copy())

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                            "prompt": str(task_description),
                        }
                        result = policy.infer(element)
                        action_chunk = result["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(
                        action.tolist() if hasattr(action, "tolist") else list(action)
                    )
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception:
                    logging.exception("Exception during episode")
                    break

            task_episodes += 1
            total_episodes += 1
            elapsed = time.time() - ep_start

            suffix = "success" if done else "failure"
            task_tag = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_tag}_ep{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info(
                "  => %s  (%.1fs)  |  running: %d/%d (%.1f%%)",
                suffix, elapsed, total_successes, total_episodes,
                total_successes / total_episodes * 100,
            )

        task_sr = task_successes / task_episodes if task_episodes > 0 else 0.0
        per_task_results[task_description] = {
            "success": task_successes,
            "total": task_episodes,
            "rate": task_sr,
        }
        logging.info(
            "Task '%s' done: %d/%d (%.1f%%)",
            task_description, task_successes, task_episodes, task_sr * 100,
        )

    # --- Summary ---
    logging.info("\n" + "=" * 70)
    logging.info("RESULTS  --  %s  (%d trials/task)", args.task_suite_name, args.num_trials_per_task)
    logging.info("=" * 70)
    for desc, r in per_task_results.items():
        logging.info("  %-60s  %d/%d  (%.1f%%)", desc, r["success"], r["total"], r["rate"] * 100)
    overall = total_successes / total_episodes * 100 if total_episodes > 0 else 0.0
    logging.info("-" * 70)
    logging.info("  OVERALL: %d/%d  (%.1f%%)", total_successes, total_episodes, overall)
    logging.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
