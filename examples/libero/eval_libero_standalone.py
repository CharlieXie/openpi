"""Standalone LIBERO evaluation script.

Loads the pi0.5 model directly (JAX or PyTorch) and runs LIBERO simulation
in a single process, without the client-server websocket architecture.

Model backend detection is automatic:
  - If `model.safetensors` exists in the checkpoint dir -> PyTorch
  - Otherwise -> JAX (loads from `params/` subdirectory)
Norm stats are loaded automatically from `<checkpoint_dir>/assets/`.

Reproducibility:
  All random seeds (numpy, torch, JAX Policy rng, MuJoCo env) are derived
  from --seed.  Two runs with identical args produce identical action chunks.
  Use --dump-actions <path.npz> to save action data for diff-comparison.

Usage (JAX, official checkpoint):
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa uv run python examples/libero/eval_libero_standalone.py \
        --config-name pi05_libero --task-suite-name libero_10 --num-trials-per-task 3

Usage (PyTorch LoRA checkpoint):
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=osmesa uv run python examples/libero/eval_libero_standalone.py \
        --config-name pi05_libero_lora_pytorch \
        --checkpoint-dir checkpoints/pi05_libero_lora_pytorch/my_exp/5000 \
        --task-suite-name libero_10 --num-trials-per-task 3
"""

import collections
import dataclasses
import functools
import json
import logging
import math
import os
import pathlib
import random
import sys
import time

# Force XLA deterministic GPU operations BEFORE importing JAX.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# Configure logging BEFORE any library imports that might call basicConfig.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("eval_libero")
for _h in logging.getLogger().handlers:
    _h.flush = lambda _orig=_h.flush: (_orig(), sys.stdout.flush())

import imageio
import jax
import numpy as np
import torch
import tqdm
import tyro

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

MAX_STEPS_TABLE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

DEFAULT_CHECKPOINTS: dict[str, str] = {
    "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
    "pi0_libero": "gs://openpi-assets/checkpoints/pi0_base",
    "pi0_fast_libero": "gs://openpi-assets/checkpoints/pi0_fast_base",
}


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_libero"
    checkpoint_dir: str | None = None

    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 3
    num_steps_wait: int = 10
    replan_steps: int = 5
    resize_size: int = 224

    video_out_path: str = "data/libero/videos"
    seed: int = 7
    log_action_every: int = 20

    # Save all action chunks to an .npz file for reproducibility verification.
    dump_actions: str | None = None
    # Limit evaluation to the first N tasks (0 = all tasks).
    max_tasks: int = 0


# ── Helpers ─────────────────────────────────────────────────────────────


def _seed_everything(seed: int) -> None:
    """Set all RNG seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info("Seeds set: random=%d  np=%d  torch=%d  jax_policy_rng=key(%d)", seed, seed, seed, seed)


def _get_max_steps(suite_name: str) -> int:
    if suite_name not in MAX_STEPS_TABLE:
        raise ValueError(f"Unknown task suite: {suite_name}")
    return MAX_STEPS_TABLE[suite_name]


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


def _fmt_array(a: np.ndarray, precision: int = 4) -> str:
    return np.array2string(a, precision=precision, suppress_small=True, separator=", ")


def _action_stats(chunk: np.ndarray) -> str:
    if chunk.ndim == 1:
        return _fmt_array(chunk)
    return (
        f"shape={list(chunk.shape)}  "
        f"mean={_fmt_array(chunk.mean(axis=0))}  "
        f"min={_fmt_array(chunk.min(axis=0))}  "
        f"max={_fmt_array(chunk.max(axis=0))}"
    )


# ── Main ────────────────────────────────────────────────────────────────


def main(args: Args) -> None:
    _seed_everything(args.seed)

    # ── Load model ──────────────────────────────────────────────────────
    log.info("Config name : %s", args.config_name)
    config = _config.get_config(args.config_name)

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        ckpt_dir = DEFAULT_CHECKPOINTS.get(args.config_name)
        if ckpt_dir is None:
            raise ValueError(
                f"No default checkpoint for config '{args.config_name}'. "
                "Please specify --checkpoint-dir."
            )

    ckpt_local = _download.maybe_download(str(ckpt_dir))
    is_pytorch = os.path.exists(os.path.join(ckpt_local, "model.safetensors"))
    backend = "PyTorch" if is_pytorch else "JAX"
    log.info("Checkpoint   : %s", ckpt_dir)
    log.info("Local path   : %s", ckpt_local)
    log.info("Backend      : %s (auto-detected)", backend)

    t0 = time.time()
    policy = _policy_config.create_trained_policy(config, ckpt_dir)

    # Pin the Policy's JAX rng to a seed-derived key for reproducibility.
    # (JAX path) Policy._rng defaults to key(0); override with our seed.
    # (PyTorch path) noise comes from torch global RNG, already seeded above.
    if hasattr(policy, "_rng"):
        policy._rng = jax.random.key(args.seed)
    log.info("Model loaded in %.1fs  [%s]", time.time() - t0, backend)

    # ── LIBERO benchmark setup ──────────────────────────────────────────
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    if args.max_tasks > 0:
        num_tasks = min(num_tasks, args.max_tasks)
    max_steps = _get_max_steps(args.task_suite_name)

    log.info("=" * 72)
    log.info("Task suite     : %s", args.task_suite_name)
    log.info("Num tasks      : %d", num_tasks)
    log.info("Trials / task  : %d", args.num_trials_per_task)
    log.info("Max steps      : %d  (+ %d wait steps)", max_steps, args.num_steps_wait)
    log.info("Replan every   : %d steps", args.replan_steps)
    log.info("Video output   : %s", args.video_out_path)
    log.info("Seed           : %d", args.seed)
    log.info("Dump actions   : %s", args.dump_actions or "(disabled)")
    log.info("=" * 72)

    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluation loop ─────────────────────────────────────────────────
    total_episodes, total_successes = 0, 0
    per_task_results: dict[str, dict] = {}
    all_infer_times: list[float] = []
    # For reproducibility verification: (task_id, episode_idx, infer_idx) -> action_chunk
    all_action_chunks: dict[str, np.ndarray] = {}

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        log.info("-" * 72)
        log.info("TASK %d/%d: %s", task_id + 1, num_tasks, task_description)
        log.info("-" * 72)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_task), desc=f"  T{task_id}", leave=False
        ):
            log.info(
                "  [Task %d  Episode %d/%d]  %s",
                task_id, episode_idx + 1, args.num_trials_per_task, task_description,
            )

            env.reset()
            action_plan: collections.deque = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False
            ep_start = time.time()
            ep_infer_count = 0
            ep_infer_time_total = 0.0

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

                    eef_pos = obs["robot0_eef_pos"]
                    eef_quat = obs["robot0_eef_quat"]
                    gripper_qpos = obs["robot0_gripper_qpos"]
                    state_vec = np.concatenate((eef_pos, _quat2axisangle(eef_quat), gripper_qpos))

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state_vec,
                            "prompt": str(task_description),
                        }

                        infer_start = time.time()
                        result = policy.infer(element)
                        infer_elapsed = time.time() - infer_start
                        ep_infer_count += 1
                        ep_infer_time_total += infer_elapsed
                        all_infer_times.append(infer_elapsed)

                        action_chunk = result["actions"]
                        chunk_arr = np.asarray(action_chunk)

                        chunk_key = f"t{task_id}_ep{episode_idx}_infer{ep_infer_count}"
                        all_action_chunks[chunk_key] = chunk_arr.copy()

                        if t % args.log_action_every < args.replan_steps or ep_infer_count <= 2:
                            log.info(
                                "    step=%d  infer #%d  (%.0fms)  chunk %s",
                                t, ep_infer_count, infer_elapsed * 1000, _action_stats(chunk_arr),
                            )
                            log.info(
                                "    state: eef_pos=%s  eef_axangle=%s  gripper=%s",
                                _fmt_array(eef_pos), _fmt_array(_quat2axisangle(eef_quat)),
                                _fmt_array(gripper_qpos),
                            )
                            log.info("    first action: %s", _fmt_array(chunk_arr[0]))

                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    action_list = action.tolist() if hasattr(action, "tolist") else list(action)
                    obs, reward, done, info = env.step(action_list)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception:
                    logging.exception("Exception during step %d", t)
                    break

            task_episodes += 1
            total_episodes += 1
            elapsed = time.time() - ep_start
            suffix = "success" if done else "failure"

            avg_infer_ms = (ep_infer_time_total / ep_infer_count * 1000) if ep_infer_count else 0
            log.info(
                "  => %s  steps=%d  time=%.1fs  infer_calls=%d  avg_infer=%.0fms  "
                "episode %d/%d  running_sr=%.1f%% (%d/%d)",
                suffix.upper(), t, elapsed, ep_infer_count, avg_infer_ms,
                total_episodes, num_tasks * args.num_trials_per_task,
                total_successes / total_episodes * 100, total_successes, total_episodes,
            )

            task_tag = task_description.replace(" ", "_")[:80]
            video_path = video_dir / f"task{task_id:02d}_ep{episode_idx:02d}_{suffix}_{task_tag}.mp4"
            if replay_images:
                imageio.mimwrite(str(video_path), [np.asarray(x) for x in replay_images], fps=10)
                log.info("  Video saved: %s  (%d frames)", video_path.name, len(replay_images))
            else:
                log.warning("  No frames captured, skipping video.")

        task_sr = task_successes / task_episodes if task_episodes > 0 else 0.0
        per_task_results[task_description] = {
            "task_id": task_id,
            "success": task_successes,
            "total": task_episodes,
            "rate": task_sr,
        }
        log.info(
            "  Task %d done: %d/%d (%.1f%%)  |  overall so far: %d/%d (%.1f%%)",
            task_id, task_successes, task_episodes, task_sr * 100,
            total_successes, total_episodes, total_successes / total_episodes * 100,
        )

    # ── Dump action chunks for reproducibility verification ─────────────
    if args.dump_actions:
        dump_path = pathlib.Path(args.dump_actions)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dump_path, **all_action_chunks)
        log.info("Action chunks saved: %s  (%d chunks)", dump_path, len(all_action_chunks))

    # ── Summary ─────────────────────────────────────────────────────────
    overall_sr = total_successes / total_episodes * 100 if total_episodes > 0 else 0.0
    avg_infer_all = np.mean(all_infer_times) * 1000 if all_infer_times else 0

    log.info("")
    log.info("=" * 80)
    log.info(
        "RESULTS  --  %s  |  %d trials/task  |  backend=%s  |  seed=%d",
        args.task_suite_name, args.num_trials_per_task, backend, args.seed,
    )
    log.info("=" * 80)
    log.info("%-4s  %-55s  %s  %s", "ID", "Task", "Result", "Rate")
    log.info("-" * 80)
    for desc, r in per_task_results.items():
        log.info(
            "T%-3d  %-55s  %d / %d   %5.1f%%",
            r["task_id"], desc[:55], r["success"], r["total"], r["rate"] * 100,
        )
    log.info("-" * 80)
    log.info("OVERALL: %d / %d  (%.1f%%)", total_successes, total_episodes, overall_sr)
    log.info(
        "Inference: %d calls  avg=%.0fms  min=%.0fms  max=%.0fms",
        len(all_infer_times),
        avg_infer_all,
        np.min(all_infer_times) * 1000 if all_infer_times else 0,
        np.max(all_infer_times) * 1000 if all_infer_times else 0,
    )
    log.info("=" * 80)

    summary = {
        "config": args.config_name,
        "checkpoint": ckpt_dir,
        "backend": backend,
        "seed": args.seed,
        "task_suite": args.task_suite_name,
        "trials_per_task": args.num_trials_per_task,
        "overall_success_rate": round(overall_sr, 2),
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "avg_infer_ms": round(avg_infer_all, 1),
        "per_task": {
            desc: {"success": r["success"], "total": r["total"], "rate": round(r["rate"] * 100, 1)}
            for desc, r in per_task_results.items()
        },
    }
    summary_path = video_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
