#!/usr/bin/env python3
"""Grid evaluation orchestrator for Waypoint VLM + AE checkpoint search.

Anchor-based strategy:
  Phase 1: Fix AE anchor, sweep all VLMs  -> find top VLMs
  Phase 2: Top VLMs x all AEs             -> find best combos
  Phase 3: Validate top combos with more trials

Usage:
  python run_grid_eval.py --phase 1
  python run_grid_eval.py --phase 2 --top-vlms 3
  python run_grid_eval.py --phase 3 --top-n 5 --trials 20
  python run_grid_eval.py --report            # show leaderboard from existing logs
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Checkpoint inventory
# ---------------------------------------------------------------------------

CKPT_ROOT = "/workspace/openpi/checkpoints"

VLM_CHECKPOINTS = {
    "t10/800":  f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t10/800",
    "t10/1000": f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t10/1000",
    "t10/1200": f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t10/1200",
    "t10/1400": f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t10/1400",
    "t11/200":  f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t11/200",
    "t11/400":  f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t11/400",
    "t11/600":  f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t11/600",
    "t11/800":  f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t11/800",
    "t11/1000": f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t11/1000",
    "t11/1200": f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t11/1200",
    "t11/1400": f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t11/1400",
    "t11/1600": f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t11/1600",
    "t11/1800": f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t11/1800",
    "t11/2000": f"{CKPT_ROOT}/waypoint_vlm_libero_spatial_t11/2000",
}

AE_CHECKPOINTS = {
    "ae03/800":  f"{CKPT_ROOT}/waypoint_ae_libero_spatial_03/800",
    "ae03/900":  f"{CKPT_ROOT}/waypoint_ae_libero_spatial_03/900",
    "ae03/1000": f"{CKPT_ROOT}/waypoint_ae_libero_spatial_03/1000",
    "ae03/1100": f"{CKPT_ROOT}/waypoint_ae_libero_spatial_03/1100",
    "ae03/1300": f"{CKPT_ROOT}/waypoint_ae_libero_spatial_03/1300",
    "ae03/1800": f"{CKPT_ROOT}/waypoint_ae_libero_spatial_03/1800",
    "ae03/1900": f"{CKPT_ROOT}/waypoint_ae_libero_spatial_03/1900",
    "ae03/2000": f"{CKPT_ROOT}/waypoint_ae_libero_spatial_03/2000",
    "ae04/100":  f"{CKPT_ROOT}/waypoint_ae_libero_spatial_04/100",
    "ae04/200":  f"{CKPT_ROOT}/waypoint_ae_libero_spatial_04/200",
    "ae04/300":  f"{CKPT_ROOT}/waypoint_ae_libero_spatial_04/300",
    "ae04/400":  f"{CKPT_ROOT}/waypoint_ae_libero_spatial_04/400",
    "ae04/500":  f"{CKPT_ROOT}/waypoint_ae_libero_spatial_04/500",
}

ANCHOR_AE = "ae03/1300"
OPENPI_ROOT = Path("/workspace/openpi")
BASE_CONFIG = "configs/eval_grid_base.yaml"   # relative to OPENPI_ROOT
LOG_DIR = Path(__file__).parent / "logs"
RESULTS_DIR = LOG_DIR
VIDEO_DIR = Path(__file__).parent / "videos"
NUM_GPUS = 2
MAX_PER_GPU = 5  # conservative; scale to 6 if stable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tag_to_safe(vlm_tag: str, ae_tag: str) -> str:
    return f"vlm_{vlm_tag.replace('/', '_')}__ae_{ae_tag.replace('/', '_')}"


def parse_log_result(log_path: Path) -> dict | None:
    """Extract overall success rate and per-task results from an eval log."""
    if not log_path.exists():
        return None
    text = log_path.read_text()

    overall = re.findall(
        r"Overall success rate:\s*([\d.]+)%\s*\((\d+)/(\d+)\)", text
    )
    if not overall:
        return None

    total_success = 0
    total_episodes = 0
    for rate_str, succ, total in overall:
        total_success += int(succ)
        total_episodes += int(total)

    if total_episodes == 0:
        return None

    task_results = {}
    for match in re.finditer(
        r"\[(\d+)\]\s+(.+?):\s*([\d.]+)%\s*\((\d+)/(\d+)\)", text
    ):
        idx, name, rate, succ, total = match.groups()
        task_results[name.strip()] = {
            "idx": int(idx),
            "success_rate": float(rate) / 100,
            "successes": int(succ),
            "trials": int(total),
        }

    return {
        "success_rate": total_success / total_episodes,
        "successes": total_success,
        "episodes": total_episodes,
        "tasks": task_results,
    }


def build_eval_cmd(vlm_path: str, ae_path: str, video_dir: str,
                    num_trials: int | None = None) -> list[str]:
    cmd = [
        ".venv/bin/python", "-u",
        "src/openpi/waypoint/eval_libero.py",
        "--config", BASE_CONFIG,
        "--vlm-checkpoint", vlm_path,
        "--ae-checkpoint", ae_path,
        "--video-out-path", video_dir,
    ]
    if num_trials is not None:
        cmd.extend(["--num-trials", str(num_trials)])
    return cmd


def make_env(gpu_id: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_GL"] = "osmesa"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONFAULTHANDLER"] = "1"
    src = str(Path.cwd() / "src")
    libero = str(Path.cwd() / "third_party" / "libero")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}:{libero}:{existing}"
    return env


# ---------------------------------------------------------------------------
# Job scheduler
# ---------------------------------------------------------------------------

class Job:
    __slots__ = ("vlm_tag", "ae_tag", "vlm_path", "ae_path",
                 "log_path", "video_dir", "num_trials",
                 "gpu_id", "proc", "log_fh", "start_time")

    def __init__(self, vlm_tag, ae_tag, vlm_path, ae_path, num_trials=None, phase=2):
        self.vlm_tag = vlm_tag
        self.ae_tag = ae_tag
        self.vlm_path = vlm_path
        self.ae_path = ae_path
        self.num_trials = num_trials
        safe = tag_to_safe(vlm_tag, ae_tag)
        self.log_path = LOG_DIR / f"phase{phase}" / f"{safe}.log"
        self.video_dir = str(VIDEO_DIR / safe)
        self.gpu_id = -1
        self.proc = None
        self.log_fh = None
        self.start_time = 0.0


def run_jobs(jobs: list[Job], max_parallel: int | None = None):
    """Execute jobs with bounded parallelism across GPUs."""
    if max_parallel is None:
        max_parallel = NUM_GPUS * MAX_PER_GPU

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("phase1", "phase2", "phase3"):
        (LOG_DIR / sub).mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    queue = list(jobs)
    running: list[Job] = []
    completed: list[Job] = []
    gpu_counts = [0] * NUM_GPUS

    def _pick_gpu() -> int:
        for g in range(NUM_GPUS):
            if gpu_counts[g] < MAX_PER_GPU:
                return g
        return -1

    def _poll():
        still_running = []
        for j in running:
            ret = j.proc.poll()
            if ret is not None:
                elapsed = time.time() - j.start_time
                j.log_fh.close()
                gpu_counts[j.gpu_id] -= 1
                result = parse_log_result(j.log_path)
                rate_str = f"{result['success_rate']:.2%}" if result else "PARSE_ERR"
                status = "OK" if ret == 0 else f"EXIT={ret}"
                print(
                    f"  [DONE] {j.vlm_tag} + {j.ae_tag}  "
                    f"rate={rate_str}  {status}  {elapsed:.0f}s  "
                    f"[{len(completed)+1}/{len(jobs)}]"
                )
                completed.append(j)
            else:
                still_running.append(j)
        running.clear()
        running.extend(still_running)

    print(f"\n{'='*70}")
    print(f"Starting {len(jobs)} eval jobs  (max_parallel={max_parallel}, "
          f"gpus={NUM_GPUS}, max_per_gpu={MAX_PER_GPU})")
    print(f"{'='*70}\n")

    t0 = time.time()
    while queue or running:
        _poll()

        while queue and len(running) < max_parallel:
            gpu = _pick_gpu()
            if gpu < 0:
                break
            j = queue.pop(0)
            j.gpu_id = gpu
            gpu_counts[gpu] += 1

            Path(j.video_dir).mkdir(parents=True, exist_ok=True)
            cmd = build_eval_cmd(j.vlm_path, j.ae_path, j.video_dir,
                                 j.num_trials)
            j.log_fh = open(j.log_path, "w")
            j.proc = subprocess.Popen(
                cmd, stdout=j.log_fh, stderr=subprocess.STDOUT,
                env=make_env(gpu),
                cwd=str(OPENPI_ROOT),
            )
            j.start_time = time.time()
            running.append(j)
            print(
                f"  [START] GPU{gpu}  {j.vlm_tag} + {j.ae_tag}  "
                f"(running={len(running)}, queued={len(queue)})"
            )

        if running:
            time.sleep(5)

    elapsed_total = time.time() - t0
    print(f"\nAll {len(completed)} jobs finished in {elapsed_total:.0f}s "
          f"({elapsed_total/60:.1f} min)\n")
    return completed


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def collect_results(jobs: list[Job]) -> list[dict]:
    rows = []
    for j in jobs:
        r = parse_log_result(j.log_path)
        rows.append({
            "vlm": j.vlm_tag,
            "ae": j.ae_tag,
            "success_rate": r["success_rate"] if r else None,
            "successes": r["successes"] if r else None,
            "episodes": r["episodes"] if r else None,
            "tasks": r["tasks"] if r else None,
            "log": str(j.log_path),
        })
    rows.sort(key=lambda x: x["success_rate"] or 0, reverse=True)
    return rows


def print_leaderboard(rows: list[dict], title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"{'Rank':>4}  {'VLM':<16} {'AE':<16} {'Rate':>8}  {'Succ/Total':<12}")
    print(f"{'-'*4}  {'-'*16} {'-'*16} {'-'*8}  {'-'*12}")
    for i, r in enumerate(rows, 1):
        rate = f"{r['success_rate']:.2%}" if r["success_rate"] is not None else "N/A"
        frac = (f"{r['successes']}/{r['episodes']}"
                if r["successes"] is not None else "N/A")
        print(f"{i:>4}  {r['vlm']:<16} {r['ae']:<16} {rate:>8}  {frac:<12}")
    print()


def save_results(rows: list[dict], path: Path):
    serializable = []
    for r in rows:
        entry = {k: v for k, v in r.items() if k != "tasks"}
        entry["tasks"] = r.get("tasks") or {}
        serializable.append(entry)
    path.write_text(json.dumps(serializable, indent=2))
    print(f"Results saved to {path}")


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase1(anchor_ae: str = ANCHOR_AE):
    """Sweep all VLMs with a fixed AE anchor."""
    ae_path = AE_CHECKPOINTS[anchor_ae]
    jobs = []
    for vlm_tag, vlm_path in sorted(VLM_CHECKPOINTS.items()):
        j = Job(vlm_tag, anchor_ae, vlm_path, ae_path, phase=1)
        if j.log_path.exists():
            r = parse_log_result(j.log_path)
            if r and r["episodes"] > 0:
                print(f"  [SKIP] {vlm_tag} + {anchor_ae} — already done "
                      f"({r['success_rate']:.2%})")
                continue
        jobs.append(j)

    if not jobs:
        print("All Phase 1 jobs already completed.")
    else:
        run_jobs(jobs)

    all_jobs = []
    for vlm_tag, vlm_path in sorted(VLM_CHECKPOINTS.items()):
        all_jobs.append(Job(vlm_tag, anchor_ae, vlm_path, ae_path, phase=1))
    rows = collect_results(all_jobs)
    print_leaderboard(rows, f"Phase 1: VLM Sweep (AE anchor: {anchor_ae})")
    save_results(rows, RESULTS_DIR / "results_phase1.json")
    return rows


def phase2(top_vlms: int = 3):
    """Sweep all AEs for the top VLMs from Phase 1."""
    p1_path = RESULTS_DIR / "results_phase1.json"
    if not p1_path.exists():
        print("ERROR: Phase 1 results not found. Run --phase 1 first.")
        sys.exit(1)

    p1 = load_results(p1_path)
    selected_vlms = [r["vlm"] for r in p1[:top_vlms] if r["success_rate"] is not None]
    print(f"Top {top_vlms} VLMs from Phase 1: {selected_vlms}\n")

    jobs = []
    for vlm_tag in selected_vlms:
        vlm_path = VLM_CHECKPOINTS[vlm_tag]
        for ae_tag, ae_path in sorted(AE_CHECKPOINTS.items()):
            j = Job(vlm_tag, ae_tag, vlm_path, ae_path)
            if j.log_path.exists():
                r = parse_log_result(j.log_path)
                if r and r["episodes"] > 0:
                    print(f"  [SKIP] {vlm_tag} + {ae_tag} — already done "
                          f"({r['success_rate']:.2%})")
                    continue
            jobs.append(j)

    if not jobs:
        print("All Phase 2 jobs already completed.")
    else:
        run_jobs(jobs)

    all_jobs = []
    for vlm_tag in selected_vlms:
        vlm_path = VLM_CHECKPOINTS[vlm_tag]
        for ae_tag, ae_path in sorted(AE_CHECKPOINTS.items()):
            all_jobs.append(Job(vlm_tag, ae_tag, vlm_path, ae_path))
    rows = collect_results(all_jobs)
    print_leaderboard(rows, f"Phase 2: AE Sweep (top {top_vlms} VLMs)")
    save_results(rows, RESULTS_DIR / "results_phase2.json")
    return rows


def phase3(top_n: int = 5, trials: int = 20):
    """Validate top combinations with more trials."""
    p2_path = RESULTS_DIR / "results_phase2.json"
    if not p2_path.exists():
        print("ERROR: Phase 2 results not found. Run --phase 2 first.")
        sys.exit(1)

    p2 = load_results(p2_path)
    selected = p2[:top_n]
    print(f"Validating top {top_n} combos with {trials} trials each:\n")
    for r in selected:
        print(f"  {r['vlm']} + {r['ae']}  (screening: {r['success_rate']:.2%})")
    print()

    jobs = []
    for r in selected:
        vlm_path = VLM_CHECKPOINTS[r["vlm"]]
        ae_path = AE_CHECKPOINTS[r["ae"]]
        j = Job(r["vlm"], r["ae"], vlm_path, ae_path, num_trials=trials)
        safe = tag_to_safe(r["vlm"], r["ae"])
        j.log_path = LOG_DIR / "phase3" / f"{safe}_t{trials}.log"
        j.video_dir = str(VIDEO_DIR / f"{safe}_t{trials}")
        jobs.append(j)

    new_jobs = []
    for j in jobs:
        if j.log_path.exists():
            res = parse_log_result(j.log_path)
            if res and res["episodes"] > 0:
                print(f"  [SKIP] {j.vlm_tag} + {j.ae_tag} — already done "
                      f"({res['success_rate']:.2%})")
                continue
        new_jobs.append(j)

    if not new_jobs:
        print("All Phase 3 jobs already completed.")
    else:
        run_jobs(new_jobs)

    rows = collect_results(jobs)
    print_leaderboard(rows, f"Phase 3: Validation ({trials} trials)")
    save_results(rows, RESULTS_DIR / "results_phase3.json")
    return rows


def report():
    """Print leaderboard from all available results."""
    for phase_num in [1, 2, 3]:
        p = RESULTS_DIR / f"results_phase{phase_num}.json"
        if p.exists():
            rows = load_results(p)
            print_leaderboard(rows, f"Phase {phase_num} Results")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global MAX_PER_GPU

    parser = argparse.ArgumentParser(description="Grid eval orchestrator")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Which phase to run")
    parser.add_argument("--anchor-ae", type=str, default=ANCHOR_AE,
                        help=f"AE anchor for Phase 1 (default: {ANCHOR_AE})")
    parser.add_argument("--top-vlms", type=int, default=3,
                        help="How many top VLMs to take into Phase 2")
    parser.add_argument("--top-n", type=int, default=5,
                        help="How many top combos to validate in Phase 3")
    parser.add_argument("--trials", type=int, default=20,
                        help="Trials per task for Phase 3 validation")
    parser.add_argument("--report", action="store_true",
                        help="Print leaderboard from existing results")
    parser.add_argument("--max-per-gpu", type=int, default=MAX_PER_GPU,
                        help=f"Max processes per GPU (default: {MAX_PER_GPU})")
    args = parser.parse_args()

    MAX_PER_GPU = args.max_per_gpu

    if args.report:
        report()
        return

    if args.phase is None:
        parser.print_help()
        return

    if args.phase == 1:
        phase1(anchor_ae=args.anchor_ae)
    elif args.phase == 2:
        phase2(top_vlms=args.top_vlms)
    elif args.phase == 3:
        phase3(top_n=args.top_n, trials=args.trials)


if __name__ == "__main__":
    main()
