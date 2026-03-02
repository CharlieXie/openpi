#!/usr/bin/env python3
"""Early-stop monitor for grid eval.

Runs alongside the orchestrator. Every POLL_INTERVAL seconds:
  - Finds all running eval_libero.py processes
  - Maps each PID -> log file via /proc/<pid>/cmdline
  - Counts FAILURE lines in the log
  - If failures >= KILL_THRESHOLD, kills the process
    (orchestrator detects exit and starts next queued job automatically)

Kill condition:
  failures >= 3  =>  max possible = (30-3)/30 = 90.00%
  i.e. job can no longer strictly exceed 90%, so it's not worth continuing.
"""

import os
import re
import signal
import time
from pathlib import Path

KILL_THRESHOLD = 3      # kill when job can no longer exceed 90% (max = 90.00%)
POLL_INTERVAL  = 30     # seconds between polls
LOG_DIR        = Path(__file__).parent / "logs"


def get_running_evals():
    """Return list of (pid, vlm_tag, ae_tag, log_path) for all running evals."""
    results = []
    for pid_str in os.listdir("/proc"):
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            cmdline = open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()
        except (FileNotFoundError, PermissionError):
            continue
        if "eval_libero.py" not in cmdline or "early_stop" in cmdline:
            continue

        vlm = re.search(r"--vlm-checkpoint (\S+)", cmdline)
        ae  = re.search(r"--ae-checkpoint (\S+)",  cmdline)
        if not vlm or not ae:
            continue

        vlm_path = vlm.group(1)
        ae_path  = ae.group(1)
        # convert paths to tags: .../waypoint_vlm_libero_spatial_t10/800 -> t10/800
        vlm_tag = re.search(r"waypoint_vlm_libero_spatial_(\w+)/(\d+)$", vlm_path)
        ae_tag  = re.search(r"waypoint_ae_libero_spatial_(\w+)/(\d+)$",  ae_path)
        if not vlm_tag or not ae_tag:
            continue
        vlm_tag = f"{vlm_tag.group(1)}/{vlm_tag.group(2)}"
        ae_tag  = f"ae{ae_tag.group(1)}/{ae_tag.group(2)}"

        safe = f"vlm_{vlm_tag.replace('/','_')}__ae_{ae_tag.replace('/','_')}"
        # look in phase subdirs first, fall back to LOG_DIR root
        log_path = next(
            (p for p in [LOG_DIR / "phase2" / f"{safe}.log",
                         LOG_DIR / "phase1" / f"{safe}.log",
                         LOG_DIR / f"{safe}.log"] if p.exists()),
            LOG_DIR / "phase2" / f"{safe}.log"
        )

        results.append((pid, vlm_tag, ae_tag, log_path))
    return results


def count_failures(log_path: Path) -> int:
    try:
        return log_path.read_text().count("-> FAILURE")
    except FileNotFoundError:
        return 0


def count_episodes(log_path: Path) -> int:
    try:
        text = log_path.read_text()
        return text.count("-> FAILURE") + text.count("-> SUCCESS")
    except FileNotFoundError:
        return 0


def main():
    killed = set()
    print(f"[early_stop] started  threshold=failures>={KILL_THRESHOLD}  "
          f"(max_possible={(30-KILL_THRESHOLD)*100//30}%, cannot exceed 90%)  "
          f"poll={POLL_INTERVAL}s")
    print(f"[early_stop] watching {LOG_DIR}\n")

    while True:
        procs = get_running_evals()
        if not procs:
            print("[early_stop] no eval processes found, waiting...")
            time.sleep(POLL_INTERVAL)
            continue

        for pid, vlm_tag, ae_tag, log_path in procs:
            if pid in killed:
                continue
            fails = count_failures(log_path)
            eps   = count_episodes(log_path)
            if fails >= KILL_THRESHOLD:
                max_pct = (30 - fails) * 100 // 30
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed.add(pid)
                    print(f"[KILL] PID={pid}  {vlm_tag} + {ae_tag}  "
                          f"fails={fails}/{eps}  max_possible={max_pct}%  "
                          f"log={log_path.name}")
                except ProcessLookupError:
                    pass  # already gone

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
