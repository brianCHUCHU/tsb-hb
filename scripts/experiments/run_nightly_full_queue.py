#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


@dataclass
class Task:
    name: str
    cmd: list[str]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(master_log: Path, msg: str) -> None:
    line = f"[{now_str()}] {msg}"
    print(line)
    with master_log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_summary(summary_path: Path, out_root: Path, walk_step: int, seed: int, m5_sample_size: int, deepar_max_steps: int) -> None:
    text = f"""Nightly full experiment queue
- OUT_ROOT: {out_root}
- WALK_STEP: {walk_step}
- SEED: {seed}
- M5_SAMPLE_SIZE: {m5_sample_size}
- DEEPAR_MAX_STEPS: {deepar_max_steps}

Batches (2 at a time):
1) point_fixed_full + point_walk_full
2) prob_fixed_full + prob_walk_full
3) prob_fixed_full_deepar + m5_point_fixed_ablation
4) prob_fixed_hurdle_none + prob_fixed_hurdle_location_scale
"""
    summary_path.write_text(text, encoding="utf-8")


def launch_task(task: Task, root_dir: Path, log_dir: Path, master_log: Path) -> dict:
    task_log = log_dir / f"{task.name}.log"
    start_ts = now_str()
    t0 = time.time()
    fh = task_log.open("w", encoding="utf-8")
    fh.write(f"[{start_ts}] START {task.name}\n")
    fh.write("CMD: " + " ".join(shlex.quote(x) for x in task.cmd) + "\n")
    fh.flush()

    proc = subprocess.Popen(
        task.cmd,
        cwd=str(root_dir),
        stdout=fh,
        stderr=subprocess.STDOUT,
    )
    log_line(master_log, f"launched {task.name} pid={proc.pid}")
    return {
        "task": task,
        "proc": proc,
        "fh": fh,
        "start_ts": start_ts,
        "t0": t0,
        "task_log": task_log,
    }


def complete_task(state: dict, status_writer: csv.writer, master_log: Path) -> None:
    task: Task = state["task"]
    proc: subprocess.Popen = state["proc"]
    fh = state["fh"]
    start_ts: str = state["start_ts"]
    t0: float = state["t0"]
    task_log: Path = state["task_log"]

    rc = proc.wait()
    end_ts = now_str()
    elapsed = int(time.time() - t0)
    status = "OK" if rc == 0 else "FAIL"
    status_writer.writerow([task.name, status, start_ts, end_ts, str(elapsed), str(task_log)])
    fh.flush()
    fh.close()
    if rc == 0:
        log_line(master_log, f"finished {task.name}")
    else:
        log_line(master_log, f"finished {task.name} with rc={rc} (see {task_log})")


def run_batch(
    batch_name: str,
    tasks: Sequence[Task],
    root_dir: Path,
    log_dir: Path,
    master_log: Path,
    status_writer: csv.writer,
) -> None:
    log_line(master_log, f"=== {batch_name} START ===")
    running = [launch_task(t, root_dir, log_dir, master_log) for t in tasks]
    for state in running:
        complete_task(state, status_writer, master_log)
    log_line(master_log, f"=== {batch_name} END ===")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--walk-step", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m5-sample-size", type=int, default=5000)
    parser.add_argument("--deepar-max-steps", type=int, default=500)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    out_root = args.out.resolve()
    log_dir = out_root / "logs"
    online_dir = out_root / "online_retail"
    m5_dir = out_root / "m5"

    log_dir.mkdir(parents=True, exist_ok=True)
    online_dir.mkdir(parents=True, exist_ok=True)
    m5_dir.mkdir(parents=True, exist_ok=True)

    master_log = out_root / "run.log"
    status_tsv = out_root / "status.tsv"
    summary_txt = out_root / "summary.txt"
    write_summary(summary_txt, out_root, args.walk_step, args.seed, args.m5_sample_size, args.deepar_max_steps)

    with status_tsv.open("w", encoding="utf-8", newline="") as sf:
        writer = csv.writer(sf, delimiter="\t")
        writer.writerow(["name", "status", "start", "end", "elapsed_sec", "log"])

        batch1 = [
            Task(
                "point_fixed_full",
                [
                    "uv", "run", "python", "-m", "experiments.run_point",
                    "--dataset", "online_retail",
                    "--seed", str(args.seed),
                    "--hb-item-variance-mode", "conjugate",
                    "--hb-variance-prior-df", "20",
                    "--protocol", "fixed",
                    "--baseline-mode", "full",
                    "--out", str(online_dir / "point_fixed_full"),
                ],
            ),
            Task(
                "point_walk_full",
                [
                    "uv", "run", "python", "-m", "experiments.run_point",
                    "--dataset", "online_retail",
                    "--seed", str(args.seed),
                    "--hb-item-variance-mode", "conjugate",
                    "--hb-variance-prior-df", "20",
                    "--protocol", "walk_forward",
                    "--walk-step", str(args.walk_step),
                    "--baseline-mode", "full",
                    "--out", str(online_dir / "point_walk_full"),
                ],
            ),
        ]

        batch2 = [
            Task(
                "prob_fixed_full",
                [
                    "uv", "run", "python", "-m", "experiments.run_prob",
                    "--seed", str(args.seed),
                    "--protocol", "fixed",
                    "--baseline-mode", "full",
                    "--hb-item-variance-mode", "conjugate",
                    "--hb-variance-prior-df", "20",
                    "--hb-bootstrap-draws", "0",
                    "--hb-calibration-mode", "none",
                    "--out", str(online_dir / "prob_fixed_full"),
                ],
            ),
            Task(
                "prob_walk_full",
                [
                    "uv", "run", "python", "-m", "experiments.run_prob",
                    "--seed", str(args.seed),
                    "--protocol", "walk_forward",
                    "--walk-step", str(args.walk_step),
                    "--baseline-mode", "full",
                    "--hb-item-variance-mode", "conjugate",
                    "--hb-variance-prior-df", "20",
                    "--hb-bootstrap-draws", "0",
                    "--hb-calibration-mode", "none",
                    "--out", str(online_dir / "prob_walk_full"),
                ],
            ),
        ]

        batch3 = [
            Task(
                "prob_fixed_full_deepar",
                [
                    "uv", "run", "python", "-m", "experiments.run_prob",
                    "--seed", str(args.seed),
                    "--protocol", "fixed",
                    "--baseline-mode", "full",
                    "--hb-item-variance-mode", "conjugate",
                    "--hb-variance-prior-df", "20",
                    "--hb-bootstrap-draws", "0",
                    "--hb-calibration-mode", "none",
                    "--with-deepar",
                    "--horizon", "10",
                    "--input-size", "14",
                    "--max-steps", str(args.deepar_max_steps),
                    "--out", str(online_dir / "prob_fixed_full_deepar"),
                ],
            ),
            Task(
                "m5_point_fixed_ablation",
                [
                    "uv", "run", "python", "-m", "experiments.run_point",
                    "--dataset", "m5",
                    "--protocol", "fixed",
                    "--seed", str(args.seed),
                    "--hb-item-variance-mode", "conjugate",
                    "--hb-variance-prior-df", "20",
                    "--m5-sample-size", str(args.m5_sample_size),
                    "--m5-hierarchy-mode", "ablation",
                    "--out", str(m5_dir / "point_fixed_ablation"),
                ],
            ),
        ]

        batch4 = [
            Task(
                "prob_fixed_hurdle_none",
                [
                    "uv", "run", "python", "-m", "experiments.run_prob",
                    "--seed", str(args.seed),
                    "--protocol", "fixed",
                    "--baseline-mode", "hurdle_only",
                    "--hb-item-variance-mode", "conjugate",
                    "--hb-variance-prior-df", "20",
                    "--hb-bootstrap-draws", "0",
                    "--hb-calibration-mode", "none",
                    "--out", str(online_dir / "prob_fixed_hurdle_none"),
                ],
            ),
            Task(
                "prob_fixed_hurdle_location_scale",
                [
                    "uv", "run", "python", "-m", "experiments.run_prob",
                    "--seed", str(args.seed),
                    "--protocol", "fixed",
                    "--baseline-mode", "hurdle_only",
                    "--hb-item-variance-mode", "conjugate",
                    "--hb-variance-prior-df", "20",
                    "--hb-bootstrap-draws", "0",
                    "--hb-calibration-mode", "location_scale",
                    "--hb-calibration-ratio", "0.20",
                    "--hb-calibration-samples", "1000",
                    "--hb-calibration-lambda-min", "0.40",
                    "--hb-calibration-lambda-max", "1.10",
                    "--hb-calibration-lambda-steps", "15",
                    "--hb-calibration-coverage-weight", "1.00",
                    "--out", str(online_dir / "prob_fixed_hurdle_location_scale"),
                ],
            ),
        ]

        run_batch("batch1_point", batch1, root_dir, log_dir, master_log, writer)
        run_batch("batch2_prob", batch2, root_dir, log_dir, master_log, writer)
        run_batch("batch3_deepar_m5", batch3, root_dir, log_dir, master_log, writer)
        run_batch("batch4_calibration_ablation", batch4, root_dir, log_dir, master_log, writer)

    log_line(master_log, "All batches completed.")
    log_line(master_log, f"Status file: {status_tsv}")
    log_line(master_log, f"Summary file: {summary_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
