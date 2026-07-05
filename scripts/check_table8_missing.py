#!/usr/bin/env python3
"""Report missing canonical Table 8 ReKV/B-ReKV runs under a snapshot root."""

from __future__ import annotations

import argparse
from pathlib import Path


TASKS = [
    "countries",
    "tipsheets",
    "hotpotqa",
    "musique",
    "multifieldqa_en",
    "twowikimqa",
    "qasper",
    "tmath",
]

RUNS = [
    ("mtc_receiver", "recv_w8_r0.3"),
    ("mtc_receiver", "recv_w8_r0.5"),
    ("mtc_receiver", "recv_w8_r0.7"),
    ("mtc_receiver", "recv_w16_r0.3"),
    ("mtc_receiver", "recv_w16_r0.5"),
    ("mtc_receiver", "recv_w16_r0.7"),
    ("coverage", "cov_t0.95_s0.75_w8"),
    ("coverage", "cov_t0.95_s0.85_w8"),
    ("coverage", "cov_t0.95_s0.90_w16"),
]


def is_done(root: Path, task: str, method_dir: str, run_name: str) -> bool:
    return any((root / task / method_dir).glob(f"{run_name}_*/per_sample.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    done = 0
    missing: list[tuple[str, list[str]]] = []
    total = len(TASKS) * len(RUNS)

    for task in TASKS:
        task_missing = []
        for method_dir, run_name in RUNS:
            if is_done(args.root, task, method_dir, run_name):
                done += 1
            else:
                task_missing.append(f"{method_dir}/{run_name}")
        if task_missing:
            missing.append((task, task_missing))

    print(f"{args.root}: {done}/{total} done, {total - done} missing")
    for task, task_missing in missing:
        print(f"- {task}: {', '.join(task_missing)}")


if __name__ == "__main__":
    main()
