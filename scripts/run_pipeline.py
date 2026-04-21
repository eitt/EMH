#!/usr/bin/env python3
"""Unified entrypoint for the EMH research pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(label: str, cmd: list[str], env_extra: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    if env_extra:
        env.update(env_extra)

    print(f"[run_pipeline] {label}: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run project pipeline in a single command."
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Download fresh raw data before preprocessing.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip diffusion model training.",
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip benchmark/diffusion experiment loop.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip report figure generation.",
    )
    parser.add_argument(
        "--skip-xai",
        action="store_true",
        help="Skip integrated-gradients explainability step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable

    if args.ingest:
        run_step("Ingest data", [python, "src/data/ingest.py"])

    run_step("Preprocess data", [python, "src/data/preprocess.py"])

    if not args.skip_train:
        run_step("Train diffusion model", [python, "src/models/trainer.py"])

    if not args.skip_experiments:
        run_step("Run experiment loop", [python, "src/experiments/run_loop.py"])

    if not args.skip_plots:
        run_step(
            "Generate figures",
            [python, "src/visualization/plot_results.py"],
            env_extra={"MPLBACKEND": "Agg"},
        )

    if not args.skip_xai:
        run_step(
            "Run explainability",
            [python, "src/xai/explain.py"],
            env_extra={"MPLBACKEND": "Agg"},
        )

    print("[run_pipeline] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
