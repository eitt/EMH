#!/usr/bin/env python3
"""Unified entrypoint for EMH robust research + manuscript pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path.cwd().resolve()


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
    parser = argparse.ArgumentParser(description="Run project pipeline in a single command.")
    parser.add_argument("--ingest", action="store_true", help="Download fresh raw data before preprocessing.")
    parser.add_argument("--skip-train", action="store_true", help="Skip standalone diffusion trainer.")
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip robust benchmark + diffusion experiment loop.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip report figure generation.")
    parser.add_argument("--skip-xai", action="store_true", help="Skip explainability step.")
    parser.add_argument(
        "--skip-manuscript-assets",
        action="store_true",
        help="Skip LaTeX asset export (tables/generated, figures/generated).",
    )
    parser.add_argument(
        "--skip-working-paper",
        action="store_true",
        help="Skip Overleaf-ready working-paper bundle generation.",
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
        run_step("Run robust experiment loop", [python, "src/experiments/run_loop.py"])

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

    if not args.skip_manuscript_assets:
        run_step("Build manuscript assets", [python, "scripts/build_manuscript_assets.py"])

    if not args.skip_working_paper:
        run_step("Build working-paper bundle", [python, "scripts/build_working_paper.py"])

    print("[run_pipeline] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
