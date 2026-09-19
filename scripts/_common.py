"""Shared argument and phase helpers for public MediPPD scripts."""

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from medippd_gvlm.config import (
    MAIN_CONFIG,
    PATIENT_CSV,
    RESULT_ROOT,
    RUN_ROOT,
    SOURCE_DATASET,
)


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset-root", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--patient-csv", type=Path, default=PATIENT_CSV)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--llava-model", default=MAIN_CONFIG["vlm"]["model"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--main-budget-min",
        type=float,
        default=float(MAIN_CONFIG["task_routed"]["main_budget_minutes"]),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-vlm-cache", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args):
    if args.main_budget_min <= 0:
        parser.error("--main-budget-min must be positive")
    return args


def run_named_phase(args, phase_name: str) -> dict:
    from medippd_gvlm.main_pipeline import PipelineContext, build_phase_plan

    context = PipelineContext(args)
    phases = {phase.name: phase for phase in build_phase_plan(context)}
    phase = phases[phase_name]
    return dict(phase.run(context) or {})
