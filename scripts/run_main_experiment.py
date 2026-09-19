"""Run the complete, resumable MediPPD main experiment."""

import argparse
import os

try:
    from scripts._common import add_common_arguments, validate_args
except ModuleNotFoundError:
    from _common import add_common_arguments, validate_args

from medippd_gvlm.main_pipeline import run_main_pipeline


def parse_args(argv=None):
    parser = add_common_arguments(
        argparse.ArgumentParser(description=__doc__)
    )
    return validate_args(parser, parser.parse_args(argv))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    run_main_pipeline(args)


if __name__ == "__main__":
    main()
