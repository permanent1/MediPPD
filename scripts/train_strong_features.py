"""Train or reuse the MediPPD blister, necrosis, and double-ring detector."""

import argparse

try:
    from scripts._common import add_common_arguments, run_named_phase, validate_args
except ModuleNotFoundError:
    from _common import add_common_arguments, run_named_phase, validate_args


def parse_args(argv=None):
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    return validate_args(parser, parser.parse_args(argv))


def main() -> None:
    run_named_phase(parse_args(), "strong_detector")


if __name__ == "__main__":
    main()
