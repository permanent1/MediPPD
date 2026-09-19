from pathlib import Path

from scripts.run_main_experiment import parse_args


def test_main_cli_exposes_only_main_experiment_options(tmp_path):
    args = parse_args(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--patient-csv",
            str(tmp_path / "patients.csv"),
        ]
    )

    assert not hasattr(args, "run_ablations")
    assert not hasattr(args, "main_only")
    assert args.dataset_root == tmp_path / "dataset"
    assert args.patient_csv == tmp_path / "patients.csv"


def test_main_cli_defaults_are_repository_relative():
    args = parse_args([])
    repository = Path(__file__).resolve().parents[1]

    assert args.dataset_root == repository / "datasets" / "ppd553_seg"
    assert args.run_root == repository / "runs" / "medippd_main"
    assert args.result_root == repository / "results" / "medippd_main"

