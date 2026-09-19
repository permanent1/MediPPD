from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_public_files_exist():
    for relative in (
        ".gitignore",
        "requirements.txt",
        "THIRD_PARTY.md",
        "configs/medippd_main.yaml",
        "datasets/README.md",
    ):
        assert (ROOT / relative).is_file(), relative


def test_forbidden_experiment_trees_are_absent():
    for name in ("baselines", "medippd_aux", "runs", "results"):
        assert not (ROOT / name).exists(), name


def test_no_weights_or_real_dataset_files_are_present():
    forbidden = {".pt", ".pth", ".ckpt", ".jpg", ".jpeg", ".png", ".csv"}
    offenders = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.suffix.lower() in forbidden
    ]
    assert offenders == []
