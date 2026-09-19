from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_reproduction_and_data_access():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "MediPPD",
        "contact the authors",
        "scripts/run_main_experiment.py",
        "conda create",
        "30.0 mm",
    ):
        assert phrase in text


def test_readme_does_not_reference_private_workspace():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "/media/" not in text
    assert "medippd" + "4" not in text


def test_readme_describes_only_the_main_method():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for forbidden in ("baseline experiment", "ablation experiment", "辅助实验"):
        assert forbidden not in text
