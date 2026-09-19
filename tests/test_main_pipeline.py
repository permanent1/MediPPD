from pathlib import Path
from types import SimpleNamespace

import numpy as np

from medippd_gvlm import main_pipeline
from medippd_gvlm.main_pipeline import (
    PipelineContext,
    build_phase_plan,
    evaluate_main,
    pack_detection_candidates,
)
from medippd_gvlm.reporting import write_main_result_bundle


def make_args(tmp_path: Path):
    return SimpleNamespace(
        dataset_root=tmp_path / "dataset",
        patient_csv=tmp_path / "patients.csv",
        run_root=tmp_path / "runs",
        result_root=tmp_path / "results",
        llava_model="test/model",
        device="cpu",
        main_budget_min=35.0,
        skip_vlm_cache=False,
        resume=False,
    )


def test_phase_plan_contains_only_main_method_phases(tmp_path):
    context = PipelineContext(make_args(tmp_path))

    assert [phase.name for phase in build_phase_plan(context)] == [
        "prepare",
        "segmentation",
        "strong_detector",
        "export_predictions",
        "vlm_cache",
        "train_main",
        "train_red_fusion",
        "evaluate_main",
        "write_results",
    ]


def test_context_uses_explicit_private_data_paths(tmp_path):
    args = make_args(tmp_path)
    context = PipelineContext(args)

    assert context.source_dataset == args.dataset_root.resolve()
    assert context.patient_csv == args.patient_csv.resolve()


def test_detection_candidates_map_contiguous_classes_to_source_ids():
    exported = pack_detection_candidates(
        boxes=np.asarray([[0, 0, 10, 10], [2, 2, 8, 8]], dtype=np.float32),
        scores=np.asarray([0.004, 0.002], dtype=np.float32),
        train_classes=np.asarray([0, 2], dtype=np.int64),
        image_shape=(20, 20),
    )

    assert exported["detector_classes"].tolist() == [2, 4]
    np.testing.assert_allclose(
        exported["detector_boxes"],
        [[0.0, 0.0, 0.5, 0.5], [0.1, 0.1, 0.4, 0.4]],
    )


def test_evaluation_writes_only_medippd_rows(tmp_path, monkeypatch):
    context = PipelineContext(make_args(tmp_path))
    validation = [object(), object()]
    monkeypatch.setattr(main_pipeline, "_datasets", lambda _context: ([], validation))
    monkeypatch.setattr(main_pipeline, "_load_model", lambda *args: object())
    monkeypatch.setattr(main_pipeline, "_load_red_fusion", lambda *args: object())
    monkeypatch.setattr(main_pipeline, "_evaluation_expected", lambda _context: {})
    monkeypatch.setattr(main_pipeline, "_main_seconds", lambda _context: 60.0)

    import medippd_gvlm.task_routed_training as training

    monkeypatch.setattr(
        training,
        "evaluate_task_routed_model",
        lambda *args, **kwargs: (
            {"mask_dice": 0.8},
            {"macro_ap": 0.5},
            [{"image_stem": "synthetic"}],
            {"mask_threshold": 0.5},
        ),
    )

    details = evaluate_main(context)
    payload = main_pipeline._read_json(Path(details["evaluation"]))

    assert [row["method"] for row in payload["main_red"]] == ["MediPPD"]
    assert [row["method"] for row in payload["main_strong"]] == ["MediPPD"]
    assert "bootstrap" not in payload
    assert "baseline" not in str(payload).lower()


def test_reporting_writes_only_main_experiment_tables(tmp_path):
    frames = write_main_result_bundle(
        tmp_path,
        [{"method": "MediPPD", "mask_dice": 0.8}],
        [{"method": "MediPPD", "macro_classification_ap": 0.5}],
        [{"phase": "train_main", "seconds": 1.0}],
        per_case_rows=[{"image_stem": "synthetic"}],
        detection_rows=[{"method": "MediPPD", "mask_threshold": 0.5}],
    )

    assert set(frames) == {
        "main_redswollen",
        "main_strong_features",
        "runtime_budget",
        "per_case_predictions",
        "detection_diagnostics",
    }
    assert not any("baseline" in path.name.lower() for path in tmp_path.iterdir())
    assert not any("ablation" in path.name.lower() for path in tmp_path.iterdir())
