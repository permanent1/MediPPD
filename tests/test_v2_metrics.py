from types import SimpleNamespace

import numpy as np
import pytest
import torch

from medippd_gvlm.data import CaseRecord
from medippd_gvlm.metrics import (
    average_precision_detections,
    compute_v2_red_metrics,
    compute_v2_strong_metrics,
)
from medippd_gvlm.task_routed_model import TaskRoutedConfig
from medippd_gvlm.red_mask_fusion import RedMaskFusionHead
from medippd_gvlm.task_routed_training import (
    evaluate_task_routed_model,
    positive_grounding_dice,
)


def _box(x1=0.1, y1=0.1, x2=0.4, y2=0.4):
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def test_perfect_scored_detections_have_ap50_one():
    ground_truth = {"case": {2: [_box()]}}
    predictions = [
        {"image": "case", "class_id": 2, "score": 0.9, "box": _box()}
    ]

    metrics = average_precision_detections(ground_truth, predictions, [0.5])

    assert metrics["macro_map50"] == pytest.approx(1.0)


def test_duplicate_predictions_match_one_ground_truth_only_once():
    ground_truth = {"case": {2: [_box()]}}
    predictions = [
        {"image": "case", "class_id": 2, "score": 0.9, "box": _box()},
        {"image": "case", "class_id": 2, "score": 0.8, "box": _box()},
    ]

    metrics = average_precision_detections(ground_truth, predictions, [0.5])

    assert metrics["class_2_tp50"] == 1
    assert metrics["class_2_candidate_count"] == 2
    assert metrics["class_2_candidate_count"] - metrics["class_2_tp50"] == 1


def test_high_score_false_positive_before_true_positive_lowers_ap():
    ground_truth = {"case": {2: [_box()]}}
    predictions = [
        {"image": "case", "class_id": 2, "score": 0.9, "box": _box(0.6, 0.6, 0.9, 0.9)},
        {"image": "case", "class_id": 2, "score": 0.8, "box": _box()},
    ]

    metrics = average_precision_detections(ground_truth, predictions, [0.5])

    assert metrics["macro_map50"] == pytest.approx(0.5)


def test_classes_without_validation_positives_are_excluded_from_macro_ap():
    ground_truth = {"case": {2: [_box()]}}
    predictions = [
        {"image": "case", "class_id": 2, "score": 0.9, "box": _box()},
        {"image": "case", "class_id": 3, "score": 0.99, "box": _box()},
    ]

    metrics = average_precision_detections(ground_truth, predictions, [0.5])

    assert metrics["macro_map50"] == pytest.approx(1.0)
    assert np.isnan(metrics["class_3_ap50"])


def test_v2_mask_metrics_reject_copied_yolo_map_argument():
    truth = np.asarray([[[1, 0], [0, 0]]], dtype=np.uint8)
    probability = np.asarray([[[0.9, 0.2], [0.1, 0.1]]], dtype=np.float32)

    with pytest.raises(TypeError):
        compute_v2_red_metrics(
            truth,
            probability,
            [10.0],
            [10.0],
            yolo_metrics={"mask_map50": 1.0},
        )


def test_v2_classification_ap_is_defined_when_every_case_is_positive():
    labels = np.ones((2, 3), dtype=np.int64)
    scores = np.asarray([[0.8, 0.7, 0.6], [0.9, 0.8, 0.7]])

    metrics = compute_v2_strong_metrics(labels, scores, np.asarray([0.8, 0.9]))

    assert metrics["blister_classification_ap"] == pytest.approx(1.0)
    assert metrics["strong_any_auprc"] == pytest.approx(1.0)


def test_positive_grounding_dice_scores_only_cases_with_a_positive_target():
    probabilities = np.asarray(
        [
            [[0.9, 0.1], [0.1, 0.1]],
            [[0.9, 0.9], [0.9, 0.9]],
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        [
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.float32,
    )

    assert positive_grounding_dice(probabilities, targets) == pytest.approx(1.0)
    assert positive_grounding_dice(
        np.zeros_like(probabilities), np.zeros_like(targets)
    ) == pytest.approx(0.0)


class _EvaluationDataset:
    def __init__(self, tmp_path, include_red=False):
        self.prediction_dir = tmp_path / "predictions"
        self.prediction_dir.mkdir()
        self.cache_dir = tmp_path / "cache"
        self.cache_dir.mkdir()
        self.cases = []
        for index, stem in enumerate(("negative", "positive")):
            label_path = tmp_path / f"{stem}.txt"
            label_path.write_text(
                ("1 0.5 0.5 1.0 1.0\n" if include_red else "")
                + ("2 0.5 0.5 0.4 0.4\n" if index == 1 else ""),
                encoding="utf-8",
            )
            self.cases.append(
                CaseRecord(
                    stem=stem,
                    image_path=tmp_path / f"{stem}.jpg",
                    label_path=label_path,
                    split="val",
                    patient={"硬结平均径": "10.0"},
                )
            )
            np.savez_compressed(
                self.prediction_dir / f"{stem}.npz",
                image_shape=np.asarray([8, 8], dtype=np.int32),
                lesion_mask=np.zeros((8, 8), dtype=np.uint8),
                morphometry=np.asarray([10.0, 0, 0, 0, 0, 1, 1], dtype=np.float32),
                detector_boxes=np.empty((0, 4), dtype=np.float32),
                detector_scores=np.empty((0,), dtype=np.float32),
                detector_classes=np.empty((0,), dtype=np.int64),
            )
            np.savez_compressed(
                self.cache_dir / f"{stem}.npz",
                view_boxes=np.asarray(
                    [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                    dtype=np.float32,
                ),
            )

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "patches": torch.zeros(3, 4, 2),
            "morphometry": torch.zeros(7),
            "diameter_baseline": torch.tensor(10.0),
            "clinical": torch.zeros(2),
            "index": torch.tensor(index),
            "grounding_targets": torch.zeros(3, 4, 2, 2),
        }


class _ContradictoryStrongModel:
    def eval(self):
        return self

    def __call__(self, batch, config):
        indices = batch["index"].long()
        class_logit = torch.where(indices[:, None] == 0, 5.0, -5.0).expand(-1, 3)
        strong_logit = torch.where(indices == 0, -5.0, 5.0)
        batch_size = len(indices)
        return SimpleNamespace(
            diameter_mm=torch.full((batch_size,), 10.0),
            class_logits=class_logit,
            strong_any_logit=strong_logit,
            red_mask_logits=torch.full((batch_size, 2, 2), -12.0),
            class_grounding_logits=torch.full((batch_size, 3, 3, 2, 2), -12.0),
        )


class _HighRedProbabilityModel(_ContradictoryStrongModel):
    def __call__(self, batch, config):
        output = super().__call__(batch, config)
        output.red_mask_logits.fill_(12.0)
        output.grounding_logits = torch.full(
            (len(batch["index"]), 3, 4, 2, 2), 12.0
        )
        return output


def test_evaluation_uses_dedicated_strong_any_logits_when_noisy_or_disagrees(tmp_path):
    red, strong, per_case, diagnostics = evaluate_task_routed_model(
        _ContradictoryStrongModel(),
        _EvaluationDataset(tmp_path),
        TaskRoutedConfig.full(),
        device="cpu",
    )

    assert strong["strong_any_auroc"] == pytest.approx(1.0)
    assert strong["strong_any_auprc"] == pytest.approx(1.0)
    assert strong["strong_any_f1"] == pytest.approx(1.0)
    assert per_case[0]["strong_any_score"] < per_case[1]["strong_any_score"]
    assert per_case[0]["strong_any_noisy_or_score"] > per_case[1]["strong_any_noisy_or_score"]
    assert np.isnan(red["mask_map50"])
    assert diagnostics["blister_candidate_count"] == 0
    assert diagnostics["rare_grounding_dice"] == pytest.approx(0.0)
    assert diagnostics["positive_patch_recall"] == pytest.approx(0.0)
    assert diagnostics["empty_grounding_prediction_rate"] == pytest.approx(1.0)


def test_evaluation_scores_the_v2_red_probability_not_the_binary_yolo_mask(tmp_path):
    red, _, _, _ = evaluate_task_routed_model(
        _HighRedProbabilityModel(),
        _EvaluationDataset(tmp_path, include_red=True),
        TaskRoutedConfig.full(),
        device="cpu",
    )

    assert red["mask_dice"] == pytest.approx(1.0)
    assert red["mask_iou"] == pytest.approx(1.0)


def test_evaluation_uses_yolo_anchored_red_fusion_when_supplied(tmp_path):
    red, _, _, diagnostics = evaluate_task_routed_model(
        _HighRedProbabilityModel(),
        _EvaluationDataset(tmp_path, include_red=True),
        TaskRoutedConfig.full(),
        device="cpu",
        red_fusion=RedMaskFusionHead(active_views=0),
    )

    assert red["mask_dice"] == pytest.approx(0.0)
    assert diagnostics["red_fusion_active_views"] == 0
