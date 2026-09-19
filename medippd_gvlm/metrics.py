import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def read_best_segmentation_map(results_csv: Path) -> Dict[str, float]:
    best = None
    with Path(results_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                map50 = float(row["metrics/mAP50(M)"])
                map50_95 = float(row["metrics/mAP50-95(M)"])
            except (KeyError, TypeError, ValueError):
                continue
            if best is None or map50_95 > best[1]:
                best = (map50, map50_95)
    if best is None:
        raise ValueError(f"no segmentation mAP columns found in {results_csv}")
    return {"mask_map50": best[0], "mask_map50_95": best[1]}


def _safe_auc(function, labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(function(labels, scores))


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if not np.any(labels == 1):
        return float("nan")
    return float(average_precision_score(labels, scores))


def compute_red_metrics(
    true_masks: np.ndarray,
    pred_masks: np.ndarray,
    true_diameter_mm: Sequence[float],
    pred_diameter_mm: Sequence[float],
    yolo_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    truth = np.asarray(true_masks, dtype=bool)
    prediction = np.asarray(pred_masks, dtype=bool)
    if truth.shape != prediction.shape:
        raise ValueError(f"mask shape mismatch: {truth.shape} versus {prediction.shape}")
    tp = float(np.logical_and(truth, prediction).sum())
    fp = float(np.logical_and(~truth, prediction).sum())
    fn = float(np.logical_and(truth, ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    dice = 2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0

    true_values = np.asarray(true_diameter_mm, dtype=np.float64)
    pred_values = np.asarray(pred_diameter_mm, dtype=np.float64)
    valid = np.isfinite(true_values) & np.isfinite(pred_values)
    true_values, pred_values = true_values[valid], pred_values[valid]
    if true_values.size:
        errors = pred_values - true_values
        absolute = np.abs(errors)
        mae = float(absolute.mean())
        rmse = float(np.sqrt(np.mean(errors**2)))
        denominator = float(np.sum((true_values - true_values.mean()) ** 2))
        r2 = 1.0 - float(np.sum(errors**2)) / denominator if denominator > 0 else float("nan")
        acc2 = float(np.mean(absolute <= 2.0))
        acc5 = float(np.mean(absolute <= 5.0))
    else:
        mae = rmse = r2 = acc2 = acc5 = float("nan")
    yolo_metrics = yolo_metrics or {}
    return {
        "mask_precision": precision,
        "mask_recall": recall,
        "mask_dice": dice,
        "mask_iou": iou,
        "mask_map50": float(yolo_metrics.get("mask_map50", float("nan"))),
        "mask_map50_95": float(yolo_metrics.get("mask_map50_95", float("nan"))),
        "diameter_mae_mm": mae,
        "diameter_rmse_mm": rmse,
        "diameter_r2": r2,
        "diameter_acc_2mm": acc2,
        "diameter_acc_5mm": acc5,
    }


def compute_v2_red_metrics(
    true_masks: np.ndarray,
    red_probability_masks: np.ndarray,
    true_diameter_mm: Sequence[float],
    pred_diameter_mm: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Measure V2's actual red probability mask without importing YOLO mAP."""

    probabilities = np.asarray(red_probability_masks, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("red probability masks must be finite")
    return compute_red_metrics(
        true_masks,
        probabilities >= float(threshold),
        true_diameter_mm,
        pred_diameter_mm,
    )


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(4)
    second = np.asarray(second, dtype=np.float64).reshape(4)
    upper_left = np.maximum(first[:2], second[:2])
    lower_right = np.minimum(first[2:], second[2:])
    intersection_size = np.maximum(lower_right - upper_left, 0.0)
    intersection = float(np.prod(intersection_size))
    first_size = np.maximum(first[2:] - first[:2], 0.0)
    second_size = np.maximum(second[2:] - second[:2], 0.0)
    union = float(np.prod(first_size) + np.prod(second_size) - intersection)
    return intersection / union if union > 0.0 else 0.0


def _class_detection_ap(
    gt_by_image: Mapping[object, Mapping[int, Sequence[np.ndarray]]],
    predictions: Sequence[Mapping[str, object]],
    class_id: int,
    iou_threshold: float,
):
    positives = sum(
        len(classes.get(class_id, ())) for classes in gt_by_image.values()
    )
    if positives == 0:
        return float("nan"), 0

    ranked = sorted(
        (
            prediction
            for prediction in predictions
            if int(prediction["class_id"]) == class_id
        ),
        key=lambda prediction: -float(prediction["score"]),
    )
    matched = {
        image: np.zeros(len(classes.get(class_id, ())), dtype=bool)
        for image, classes in gt_by_image.items()
    }
    true_positives = []
    false_positives = []
    for prediction in ranked:
        image = prediction["image"]
        boxes = gt_by_image.get(image, {}).get(class_id, ())
        available = matched.get(image)
        best_index = -1
        best_iou = -1.0
        for index, box in enumerate(boxes):
            if available[index]:
                continue
            iou = _box_iou(prediction["box"], box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        is_match = best_index >= 0 and best_iou >= iou_threshold
        if is_match:
            available[best_index] = True
        true_positives.append(float(is_match))
        false_positives.append(float(not is_match))

    if not ranked:
        return 0.0, 0
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recall = cumulative_tp / float(positives)
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    average_precision = np.sum(
        (recall[changes + 1] - recall[changes]) * precision[changes + 1]
    )
    return float(average_precision), int(cumulative_tp[-1])


def average_precision_detections(
    gt_by_image: Mapping[object, Mapping[int, Sequence[np.ndarray]]],
    predictions: Sequence[Mapping[str, object]],
    iou_thresholds: Sequence[float],
) -> Dict[str, float]:
    """Compute globally ranked, one-to-one matched detection AP by class."""

    thresholds = tuple(float(value) for value in iou_thresholds)
    if not thresholds or any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("iou_thresholds must contain values in [0, 1]")
    predictions = list(predictions)
    for prediction in predictions:
        missing = {"image", "class_id", "score", "box"} - set(prediction)
        if missing:
            raise KeyError(f"detection prediction missing fields: {sorted(missing)}")
        if not np.isfinite(float(prediction["score"])):
            raise ValueError("detection scores must be finite")
        np.asarray(prediction["box"], dtype=np.float64).reshape(4)

    class_ids = sorted(
        {
            int(class_id)
            for classes in gt_by_image.values()
            for class_id in classes
        }
        | {int(prediction["class_id"]) for prediction in predictions}
    )
    result: Dict[str, float] = {}
    positive_class_ap50 = []
    positive_class_mean_ap = []
    for class_id in class_ids:
        positives = sum(
            len(classes.get(class_id, ())) for classes in gt_by_image.values()
        )
        candidate_count = sum(
            int(int(prediction["class_id"]) == class_id)
            for prediction in predictions
        )
        ap50, tp50 = _class_detection_ap(
            gt_by_image, predictions, class_id, 0.5
        )
        threshold_aps = [
            _class_detection_ap(gt_by_image, predictions, class_id, threshold)[0]
            for threshold in thresholds
        ]
        mean_ap = float(np.mean(threshold_aps)) if positives else float("nan")
        prefix = f"class_{class_id}"
        result[f"{prefix}_ap50"] = ap50
        result[f"{prefix}_map50_95"] = mean_ap
        result[f"{prefix}_positive_count"] = int(positives)
        result[f"{prefix}_candidate_count"] = int(candidate_count)
        result[f"{prefix}_tp50"] = int(tp50)
        if positives:
            positive_class_ap50.append(ap50)
            positive_class_mean_ap.append(mean_ap)

    result["macro_map50"] = (
        float(np.mean(positive_class_ap50))
        if positive_class_ap50
        else float("nan")
    )
    result["macro_map50_95"] = (
        float(np.mean(positive_class_mean_ap))
        if positive_class_mean_ap
        else float("nan")
    )
    return result


def compute_strong_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    detection_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("strong labels and scores must both have shape [N, 3]")
    predictions = (scores >= threshold).astype(np.int64)
    class_ap = [_safe_auc(average_precision_score, labels[:, index], scores[:, index]) for index in range(3)]
    macro_precision = float(precision_score(labels, predictions, average="macro", zero_division=0))
    macro_recall = float(recall_score(labels, predictions, average="macro", zero_division=0))
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    any_labels = labels.max(axis=1)
    any_scores = 1.0 - np.prod(1.0 - np.clip(scores, 0.0, 1.0), axis=1)
    any_predictions = (any_scores >= threshold).astype(np.int64)
    positive = any_labels == 1
    negative = ~positive
    sensitivity = float(np.mean(any_predictions[positive] == 1)) if positive.any() else float("nan")
    specificity = float(np.mean(any_predictions[negative] == 0)) if negative.any() else float("nan")
    detection_metrics = detection_metrics or {}
    return {
        "blister_ap50": float(detection_metrics.get("blister_ap50", class_ap[0])),
        "necrosis_ap50": float(detection_metrics.get("necrosis_ap50", class_ap[1])),
        "double_ring_ap50": float(detection_metrics.get("double_ring_ap50", class_ap[2])),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_map50": float(detection_metrics.get("macro_map50", np.nanmean(class_ap))),
        "macro_map50_95": float(detection_metrics.get("macro_map50_95", float("nan"))),
        "strong_any_sensitivity": sensitivity,
        "strong_any_specificity": specificity,
        "strong_any_f1": float(f1_score(any_labels, any_predictions, zero_division=0)),
        "strong_any_auroc": _safe_auc(roc_auc_score, any_labels, any_scores),
        "strong_any_auprc": _safe_auc(average_precision_score, any_labels, any_scores),
    }


def compute_v2_strong_metrics(
    labels: np.ndarray,
    class_scores: np.ndarray,
    strong_any_scores: np.ndarray,
    threshold: float = 0.5,
    detection_metrics: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Measure V2 classification and localization with unambiguous names."""

    labels = np.asarray(labels, dtype=np.int64)
    class_scores = np.asarray(class_scores, dtype=np.float64)
    strong_any_scores = np.asarray(strong_any_scores, dtype=np.float64)
    if labels.shape != class_scores.shape or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("strong labels and class scores must both have shape [N, 3]")
    if strong_any_scores.shape != (labels.shape[0],):
        raise ValueError("strong-any scores must have shape [N]")

    class_predictions = (class_scores >= threshold).astype(np.int64)
    class_ap = [
        _safe_average_precision(labels[:, index], class_scores[:, index])
        for index in range(3)
    ]
    any_labels = labels.max(axis=1)
    any_predictions = (strong_any_scores >= threshold).astype(np.int64)
    positive = any_labels == 1
    negative = ~positive
    detection_metrics = detection_metrics or {}
    class_names = ("blister", "necrosis", "double_ring")
    result = {
        f"{name}_classification_ap": class_ap[index]
        for index, name in enumerate(class_names)
    }
    result.update(
        {
            f"{name}_localized_ap50": float(
                detection_metrics.get(f"class_{index + 2}_ap50", float("nan"))
            )
            for index, name in enumerate(class_names)
        }
    )
    finite_class_ap = [value for value in class_ap if np.isfinite(value)]
    result.update(
        {
            "macro_classification_ap": (
                float(np.mean(finite_class_ap))
                if finite_class_ap
                else float("nan")
            ),
            "macro_precision": float(
                precision_score(labels, class_predictions, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(labels, class_predictions, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(labels, class_predictions, average="macro", zero_division=0)
            ),
            "localized_map50": float(
                detection_metrics.get("macro_map50", float("nan"))
            ),
            "localized_map50_95": float(
                detection_metrics.get("macro_map50_95", float("nan"))
            ),
            "strong_any_sensitivity": (
                float(np.mean(any_predictions[positive] == 1))
                if positive.any()
                else float("nan")
            ),
            "strong_any_specificity": (
                float(np.mean(any_predictions[negative] == 0))
                if negative.any()
                else float("nan")
            ),
            "strong_any_f1": float(
                f1_score(any_labels, any_predictions, zero_division=0)
            ),
            "strong_any_auroc": _safe_auc(
                roc_auc_score, any_labels, strong_any_scores
            ),
            "strong_any_auprc": _safe_average_precision(
                any_labels, strong_any_scores
            ),
        }
    )
    return result


def bootstrap_interval(metric_function, labels, scores, samples: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(labels), len(labels))
        value = float(metric_function(labels[indices], scores[indices]))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))
