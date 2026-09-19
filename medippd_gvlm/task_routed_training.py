"""Deterministic sampling and staged training for MediPPD V2."""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import class_mask, read_yolo_segments
from .metrics import (
    average_precision_detections,
    compute_v2_red_metrics,
    compute_v2_strong_metrics,
)
from .task_routed_losses import compute_task_routed_losses
from .task_routed_model import (
    TaskRoutedConfig,
    TaskRoutedMediPPD,
)


def make_class_aware_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    """Return a reproducible sampler that increases exposure of rare positives."""

    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim != 2 or labels.shape[0] == 0:
        raise ValueError("labels must be a non-empty [samples, classes] array")
    if np.any((labels < 0) | (labels > 1)):
        raise ValueError("labels must contain binary values")

    positive_counts = labels.sum(axis=0)
    class_weights = np.clip(
        labels.shape[0] / np.maximum(positive_counts, 1.0), 1.0, 25.0
    )
    sample_weights = np.maximum(1.0, (labels * class_weights).max(axis=1))
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=labels.shape[0],
        replacement=True,
        generator=generator,
    )


def make_training_sampler(labels: np.ndarray, strategy: str, seed: int):
    if strategy == "uniform":
        return None
    if strategy == "weighted":
        return make_class_aware_sampler(labels, seed)
    raise ValueError(f"unknown sampler strategy: {strategy}")


def freeze_diameter_head(model: TaskRoutedMediPPD) -> None:
    """Freeze the complete private diameter branch before stage two."""

    for name, parameter in model.named_parameters():
        if name.startswith("diameter_"):
            parameter.requires_grad_(False)


def task_routed_training_objectives(
    config: TaskRoutedConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return stage-one and stage-two objectives enabled by the configuration."""

    stage_one = []
    stage_two = []
    if config.train_diameter:
        stage_one.append("diameter")
    if config.train_grounding:
        stage_one.append("red_ground")
    if config.train_case:
        stage_two.extend(("class_bce", "strong_any", "reaction"))
    if config.train_grounding:
        stage_two.append("class_ground")
    if (
        config.train_case
        and config.train_grounding
        and config.use_task_consistency
    ):
        stage_two.append("consistency")
    return tuple(stage_one), tuple(stage_two)


def checkpoint_selection_objectives(
    config: TaskRoutedConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select checkpoints using the task owned by each training stage."""

    stage_one, stage_two = task_routed_training_objectives(config)
    case_selection = tuple(
        name for name in ("class_bce", "strong_any") if name in stage_two
    )
    return stage_one, case_selection or stage_two


def _dataset_labels(dataset) -> np.ndarray:
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        result = np.asarray(labels, dtype=np.float32)
    else:
        rows = []
        for index in range(len(dataset)):
            sample = dataset[index]
            target = sample.get("class_target", sample.get("dermal_target"))
            if target is None:
                raise KeyError("dataset samples must contain class_target or dermal_target")
            rows.append(torch.as_tensor(target).cpu().numpy())
        result = np.asarray(rows, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != len(dataset):
        raise ValueError("dataset labels must have shape [samples, classes]")
    return result


def _train_calibration(labels: np.ndarray) -> Dict[str, object]:
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    class_pos_weight = np.clip(
        negatives / np.maximum(positives, 1.0), 1.0, 25.0
    ).astype(np.float32)
    strong_positive = float((labels.max(axis=1) > 0).sum())
    strong_pos_weight = float(
        np.clip(
            (len(labels) - strong_positive) / max(strong_positive, 1.0),
            1.0,
            25.0,
        )
    )
    return {
        "num_train_samples": int(len(labels)),
        "class_positive_counts": positives.astype(int).tolist(),
        "class_pos_weight": class_pos_weight.tolist(),
        "strong_positive_count": int(strong_positive),
        "strong_pos_weight": strong_pos_weight,
        "focal_gamma": 2.0,
        "positive_weight_clip": [1.0, 25.0],
    }


def _to_device(batch, device: torch.device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _add_calibration(batch, calibration: Dict[str, object], device: torch.device):
    batch["class_pos_weight"] = torch.as_tensor(
        calibration["class_pos_weight"], dtype=torch.float32, device=device
    )
    batch["strong_pos_weight"] = torch.tensor(
        calibration["strong_pos_weight"], dtype=torch.float32, device=device
    )
    return batch


def _make_model(
    dataset,
    semantic_initializers: np.ndarray,
    d_model: int,
) -> TaskRoutedMediPPD:
    if len(dataset) == 0:
        raise ValueError("train_dataset must not be empty")
    first = dataset[0]
    grounding_targets = torch.as_tensor(first["grounding_targets"])
    return TaskRoutedMediPPD(
        patch_dim=int(torch.as_tensor(first["patches"]).shape[-1]),
        semantic_initializers=semantic_initializers,
        clinical_dim=int(torch.as_tensor(first["clinical"]).numel()),
        d_model=int(d_model),
        grid_size=int(grounding_targets.shape[-1]),
        num_reaction_classes=4,
    )


def _stage_one_parameters(
    model: TaskRoutedMediPPD, config: TaskRoutedConfig
) -> Iterable[torch.nn.Parameter]:
    prefixes = ("diameter_", "ground_")
    if not config.use_task_routing:
        prefixes += ("shared_",)
    exact = {"query_delta"}
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes) or name.startswith("semantic_projection.") or name in exact:
            yield parameter


def _validation_loss(
    model,
    loader,
    config,
    calibration,
    device,
    loss_names,
) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for batch in loader:
            batch = _add_calibration(_to_device(batch, device), calibration, device)
            losses = compute_task_routed_losses(model(batch, config), batch, config)
            value = sum(losses[name] for name in loss_names)
            values.append(float(value.item()))
    if not values:
        raise ValueError("val_dataset must not be empty")
    return float(np.mean(values))


def train_task_routed_model(
    train_dataset,
    val_dataset,
    semantic_initializers: np.ndarray,
    config: TaskRoutedConfig,
    checkpoint: Path,
    stage_one_epochs: int = 20,
    stage_two_epochs: int = 40,
    batch_size: int = 32,
    val_batch_size: int = 64,
    d_model: int = 128,
    device: str = "cuda:0",
    seed: int = 42,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    max_seconds: float = 0.0,
    num_workers: int = 0,
) -> Tuple[TaskRoutedMediPPD, float]:
    """Train the isolated V2 model in measurement then case-focused stages."""

    if stage_one_epochs < 0 or stage_two_epochs < 0:
        raise ValueError("stage epoch counts must be non-negative")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device_object = torch.device(device)
    labels = _dataset_labels(train_dataset)
    calibration = _train_calibration(labels)
    model = _make_model(train_dataset, semantic_initializers, d_model).to(device_object)
    sampler = make_training_sampler(labels, config.sampler_strategy, seed)
    loader_generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=sampler is None,
        generator=loader_generator if sampler is None else None,
        num_workers=int(num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(val_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    best_loss = float("inf")
    stale = 0
    stage_objectives = task_routed_training_objectives(config)
    selection_objectives = checkpoint_selection_objectives(config)
    if not (stage_objectives[0] or stage_objectives[1]):
        raise ValueError("task-routed config must enable at least one training objective")

    def save_if_best(validation_objectives) -> bool:
        nonlocal best_loss, stale
        validation_loss = _validation_loss(
            model,
            val_loader,
            config,
            calibration,
            device_object,
            validation_objectives,
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "train_calibration": calibration,
                    "best_validation_loss": best_loss,
                },
                checkpoint,
            )
            return True
        stale += 1
        return False

    stages = (
        (
            int(stage_one_epochs),
            list(_stage_one_parameters(model, config)),
            stage_objectives[0],
            selection_objectives[0],
        ),
        (
            int(stage_two_epochs),
            None,
            stage_objectives[1],
            selection_objectives[1],
        ),
    )
    for stage_index, (
        epochs,
        parameters,
        loss_names,
        validation_objectives,
    ) in enumerate(stages):
        if stage_index == 1:
            freeze_diameter_head(model)
            parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
        if not loss_names:
            continue
        best_loss = float("inf")
        stale = 0
        save_if_best(validation_objectives)
        if epochs == 0:
            continue
        optimizer = torch.optim.AdamW(
            parameters, lr=float(learning_rate), weight_decay=float(weight_decay)
        )
        for _ in range(epochs):
            model.train()
            for batch in train_loader:
                batch = _add_calibration(
                    _to_device(batch, device_object), calibration, device_object
                )
                optimizer.zero_grad(set_to_none=True)
                losses = compute_task_routed_losses(model(batch, config), batch, config)
                objective = sum(losses[name] for name in loss_names)
                objective.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                if max_seconds > 0 and time.monotonic() - started >= max_seconds:
                    break
            save_if_best(validation_objectives)
            if stale >= int(patience):
                break
            if max_seconds > 0 and time.monotonic() - started >= max_seconds:
                break
        state = torch.load(checkpoint, map_location=device_object, weights_only=False)
        model.load_state_dict(state["model"])
        if max_seconds > 0 and time.monotonic() - started >= max_seconds:
            break

    state = torch.load(checkpoint, map_location=device_object, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, time.monotonic() - started


def _normalized_polygon_box(polygon: np.ndarray, image_shape) -> np.ndarray:
    height, width = image_shape
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    return np.asarray(
        [
            points[:, 0].min() / width,
            points[:, 1].min() / height,
            (points[:, 0].max() + 1.0) / width,
            (points[:, 1].max() + 1.0) / height,
        ],
        dtype=np.float32,
    )


def _grounding_evidence(probability_map: np.ndarray, box: np.ndarray) -> float:
    height, width = probability_map.shape
    box = np.clip(np.asarray(box, dtype=np.float64), 0.0, 1.0)
    x1 = min(width - 1, max(0, int(np.floor(box[0] * width))))
    y1 = min(height - 1, max(0, int(np.floor(box[1] * height))))
    x2 = min(width, max(x1 + 1, int(np.ceil(box[2] * width))))
    y2 = min(height, max(y1 + 1, int(np.ceil(box[3] * height))))
    return float(probability_map[y1:y2, x1:x2].mean())


def positive_grounding_dice(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Return sample-mean Dice over cases where the semantic target exists."""

    probabilities = np.asarray(probabilities, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if probabilities.shape != targets.shape or probabilities.ndim != 3:
        raise ValueError("probabilities and targets must have equal [cases, H, W] shape")
    truth = targets >= 0.5
    positive_cases = truth.sum(axis=(1, 2)) > 0
    if not positive_cases.any():
        return 0.0
    prediction = probabilities >= float(threshold)
    intersection = (prediction & truth).sum(axis=(1, 2)).astype(np.float64)
    denominator = prediction.sum(axis=(1, 2)) + truth.sum(axis=(1, 2))
    return float(
        np.mean(2.0 * intersection[positive_cases] / denominator[positive_cases])
    )


def _case_diameter(case, fallback: float) -> float:
    try:
        value = float(case.patient.get("硬结平均径", fallback))
    except (TypeError, ValueError):
        value = float(fallback)
    return value if np.isfinite(value) else float(fallback)


def evaluate_task_routed_model(
    model,
    dataset,
    config: TaskRoutedConfig,
    device: str = "cuda:0",
    batch_size: int = 64,
    mask_threshold: float = 0.5,
    red_fusion=None,
) -> tuple[dict, dict, list[dict], dict]:
    """Evaluate native V2 outputs and every exported scored detection candidate."""

    if len(dataset) == 0:
        raise ValueError("evaluation dataset must not be empty")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    device_object = torch.device(device)
    model.eval()
    ordered = {}
    model_keys = ("patches", "morphometry", "diameter_baseline", "clinical", "index")
    with torch.no_grad():
        for start in range(0, len(dataset), int(batch_size)):
            samples = [
                dataset[index]
                for index in range(start, min(start + int(batch_size), len(dataset)))
            ]
            batch = {
                key: torch.stack([torch.as_tensor(sample[key]) for sample in samples]).to(
                    device_object
                )
                for key in model_keys
                if key in samples[0]
            }
            if "index" not in batch:
                batch["index"] = torch.arange(
                    start, start + len(samples), device=device_object
                )
            output = model(batch, config)
            indices = batch["index"].detach().cpu().numpy()
            diameters = output.diameter_mm.detach().cpu().numpy()
            class_scores = torch.sigmoid(output.class_logits).detach().cpu().numpy()
            strong_scores = (
                torch.sigmoid(output.strong_any_logit).detach().cpu().numpy()
            )
            red_probabilities = (
                torch.sigmoid(output.red_mask_logits).detach().cpu().numpy()
            )
            grounding_probabilities = (
                torch.sigmoid(output.class_grounding_logits).detach().cpu().numpy()
            )
            all_grounding_logits = getattr(output, "grounding_logits", None)
            masked_red_probabilities = (
                torch.sigmoid(all_grounding_logits[:, 2, 0]).detach().cpu().numpy()
                if all_grounding_logits is not None and config.use_masked_grounding
                else None
            )
            for offset, index in enumerate(indices):
                ordered[int(index)] = {
                    "diameter": float(diameters[offset]),
                    "class_scores": class_scores[offset],
                    "strong_score": float(strong_scores[offset]),
                    "red_probability": red_probabilities[offset],
                    "class_grounding": grounding_probabilities[offset],
                    "masked_red_probability": (
                        masked_red_probabilities[offset]
                        if masked_red_probabilities is not None
                        else None
                    ),
                    "red_grounding_logits": (
                        all_grounding_logits[offset, :, 0].detach().cpu().numpy()
                        if all_grounding_logits is not None
                        else None
                    ),
                }

    true_masks = []
    red_probability_masks = []
    masked_red_probabilities = []
    masked_red_targets = []
    true_diameters = []
    pred_diameters = []
    labels = []
    class_score_rows = []
    strong_score_rows = []
    gt_by_image = {}
    localized_predictions = []
    per_case = []
    rare_grounding_probabilities = [[], [], []]
    rare_grounding_targets = [[], [], []]
    grounding_case_scores = []
    class_names = ("blister", "necrosis", "double_ring")
    for index, case in enumerate(dataset.cases):
        values = ordered[index]
        with np.load(dataset.prediction_dir / f"{case.stem}.npz") as prediction:
            image_shape = tuple(int(value) for value in prediction["image_shape"][:2])
            raw_diameter = float(prediction["morphometry"][0])
            detector_boxes = prediction["detector_boxes"].astype(np.float32)
            detector_scores = prediction["detector_scores"].astype(np.float32)
            detector_classes = prediction["detector_classes"].astype(np.int64)
            lesion_mask = prediction["lesion_mask"].astype(np.uint8)

        objects = read_yolo_segments(case.label_path, image_shape)
        truth_mask = class_mask(objects, 1, image_shape)
        red_probability = cv2.resize(
            values["red_probability"],
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        evaluation_size = (128, 128)
        true_masks.append(
            cv2.resize(truth_mask, evaluation_size, interpolation=cv2.INTER_NEAREST)
            > 0
        )
        if red_fusion is None:
            evaluated_red_probability = cv2.resize(
                red_probability, evaluation_size, interpolation=cv2.INTER_LINEAR
            )
        else:
            from .red_mask_fusion import prepare_fusion_inputs

            with np.load(dataset.cache_dir / f"{case.stem}.npz") as cached:
                roi_box = cached["view_boxes"][1].copy()
            probabilities, prior = prepare_fusion_inputs(
                lesion_mask, values["red_grounding_logits"], roi_box
            )
            with torch.no_grad():
                fused_logits = red_fusion(
                    torch.from_numpy(probabilities[None]).to(
                        device_object, dtype=torch.float32
                    ),
                    torch.from_numpy(prior[None]).to(
                        device_object, dtype=torch.float32
                    ),
                )
                evaluated_red_probability = torch.sigmoid(fused_logits)[0, 0].cpu().numpy()
        red_probability_masks.append(evaluated_red_probability)
        if values["masked_red_probability"] is not None:
            sample = dataset[index]
            masked_red_probabilities.append(values["masked_red_probability"])
            masked_red_targets.append(
                torch.as_tensor(sample["grounding_targets"])[2, 0].cpu().numpy()
            )
        true_diameter = _case_diameter(case, raw_diameter)
        true_diameters.append(true_diameter)
        pred_diameters.append(values["diameter"])

        class_labels = np.asarray(
            [int(bool(objects.get(class_id))) for class_id in (2, 3, 4)],
            dtype=np.int64,
        )
        labels.append(class_labels)
        class_score_rows.append(values["class_scores"])
        strong_score_rows.append(values["strong_score"])
        sample = dataset[index]
        sample_targets = torch.as_tensor(sample["grounding_targets"]).cpu().numpy()
        configured_views = tuple(
            view
            for view in config.active_views
            if config.use_masked_grounding or view != 2
        )
        class_views = (0, 1) if configured_views == (0, 1, 2) else configured_views
        case_evidence = []
        for class_index in range(3):
            selected_probabilities = values["class_grounding"][list(class_views), class_index]
            selected_targets = sample_targets[list(class_views), class_index + 1]
            rare_grounding_probabilities[class_index].extend(selected_probabilities)
            rare_grounding_targets[class_index].extend(selected_targets)
            flattened = selected_probabilities.reshape(-1)
            count = max(1, int(np.ceil(len(flattened) * float(config.topk_ratio))))
            case_evidence.append(float(np.partition(flattened, -count)[-count:].mean()))
        grounding_case_scores.append(case_evidence)
        gt_by_image[case.stem] = {
            class_id: [
                _normalized_polygon_box(polygon, image_shape)
                for polygon in objects.get(class_id, ())
            ]
            for class_id in (2, 3, 4)
        }

        active_views = 3 if config.use_masked_grounding else 2
        mean_grounding = values["class_grounding"][:active_views].mean(axis=0)
        for box, detector_score, class_id in zip(
            detector_boxes, detector_scores, detector_classes
        ):
            if int(class_id) not in (2, 3, 4):
                continue
            class_index = int(class_id) - 2
            grounding_score = _grounding_evidence(
                mean_grounding[class_index], box
            )
            localized_score = (
                0.50 * float(detector_score)
                + 0.30 * float(values["class_scores"][class_index])
                + 0.20 * grounding_score
            )
            localized_predictions.append(
                {
                    "image": case.stem,
                    "class_id": int(class_id),
                    "score": localized_score,
                    "box": box,
                }
            )

        noisy_or = 1.0 - float(
            np.prod(1.0 - np.clip(values["class_scores"], 0.0, 1.0))
        )
        row = {
            "image_stem": case.stem,
            "split": case.split,
            "true_diameter_mm": true_diameter,
            "pred_diameter_mm": values["diameter"],
            "strong_any_true": int(class_labels.max()),
            "strong_any_score": values["strong_score"],
            "strong_any_noisy_or_score": noisy_or,
        }
        for class_index, class_name in enumerate(class_names):
            row[f"{class_name}_true"] = int(class_labels[class_index])
            row[f"{class_name}_classification_score"] = float(
                values["class_scores"][class_index]
            )
        per_case.append(row)

    detection_metrics = average_precision_detections(
        gt_by_image, localized_predictions, np.arange(0.5, 1.0, 0.05)
    )
    red_metrics = compute_v2_red_metrics(
        np.asarray(true_masks),
        np.asarray(red_probability_masks),
        true_diameters,
        pred_diameters,
        threshold=mask_threshold,
    )
    red_metrics["masked_view_grounding_dice"] = (
        positive_grounding_dice(
            np.asarray(masked_red_probabilities),
            np.asarray(masked_red_targets),
            threshold=mask_threshold,
        )
        if masked_red_probabilities
        else 0.0
    )
    strong_metrics = compute_v2_strong_metrics(
        np.asarray(labels),
        np.asarray(class_score_rows),
        np.asarray(strong_score_rows),
        detection_metrics=detection_metrics,
    )
    diagnostics = {
        "mask_threshold": float(mask_threshold),
        "red_fusion_active_views": (
            int(red_fusion.active_views) if red_fusion is not None else None
        ),
    }
    class_grounding_dice = []
    positive_patch_true = []
    positive_patch_pred = []
    empty_maps = []
    for probabilities, targets in zip(
        rare_grounding_probabilities, rare_grounding_targets
    ):
        probability_array = np.asarray(probabilities, dtype=np.float32)
        target_array = np.asarray(targets, dtype=np.float32)
        class_grounding_dice.append(
            positive_grounding_dice(probability_array, target_array)
            if probability_array.size
            else 0.0
        )
        truth = target_array >= 0.5
        prediction = probability_array >= 0.5
        if truth.any():
            positive_patch_true.append(truth[truth])
            positive_patch_pred.append(prediction[truth])
        if prediction.ndim == 3:
            empty_maps.extend(~prediction.any(axis=(1, 2)))
    diagnostics.update(
        blister_grounding_dice=float(class_grounding_dice[0]),
        necrosis_grounding_dice=float(class_grounding_dice[1]),
        double_ring_grounding_dice=float(class_grounding_dice[2]),
        rare_grounding_dice=float(np.mean(class_grounding_dice)),
        positive_patch_recall=(
            float(np.mean(np.concatenate(positive_patch_pred)))
            if positive_patch_pred
            else 0.0
        ),
        empty_grounding_prediction_rate=(
            float(np.mean(empty_maps)) if empty_maps else 0.0
        ),
    )
    try:
        from scipy.stats import spearmanr

        classification_values = np.asarray(class_score_rows).reshape(-1)
        grounding_values = np.asarray(grounding_case_scores).reshape(-1)
        correlation = (
            spearmanr(classification_values, grounding_values).statistic
            if np.std(classification_values) > 0 and np.std(grounding_values) > 0
            else float("nan")
        )
        diagnostics["classification_grounding_spearman"] = float(correlation)
    except (ImportError, ValueError):
        diagnostics["classification_grounding_spearman"] = float("nan")
    for class_id, class_name in zip((2, 3, 4), class_names):
        diagnostics[f"{class_name}_positive_count"] = detection_metrics.get(
            f"class_{class_id}_positive_count", 0
        )
        diagnostics[f"{class_name}_candidate_count"] = detection_metrics.get(
            f"class_{class_id}_candidate_count", 0
        )
        diagnostics[f"{class_name}_tp50"] = detection_metrics.get(
            f"class_{class_id}_tp50", 0
        )
    return red_metrics, strong_metrics, per_case, diagnostics
