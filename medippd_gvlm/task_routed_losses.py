"""Task-aligned losses for the isolated MediPPD V2 model."""

import math
from typing import Dict

import torch
import torch.nn.functional as functional

from .task_routed_model import TaskRoutedConfig, TaskRoutedOutput


def _dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    intersection = (probabilities * targets).sum(dim=(-1, -2))
    denominator = probabilities.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def sparse_grounding_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Balance sparse spatial supervision without rewarding empty-map collapse."""

    return grounding_loss(logits, targets, "dice_balanced_bce")


def grounding_loss(
    logits: torch.Tensor, targets: torch.Tensor, strategy: str
) -> torch.Tensor:
    """Compute one explicitly named sparse-grounding objective."""

    targets = targets.to(device=logits.device, dtype=logits.dtype)
    flat_logits = logits.reshape(-1, *logits.shape[-2:])
    flat_targets = targets.reshape_as(flat_logits)
    positive_maps = flat_targets.sum(dim=(-1, -2)) > 0
    if positive_maps.any():
        positive_dice = _dice_loss(
            flat_logits[positive_maps], flat_targets[positive_maps]
        )
    else:
        positive_dice = logits.sum() * 0.0

    positive_count = flat_targets.sum()
    negative_count = flat_targets.numel() - positive_count
    if positive_count.item() > 0:
        pos_weight = (negative_count / positive_count).clamp(1.0, 25.0)
    else:
        pos_weight = torch.ones((), device=logits.device, dtype=logits.dtype)
    balanced_bce = functional.binary_cross_entropy_with_logits(
        flat_logits, flat_targets, pos_weight=pos_weight
    )
    if strategy == "bce":
        return functional.binary_cross_entropy_with_logits(flat_logits, flat_targets)
    if strategy == "balanced_bce":
        return balanced_bce
    if strategy == "focal":
        unit_weight = torch.ones((), device=logits.device, dtype=logits.dtype)
        return _focal_bce_with_logits(flat_logits, flat_targets, unit_weight)
    if strategy == "positive_dice":
        return positive_dice
    if strategy == "dice_balanced_bce":
        return positive_dice + balanced_bce
    raise ValueError(f"unknown grounding loss strategy: {strategy}")


def _class_targets(batch, reference: torch.Tensor) -> torch.Tensor:
    targets = batch.get("class_target", batch.get("dermal_target"))
    if targets is None:
        raise KeyError("batch must contain class_target or dermal_target")
    return targets.to(device=reference.device, dtype=reference.dtype)


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    elementwise = functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    probabilities = torch.sigmoid(logits)
    target_probability = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    return (elementwise * (1.0 - target_probability).pow(gamma)).mean()


def _classification_loss(logits, targets, pos_weight, strategy):
    unit_weight = torch.ones_like(pos_weight)
    if strategy == "bce":
        return functional.binary_cross_entropy_with_logits(logits, targets)
    if strategy == "weighted_bce":
        return functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )
    if strategy == "focal_unweighted":
        return _focal_bce_with_logits(logits, targets, unit_weight)
    if strategy == "class_balanced_focal":
        return _focal_bce_with_logits(logits, targets, torch.sqrt(pos_weight))
    if strategy == "focal":
        return _focal_bce_with_logits(logits, targets, pos_weight)
    raise ValueError(f"unknown class loss strategy: {strategy}")


def _topk_class_evidence(
    class_grounding_logits: torch.Tensor, active_views: int, topk_ratio: float = 0.10
) -> torch.Tensor:
    probabilities = torch.sigmoid(class_grounding_logits[:, :active_views])
    # [batch, view, class, height, width] -> [batch, class, evidence]
    flattened = probabilities.permute(0, 2, 1, 3, 4).flatten(start_dim=2)
    if not 0 < float(topk_ratio) <= 1:
        raise ValueError("topk_ratio must lie in (0, 1]")
    topk = max(1, int(math.ceil(flattened.shape[-1] * float(topk_ratio))))
    return flattened.topk(topk, dim=-1).values.mean(dim=-1)


def compute_task_routed_losses(
    output: TaskRoutedOutput, batch, config: TaskRoutedConfig
) -> Dict[str, torch.Tensor]:
    """Compute V2 objectives with class weights supplied from train data only."""

    active_views = tuple(
        index
        for index in config.active_views
        if config.use_masked_grounding or index != 2
    )
    if not active_views:
        raise ValueError("configuration disables every loss view")
    red_indices = active_views
    class_indices = active_views
    grounding_targets = batch["grounding_targets"].to(
        device=output.grounding_logits.device, dtype=output.grounding_logits.dtype
    )

    diameter = functional.smooth_l1_loss(
        output.diameter_mm,
        batch["diameter_target"].to(
            device=output.diameter_mm.device, dtype=output.diameter_mm.dtype
        ),
    )
    red_ground = grounding_loss(
        output.grounding_logits[:, red_indices, 0],
        grounding_targets[:, red_indices, 0],
        config.grounding_loss,
    )

    class_targets = _class_targets(batch, output.class_logits)
    class_pos_weight = batch.get("class_pos_weight")
    if class_pos_weight is None:
        class_pos_weight = torch.ones(
            output.class_logits.shape[-1],
            device=output.class_logits.device,
            dtype=output.class_logits.dtype,
        )
    else:
        class_pos_weight = class_pos_weight.to(
            device=output.class_logits.device, dtype=output.class_logits.dtype
        )
    class_bce = _classification_loss(
        output.class_logits, class_targets, class_pos_weight, config.class_loss
    )

    strong_target = class_targets.amax(dim=1)
    strong_pos_weight = batch.get("strong_pos_weight")
    if strong_pos_weight is None:
        strong_pos_weight = torch.ones(
            (), device=output.strong_any_logit.device, dtype=output.strong_any_logit.dtype
        )
    else:
        strong_pos_weight = strong_pos_weight.to(
            device=output.strong_any_logit.device, dtype=output.strong_any_logit.dtype
        )
    strong_weighted = config.class_loss in {
        "weighted_bce",
        "class_balanced_focal",
        "focal",
    }
    effective_strong_weight = (
        torch.sqrt(strong_pos_weight)
        if config.class_loss == "class_balanced_focal"
        else strong_pos_weight
    )
    strong_any = functional.binary_cross_entropy_with_logits(
        output.strong_any_logit,
        strong_target,
        pos_weight=(effective_strong_weight if strong_weighted else None),
    )
    reaction = functional.cross_entropy(
        output.reaction_logits,
        batch["reaction_target"].to(device=output.reaction_logits.device).long(),
    )

    class_ground = grounding_loss(
        output.class_grounding_logits[:, class_indices],
        grounding_targets[:, class_indices, 1:],
        config.grounding_loss,
    )
    class_probabilities = torch.sigmoid(output.class_logits)
    grounded_evidence = _topk_class_evidence(
        output.class_grounding_logits[:, class_indices], len(class_indices), config.topk_ratio
    )
    consistency = functional.smooth_l1_loss(class_probabilities, grounded_evidence)
    if not config.use_task_consistency:
        consistency = consistency * 0.0
    consistency = consistency * float(config.consistency_weight)

    total = (
        diameter
        + red_ground
        + class_bce
        + strong_any
        + reaction
        + class_ground
        + consistency
    )
    return {
        "total": total,
        "diameter": diameter,
        "red_ground": red_ground,
        "class_bce": class_bce,
        "strong_any": strong_any,
        "reaction": reaction,
        "class_ground": class_ground,
        "consistency": consistency,
    }
