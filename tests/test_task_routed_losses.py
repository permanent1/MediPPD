from dataclasses import replace

import torch

from medippd_gvlm.task_routed_losses import (
    compute_task_routed_losses,
    grounding_loss,
    sparse_grounding_loss,
)
from medippd_gvlm.task_routed_model import TaskRoutedConfig, TaskRoutedOutput


def make_output(class_logit=5.0, grounding_logit=5.0, strong_any_logit=0.0):
    batch_size, grid_size, feature_dim = 1, 2, 4
    class_logits = torch.full((batch_size, 3), -5.0)
    class_logits[:, 0] = class_logit
    class_grounding_logits = torch.full(
        (batch_size, 3, 3, grid_size, grid_size), -5.0
    )
    class_grounding_logits[:, :, 0] = grounding_logit
    grounding_logits = torch.cat(
        [
            torch.zeros(batch_size, 3, 1, grid_size, grid_size),
            class_grounding_logits,
        ],
        dim=2,
    )
    return TaskRoutedOutput(
        grounding_logits=grounding_logits,
        diameter_mm=torch.tensor([10.0]),
        class_logits=class_logits,
        strong_any_logit=torch.tensor([strong_any_logit]),
        reaction_logits=torch.zeros(batch_size, 4),
        red_mask_logits=torch.zeros(batch_size, grid_size, grid_size),
        class_grounding_logits=class_grounding_logits,
        diameter_delta_mm=torch.zeros(batch_size),
        diameter_features=torch.zeros(batch_size, feature_dim),
        case_features=torch.zeros(batch_size, feature_dim),
        red_roi_features=torch.zeros(batch_size, feature_dim),
        clinical_residual=None,
        size_prior_residual=None,
    )


def make_batch():
    return {
        "grounding_targets": torch.zeros(1, 3, 4, 2, 2),
        "diameter_target": torch.tensor([10.0]),
        "dermal_target": torch.tensor([[1.0, 0.0, 0.0]]),
        "reaction_target": torch.tensor([0]),
        "class_pos_weight": torch.ones(3),
        "strong_pos_weight": torch.tensor(1.0),
    }


def test_consistency_rewards_agreement_between_case_score_and_corresponding_map():
    config = TaskRoutedConfig.full()
    agreed = compute_task_routed_losses(
        make_output(class_logit=5.0, grounding_logit=5.0), make_batch(), config
    )
    disagreed = compute_task_routed_losses(
        make_output(class_logit=5.0, grounding_logit=-5.0), make_batch(), config
    )

    assert agreed["consistency"] < disagreed["consistency"]


def test_strong_any_loss_uses_the_dedicated_strong_any_logit():
    config = TaskRoutedConfig.full()
    low = compute_task_routed_losses(
        make_output(strong_any_logit=-5.0), make_batch(), config
    )
    high = compute_task_routed_losses(
        make_output(strong_any_logit=5.0), make_batch(), config
    )

    assert high["strong_any"] < low["strong_any"]
    torch.testing.assert_close(high["class_bce"], low["class_bce"])


def test_task_consistency_flag_removes_consistency_from_the_objective():
    enabled = compute_task_routed_losses(
        make_output(class_logit=5.0, grounding_logit=-5.0),
        make_batch(),
        TaskRoutedConfig.full(),
    )
    disabled = compute_task_routed_losses(
        make_output(class_logit=5.0, grounding_logit=-5.0),
        make_batch(),
        replace(TaskRoutedConfig.full(), use_task_consistency=False),
    )

    assert enabled["consistency"] > 0
    assert disabled["consistency"].item() == 0.0
    torch.testing.assert_close(
        enabled["total"] - enabled["consistency"], disabled["total"]
    )


def test_sparse_grounding_loss_does_not_reward_all_negative_collapse():
    targets = torch.zeros(8, 4, 4)
    targets[0, 1:3, 1:3] = 1.0
    collapsed = torch.full_like(targets, -20.0)
    localized = torch.full_like(targets, -8.0)
    localized[0, 1:3, 1:3] = 8.0

    assert sparse_grounding_loss(localized, targets) < sparse_grounding_loss(
        collapsed, targets
    )


def test_sparse_grounding_loss_is_finite_for_an_all_empty_batch():
    logits = torch.zeros(4, 3, 3)
    targets = torch.zeros_like(logits)

    assert torch.isfinite(sparse_grounding_loss(logits, targets))


def test_all_grounding_loss_strategies_are_finite():
    logits = torch.zeros(1, 2, 2)
    targets = torch.zeros_like(logits)
    targets[:, 0, 0] = 1.0
    values = {}
    for strategy in (
        "bce",
        "balanced_bce",
        "focal",
        "positive_dice",
        "dice_balanced_bce",
    ):
        values[strategy] = grounding_loss(logits, targets, strategy)
        assert torch.isfinite(values[strategy]), strategy
    assert values["bce"] != values["positive_dice"]
    assert values["dice_balanced_bce"] > values["balanced_bce"]


def test_weighted_class_bce_uses_training_positive_weights():
    batch = make_batch()
    batch["class_pos_weight"] = torch.tensor([5.0, 1.0, 1.0])
    output = make_output(class_logit=-1.0)
    unweighted = compute_task_routed_losses(
        output, batch, replace(TaskRoutedConfig.full(), class_loss="bce")
    )
    weighted = compute_task_routed_losses(
        output, batch, replace(TaskRoutedConfig.full(), class_loss="weighted_bce")
    )
    assert weighted["class_bce"] > unweighted["class_bce"]


def test_masked_view_contributes_to_strong_class_grounding_loss():
    batch = make_batch()
    batch["grounding_targets"][:, 2, 1, :, :] = 1.0
    matched = make_output(grounding_logit=-5.0)
    mismatched = make_output(grounding_logit=-5.0)
    matched.class_grounding_logits[:, 2, 0, :, :] = 5.0

    matched_losses = compute_task_routed_losses(
        matched, batch, TaskRoutedConfig.full()
    )
    mismatched_losses = compute_task_routed_losses(
        mismatched, batch, TaskRoutedConfig.full()
    )

    assert matched_losses["class_ground"] < mismatched_losses["class_ground"]
