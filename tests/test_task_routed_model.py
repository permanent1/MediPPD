from dataclasses import replace

import numpy as np
import torch

from medippd_gvlm.task_routed_model import TaskRoutedConfig, TaskRoutedMediPPD


def make_task_routed_fixture(batch_size=2):
    torch.manual_seed(7)
    semantic_init = np.arange(4 * 10, dtype=np.float32).reshape(4, 10) / 40.0
    model = TaskRoutedMediPPD(
        patch_dim=8,
        semantic_initializers=semantic_init,
        clinical_dim=6,
        d_model=16,
        grid_size=12,
        num_reaction_classes=4,
    ).eval()
    batch = {
        "patches": torch.randn(batch_size, 3, 12 * 12, 8),
        "morphometry": torch.randn(batch_size, 7),
        "diameter_baseline": torch.tensor([12.0, 18.0]),
        "clinical": torch.randn(batch_size, 6),
    }
    return model, batch


def clone_batch(batch):
    return {key: value.clone() for key, value in batch.items()}


def test_strong_branch_is_invariant_to_morphometry_when_size_prior_disabled():
    model, batch = make_task_routed_fixture()
    config = TaskRoutedConfig.full()
    assert not config.use_size_prior

    with torch.no_grad():
        first = model(batch, config).strong_any_logit
        changed = clone_batch(batch)
        changed["morphometry"] += 100.0
        changed["diameter_baseline"] += 100.0
        second = model(changed, config).strong_any_logit

    torch.testing.assert_close(first, second)


def test_masked_view_changes_grounding_but_not_case_logits():
    model, batch = make_task_routed_fixture()
    config = TaskRoutedConfig.full()

    with torch.no_grad():
        first = model(batch, config)
        changed = clone_batch(batch)
        changed["patches"][:, 2] += torch.linspace(1.0, 10.0, 8)
        second = model(changed, config)

    torch.testing.assert_close(first.class_logits, second.class_logits)
    torch.testing.assert_close(first.strong_any_logit, second.strong_any_logit)
    torch.testing.assert_close(first.reaction_logits, second.reaction_logits)
    assert not torch.allclose(first.grounding_logits[:, 2], second.grounding_logits[:, 2])


def test_direct_clinical_residual_changes_case_logits():
    model, batch = make_task_routed_fixture()
    config = TaskRoutedConfig.full()

    with torch.no_grad():
        first = model(batch, config)
        changed = clone_batch(batch)
        changed["clinical"] += torch.linspace(2.0, 12.0, 6)
        second = model(changed, config)

    assert not torch.allclose(first.class_logits, second.class_logits)
    assert not torch.allclose(first.strong_any_logit, second.strong_any_logit)
    assert not torch.allclose(first.reaction_logits, second.reaction_logits)


def test_diameter_stays_within_bounded_residual_of_physical_baseline():
    model, batch = make_task_routed_fixture()
    config = TaskRoutedConfig.full()

    with torch.no_grad():
        model.diameter_delta.weight.fill_(100.0)
        model.diameter_delta.bias.fill_(100.0)
        output = model(batch, config)

    difference = (output.diameter_mm - batch["diameter_baseline"]).abs()
    assert torch.all(difference <= config.max_delta_mm + 1e-6)


def test_task_routed_output_shapes_are_explicit():
    model, batch = make_task_routed_fixture()

    with torch.no_grad():
        output = model(batch, TaskRoutedConfig.full())

    assert output.grounding_logits.shape == (2, 3, 4, 12, 12)
    assert output.diameter_mm.shape == (2,)
    assert output.class_logits.shape == (2, 3)
    assert output.strong_any_logit.shape == (2,)
    assert output.reaction_logits.shape == (2, 4)
    assert output.red_mask_logits.shape == (2, 12, 12)


def test_disabling_task_routing_uses_shared_fusion_for_both_primary_tasks():
    model, batch = make_task_routed_fixture()

    with torch.no_grad():
        model.diameter_delta.weight.zero_()
        model.diameter_delta.weight[0, 0] = 1.0
        routed = model(batch, TaskRoutedConfig.full())
        shared = model(
            batch,
            replace(TaskRoutedConfig.full(), use_task_routing=False),
        )

    assert not torch.allclose(routed.diameter_mm, shared.diameter_mm)
    assert not torch.allclose(routed.strong_any_logit, shared.strong_any_logit)


def test_active_views_mask_inactive_grounding_logits():
    model, batch = make_task_routed_fixture()
    config = replace(TaskRoutedConfig.full(), active_views=(2,))

    with torch.no_grad():
        output = model(batch, config)

    assert torch.all(output.grounding_logits[:, :2] == -12.0)
    assert not torch.all(output.grounding_logits[:, 2] == -12.0)


def test_full_morphometry_case_input_changes_case_logits():
    model, batch = make_task_routed_fixture()
    config = replace(
        TaskRoutedConfig.full(), use_direct_clinical=False, case_morphometry="full"
    )

    with torch.no_grad():
        first = model(batch, config).strong_any_logit
        changed = clone_batch(batch)
        changed["morphometry"][:, 0] += 10.0
        second = model(changed, config).strong_any_logit

    assert not torch.allclose(first, second)
