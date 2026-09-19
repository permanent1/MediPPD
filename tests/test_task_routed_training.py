from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from medippd_gvlm.task_routed_losses import compute_task_routed_losses
from medippd_gvlm.task_routed_model import TaskRoutedConfig, TaskRoutedMediPPD
from medippd_gvlm.task_routed_training import (
    _stage_one_parameters,
    checkpoint_selection_objectives,
    freeze_diameter_head,
    make_class_aware_sampler,
    make_training_sampler,
    task_routed_training_objectives,
    train_task_routed_model,
)


class TinyTaskRoutedDataset(Dataset):
    def __init__(self):
        self.labels = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(100 + index)
        labels = torch.from_numpy(self.labels[index])
        targets = torch.zeros(3, 4, 2, 2)
        targets[:, 0, index % 2, index // 2] = 1.0
        for class_index in range(3):
            if labels[class_index]:
                targets[:, class_index + 1, index % 2, index // 2] = 1.0
        return {
            "patches": torch.randn(3, 4, 3, generator=generator),
            "morphometry": torch.randn(7, generator=generator),
            "diameter_baseline": torch.tensor(10.0 + index),
            "clinical": torch.randn(2, generator=generator),
            "grounding_targets": targets,
            "diameter_target": torch.tensor(10.5 + index),
            "dermal_target": labels,
            "reaction_target": torch.tensor(index % 4),
            "index": torch.tensor(index),
        }


def semantic_initializers():
    return np.arange(20, dtype=np.float32).reshape(4, 5) / 20.0


def test_uniform_sampler_returns_none_and_weighted_sampler_is_reproducible():
    labels = TinyTaskRoutedDataset().labels
    assert make_training_sampler(labels, "uniform", seed=7) is None
    first = list(make_training_sampler(labels, "weighted", seed=7))
    second = list(make_training_sampler(labels, "weighted", seed=7))
    assert first == second

def test_class_aware_sampler_assigns_more_weight_to_rare_positive_cases():
    labels = np.asarray(
        [[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32
    )

    sampler = make_class_aware_sampler(labels, seed=42)

    assert sampler.weights[2] > sampler.weights[0]
    assert list(iter(sampler)) == list(iter(make_class_aware_sampler(labels, seed=42)))


def test_training_objectives_omit_disabled_b1_and_b2_branches():
    physical_only = TaskRoutedConfig(
        "B1_physical_only",
        train_case=False,
        train_grounding=False,
        use_task_consistency=False,
    )
    global_only = TaskRoutedConfig(
        "B2_global_only",
        train_diameter=False,
        train_grounding=False,
        use_roi_classification=False,
        use_task_consistency=False,
    )

    assert task_routed_training_objectives(physical_only) == (("diameter",), ())
    assert task_routed_training_objectives(global_only) == (
        (),
        ("class_bce", "strong_any", "reaction"),
    )


def test_checkpoint_selection_is_task_specific_in_each_training_stage():
    assert checkpoint_selection_objectives(TaskRoutedConfig.full()) == (
        ("diameter", "red_ground"),
        ("class_bce", "strong_any"),
    )


def test_b8_trains_shared_visual_fusion_during_stage_one():
    model = TaskRoutedMediPPD(
        patch_dim=3,
        semantic_initializers=semantic_initializers(),
        clinical_dim=2,
        d_model=8,
        grid_size=2,
    )
    shared_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("shared_")
    }

    selected_ids = {
        id(parameter)
        for parameter in _stage_one_parameters(
            model,
            TaskRoutedConfig("B8", use_task_routing=False),
        )
    }

    assert shared_ids
    assert shared_ids <= selected_ids


def test_stage_two_optimizer_step_leaves_all_diameter_parameters_bitwise_unchanged():
    dataset = TinyTaskRoutedDataset()
    model = TaskRoutedMediPPD(
        patch_dim=3,
        semantic_initializers=semantic_initializers(),
        clinical_dim=2,
        d_model=8,
        grid_size=2,
    )
    freeze_diameter_head(model)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("diameter_")
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-2
    )
    batch = {key: value.unsqueeze(0) for key, value in dataset[1].items()}
    losses = compute_task_routed_losses(model(batch, TaskRoutedConfig.full()), batch, TaskRoutedConfig.full())

    optimizer.zero_grad(set_to_none=True)
    stage_two_loss = sum(
        losses[key]
        for key in ("class_bce", "strong_any", "reaction", "class_ground", "consistency")
    )
    stage_two_loss.backward()
    optimizer.step()

    assert before
    for name, parameter in model.named_parameters():
        if name in before:
            assert torch.equal(parameter, before[name]), name


def test_tiny_cpu_training_writes_and_restores_a_finite_checkpoint(tmp_path: Path):
    dataset = TinyTaskRoutedDataset()
    checkpoint = tmp_path / "task-routed.pt"

    model, elapsed = train_task_routed_model(
        dataset,
        dataset,
        semantic_initializers(),
        TaskRoutedConfig.full(),
        checkpoint,
        stage_one_epochs=1,
        stage_two_epochs=1,
        batch_size=2,
        d_model=8,
        device="cpu",
        seed=7,
    )

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    batch = {key: value.unsqueeze(0) for key, value in dataset[0].items()}
    with torch.no_grad():
        output = model(batch, TaskRoutedConfig.full())
    assert elapsed >= 0.0
    assert {"model", "config", "train_calibration", "best_validation_loss"} <= state.keys()
    assert np.isfinite(state["best_validation_loss"])
    assert torch.isfinite(output.diameter_mm).all()
    assert torch.isfinite(output.class_logits).all()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), state["model"][name].cpu()), name
