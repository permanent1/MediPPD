import numpy as np
import pytest
import torch

from medippd_gvlm.red_mask_fusion import (
    RedMaskFusionHead,
    open_mask_prior,
    prepare_fusion_inputs,
    project_roi_grounding,
    restore_square_padded_map,
)


def test_roi_grounding_is_projected_back_into_global_coordinates():
    roi = np.ones((2, 2), dtype=np.float32)

    projected = project_roi_grounding(roi, [0.25, 0.25, 0.75, 0.75], (8, 8))

    assert projected.shape == (8, 8)
    assert projected[2:6, 2:6].mean() == pytest.approx(1.0)
    assert projected[:2].max() == pytest.approx(0.0)


def test_open_mask_prior_removes_isolated_noise_but_preserves_region():
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[2:7, 2:7] = 1
    mask[0, 0] = 1

    opened = open_mask_prior(mask)

    assert opened[0, 0] == 0
    assert opened[4, 4] == 1


def test_zero_initialized_fusion_head_starts_from_yolo_prior():
    head = RedMaskFusionHead(active_views=3)
    vlm = torch.rand(2, 3, 8, 8)
    prior = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1).expand(2, 1, 8, 8)

    prediction = torch.sigmoid(head(vlm, prior)) >= 0.5

    torch.testing.assert_close(prediction, prior.bool())


def test_roi_probability_has_no_evidence_outside_the_crop():
    logits = np.zeros((3, 2, 2), dtype=np.float32)
    logits[1] = 10.0

    probabilities, _ = prepare_fusion_inputs(
        np.zeros((8, 8), dtype=np.uint8),
        logits,
        [0.25, 0.25, 0.75, 0.75],
        image_size=8,
    )

    assert probabilities[1, 0, 0] < 1e-4
    assert probabilities[1, 3, 3] > 0.99


def test_square_padded_heatmap_is_unpadded_before_image_projection():
    square_map = np.zeros((8, 8), dtype=np.float32)
    square_map[0, 2] = 1.0

    restored = restore_square_padded_map(square_map, (8, 4))

    assert restored.shape == (8, 4)
    assert restored[0, 0] > 0.9
    assert restored[:, -1].max() == pytest.approx(0.0)
