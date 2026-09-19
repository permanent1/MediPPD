from pathlib import Path

import cv2
import numpy as np
import pytest

from medippd_gvlm.vlm_cache import (
    _downsample_binary_mask_preserve_any,
    _grounding_targets,
    _ring_boundary_target,
    build_three_views,
    cache_fingerprint,
    cache_one_case,
    pad_to_square,
    validate_cache_manifest,
)


class FakeExtractor:
    model_id = "fake-llava"
    grid_size = 4

    def extract(self, images):
        assert len(images) == 3
        return np.stack(
            [np.full((16, 8), fill_value=index + 1, dtype=np.float32) for index in range(3)]
        )


def test_cache_writes_three_views_and_four_grounding_targets(tmp_path: Path):
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[8:24, 10:30] = (80, 120, 180)
    image_path = tmp_path / "case.jpg"
    cv2.imwrite(str(image_path), image)
    masks = np.zeros((4, 32, 40), dtype=np.uint8)
    masks[0, 8:24, 10:30] = 1
    masks[1, 12:16, 14:18] = 1

    output = cache_one_case("case", image_path, masks[0], masks, tmp_path / "cache", FakeExtractor())
    cached = np.load(output)

    assert cached["patches"].shape == (3, 16, 8)
    assert cached["targets"].shape == (3, 4, 4, 4)
    assert cached["view_boxes"].shape == (3, 4)
    assert cached["patches"][0, 0, 0] == pytest.approx(1.0)
    assert cached["patches"][2, 0, 0] == pytest.approx(3.0)


def test_cache_fingerprint_changes_with_model_or_split():
    first = cache_fingerprint("model-a", ["a", "b"], 12)
    assert first != cache_fingerprint("model-b", ["a", "b"], 12)
    assert first != cache_fingerprint("model-a", ["a", "c"], 12)


def test_cache_fingerprint_changes_with_roi_margin():
    first = cache_fingerprint("model-a", ["a"], 12, roi_margin=0.15)
    second = cache_fingerprint("model-a", ["a"], 12, roi_margin=0.30)
    assert first != second


def test_roi_margin_changes_roi_box_but_not_global_or_masked_boxes():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[15:25, 20:40] = 1
    _, tight_boxes, _ = build_three_views(image, mask, margin=0.0)
    _, wide_boxes, _ = build_three_views(image, mask, margin=0.30)
    np.testing.assert_array_equal(tight_boxes[[0, 2]], wide_boxes[[0, 2]])
    assert wide_boxes[1, 0] < tight_boxes[1, 0]
    assert wide_boxes[1, 2] > tight_boxes[1, 2]


def test_stale_manifest_is_rejected(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"fingerprint": "old"}', encoding="utf-8")

    with pytest.raises(ValueError, match="stale"):
        validate_cache_manifest(manifest, "new")


def test_pad_to_square_centers_portrait_image_without_cropping_content():
    image = np.zeros((8, 4), dtype=np.uint8)
    image[0, 0] = 255
    padded = pad_to_square(image, fill_value=0)

    assert padded.shape == (8, 8)
    assert padded[0, 2] == 255
    assert padded.sum() == 255


def test_tiny_lesion_survives_grounding_target_downsampling():
    mask = np.zeros((120, 240), dtype=np.uint8)
    mask[3, 237] = 1

    target = _downsample_binary_mask_preserve_any(mask, grid_size=12)

    assert target.shape == (12, 12)
    assert target.sum() >= 1


def test_grounding_targets_apply_same_square_geometry_to_global_view():
    masks = np.zeros((4, 8, 4), dtype=np.uint8)
    masks[0, 0, 0] = 1

    targets = _grounding_targets(masks, (0, 0, 4, 8), grid_size=8)

    # A portrait image is padded two pixels on both horizontal sides before
    # LLaVA sees it, so its top-left source pixel lands at column two.
    assert targets[0, 0, 0, 2] == 1
    assert targets[0, 0, 0, 0] == 0


def test_double_ring_target_is_annular_instead_of_a_filled_box():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[12:52, 10:54] = 1

    ring = _ring_boundary_target(mask)

    assert ring[32, 32] == 0
    assert ring[12, 32] == 1
    assert ring[32, 10] == 1
    assert 0 < ring.sum() < mask.sum() * 0.6


def test_cache_fingerprint_tracks_grounding_preprocessing_version():
    first = cache_fingerprint("model-a", ["a"], 12, preprocessing_version="v1")
    second = cache_fingerprint("model-a", ["a"], 12, preprocessing_version="v2")
    assert first != second
