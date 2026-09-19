import cv2
import numpy as np
import pytest

from medippd_gvlm.metrology import measure_masks


def test_cap_scale_converts_lesion_pixels_to_physical_area():
    cap = np.zeros((80, 80), dtype=np.uint8)
    lesion = np.zeros_like(cap)
    cv2.circle(cap, (20, 20), 10, 1, -1)
    cv2.rectangle(lesion, (40, 30), (59, 39), 1, -1)

    result = measure_masks(cap, lesion, confidence=0.8)
    expected_cap_equivalent_px = np.sqrt(4.0 * cap.sum() / np.pi)
    expected_scale = 30.0 / expected_cap_equivalent_px

    assert result.scale_valid is True
    assert result.mm_per_px == pytest.approx(expected_scale)
    assert result.area_mm2 == pytest.approx(float(lesion.sum()) * expected_scale**2)
    assert result.confidence == pytest.approx(0.8)
    assert result.major_mm >= result.minor_mm > 0


def test_missing_cap_is_explicit_and_never_invents_millimetres():
    lesion = np.zeros((32, 32), dtype=np.uint8)
    lesion[8:20, 6:24] = 1

    result = measure_masks(np.zeros_like(lesion), lesion, confidence=0.5)

    assert result.scale_valid is False
    assert result.lesion_valid is True
    assert result.mm_per_px == 0.0
    assert result.diameter_mm == 0.0


def test_empty_lesion_is_explicit():
    cap = np.zeros((32, 32), dtype=np.uint8)
    cv2.circle(cap, (10, 10), 5, 1, -1)

    result = measure_masks(cap, np.zeros_like(cap), confidence=0.1)

    assert result.scale_valid is True
    assert result.lesion_valid is False
    assert result.area_mm2 == 0.0


def test_morphometry_exposes_mean_and_equivalent_diameter_without_changing_vector():
    cap = np.zeros((80, 80), dtype=np.uint8)
    lesion = np.zeros_like(cap)
    cv2.circle(cap, (20, 20), 10, 1, -1)
    cv2.rectangle(lesion, (40, 30), (59, 39), 1, -1)

    result = measure_masks(cap, lesion, confidence=0.8, cap_diameter_mm=30.0)

    assert result.mean_axis_mm == pytest.approx(
        (result.major_mm + result.minor_mm) / 2.0
    )
    assert result.equivalent_diameter_mm == pytest.approx(
        2.0 * np.sqrt(result.area_mm2 / np.pi)
    )
    assert result.vector().shape == (7,)
