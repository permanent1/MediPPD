import math
from dataclasses import asdict, dataclass
from typing import Dict

import cv2
import numpy as np

from .config import CAP_DIAMETER_MM


@dataclass(frozen=True)
class Morphometry:
    diameter_mm: float
    area_mm2: float
    major_mm: float
    minor_mm: float
    aspect_ratio: float
    compactness: float
    confidence: float
    mm_per_px: float
    scale_valid: bool
    lesion_valid: bool

    @property
    def mean_axis_mm(self) -> float:
        return (self.major_mm + self.minor_mm) / 2.0

    @property
    def equivalent_diameter_mm(self) -> float:
        return 2.0 * math.sqrt(self.area_mm2 / math.pi) if self.area_mm2 > 0 else 0.0

    def vector(self) -> np.ndarray:
        return np.asarray(
            [self.diameter_mm, self.area_mm2, self.major_mm, self.minor_mm, self.aspect_ratio, self.compactness, self.confidence],
            dtype=np.float32,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def measure_masks(cap_mask: np.ndarray, lesion_mask: np.ndarray, confidence: float, cap_diameter_mm: float = CAP_DIAMETER_MM) -> Morphometry:
    cap_area = float(np.count_nonzero(cap_mask))
    lesion_area = float(np.count_nonzero(lesion_mask))
    scale_valid = cap_area > 0
    lesion_valid = lesion_area > 0
    cap_equivalent_px = math.sqrt(4.0 * cap_area / math.pi) if scale_valid else 0.0
    mm_per_px = cap_diameter_mm / cap_equivalent_px if cap_equivalent_px > 0 else 0.0

    major_px = minor_px = compactness = aspect_ratio = 0.0
    contour = _largest_contour(lesion_mask)
    if contour is not None:
        (_, _), (width, height), _ = cv2.minAreaRect(contour)
        major_px, minor_px = max(float(width), float(height)), min(float(width), float(height))
        perimeter = float(cv2.arcLength(contour, True))
        contour_area = float(cv2.contourArea(contour))
        compactness = 4.0 * math.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0.0
        aspect_ratio = major_px / minor_px if minor_px > 0 else 0.0

    major_mm = major_px * mm_per_px if scale_valid else 0.0
    minor_mm = minor_px * mm_per_px if scale_valid else 0.0
    return Morphometry(
        diameter_mm=major_mm,
        area_mm2=lesion_area * mm_per_px * mm_per_px if scale_valid else 0.0,
        major_mm=major_mm,
        minor_mm=minor_mm,
        aspect_ratio=aspect_ratio,
        compactness=float(np.clip(compactness, 0.0, 1.0)),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        mm_per_px=mm_per_px,
        scale_valid=scale_valid,
        lesion_valid=lesion_valid,
    )
