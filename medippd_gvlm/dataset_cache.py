"""Dataset adapter for cached MediPPD visual and prediction features."""

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import CaseRecord, ClinicalEncoderSpec


REACTION_LABELS = {"阴性": 0, "一般阳性": 1, "中度阳性": 2, "强阳性": 3}
MORPH_SCALES = np.asarray(
    [50.0, 2500.0, 50.0, 50.0, 5.0, 1.0, 1.0], dtype=np.float32
)


def safe_float(value, default=0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(number) if math.isfinite(number) else float(default)


def dermal_labels(case: CaseRecord) -> np.ndarray:
    labels = np.zeros(3, dtype=np.float32)
    if case.label_path.exists():
        for line in case.label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                class_id = int(float(parts[0]))
                if class_id in (2, 3, 4):
                    labels[class_id - 2] = 1.0
    return labels


def reaction_label(case: CaseRecord) -> int:
    return REACTION_LABELS.get(str(case.patient.get("结果评判文本", "")), 0)


class CachedMediPPDDataset(Dataset):
    """Join per-case VLM caches, YOLO predictions, and clinical metadata."""

    def __init__(
        self,
        cases: Sequence[CaseRecord],
        cache_dir: Path,
        prediction_dir: Path,
        clinical_spec: ClinicalEncoderSpec,
    ):
        self.cases = list(cases)
        self.cache_dir = Path(cache_dir)
        self.prediction_dir = Path(prediction_dir)
        self.clinical_spec = clinical_spec
        self.labels = np.stack([dermal_labels(case) for case in self.cases])

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        with np.load(self.cache_dir / f"{case.stem}.npz") as cache:
            patches = cache["patches"].astype(np.float32)
            grounding_targets = cache["targets"].astype(np.float32)
        with np.load(self.prediction_dir / f"{case.stem}.npz") as prediction:
            raw_morph = prediction["morphometry"].astype(np.float32)
            detector_scores = prediction["detector_scores"].astype(np.float32)
        true_diameter = safe_float(case.patient.get("硬结平均径"), raw_morph[0])
        return {
            "patches": torch.from_numpy(patches),
            "grounding_targets": torch.from_numpy(grounding_targets),
            "morphometry": torch.from_numpy(raw_morph / MORPH_SCALES),
            "diameter_baseline": torch.tensor(raw_morph[0], dtype=torch.float32),
            "clinical": torch.from_numpy(
                self.clinical_spec.transform_one(case.patient)
            ),
            "diameter_target": torch.tensor(true_diameter, dtype=torch.float32),
            "dermal_target": torch.from_numpy(self.labels[index]),
            "reaction_target": torch.tensor(
                reaction_label(case), dtype=torch.long
            ),
            "detector_scores": torch.from_numpy(detector_scores),
            "index": torch.tensor(index, dtype=torch.long),
        }
