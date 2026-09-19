import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import ALLOWED_CLINICAL_COLUMNS, FORBIDDEN_CLINICAL_FRAGMENTS


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
STRONG_SOURCE_TO_TRAIN = {2: 0, 3: 1, 4: 2}
STRONG_TRAIN_TO_SOURCE = {value: key for key, value in STRONG_SOURCE_TO_TRAIN.items()}


def normalize_stem(value: str) -> str:
    return Path(str(value).strip()).stem


def assert_disjoint_split(train_stems: Iterable[str], val_stems: Iterable[str]) -> None:
    overlap = set(train_stems) & set(val_stems)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"train/validation overlap detected: {sample}")


def _safe_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class CaseRecord:
    stem: str
    image_path: Path
    label_path: Path
    split: str
    patient: Mapping[str, str]


@dataclass
class ClinicalEncoderSpec:
    numeric_mean: Dict[str, float]
    numeric_std: Dict[str, float]
    categorical_values: Dict[str, List[str]]
    feature_names: List[str]

    NUMERIC_COLUMNS = ("年龄", "测量间隔小时")
    CATEGORICAL_COLUMNS = ("性别文本", "色泽文本", "硬结触感文本")

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, object]]) -> "ClinicalEncoderSpec":
        for column in ALLOWED_CLINICAL_COLUMNS:
            if any(fragment in column for fragment in FORBIDDEN_CLINICAL_FRAGMENTS):
                raise ValueError(f"forbidden clinical column configured: {column}")

        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for column in cls.NUMERIC_COLUMNS:
            values = [_safe_float(row.get(column)) for row in rows]
            clean = np.asarray([value for value in values if value is not None], dtype=np.float32)
            means[column] = float(clean.mean()) if clean.size else 0.0
            std = float(clean.std()) if clean.size else 1.0
            stds[column] = std if std > 1e-6 else 1.0

        categories: Dict[str, List[str]] = {}
        for column in cls.CATEGORICAL_COLUMNS:
            categories[column] = sorted({str(row.get(column, "") or "<MISSING>") for row in rows})
            if "<MISSING>" not in categories[column]:
                categories[column].append("<MISSING>")

        names = list(cls.NUMERIC_COLUMNS)
        for column in cls.CATEGORICAL_COLUMNS:
            names.extend(f"{column}={value}" for value in categories[column])
        return cls(means, stds, categories, names)

    def transform_one(self, row: Mapping[str, object]) -> np.ndarray:
        values: List[float] = []
        for column in self.NUMERIC_COLUMNS:
            number = _safe_float(row.get(column))
            number = self.numeric_mean[column] if number is None else number
            values.append((number - self.numeric_mean[column]) / self.numeric_std[column])
        for column in self.CATEGORICAL_COLUMNS:
            raw = str(row.get(column, "") or "<MISSING>")
            known = self.categorical_values[column]
            raw = raw if raw in known else "<MISSING>"
            values.extend(1.0 if raw == category else 0.0 for category in known)
        return np.asarray(values, dtype=np.float32)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "ClinicalEncoderSpec":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def load_patient_rows(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for raw_key in (row.get("image_name", ""), row.get("image_path", "")):
                key = normalize_stem(raw_key)
                if key:
                    rows.setdefault(key, row)
    return rows


def list_cases(dataset_root: Path, patient_csv: Path) -> Tuple[List[CaseRecord], List[CaseRecord]]:
    patient_rows = load_patient_rows(patient_csv)
    by_split: Dict[str, List[CaseRecord]] = {"train": [], "val": []}
    for split in by_split:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            by_split[split].append(
                CaseRecord(
                    stem=image_path.stem,
                    image_path=image_path,
                    label_path=label_dir / f"{image_path.stem}.txt",
                    split=split,
                    patient=patient_rows.get(image_path.stem, {}),
                )
            )
    assert_disjoint_split((case.stem for case in by_split["train"]), (case.stem for case in by_split["val"]))
    return by_split["train"], by_split["val"]


def read_yolo_segments(label_path: Path, image_shape: Tuple[int, int]) -> Dict[int, List[np.ndarray]]:
    height, width = image_shape
    objects: Dict[int, List[np.ndarray]] = {}
    if not label_path.exists():
        return objects
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        values = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
        if values.size == 4:
            x, y, w, h = values
            points = np.asarray(
                [[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2], [x - w / 2, y + h / 2]],
                dtype=np.float32,
            )
        elif values.size >= 6 and values.size % 2 == 0:
            points = values.reshape(-1, 2)
        else:
            continue
        points[:, 0] = np.clip(points[:, 0] * width, 0, width - 1)
        points[:, 1] = np.clip(points[:, 1] * height, 0, height - 1)
        objects.setdefault(cls_id, []).append(np.round(points).astype(np.int32))
    return objects


def class_mask(objects: Mapping[int, Sequence[np.ndarray]], cls_id: int, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    polygons = objects.get(cls_id, [])
    if polygons:
        cv2.fillPoly(mask, list(polygons), 1)
    return mask


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_contiguous_strong_dataset(cases: Sequence[CaseRecord], output_root: Path) -> Path:
    """Build a strong-feature detection dataset with contiguous training IDs."""
    for case in cases:
        _link_or_copy(case.image_path, output_root / "images" / case.split / case.image_path.name)
        lines: List[str] = []
        if case.label_path.exists():
            for line in case.label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                train_id = STRONG_SOURCE_TO_TRAIN.get(cls_id)
                if train_id is None:
                    continue
                coordinates = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
                if coordinates.size == 4:
                    lines.append(f"{train_id} " + " ".join(parts[1:]))
                elif coordinates.size >= 6 and coordinates.size % 2 == 0:
                    points = coordinates.reshape(-1, 2)
                    xmin, ymin = points.min(axis=0)
                    xmax, ymax = points.max(axis=0)
                    lines.append(
                        f"{train_id} {(xmin + xmax) / 2:.6f} {(ymin + ymax) / 2:.6f} "
                        f"{xmax - xmin:.6f} {ymax - ymin:.6f}"
                    )
        destination = output_root / "labels" / case.split / f"{case.stem}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: blister",
                "  1: necrosis",
                "  2: double_ring",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def build_task_datasets(cases: Sequence[CaseRecord], output_root: Path) -> Tuple[Path, Path]:
    """Build clean task-specific datasets without mutating source labels."""
    seg_root = output_root / "seg01"
    det_root = output_root / "det234"
    names_seg = {0: "bottleCap", 1: "redSwollen"}
    names_det = {0: "unused0", 1: "unused1", 2: "blister", 3: "necrosis", 4: "doubleCircle"}
    for case in cases:
        for root in (seg_root, det_root):
            _link_or_copy(case.image_path, root / "images" / case.split / case.image_path.name)
        seg_lines: List[str] = []
        det_lines: List[str] = []
        if case.label_path.exists():
            for line in case.label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id in (0, 1):
                    coordinates = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
                    if coordinates.size == 4:
                        x, y, width, height = coordinates
                        polygon = [
                            x - width / 2,
                            y - height / 2,
                            x + width / 2,
                            y - height / 2,
                            x + width / 2,
                            y + height / 2,
                            x - width / 2,
                            y + height / 2,
                        ]
                        seg_lines.append(f"{cls_id} " + " ".join(f"{value:.6f}" for value in polygon))
                    else:
                        seg_lines.append(line)
                if cls_id in (2, 3, 4):
                    coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
                    if coords.size == 4:
                        det_lines.append(line)
                    elif coords.size >= 6 and coords.size % 2 == 0:
                        xy = coords.reshape(-1, 2)
                        xmin, ymin = xy.min(axis=0)
                        xmax, ymax = xy.max(axis=0)
                        det_lines.append(f"{cls_id} {(xmin+xmax)/2:.6f} {(ymin+ymax)/2:.6f} {xmax-xmin:.6f} {ymax-ymin:.6f}")
        for root, lines in ((seg_root, seg_lines), (det_root, det_lines)):
            destination = root / "labels" / case.split / f"{case.stem}.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def write_yaml(root: Path, names: Mapping[int, str]) -> Path:
        yaml_path = root / "data.yaml"
        text = [f"path: {root.resolve().as_posix()}", "train: images/train", "val: images/val", "", "names:"]
        text.extend(f"  {key}: {value}" for key, value in names.items())
        yaml_path.write_text("\n".join(text) + "\n", encoding="utf-8")
        return yaml_path

    return write_yaml(seg_root, names_seg), write_yaml(det_root, names_det)
