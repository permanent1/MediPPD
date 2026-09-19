"""Resumable orchestration for the MediPPD main experiment."""

import gc
import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from core.models import YOLO

from .config import (
    CAP_DIAMETER_MM,
    PATIENT_CSV,
    PROJECT_ROOT,
    RESULT_ROOT,
    RUN_ROOT,
    SEED,
    SOURCE_DATASET,
)
from .data import (
    ClinicalEncoderSpec,
    STRONG_TRAIN_TO_SOURCE,
    build_contiguous_strong_dataset,
    build_task_datasets,
    class_mask,
    list_cases,
    read_yolo_segments,
)
from .metrology import measure_masks


class PredictionCacheValidationError(ValueError):
    """Raised when cached V2 predictions were built with different inputs."""


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def pack_detection_candidates(
    boxes,
    scores,
    train_classes,
    image_shape: Tuple[int, int],
) -> dict:
    """Normalize and map every post-NMS contiguous detector candidate."""

    boxes_array = _as_numpy(boxes).astype(np.float32, copy=False).reshape(-1, 4)
    scores_array = _as_numpy(scores).astype(np.float32, copy=False).reshape(-1)
    classes_array = _as_numpy(train_classes).astype(np.int64, copy=False).reshape(-1)
    if not (len(boxes_array) == len(scores_array) == len(classes_array)):
        raise ValueError("detection boxes, scores, and classes must have equal lengths")

    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape height and width must be positive")

    unknown = sorted(set(classes_array.tolist()) - set(STRONG_TRAIN_TO_SOURCE))
    if unknown:
        raise ValueError(f"unknown contiguous strong-feature class IDs: {unknown}")
    source_classes = np.asarray(
        [STRONG_TRAIN_TO_SOURCE[int(class_id)] for class_id in classes_array],
        dtype=np.int64,
    )
    normalized_boxes = boxes_array / np.asarray(
        [width, height, width, height], dtype=np.float32
    )
    normalized_boxes = np.clip(normalized_boxes, 0.0, 1.0).astype(
        np.float32, copy=False
    )
    return {
        "detector_boxes": normalized_boxes.reshape(-1, 4),
        "detector_scores": scores_array.astype(np.float32, copy=False),
        "detector_classes": source_classes,
    }


def validate_prediction_manifest(path: Path, expected: Mapping[str, object]) -> None:
    """Reject a V2 prediction cache whose calibration or inputs differ."""

    path = Path(path)
    if not path.exists():
        raise PredictionCacheValidationError(
            f"prediction cache manifest mismatch at {path}: missing manifest"
        )
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PredictionCacheValidationError(
            f"prediction cache manifest mismatch at {path}: unreadable manifest"
        ) from error
    if not isinstance(actual, Mapping):
        raise PredictionCacheValidationError(
            f"prediction cache manifest mismatch at {path}: expected JSON object"
        )

    differences = [
        key for key, expected_value in expected.items() if actual.get(key) != expected_value
    ]
    if differences:
        fields = ", ".join(sorted(differences))
        raise PredictionCacheValidationError(
            f"prediction cache manifest mismatch at {path}: {fields}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_fingerprint(cases: Sequence[object]) -> str:
    records = sorted(
        (
            {
                "split": str(case.split),
                "stem": str(case.stem),
            }
            for case in cases
        ),
        key=lambda record: (record["split"], record["stem"]),
    )
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prediction_manifest(
    cases: Sequence[object], seg_weights: Path, det_weights: Path, conf: float
) -> dict:
    manifest = {
        "cap_diameter_mm": float(CAP_DIAMETER_MM),
        "segmentation_weight_hash": _file_sha256(seg_weights),
        "detection_weight_hash": _file_sha256(det_weights),
        "confidence_floor": float(conf),
        "image_count": len(cases),
        "split_fingerprint": _split_fingerprint(cases),
    }
    for field, weights in (
        ("segmentation_training_manifest_hash", Path(seg_weights)),
        ("detection_training_manifest_hash", Path(det_weights)),
    ):
        if weights.parent.name != "weights":
            continue
        training_manifest = weights.parents[1] / "training_manifest.json"
        if training_manifest.is_file():
            manifest[field] = _file_sha256(training_manifest)
    return manifest


def _mask_from_contour(shape: Tuple[int, int], contour) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if contour is not None and len(contour) >= 3:
        points = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [points], 1)
    return mask


def _save_prediction(path: Path, values: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def export_v2_predictions(
    cases,
    seg_weights: Path,
    det_weights: Path,
    prediction_dir: Path,
    conf: float = 0.001,
) -> float:
    """Export isolated V2 masks, 30 mm metrology, and all detections."""

    from predict_01 import collect_detections, select_measurement_detections

    started = time.monotonic()
    cases = list(cases)
    seg_weights = Path(seg_weights)
    det_weights = Path(det_weights)
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    expected_manifest = _prediction_manifest(cases, seg_weights, det_weights, conf)
    manifest_path = prediction_dir / "prediction_manifest.json"
    if manifest_path.exists():
        validate_prediction_manifest(manifest_path, expected_manifest)
        manifest_path.unlink()

    cases_by_stem = {str(case.stem): case for case in cases}
    if len(cases_by_stem) != len(cases):
        raise ValueError("V2 prediction export requires unique case stems")
    sources = [str(case.image_path) for case in cases]
    source_manifest_path = prediction_dir / "prediction_sources.txt"
    source_manifest_path.write_text("\n".join(sources) + "\n", encoding="utf-8")

    seg_model = YOLO(str(seg_weights))
    segmented = set()
    segmentation_results = seg_model.predict(
        source=str(source_manifest_path),
        imgsz=768,
        conf=0.10,
        iou=0.5,
        device=0,
        stream=True,
        verbose=False,
    )
    for result in segmentation_results:
        stem = Path(result.path).stem
        if stem not in cases_by_stem:
            raise ValueError(f"segmentation produced an unknown case: {stem}")
        shape = tuple(int(value) for value in result.orig_shape[:2])
        selected = select_measurement_detections(
            result.orig_img, collect_detections(result)
        )
        cap = _mask_from_contour(shape, selected.get(0, {}).get("contour"))
        lesion = _mask_from_contour(shape, selected.get(1, {}).get("contour"))
        confidence = float(selected.get(1, {}).get("score", 0.0))
        morphometry = measure_masks(cap, lesion, confidence)
        empty_candidates = pack_detection_candidates(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            shape,
        )
        _save_prediction(
            prediction_dir / f"{stem}.npz",
            {
                "cap_mask": cap,
                "lesion_mask": lesion,
                "morphometry": morphometry.vector(),
                "scale_valid": np.asarray(
                    int(morphometry.scale_valid), dtype=np.int8
                ),
                "lesion_valid": np.asarray(
                    int(morphometry.lesion_valid), dtype=np.int8
                ),
                "image_shape": np.asarray(shape, dtype=np.int32),
                **empty_candidates,
            },
        )
        segmented.add(stem)
    missing_segmentations = sorted(set(cases_by_stem) - segmented)
    if missing_segmentations:
        raise RuntimeError(
            "segmentation export omitted cases: " + ", ".join(missing_segmentations)
        )

    det_model = YOLO(str(det_weights))
    detected = set()
    detection_results = det_model.predict(
        source=str(source_manifest_path),
        imgsz=960,
        conf=float(conf),
        iou=0.5,
        device=0,
        stream=True,
        verbose=False,
    )
    for result in detection_results:
        stem = Path(result.path).stem
        if stem not in cases_by_stem:
            raise ValueError(f"detection produced an unknown case: {stem}")
        shape = tuple(int(value) for value in result.orig_shape[:2])
        result_boxes = getattr(result, "boxes", None)
        if result_boxes is None:
            candidates = pack_detection_candidates([], [], [], shape)
        else:
            candidates = pack_detection_candidates(
                result_boxes.xyxy,
                result_boxes.conf,
                result_boxes.cls,
                shape,
            )
        path = prediction_dir / f"{stem}.npz"
        with np.load(path) as stored:
            values = {key: stored[key] for key in stored.files}
        values.update(candidates)
        _save_prediction(path, values)
        detected.add(stem)
    missing_detections = sorted(set(cases_by_stem) - detected)
    if missing_detections:
        raise RuntimeError(
            "detection export omitted cases: " + ", ".join(missing_detections)
        )

    manifest_path.write_text(
        json.dumps(expected_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return time.monotonic() - started


MAIN_RUN_ROOT = RUN_ROOT
MAIN_RESULT_ROOT = RESULT_ROOT
GRID_SIZE = 12
DETECTION_CONFIDENCE = 0.001


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def manifest_matches(path: Path, expected: Mapping[str, object]) -> bool:
    """Return whether a JSON manifest contains every expected field and value."""

    actual = _read_json(path)
    return isinstance(actual, Mapping) and all(
        actual.get(key) == value for key, value in expected.items()
    )


@dataclass
class PipelineContext:
    """Resolved paths and command options shared by all main phases."""

    args: object

    def __post_init__(self) -> None:
        self.source_dataset = Path(self.args.dataset_root).resolve()
        self.patient_csv = Path(self.args.patient_csv).resolve()
        self.run_root = Path(self.args.run_root).resolve()
        self.result_root = Path(self.args.result_root).resolve()
        self.status_path = self.run_root / "experiment_status.json"
        self.runtime_path = self.run_root / "runtime.json"
        self.manifest_dir = self.run_root / "manifests"
        self.data_root = self.run_root / "data"
        self.seg_yaml = self.data_root / "seg01" / "data.yaml"
        self.strong_yaml = self.data_root / "strong_contiguous" / "data.yaml"
        self.seg_weights = (
            self.run_root / "stage1" / "redswollen_seg" / "weights" / "best.pt"
        )
        self.det_weights = (
            self.run_root
            / "stage1"
            / "strong_features_contiguous"
            / "weights"
            / "best.pt"
        )
        self.prediction_dir = self.run_root / "predictions_30mm"
        self.cache_dir = self.run_root / "vlm_cache"
        self.main_checkpoint = (
            self.run_root / "task_routed" / "MediPPD" / "best.pt"
        )
        self.red_fusion_dir = self.run_root / "red_mask_fusion"
        self.red_fusion_checkpoint = self.red_fusion_dir / "three_view_grounding.pt"
        self.red_fusion_features = self.red_fusion_dir / "train_features.pt"


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    counted_in_main: bool
    run: Callable[[PipelineContext], Optional[Mapping[str, object]]]
    validate: Callable[[PipelineContext], bool]


class _PipelineState:
    def __init__(self, path: Path, resume: bool):
        self.path = Path(path)
        loaded = _read_json(self.path) if resume else None
        self.data = loaded if isinstance(loaded, Mapping) else {"phases": {}}
        self.data = dict(self.data)
        self.data.setdefault("phases", {})
        self.data.update({"status": "running", "pid": os.getpid()})
        self.data.pop("exception", None)
        self._write()

    def _write(self) -> None:
        _write_json(self.path, self.data)

    def is_complete(self, phase: str) -> bool:
        return (
            self.data.get("phases", {}).get(phase, {}).get("status") == "complete"
        )

    def start(self, phase: str) -> None:
        self.data["active_phase"] = phase
        self.data["phases"][phase] = {
            "status": "running",
            "started_at": _utc_now(),
        }
        self._write()

    def skipped(self, phase: str) -> None:
        entry = self.data["phases"][phase]
        entry["resume_skipped_at"] = _utc_now()
        self.data["active_phase"] = phase
        self._write()

    def complete(
        self,
        phase: str,
        seconds: float,
        counted_in_main: bool,
        details: Optional[Mapping[str, object]],
    ) -> None:
        entry = self.data["phases"].setdefault(phase, {})
        entry.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "wall_seconds": round(float(seconds), 3),
                "counted_in_main": bool(counted_in_main),
                "details": dict(details or {}),
            }
        )
        self._write()

    def fail(
        self,
        phase: str,
        error: Exception,
        traceback_path: Path,
        seconds: float,
        counted_in_main: bool,
    ) -> None:
        timestamp = _utc_now()
        entry = self.data["phases"].setdefault(phase, {})
        entry.update(
            {
                "status": "failed",
                "failed_at": timestamp,
                "wall_seconds": round(float(seconds), 3),
                "counted_in_main": bool(counted_in_main),
            }
        )
        self.data.update(
            {
                "status": "failed",
                "active_phase": phase,
                "exception": {
                    "phase": phase,
                    "type": type(error).__name__,
                    "message": str(error),
                    "timestamp": timestamp,
                    "traceback_path": str(Path(traceback_path).resolve()),
                },
            }
        )
        self._write()

    def finish(self) -> None:
        self.data.update(
            {"status": "complete", "active_phase": None, "completed_at": _utc_now()}
        )
        self._write()


def _runtime_rows(context: PipelineContext) -> list[dict]:
    rows = _read_json(context.runtime_path, [])
    return rows if isinstance(rows, list) else []


def _record_runtime(
    context: PipelineContext, phase: str, seconds: float, counted_in_main: bool
) -> None:
    rows = [row for row in _runtime_rows(context) if row.get("phase") != phase]
    rows.append(
        {
            "phase": phase,
            "seconds": round(float(seconds), 3),
            "minutes": round(float(seconds) / 60.0, 3),
            "counted_in_main": bool(counted_in_main),
        }
    )
    _write_json(context.runtime_path, rows)


def _main_seconds(context: PipelineContext) -> float:
    return sum(
        float(row.get("seconds", 0.0))
        for row in _runtime_rows(context)
        if row.get("counted_in_main")
    )


def _remaining_main_seconds(context: PipelineContext, reserve_minutes=2.0) -> float:
    limit = float(context.args.main_budget_min) * 60.0
    return max(0.0, limit - _main_seconds(context) - float(reserve_minutes) * 60.0)


def _all_cases(context: PipelineContext):
    train_cases, val_cases = list_cases(context.source_dataset, context.patient_csv)
    return train_cases, val_cases, train_cases + val_cases


def _fingerprint_payload(records: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(
        list(records), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _optional_file_hash(path: Path) -> Optional[str]:
    path = Path(path)
    return _file_sha256(path) if path.is_file() else None


def _source_data_fingerprint(cases: Sequence[object]) -> str:
    records = []
    for case in sorted(cases, key=lambda item: (item.split, item.stem)):
        records.append(
            {
                "split": str(case.split),
                "stem": str(case.stem),
                "image_name": Path(case.image_path).name,
                "image_hash": _optional_file_hash(case.image_path),
                "label_hash": _optional_file_hash(case.label_path),
                "patient": dict(case.patient),
            }
        )
    return _fingerprint_payload(records)


def _prepared_data_fingerprint(
    context: PipelineContext, cases: Sequence[object]
) -> str:
    records = []
    for case in sorted(cases, key=lambda item: (item.split, item.stem)):
        for dataset in ("seg01", "strong_contiguous"):
            root = context.data_root / dataset
            records.append(
                {
                    "dataset": dataset,
                    "split": str(case.split),
                    "stem": str(case.stem),
                    "image_hash": _optional_file_hash(
                        root / "images" / case.split / Path(case.image_path).name
                    ),
                    "label_hash": _optional_file_hash(
                        root / "labels" / case.split / f"{case.stem}.txt"
                    ),
                }
            )
    records.extend(
        [
            {
                "dataset": "seg01",
                "yaml_hash": _optional_file_hash(context.seg_yaml),
            },
            {
                "dataset": "strong_contiguous",
                "yaml_hash": _optional_file_hash(context.strong_yaml),
            },
        ]
    )
    return _fingerprint_payload(records)


def _prepare_expected() -> dict:
    train_cases, val_cases, all_cases = _all_cases(context)
    return {
        "seed": SEED,
        "cap_diameter_mm": float(CAP_DIAMETER_MM),
        "train_stems": [case.stem for case in train_cases],
        "val_stems": [case.stem for case in val_cases],
        "source_data_fingerprint": _source_data_fingerprint(all_cases),
    }


def _prepare_valid(context: PipelineContext) -> bool:
    try:
        _, _, all_cases = _all_cases(context)
        expected = _prepare_expected()
        prepared_fingerprint = _prepared_data_fingerprint(context, all_cases)
    except OSError:
        return False
    manifest = context.manifest_dir / "split_manifest.json"
    if not manifest_matches(manifest, expected):
        return False
    if not manifest_matches(
        manifest, {"prepared_data_fingerprint": prepared_fingerprint}
    ):
        return False
    clinical = context.manifest_dir / "clinical_spec.json"
    try:
        ClinicalEncoderSpec.from_json(clinical)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return context.seg_yaml.is_file() and context.strong_yaml.is_file()


def prepare_v2_data(context: PipelineContext) -> dict:
    train_cases, val_cases, all_cases = _all_cases(context)
    seg_yaml, _ = build_task_datasets(all_cases, context.data_root)
    strong_yaml = build_contiguous_strong_dataset(
        all_cases, context.data_root / "strong_contiguous"
    )
    clinical = ClinicalEncoderSpec.fit([case.patient for case in train_cases])
    clinical.to_json(context.manifest_dir / "clinical_spec.json")
    manifest = _prepare_expected()
    manifest.update(
        {
            "prepared_data_fingerprint": _prepared_data_fingerprint(
                context, all_cases
            ),
            "train_count": len(train_cases),
            "val_count": len(val_cases),
            "forbidden_input_fragments": [
                "结果评判",
                "硬结横径",
                "硬结纵径",
                "硬结平均径",
                "特征描述",
            ],
        }
    )
    _write_json(context.manifest_dir / "split_manifest.json", manifest)
    return {
        "train": len(train_cases),
        "val": len(val_cases),
        "seg_yaml": str(seg_yaml),
        "strong_yaml": str(strong_yaml),
    }


def _training_dataset_fingerprint(data_yaml: Path) -> str:
    data_yaml = Path(data_yaml)
    dataset_root = data_yaml.parent
    records = [
        {
            "path": "data.yaml",
            "sha256": _optional_file_hash(data_yaml),
        }
    ]
    for directory in ("images", "labels"):
        root = dataset_root / directory
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.suffix == ".cache":
                continue
            records.append(
                {
                    "path": path.relative_to(dataset_root).as_posix(),
                    "sha256": _file_sha256(path),
                }
            )
    split_manifest = data_yaml.parents[2] / "manifests" / "split_manifest.json"
    records.append(
        {
            "path": "../manifests/split_manifest.json",
            "sha256": _optional_file_hash(split_manifest),
        }
    )
    return _fingerprint_payload(records)


def _training_manifest_expected(
    model_path: Path, data_yaml: Path, task: str, image_size: int
) -> dict:
    return {
        "model_hash": _file_sha256(model_path),
        "data_hash": _file_sha256(data_yaml),
        "dataset_fingerprint": _training_dataset_fingerprint(data_yaml),
        "task": task,
        "image_size": int(image_size),
        "seed": SEED,
    }


def _weights_valid(weights: Path, manifest: Path, expected: Mapping[str, object]) -> bool:
    weights = Path(weights)
    actual = _read_json(manifest)
    return (
        weights.is_file()
        and isinstance(actual, Mapping)
        and all(actual.get(key) == value for key, value in expected.items())
        and actual.get("weight_hash") == _file_sha256(weights)
    )


def _train_yolo(
    context: PipelineContext,
    model_path: Path,
    data_yaml: Path,
    task: str,
    name: str,
    weights: Path,
    epochs: int,
    image_size: int,
    batch: int,
    time_minutes: float,
) -> dict:
    manifest_path = weights.parents[1] / "training_manifest.json"
    expected = _training_manifest_expected(model_path, data_yaml, task, image_size)
    if _weights_valid(weights, manifest_path, expected):
        return {"weights": str(weights), "reused": True}
    if weights.exists():
        weights.unlink()
    model = YOLO(str(model_path))
    model.train(
        data=str(data_yaml),
        task=task,
        imgsz=int(image_size),
        epochs=int(epochs),
        time=max(0.01, float(time_minutes) / 60.0),
        batch=int(batch),
        seed=SEED,
        deterministic=True,
        device=0,
        workers=4,
        project=str(context.run_root / "stage1"),
        name=name,
        exist_ok=True,
        pretrained=True,
        amp=False,
        cache=False,
        optimizer="AdamW",
        cos_lr=True,
        patience=max(15, int(epochs) // 3),
        close_mosaic=max(5, int(epochs) // 5),
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=8.0,
        translate=0.06,
        scale=0.20,
        fliplr=0.5,
        flipud=0.05,
        mosaic=1.0,
        mixup=0.05 if task == "detect" else 0.0,
        plots=True,
        verbose=False,
    )
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO training did not produce {weights}")
    payload = dict(expected)
    payload["weight_hash"] = _file_sha256(weights)
    _write_json(manifest_path, payload)
    return {"weights": str(weights), "reused": False}


def _segmentation_expected(context: PipelineContext) -> dict:
    return _training_manifest_expected(
        PROJECT_ROOT / "yolov8s-seg.pt", context.seg_yaml, "segment", 768
    )


def _segmentation_valid(context: PipelineContext) -> bool:
    try:
        expected = _segmentation_expected(context)
    except OSError:
        return False
    return _weights_valid(
        context.seg_weights,
        context.seg_weights.parents[1] / "training_manifest.json",
        expected,
    )


def train_or_reuse_segmentation(context: PipelineContext) -> dict:
    return _train_yolo(
        context,
        PROJECT_ROOT / "yolov8s-seg.pt",
        context.seg_yaml,
        "segment",
        "redswollen_seg",
        context.seg_weights,
        60,
        768,
        12,
        min(10.0, max(1.0, _remaining_main_seconds(context) / 60.0)),
    )


def _detector_expected(context: PipelineContext) -> dict:
    expected = _training_manifest_expected(
        PROJECT_ROOT / "yolo11n.pt", context.strong_yaml, "detect", 960
    )
    expected.update(
        {
            "source_to_train": {"2": 0, "3": 1, "4": 2},
            "confidence_floor": DETECTION_CONFIDENCE,
        }
    )
    return expected


def _detector_valid(context: PipelineContext) -> bool:
    try:
        expected = _detector_expected(context)
    except OSError:
        return False
    return _weights_valid(
        context.det_weights,
        context.det_weights.parents[1] / "training_manifest.json",
        expected,
    )


def train_v2_strong_detector(context: PipelineContext) -> dict:
    details = _train_yolo(
        context,
        PROJECT_ROOT / "yolo11n.pt",
        context.strong_yaml,
        "detect",
        "strong_features_contiguous",
        context.det_weights,
        80,
        960,
        16,
        min(10.0, max(1.0, _remaining_main_seconds(context) / 60.0)),
    )
    manifest_path = context.det_weights.parents[1] / "training_manifest.json"
    payload = _read_json(manifest_path, {})
    payload.update(
        {
            "source_to_train": {"2": 0, "3": 1, "4": 2},
            "confidence_floor": DETECTION_CONFIDENCE,
        }
    )
    _write_json(manifest_path, payload)
    return details


def _prediction_expected(context: PipelineContext) -> dict:
    _, _, cases = _all_cases(context)
    return _prediction_manifest(
        cases,
        context.seg_weights,
        context.det_weights,
        DETECTION_CONFIDENCE,
    )


def _prediction_artifact_valid(path: Path) -> bool:
    required = {
        "cap_mask",
        "lesion_mask",
        "morphometry",
        "scale_valid",
        "lesion_valid",
        "image_shape",
        "detector_boxes",
        "detector_scores",
        "detector_classes",
    }
    try:
        with np.load(path, allow_pickle=False) as stored:
            if not required <= set(stored.files):
                return False
            image_shape = stored["image_shape"]
            if image_shape.shape != (2,):
                return False
            height, width = (int(value) for value in image_shape)
            boxes = stored["detector_boxes"]
            scores = stored["detector_scores"]
            classes = stored["detector_classes"]
            return bool(
                height > 0
                and width > 0
                and stored["cap_mask"].shape == (height, width)
                and stored["lesion_mask"].shape == (height, width)
                and stored["morphometry"].shape == (7,)
                and stored["scale_valid"].shape in ((), (1,))
                and stored["lesion_valid"].shape in ((), (1,))
                and boxes.ndim == 2
                and boxes.shape[1:] == (4,)
                and scores.ndim == 1
                and classes.ndim == 1
                and len(boxes) == len(scores) == len(classes)
            )
    except (OSError, ValueError, EOFError, KeyError, TypeError):
        return False


def _predictions_valid(context: PipelineContext) -> bool:
    try:
        _, _, cases = _all_cases(context)
        expected = _prediction_expected(context)
    except OSError:
        return False
    if not manifest_matches(
        context.prediction_dir / "prediction_manifest.json", expected
    ):
        return False
    return all(
        _prediction_artifact_valid(context.prediction_dir / f"{case.stem}.npz")
        for case in cases
    )


def export_predictions_phase(context: PipelineContext) -> dict:
    _, _, cases = _all_cases(context)
    manifest = context.prediction_dir / "prediction_manifest.json"
    if manifest.exists() and not manifest_matches(manifest, _prediction_expected(context)):
        manifest.unlink()
    seconds = export_v2_predictions(
        cases,
        context.seg_weights,
        context.det_weights,
        context.prediction_dir,
        DETECTION_CONFIDENCE,
    )
    return {"cases": len(cases), "export_seconds": seconds}


def _cache_expected(context: PipelineContext) -> tuple[str, list]:
    from .vlm_cache import cache_fingerprint

    _, _, cases = _all_cases(context)
    prediction_manifest = _read_json(
        context.prediction_dir / "prediction_manifest.json", {}
    )
    prediction_digest = hashlib.sha256(
        json.dumps(prediction_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    model_key = f"{context.args.llava_model}|prediction={prediction_digest}"
    fingerprint = cache_fingerprint(
        model_key, [case.stem for case in cases], GRID_SIZE
    )
    return fingerprint, cases


def _semantic_initializers_valid(path: Path) -> bool:
    try:
        semantic = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError, TypeError):
        return False
    return bool(
        semantic.ndim == 2
        and semantic.shape[0] == 4
        and semantic.shape[1] > 0
        and np.isfinite(semantic).all()
    )


def _vlm_artifact_valid(path: Path) -> bool:
    required = {"patches", "targets", "view_boxes", "image_shape"}
    try:
        with np.load(path, allow_pickle=False) as stored:
            if not required <= set(stored.files):
                return False
            patches = stored["patches"]
            targets = stored["targets"]
            view_boxes = stored["view_boxes"]
            image_shape = stored["image_shape"]
            return bool(
                patches.ndim == 3
                and patches.shape[0] == 3
                and patches.shape[1] == GRID_SIZE * GRID_SIZE
                and patches.shape[2] > 0
                and targets.shape == (3, 4, GRID_SIZE, GRID_SIZE)
                and view_boxes.shape == (3, 4)
                and image_shape.shape == (2,)
                and np.all(image_shape > 0)
            )
    except (OSError, ValueError, EOFError, KeyError, TypeError):
        return False


def _vlm_cache_valid(context: PipelineContext) -> bool:
    manifest_path = context.cache_dir / "cache_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        fingerprint, cases = _cache_expected(context)
    except (OSError, ValueError):
        return False
    if not manifest_matches(manifest_path, {"fingerprint": fingerprint}):
        return False
    if not _semantic_initializers_valid(
        context.cache_dir / "semantic_initializers.npy"
    ):
        return False
    return all(
        _vlm_artifact_valid(context.cache_dir / f"{case.stem}.npz")
        for case in cases
    )


def build_or_reuse_vlm_cache(context: PipelineContext) -> dict:
    from .vlm_cache import (
        DERMAL_PROMPTS,
        FrozenLlavaPatchExtractor,
        cache_one_case,
        write_cache_manifest,
    )

    manifest_path = context.cache_dir / "cache_manifest.json"
    if context.args.skip_vlm_cache and not manifest_path.is_file():
        raise RuntimeError(
            "--skip-vlm-cache requires a cache manifest with the expected fingerprint"
        )
    if _vlm_cache_valid(context):
        return {"reused": True, "fingerprint": _read_json(manifest_path)["fingerprint"]}
    if context.args.skip_vlm_cache:
        raise RuntimeError(
            "--skip-vlm-cache requested, but the cache fingerprint is absent or stale"
        )

    fingerprint, cases = _cache_expected(context)
    context.cache_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        stale = context.cache_dir / f"{case.stem}.npz"
        if stale.exists():
            stale.unlink()
    semantic_path = context.cache_dir / "semantic_initializers.npy"
    if semantic_path.exists():
        semantic_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()

    extractor = FrozenLlavaPatchExtractor.from_pretrained(
        context.args.llava_model, device=context.args.device, grid_size=GRID_SIZE
    )
    np.save(semantic_path, extractor.semantic_initializers(DERMAL_PROMPTS))
    for index, case in enumerate(cases, start=1):
        with np.load(context.prediction_dir / f"{case.stem}.npz") as prediction:
            shape = tuple(int(value) for value in prediction["image_shape"][:2])
            lesion = prediction["lesion_mask"].astype(np.uint8)
        objects = read_yolo_segments(case.label_path, shape)
        targets = np.stack(
            [class_mask(objects, class_id, shape) for class_id in (1, 2, 3, 4)]
        )
        if not lesion.any():
            lesion = targets[0]
        cache_one_case(
            case.stem,
            case.image_path,
            lesion,
            targets,
            context.cache_dir,
            extractor,
        )
        if index % 25 == 0:
            print(f"[V2 VLM] cached {index}/{len(cases)}", flush=True)
    write_cache_manifest(
        manifest_path,
        fingerprint,
        context.args.llava_model,
        [case.stem for case in cases],
        GRID_SIZE,
    )
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"reused": False, "fingerprint": fingerprint, "cases": len(cases)}


def _datasets(context: PipelineContext):
    from .dataset_cache import CachedMediPPDDataset

    class TaskRoutedDataset(CachedMediPPDDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            # Candidate arrays are deliberately variable length and are consumed
            # only by evaluation, not by the adapter DataLoader.
            sample.pop("detector_scores", None)
            return sample

    train_cases, val_cases, _ = _all_cases(context)
    clinical = ClinicalEncoderSpec.from_json(
        context.manifest_dir / "clinical_spec.json"
    )
    return (
        TaskRoutedDataset(
            train_cases, context.cache_dir, context.prediction_dir, clinical
        ),
        TaskRoutedDataset(
            val_cases, context.cache_dir, context.prediction_dir, clinical
        ),
    )


def _adapter_manifest_expected(context: PipelineContext, config) -> dict:
    return {
        "config": asdict(config),
        "training_objective_version": "stage-specific-selection-v3",
        "prediction_manifest_hash": _file_sha256(
            context.prediction_dir / "prediction_manifest.json"
        ),
        "cache_manifest_hash": _file_sha256(
            context.cache_dir / "cache_manifest.json"
        ),
        "seed": SEED,
    }


def _adapter_valid(context: PipelineContext, config, checkpoint: Path) -> bool:
    manifest = checkpoint.parent / "training_manifest.json"
    try:
        expected = _adapter_manifest_expected(context, config)
    except OSError:
        return False
    actual = _read_json(manifest)
    return (
        checkpoint.is_file()
        and context.red_fusion_features.is_file()
        and isinstance(actual, Mapping)
        and all(actual.get(key) == value for key, value in expected.items())
        and actual.get("checkpoint_hash") == _file_sha256(checkpoint)
    )


def _train_configuration(
    context: PipelineContext,
    config,
    checkpoint: Path,
    stage_one_epochs: int,
    stage_two_epochs: int,
    max_seconds: float,
) -> dict:
    from .task_routed_training import train_task_routed_model

    if _adapter_valid(context, config, checkpoint):
        return {"checkpoint": str(checkpoint), "reused": True}
    train_data, val_data = _datasets(context)
    semantic = np.load(context.cache_dir / "semantic_initializers.npy")
    _, seconds = train_task_routed_model(
        train_data,
        val_data,
        semantic,
        config,
        checkpoint,
        stage_one_epochs=stage_one_epochs,
        stage_two_epochs=stage_two_epochs,
        batch_size=32,
        val_batch_size=64,
        device=context.args.device,
        seed=SEED,
        max_seconds=max_seconds,
    )
    payload = _adapter_manifest_expected(context, config)
    payload["checkpoint_hash"] = _file_sha256(checkpoint)
    _write_json(checkpoint.parent / "training_manifest.json", payload)
    return {"checkpoint": str(checkpoint), "reused": False, "train_seconds": seconds}


def _main_valid(context: PipelineContext) -> bool:
    from .task_routed_model import TaskRoutedConfig

    return _adapter_valid(context, TaskRoutedConfig.full(), context.main_checkpoint)


def train_main_method(context: PipelineContext) -> dict:
    from .task_routed_model import TaskRoutedConfig

    available = _remaining_main_seconds(context)
    if available <= 0:
        raise RuntimeError("main budget exhausted before B9 training")
    return _train_configuration(
        context,
        TaskRoutedConfig.full(),
        context.main_checkpoint,
        stage_one_epochs=20,
        stage_two_epochs=40,
        max_seconds=available,
    )


def _red_fusion_expected(context: PipelineContext, active_views: int) -> dict:
    return {
        "fusion_version": "yolo-anchored-v2-roi-background",
        "active_views": int(active_views),
        "main_checkpoint_hash": _file_sha256(context.main_checkpoint),
        "prediction_manifest_hash": _file_sha256(
            context.prediction_dir / "prediction_manifest.json"
        ),
        "cache_manifest_hash": _file_sha256(
            context.cache_dir / "cache_manifest.json"
        ),
        "seed": SEED,
    }


def _red_fusion_manifest(checkpoint: Path) -> Path:
    return Path(checkpoint).with_suffix(".json")


def _red_fusion_valid(
    context: PipelineContext,
    active_views: int = 3,
    checkpoint: Optional[Path] = None,
) -> bool:
    checkpoint = Path(checkpoint or context.red_fusion_checkpoint)
    try:
        expected = _red_fusion_expected(context, active_views)
    except OSError:
        return False
    actual = _read_json(_red_fusion_manifest(checkpoint))
    return (
        checkpoint.is_file()
        and isinstance(actual, Mapping)
        and all(actual.get(key) == value for key, value in expected.items())
        and actual.get("checkpoint_hash") == _file_sha256(checkpoint)
    )


def _write_red_fusion_manifest(
    context: PipelineContext,
    checkpoint: Path,
    active_views: int,
    train_seconds: float,
) -> None:
    payload = _red_fusion_expected(context, active_views)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.update(
        {
            "checkpoint_hash": _file_sha256(checkpoint),
            "internal_holdout_dice": float(state["internal_holdout_dice"]),
            "train_seconds": float(train_seconds),
        }
    )
    _write_json(_red_fusion_manifest(checkpoint), payload)


def train_red_mask_fusion(context: PipelineContext) -> dict:
    """Train the YOLO-anchored three-view semantic residual head."""

    from .red_mask_fusion import build_fusion_tensors, train_fusion_head
    from .task_routed_model import TaskRoutedConfig

    if _red_fusion_valid(context):
        return {"checkpoint": str(context.red_fusion_checkpoint), "reused": True}
    train_data, _ = _datasets(context)
    config = TaskRoutedConfig.full()
    model = _load_model(context, config, context.main_checkpoint)
    probabilities, priors, targets = build_fusion_tensors(
        model, train_data, config, device=context.args.device
    )
    context.red_fusion_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"probabilities": probabilities, "priors": priors, "targets": targets},
        context.red_fusion_features,
    )
    _, seconds = train_fusion_head(
        probabilities,
        priors,
        targets,
        active_views=3,
        checkpoint=context.red_fusion_checkpoint,
        device=context.args.device,
        epochs=40,
        seed=SEED,
    )
    _write_red_fusion_manifest(
        context, context.red_fusion_checkpoint, active_views=3, train_seconds=seconds
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "checkpoint": str(context.red_fusion_checkpoint),
        "reused": False,
        "train_seconds": seconds,
    }


def _load_red_fusion(context: PipelineContext, checkpoint: Optional[Path] = None):
    from .red_mask_fusion import load_fusion_head

    return load_fusion_head(
        checkpoint or context.red_fusion_checkpoint, device=context.args.device
    )


def _load_model(context: PipelineContext, config, checkpoint: Path):
    from .task_routed_training import _make_model

    train_data, _ = _datasets(context)
    semantic = np.load(context.cache_dir / "semantic_initializers.npy")
    model = _make_model(train_data, semantic, d_model=128).to(context.args.device)
    state = torch.load(
        checkpoint, map_location=context.args.device, weights_only=False
    )
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _metric_row(method: str, metrics: Mapping[str, object], runtime: float) -> dict:
    row = dict(metrics)
    row.update({"method": method, "runtime_minutes": float(runtime)})
    return row


def _evaluation_expected(context: PipelineContext) -> dict:
    return {
        "evaluation_version": "yolo-anchored-red-fusion-v3",
        "checkpoint_hash": _file_sha256(context.main_checkpoint),
        "red_fusion_checkpoint_hash": _file_sha256(
            context.red_fusion_checkpoint
        ),
        "prediction_manifest_hash": _file_sha256(
            context.prediction_dir / "prediction_manifest.json"
        ),
    }


def _evaluation_valid(context: PipelineContext) -> bool:
    path = context.run_root / "evaluation" / "main.json"
    try:
        expected = _evaluation_expected(context)
    except OSError:
        return False
    actual = _read_json(path)
    return (
        isinstance(actual, Mapping)
        and all(actual.get(key) == value for key, value in expected.items())
        and bool(actual.get("main_red"))
        and bool(actual.get("main_strong"))
    )


def evaluate_main(context: PipelineContext) -> dict:
    """Evaluate only the production MediPPD method on the validation split."""

    from .task_routed_model import TaskRoutedConfig
    from .task_routed_training import evaluate_task_routed_model

    _, val_data = _datasets(context)
    config = TaskRoutedConfig.full()
    model = _load_model(context, config, context.main_checkpoint)
    red_fusion = _load_red_fusion(context)
    red_metrics, strong_metrics, per_case, diagnostics = (
        evaluate_task_routed_model(
            model,
            val_data,
            config,
            device=context.args.device,
            red_fusion=red_fusion,
        )
    )
    runtime = _main_seconds(context) / 60.0
    payload = _evaluation_expected(context)
    payload.update(
        {
            "main_red": [_metric_row("MediPPD", red_metrics, runtime)],
            "main_strong": [_metric_row("MediPPD", strong_metrics, runtime)],
            "per_case": per_case,
            "detection_diagnostics": [
                {"method": "MediPPD", **diagnostics}
            ],
        }
    )
    path = context.run_root / "evaluation" / "main.json"
    _write_json(path, payload)
    del model
    del red_fusion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"evaluation": str(path), "validation_cases": len(val_data)}

def _report_manifest_expected(context: PipelineContext) -> dict:
    return {
        "main_evaluation_hash": _file_sha256(
            context.run_root / "evaluation" / "main.json"
        ),
        "method": "MediPPD",
    }


def _reports_valid(context: PipelineContext) -> bool:
    required = (
        "tables.xlsx",
        "main_redswollen.csv",
        "main_strong_features.csv",
        "runtime_budget.csv",
        "per_case_predictions.csv",
        "detection_diagnostics.csv",
    )
    if not all(
        (context.result_root / name).is_file()
        and (context.result_root / name).stat().st_size > 0
        for name in required
    ):
        return False
    try:
        expected = _report_manifest_expected(context)
    except OSError:
        return False
    return manifest_matches(
        context.result_root / "result_manifest.json", expected
    )


def write_results_phase(context: PipelineContext) -> dict:
    from .reporting import write_main_result_bundle

    main = _read_json(context.run_root / "evaluation" / "main.json")
    if not isinstance(main, Mapping):
        raise RuntimeError("main evaluation is missing; cannot write results")
    frames = write_main_result_bundle(
        context.result_root,
        main["main_red"],
        main["main_strong"],
        _runtime_rows(context),
        per_case_rows=main.get("per_case", []),
        detection_rows=main.get("detection_diagnostics", []),
    )
    _write_json(
        context.result_root / "result_manifest.json",
        _report_manifest_expected(context),
    )
    return {"result_root": str(context.result_root), "tables": len(frames)}

def build_phase_plan(context: PipelineContext) -> list[PhaseSpec]:
    """Return the fixed sequence for the production main experiment."""

    return [
        PhaseSpec("prepare", True, prepare_v2_data, _prepare_valid),
        PhaseSpec(
            "segmentation",
            True,
            train_or_reuse_segmentation,
            _segmentation_valid,
        ),
        PhaseSpec(
            "strong_detector",
            True,
            train_v2_strong_detector,
            _detector_valid,
        ),
        PhaseSpec(
            "export_predictions",
            True,
            export_predictions_phase,
            _predictions_valid,
        ),
        PhaseSpec(
            "vlm_cache", False, build_or_reuse_vlm_cache, _vlm_cache_valid
        ),
        PhaseSpec("train_main", True, train_main_method, _main_valid),
        PhaseSpec(
            "train_red_fusion",
            True,
            train_red_mask_fusion,
            _red_fusion_valid,
        ),
        PhaseSpec("evaluate_main", True, evaluate_main, _evaluation_valid),
        PhaseSpec("write_results", True, write_results_phase, _reports_valid),
    ]


def run_main_pipeline(args) -> None:
    """Run the ordered, resumable MediPPD main experiment."""

    context = PipelineContext(args)
    state = _PipelineState(context.status_path, resume=bool(args.resume))
    for phase in build_phase_plan(context):
        started = time.monotonic()
        try:
            if (
                args.resume
                and state.is_complete(phase.name)
                and phase.validate(context)
            ):
                state.skipped(phase.name)
                continue
            state.start(phase.name)
            details = phase.run(context)
        except Exception as error:
            seconds = time.monotonic() - started
            _record_runtime(context, phase.name, seconds, phase.counted_in_main)
            traceback_path = (
                context.run_root / "tracebacks" / f"{phase.name}.txt"
            )
            traceback_path.parent.mkdir(parents=True, exist_ok=True)
            traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
            state.fail(
                phase.name,
                error,
                traceback_path,
                seconds,
                phase.counted_in_main,
            )
            raise
        seconds = time.monotonic() - started
        _record_runtime(context, phase.name, seconds, phase.counted_in_main)
        state.complete(
            phase.name, seconds, phase.counted_in_main, details or {}
        )
    state.finish()
