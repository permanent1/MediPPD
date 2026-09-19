import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


DERMAL_PROMPTS = (
    "red swollen skin reaction",
    "blister on the skin",
    "skin necrosis",
    "double-ring skin reaction",
)

GROUNDING_PREPROCESSING_VERSION = "square-pad_any-overlap_double-ring-v2"


def cache_fingerprint(
    model_id: str,
    stems: Iterable[str],
    grid_size: int,
    roi_margin: float = 0.15,
    preprocessing_version: str = GROUNDING_PREPROCESSING_VERSION,
) -> str:
    payload = json.dumps(
        {
            "model_id": model_id,
            "stems": sorted(stems),
            "grid_size": int(grid_size),
            "roi_margin": float(roi_margin),
            "views": ["global", "roi", "masked"],
            "preprocessing_version": str(preprocessing_version),
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_cache_manifest(path: Path, expected_fingerprint: str) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("fingerprint") != expected_fingerprint:
        raise ValueError(f"stale VLM cache manifest at {path}")


def _roi_box(mask: np.ndarray, margin: float = 0.15) -> Tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return 0, 0, width, height
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(2, int(round((x2 - x1) * margin)))
    pad_y = max(2, int(round((y2 - y1) * margin)))
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def build_three_views(
    image: np.ndarray, lesion_mask: np.ndarray, margin: float = 0.15
):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _roi_box(lesion_mask, margin=margin)
    roi = image[y1:y2, x1:x2].copy()
    background = np.full_like(image, 114)
    masked = np.where(lesion_mask[:, :, None].astype(bool), image, background)
    boxes = np.asarray(
        [[0.0, 0.0, 1.0, 1.0], [x1 / width, y1 / height, x2 / width, y2 / height], [0.0, 0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    return [image, roi, masked], boxes, (x1, y1, x2, y2)


def pad_to_square(array: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """Center-pad an image or mask so LLaVA and supervision share geometry."""

    height, width = array.shape[:2]
    side = max(height, width)
    shape = (side, side) + tuple(array.shape[2:])
    padded = np.full(shape, fill_value, dtype=array.dtype)
    top = (side - height) // 2
    left = (side - width) // 2
    padded[top : top + height, left : left + width] = array
    return padded


def _downsample_binary_mask_preserve_any(mask: np.ndarray, grid_size: int) -> np.ndarray:
    """Max-pool arbitrary image geometry into a grid without losing tiny lesions."""

    binary = np.asarray(mask) > 0
    height, width = binary.shape
    output = np.zeros((grid_size, grid_size), dtype=np.float32)
    for row in range(grid_size):
        y1 = int(np.floor(row * height / grid_size))
        y2 = max(y1 + 1, int(np.ceil((row + 1) * height / grid_size)))
        for column in range(grid_size):
            x1 = int(np.floor(column * width / grid_size))
            x2 = max(x1 + 1, int(np.ceil((column + 1) * width / grid_size)))
            output[row, column] = float(binary[y1:y2, x1:x2].any())
    return output


def _ring_boundary_target(mask: np.ndarray) -> np.ndarray:
    """Represent double-ring annotations with two elliptical boundary bands."""

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    result = np.zeros_like(binary)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    for component in range(1, count):
        x, y, width, height, area = stats[component]
        if area <= 0:
            continue
        center = tuple(int(round(value)) for value in centroids[component])
        outer_axes = (max(1, width // 2), max(1, height // 2))
        inner_axes = (
            max(1, int(round(outer_axes[0] * 0.62))),
            max(1, int(round(outer_axes[1] * 0.62))),
        )
        thickness = max(1, int(round(min(width, height) * 0.035)))
        cv2.ellipse(result, center, outer_axes, 0, 0, 360, 1, thickness)
        cv2.ellipse(result, center, inner_axes, 0, 0, 360, 1, thickness)
    return result


def _grounding_targets(target_masks: np.ndarray, roi: Tuple[int, int, int, int], grid_size: int) -> np.ndarray:
    x1, y1, x2, y2 = roi
    prepared_targets = [
        _ring_boundary_target(target) if index == 3 else (target > 0).astype(np.uint8)
        for index, target in enumerate(target_masks)
    ]
    view_targets: List[np.ndarray] = []
    for view_index in range(3):
        per_token = []
        for target in prepared_targets:
            source = target[y1:y2, x1:x2] if view_index == 1 else target
            source = pad_to_square(source, fill_value=0)
            per_token.append(_downsample_binary_mask_preserve_any(source, grid_size))
        view_targets.append(np.stack(per_token))
    return np.stack(view_targets)


def cache_one_case(
    stem: str,
    image_path: Path,
    lesion_mask: np.ndarray,
    target_masks: np.ndarray,
    cache_dir: Path,
    extractor,
    roi_margin: float = 0.15,
) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"unable to read image for VLM cache: {image_path}")
    if target_masks.shape != (4, image.shape[0], image.shape[1]):
        raise ValueError(f"target mask shape mismatch for {stem}: {target_masks.shape}")
    views, view_boxes, roi = build_three_views(
        image, lesion_mask, margin=roi_margin
    )
    patches = np.asarray(extractor.extract(views), dtype=np.float16)
    expected = (3, extractor.grid_size * extractor.grid_size)
    if patches.ndim != 3 or patches.shape[:2] != expected:
        raise ValueError(f"unexpected patch feature shape for {stem}: {patches.shape}")
    targets = _grounding_targets(target_masks, roi, extractor.grid_size)

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{stem}.npz"
    temporary = cache_dir / f".{stem}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        temporary,
        patches=patches,
        targets=targets.astype(np.float16),
        view_boxes=view_boxes,
        image_shape=np.asarray(image.shape[:2], dtype=np.int32),
    )
    os.replace(temporary, destination)
    return destination


class FrozenLlavaPatchExtractor:
    """Extract frozen LLaVA vision-tower patches without text generation."""

    def __init__(self, model, processor, device: str = "cuda:0", grid_size: int = 12, model_id: str = ""):
        self.model = model
        self.processor = processor
        self.device = device
        self.grid_size = int(grid_size)
        self.model_id = model_id

    @classmethod
    def from_pretrained(cls, model_id: str, device: str = "cuda:0", grid_size: int = 12):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval().to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(model, processor, device=device, grid_size=grid_size, model_id=model_id)

    def extract(self, images: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        rgb_images = [
            cv2.cvtColor(pad_to_square(image, fill_value=114), cv2.COLOR_BGR2RGB)
            for image in images
        ]
        encoded = self.processor.image_processor(images=rgb_images, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(self.device, dtype=torch.float16)
        with torch.inference_mode():
            output = self.model.vision_tower(pixel_values, output_hidden_states=True)
            layer = int(getattr(self.model.config, "vision_feature_layer", -2))
            features = output.hidden_states[layer]
            strategy = getattr(self.model.config, "vision_feature_select_strategy", "default")
            if strategy == "default" and features.shape[1] > 1:
                features = features[:, 1:]
            side = int(round(features.shape[1] ** 0.5))
            if side * side != features.shape[1]:
                raise ValueError(f"LLaVA vision patches are not square: {features.shape}")
            features = features.transpose(1, 2).reshape(features.shape[0], features.shape[2], side, side)
            features = functional.adaptive_avg_pool2d(features, (self.grid_size, self.grid_size))
            features = features.flatten(2).transpose(1, 2).float().cpu().numpy()
        return features

    def semantic_initializers(self, prompts: Sequence[str] = DERMAL_PROMPTS) -> np.ndarray:
        import torch

        tokenizer = self.processor.tokenizer
        embeddings = self.model.get_input_embeddings()
        results = []
        with torch.inference_mode():
            for prompt in prompts:
                ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
                results.append(embeddings(ids).mean(dim=1).squeeze(0).float().cpu().numpy())
        return np.stack(results)


def write_cache_manifest(
    path: Path,
    fingerprint: str,
    model_id: str,
    stems: Sequence[str],
    grid_size: int,
    roi_margin: float = 0.15,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "model_id": model_id,
        "grid_size": int(grid_size),
        "roi_margin": float(roi_margin),
        "views": ["global", "roi", "masked"],
        "preprocessing_version": GROUNDING_PREPROCESSING_VERSION,
        "case_count": len(stems),
        "stems": sorted(stems),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
