"""YOLO-anchored semantic residual fusion for the red-swollen mask."""

import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import class_mask, read_yolo_segments


FUSION_IMAGE_SIZE = 128


def open_mask_prior(mask: np.ndarray) -> np.ndarray:
    """Suppress isolated YOLO pixels with a fixed, training-selected opening."""

    source = (np.asarray(mask) > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(source, cv2.MORPH_OPEN, kernel)


def project_roi_grounding(
    roi_map: np.ndarray,
    normalized_box: Sequence[float],
    output_shape: tuple[int, int],
    fill_value: float = 0.0,
) -> np.ndarray:
    """Project an ROI-coordinate grounding map into global image coordinates."""

    height, width = (int(output_shape[0]), int(output_shape[1]))
    box = np.asarray(normalized_box, dtype=np.float32)
    x1, y1, x2, y2 = np.round(
        box * np.asarray([width, height, width, height], dtype=np.float32)
    ).astype(int)
    x1, y1 = np.clip([x1, y1], [0, 0], [width - 1, height - 1])
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    projected = np.full(
        (height, width), float(fill_value), dtype=np.float32
    )
    projected[y1:y2, x1:x2] = cv2.resize(
        np.asarray(roi_map, dtype=np.float32),
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_LINEAR,
    )
    return projected


def restore_square_padded_map(
    square_map: np.ndarray, original_shape: tuple[int, int]
) -> np.ndarray:
    """Undo the center square-padding used before LLaVA preprocessing."""

    height, width = (int(original_shape[0]), int(original_shape[1]))
    side = max(height, width)
    restored_square = cv2.resize(
        np.asarray(square_map, dtype=np.float32),
        (side, side),
        interpolation=cv2.INTER_LINEAR,
    )
    top = (side - height) // 2
    left = (side - width) // 2
    return restored_square[top : top + height, left : left + width]


def prepare_fusion_inputs(
    lesion_mask: np.ndarray,
    red_grounding_logits: np.ndarray,
    roi_box: Sequence[float],
    image_size: int = FUSION_IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned VLM probabilities and the conservative YOLO prior."""

    shape = (int(image_size), int(image_size))
    lesion = cv2.resize(
        (np.asarray(lesion_mask) > 0).astype(np.uint8),
        shape[::-1],
        interpolation=cv2.INTER_NEAREST,
    )
    prior = open_mask_prior(lesion).astype(np.float32)[None]
    logits = np.asarray(red_grounding_logits, dtype=np.float32)
    if logits.shape[0] != 3:
        raise ValueError("red_grounding_logits must contain global, ROI, and masked views")
    image_height, image_width = np.asarray(lesion_mask).shape[:2]
    box = np.asarray(roi_box, dtype=np.float32)
    x1, y1, x2, y2 = np.round(
        box
        * np.asarray(
            [image_width, image_height, image_width, image_height],
            dtype=np.float32,
        )
    ).astype(int)
    roi_shape = (max(1, y2 - y1), max(1, x2 - x1))
    global_logits = restore_square_padded_map(
        logits[0], (image_height, image_width)
    )
    roi_logits = restore_square_padded_map(logits[1], roi_shape)
    masked_logits = restore_square_padded_map(
        logits[2], (image_height, image_width)
    )
    aligned_logits = np.stack(
        [
            cv2.resize(global_logits, shape[::-1], interpolation=cv2.INTER_LINEAR),
            project_roi_grounding(roi_logits, roi_box, shape, fill_value=-12.0),
            cv2.resize(masked_logits, shape[::-1], interpolation=cv2.INTER_LINEAR),
        ]
    )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(aligned_logits, -20.0, 20.0)))
    return probabilities.astype(np.float32), prior


def build_fusion_tensors(
    model,
    dataset,
    config,
    device: str = "cuda:0",
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize train-only aligned inputs for fast residual-head training."""

    all_probabilities = []
    all_priors = []
    all_targets = []
    model.eval()
    model_keys = ("patches", "morphometry", "diameter_baseline", "clinical", "index")
    with torch.no_grad():
        for start in range(0, len(dataset), int(batch_size)):
            samples = [
                dataset[index]
                for index in range(start, min(start + int(batch_size), len(dataset)))
            ]
            batch = {
                key: torch.stack([torch.as_tensor(sample[key]) for sample in samples]).to(
                    device
                )
                for key in model_keys
            }
            red_logits = model(batch, config).grounding_logits[:, :, 0].cpu().numpy()
            for offset, case in enumerate(dataset.cases[start : start + len(samples)]):
                with np.load(dataset.prediction_dir / f"{case.stem}.npz") as prediction:
                    image_shape = tuple(int(value) for value in prediction["image_shape"][:2])
                    lesion_mask = prediction["lesion_mask"].copy()
                with np.load(dataset.cache_dir / f"{case.stem}.npz") as cached:
                    roi_box = cached["view_boxes"][1].copy()
                probabilities, prior = prepare_fusion_inputs(
                    lesion_mask, red_logits[offset], roi_box
                )
                objects = read_yolo_segments(case.label_path, image_shape)
                target = cv2.resize(
                    class_mask(objects, 1, image_shape),
                    (FUSION_IMAGE_SIZE, FUSION_IMAGE_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.float32)[None]
                all_probabilities.append(probabilities.astype(np.float16))
                all_priors.append(prior.astype(np.float16))
                all_targets.append(target.astype(np.float16))
    return (
        torch.from_numpy(np.stack(all_probabilities)),
        torch.from_numpy(np.stack(all_priors)),
        torch.from_numpy(np.stack(all_targets)),
    )


class RedMaskFusionHead(nn.Module):
    """Learn a bounded semantic residual while preserving the YOLO prior."""

    def __init__(self, active_views: int = 3, hidden_channels: int = 16):
        super().__init__()
        if active_views < 0 or active_views > 3:
            raise ValueError("active_views must be between zero and three")
        self.active_views = int(active_views)
        self.residual = nn.Sequential(
            nn.Conv2d(self.active_views + 1, hidden_channels, 5, padding=2),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, vlm_probabilities: torch.Tensor, prior: torch.Tensor):
        views = vlm_probabilities[:, : self.active_views]
        inputs = torch.cat([prior, views], dim=1)
        prior_logit = (prior * 2.0 - 1.0) * 2.0
        return prior_logit + 4.0 * torch.tanh(self.residual(inputs))


def mask_dice_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    prediction = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    true_positive = float((prediction & truth).sum().item())
    false_positive = float((prediction & ~truth).sum().item())
    false_negative = float((~prediction & truth).sum().item())
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 1.0


def train_fusion_head(
    vlm_probabilities: torch.Tensor,
    priors: torch.Tensor,
    targets: torch.Tensor,
    active_views: int,
    checkpoint: Path,
    device: str = "cuda:0",
    epochs: int = 40,
    seed: int = 42,
) -> tuple[RedMaskFusionHead, float]:
    """Train with a deterministic case-level internal holdout, never test labels."""

    from sklearn.model_selection import train_test_split

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    started = time.monotonic()
    labels = targets.flatten(start_dim=1).amax(dim=1).cpu().numpy()
    fit_indices, holdout_indices = train_test_split(
        np.arange(len(targets)),
        test_size=0.2,
        random_state=int(seed),
        stratify=labels,
    )
    fit_indices = torch.as_tensor(fit_indices)
    holdout_indices = torch.as_tensor(holdout_indices)
    loader = DataLoader(
        TensorDataset(
            vlm_probabilities[fit_indices],
            priors[fit_indices],
            targets[fit_indices],
        ),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    model = RedMaskFusionHead(active_views=active_views).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_dice = -1.0
    best_state = None
    for epoch in range(int(epochs) + 1):
        if epoch:
            model.train()
            for probabilities, prior, target in loader:
                probabilities = probabilities.to(device=device, dtype=torch.float32)
                prior = prior.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.float32)
                logits = model(probabilities, prior)
                prediction = torch.sigmoid(logits)
                intersection = (prediction * target).sum(dim=(-1, -2))
                denominator = prediction.sum(dim=(-1, -2)) + target.sum(
                    dim=(-1, -2)
                )
                dice_loss = 1.0 - (
                    (2.0 * intersection + 1.0) / (denominator + 1.0)
                ).mean()
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    target,
                    pos_weight=torch.tensor(2.0, device=device),
                )
                optimizer.zero_grad(set_to_none=True)
                (dice_loss + bce).backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(
                vlm_probabilities[holdout_indices].to(device, dtype=torch.float32),
                priors[holdout_indices].to(device, dtype=torch.float32),
            ).cpu()
        score = mask_dice_from_logits(logits, targets[holdout_indices])
        if score > best_dice:
            best_dice = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "active_views": int(active_views),
            "internal_holdout_dice": float(best_dice),
            "seed": int(seed),
        },
        checkpoint,
    )
    model.eval()
    return model, time.monotonic() - started


def load_fusion_head(checkpoint: Path, device: str = "cuda:0") -> RedMaskFusionHead:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = RedMaskFusionHead(active_views=int(state["active_views"])).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model
