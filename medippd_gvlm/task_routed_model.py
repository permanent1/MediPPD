"""Task-routed V2 model with isolated measurement, case, and grounding paths."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TaskRoutedConfig:
    """Configuration for the production task-routed MediPPD model."""

    name: str
    max_delta_mm: float = 3.0
    train_diameter: bool = True
    train_case: bool = True
    train_grounding: bool = True
    use_roi_classification: bool = True
    use_masked_grounding: bool = True
    use_direct_clinical: bool = True
    use_size_prior: bool = False
    use_task_consistency: bool = True
    use_task_routing: bool = True
    active_views: tuple = (0, 1, 2)
    consistency_weight: float = 1.0
    topk_ratio: float = 0.10
    grounding_loss: str = "dice_balanced_bce"
    class_loss: str = "focal"
    sampler_strategy: str = "weighted"
    sharing_strategy: str = "full_routing"
    case_morphometry: str = "none"

    @classmethod
    def full(cls) -> "TaskRoutedConfig":
        return cls(name="MediPPD")


@dataclass
class TaskRoutedOutput:
    """Predictions plus branch diagnostics consumed by task-aligned losses."""

    grounding_logits: torch.Tensor
    diameter_mm: torch.Tensor
    class_logits: torch.Tensor
    strong_any_logit: torch.Tensor
    reaction_logits: torch.Tensor
    red_mask_logits: torch.Tensor
    class_grounding_logits: torch.Tensor
    diameter_delta_mm: torch.Tensor
    diameter_features: torch.Tensor
    case_features: torch.Tensor
    red_roi_features: torch.Tensor
    clinical_residual: Optional[torch.Tensor]
    size_prior_residual: Optional[torch.Tensor]


class TaskRoutedMediPPD(nn.Module):
    """Compact model whose inputs are explicitly routed by downstream task."""

    def __init__(
        self,
        patch_dim: int,
        semantic_initializers: np.ndarray,
        clinical_dim: int,
        d_model: int = 128,
        grid_size: int = 12,
        num_reaction_classes: int = 4,
    ):
        super().__init__()
        semantic = torch.as_tensor(semantic_initializers, dtype=torch.float32)
        if semantic.ndim != 2 or semantic.shape[0] != 4:
            raise ValueError("semantic_initializers must have shape [4, language_dim]")
        if grid_size < 1:
            raise ValueError("grid_size must be positive")

        self.grid_size = int(grid_size)
        self.d_model = int(d_model)

        # Grounding owns its projection so all three spatial views can be used
        # without entering either case-level branch.
        self.ground_patch_projection = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.semantic_projection = nn.Linear(semantic.shape[1], d_model)
        self.register_buffer("semantic_initializers", semantic)
        self.query_delta = nn.Parameter(torch.zeros(4, d_model))
        self.ground_view_embedding = nn.Parameter(torch.randn(3, d_model) * 0.02)

        # The diameter branch has private global/ROI visual projections and
        # normalizes the seven physical inputs before encoding them.
        self.diameter_patch_projection = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.diameter_morphometry_encoder = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.diameter_fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.diameter_delta = nn.Linear(d_model, 1)
        nn.init.zeros_(self.diameter_delta.weight)
        nn.init.zeros_(self.diameter_delta.bias)

        # The case branch sees only the global and ROI views. Clinical and the
        # optional detached scalar prior are additive residuals after encoding.
        self.case_patch_projection = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.case_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.global_case_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.shared_patch_projection = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.shared_visual_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.clinical_encoder = nn.Sequential(
            nn.LayerNorm(clinical_dim),
            nn.Linear(clinical_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
        )
        self.size_prior_projection = nn.Linear(1, d_model, bias=False)
        self.case_residual_norm = nn.LayerNorm(d_model)
        self.class_head = nn.Linear(d_model, 3)
        self.strong_any_head = nn.Linear(d_model, 1)
        self.reaction_head = nn.Linear(d_model, num_reaction_classes)

    def _queries(self, batch_size: int) -> torch.Tensor:
        queries = self.semantic_projection(self.semantic_initializers) + self.query_delta
        return queries.unsqueeze(0).expand(batch_size, -1, -1)

    def _grounding_branch(self, patches: torch.Tensor, config: TaskRoutedConfig) -> torch.Tensor:
        batch_size, _, patch_count, _ = patches.shape
        configured = tuple(int(index) for index in config.active_views)
        if not configured or len(set(configured)) != len(configured) or any(index not in (0, 1, 2) for index in configured):
            raise ValueError("active_views must contain unique indices from {0, 1, 2}")
        active_views = tuple(
            index for index in configured if config.use_masked_grounding or index != 2
        )
        if not active_views:
            raise ValueError("configuration disables every grounding view")
        view_index = torch.as_tensor(active_views, device=patches.device)
        projected = self.ground_patch_projection(patches.index_select(1, view_index))
        projected = projected + self.ground_view_embedding.index_select(0, view_index).view(
            1, len(active_views), 1, self.d_model
        )
        active_logits = torch.einsum(
            "bkd,bvpd->bvkp", self._queries(batch_size), projected
        ) * (self.d_model**-0.5)
        logits = torch.full(
            (batch_size, 3, 4, patch_count),
            -12.0,
            dtype=active_logits.dtype,
            device=active_logits.device,
        )
        logits[:, view_index] = active_logits
        return logits.reshape(batch_size, 3, 4, self.grid_size, self.grid_size)

    def _diameter_branch(
        self,
        patches: torch.Tensor,
        morphometry: torch.Tensor,
        baseline: torch.Tensor,
        max_delta_mm: float,
        shared_visual: Optional[torch.Tensor] = None,
    ):
        if shared_visual is None:
            visual = self.diameter_patch_projection(patches[:, :2]).mean(dim=2)
            global_visual, roi_visual = visual[:, 0], visual[:, 1]
        else:
            global_visual = roi_visual = shared_visual
        physical = self.diameter_morphometry_encoder(morphometry)
        features = self.diameter_fusion(
            torch.cat([global_visual, roi_visual, physical], dim=-1)
        )
        delta_mm = float(max_delta_mm) * torch.tanh(self.diameter_delta(features).squeeze(-1))
        return baseline + delta_mm, delta_mm, features

    def _case_branch(
        self,
        patches: torch.Tensor,
        clinical: torch.Tensor,
        morphometry: torch.Tensor,
        baseline: torch.Tensor,
        config: TaskRoutedConfig,
        shared_visual: Optional[torch.Tensor] = None,
    ):
        projected = self.case_patch_projection(patches).mean(dim=2)
        active_views = tuple(int(index) for index in config.active_views)
        if shared_visual is not None:
            case_features = shared_visual
        elif config.use_roi_classification and 0 in active_views and 1 in active_views:
            case_features = self.case_fusion(
                torch.cat([projected[:, 0], projected[:, 1]], dim=-1)
            )
        elif len(active_views) >= 2:
            case_features = self.case_fusion(
                torch.cat([projected[:, active_views[0]], projected[:, active_views[1]]], dim=-1)
            )
        else:
            case_features = self.global_case_fusion(projected[:, active_views[0]])

        clinical_residual = None
        if config.use_direct_clinical:
            clinical_residual = self.clinical_encoder(clinical)
            case_features = case_features + clinical_residual

        size_prior_residual = None
        if config.case_morphometry == "full":
            size_prior_residual = self.diameter_morphometry_encoder(morphometry)
            case_features = case_features + size_prior_residual
        elif config.use_size_prior or config.case_morphometry == "diameter":
            normalized_size = baseline.detach().unsqueeze(-1) / 50.0
            size_prior_residual = self.size_prior_projection(normalized_size)
            case_features = case_features + size_prior_residual
        elif config.case_morphometry != "none":
            raise ValueError("case_morphometry must be one of: none, diameter, full")

        case_features = self.case_residual_norm(case_features)
        return case_features, projected[:, 1], clinical_residual, size_prior_residual

    def _shared_visual_branch(self, patches: torch.Tensor) -> torch.Tensor:
        projected = self.shared_patch_projection(patches[:, :2]).mean(dim=2)
        return self.shared_visual_fusion(
            torch.cat([projected[:, 0], projected[:, 1]], dim=-1)
        )

    def forward(self, batch, config: TaskRoutedConfig) -> TaskRoutedOutput:
        patches = batch["patches"].float()
        if patches.ndim != 4 or patches.shape[1] != 3:
            raise ValueError("patches must have shape [batch, 3, patches, patch_dim]")
        patch_count = patches.shape[2]
        if patch_count != self.grid_size * self.grid_size:
            raise ValueError(f"expected {self.grid_size**2} patches, got {patch_count}")
        if config.max_delta_mm < 0:
            raise ValueError("max_delta_mm must be non-negative")

        morphometry = batch["morphometry"].float()
        baseline = batch.get("diameter_baseline", morphometry[:, 0]).float()
        grounding_logits = self._grounding_branch(patches, config)
        shared_visual = (
            None if config.use_task_routing else self._shared_visual_branch(patches)
        )
        diameter_mm, diameter_delta_mm, diameter_features = self._diameter_branch(
            patches,
            morphometry,
            baseline,
            config.max_delta_mm,
            shared_visual=shared_visual,
        )
        case_features, red_roi_features, clinical_residual, size_prior_residual = self._case_branch(
            patches,
            batch["clinical"].float(),
            morphometry,
            baseline,
            config,
            shared_visual=shared_visual,
        )

        configured_views = tuple(int(index) for index in config.active_views)
        if configured_views == (0, 1, 2) and config.use_masked_grounding:
            red_indices = (0, 2)
        else:
            red_indices = tuple(
                index for index in configured_views if config.use_masked_grounding or index != 2
            )
        red_views = grounding_logits[:, red_indices, 0]
        red_mask_logits = red_views.mean(dim=1)
        return TaskRoutedOutput(
            grounding_logits=grounding_logits,
            diameter_mm=diameter_mm,
            class_logits=self.class_head(case_features),
            strong_any_logit=self.strong_any_head(case_features).squeeze(-1),
            reaction_logits=self.reaction_head(case_features),
            red_mask_logits=red_mask_logits,
            class_grounding_logits=grounding_logits[:, :, 1:],
            diameter_delta_mm=diameter_delta_mm,
            diameter_features=diameter_features,
            case_features=case_features,
            red_roi_features=red_roi_features,
            clinical_residual=clinical_residual,
            size_prior_residual=size_prior_residual,
        )
