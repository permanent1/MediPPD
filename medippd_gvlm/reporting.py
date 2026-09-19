"""Tabular result export for the MediPPD main experiment."""

from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd


RED_COLUMNS = [
    "method",
    "mask_precision",
    "mask_recall",
    "mask_dice",
    "mask_iou",
    "mask_map50",
    "mask_map50_95",
    "masked_view_grounding_dice",
    "diameter_mae_mm",
    "diameter_rmse_mm",
    "diameter_r2",
    "diameter_acc_2mm",
    "diameter_acc_5mm",
    "runtime_minutes",
]

STRONG_COLUMNS = [
    "method",
    "blister_classification_ap",
    "necrosis_classification_ap",
    "double_ring_classification_ap",
    "macro_classification_ap",
    "blister_localized_ap50",
    "necrosis_localized_ap50",
    "double_ring_localized_ap50",
    "localized_map50",
    "localized_map50_95",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "strong_any_sensitivity",
    "strong_any_specificity",
    "strong_any_f1",
    "strong_any_auroc",
    "strong_any_auprc",
    "runtime_minutes",
]


def _frame(
    rows: Sequence[Mapping[str, object]], columns
) -> pd.DataFrame:
    return pd.DataFrame(rows).reindex(columns=columns)


def write_main_result_bundle(
    result_root: Path,
    red_rows,
    strong_rows,
    runtime_rows,
    per_case_rows=(),
    detection_rows=(),
) -> Dict[str, pd.DataFrame]:
    """Write only the production method's tables and case-level outputs."""

    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    frames = {
        "main_redswollen": _frame(red_rows, RED_COLUMNS),
        "main_strong_features": _frame(strong_rows, STRONG_COLUMNS),
        "runtime_budget": pd.DataFrame(runtime_rows),
        "per_case_predictions": pd.DataFrame(per_case_rows),
        "detection_diagnostics": pd.DataFrame(detection_rows),
    }
    for name, frame in frames.items():
        frame.to_csv(
            result_root / f"{name}.csv", index=False, encoding="utf-8-sig"
        )
    with pd.ExcelWriter(result_root / "tables.xlsx", engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return frames
