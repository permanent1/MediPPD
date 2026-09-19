"""Public configuration and repository-relative paths for MediPPD."""

from pathlib import Path
from typing import Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "medippd_main.yaml"


def load_main_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the main experiment YAML without introducing private defaults."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a YAML mapping in {path}")
    return dict(payload)


MAIN_CONFIG = load_main_config()
SEED = int(MAIN_CONFIG["seed"])
CAP_DIAMETER_MM = float(MAIN_CONFIG["cap_diameter_mm"])
SOURCE_DATASET = PROJECT_ROOT / MAIN_CONFIG["data"]["dataset_root"]
PATIENT_CSV = PROJECT_ROOT / MAIN_CONFIG["data"]["patient_csv"]
RESULT_ROOT = PROJECT_ROOT / MAIN_CONFIG["outputs"]["result_root"]
RUN_ROOT = PROJECT_ROOT / MAIN_CONFIG["outputs"]["run_root"]

ALLOWED_CLINICAL_COLUMNS = (
    "性别文本",
    "年龄",
    "测量间隔小时",
    "色泽文本",
    "硬结触感文本",
)

FORBIDDEN_CLINICAL_FRAGMENTS = (
    "结果评判",
    "硬结横径",
    "硬结纵径",
    "硬结平均径",
    "特征描述",
)
