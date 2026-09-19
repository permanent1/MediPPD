import numpy as np
import pytest
import cv2

from medippd_gvlm import data
from medippd_gvlm.data import CaseRecord, ClinicalEncoderSpec, assert_disjoint_split, build_task_datasets, load_patient_rows


def test_split_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_split(["case-a", "case-b"], ["case-b", "case-c"])


def test_forbidden_target_fields_cannot_change_clinical_tensor():
    train_rows = [
        {
            "性别文本": "男",
            "年龄": "40",
            "测量间隔小时": "48",
            "色泽文本": "淡红",
            "硬结触感文本": "硬度适中",
            "结果评判文本": "强阳性",
            "硬结平均径": "20",
            "特征描述文本": "水疱",
        },
        {
            "性别文本": "女",
            "年龄": "60",
            "测量间隔小时": "72",
            "色泽文本": "无变化",
            "硬结触感文本": "偏软",
            "结果评判文本": "阴性",
            "硬结平均径": "0",
            "特征描述文本": "无异常",
        },
    ]
    spec = ClinicalEncoderSpec.fit(train_rows)
    original = spec.transform_one(train_rows[0])
    changed = dict(train_rows[0])
    changed.update({"结果评判文本": "阴性", "硬结平均径": "0", "特征描述文本": "坏死"})

    np.testing.assert_allclose(original, spec.transform_one(changed))
    assert not any("结果评判" in name or "硬结平均径" in name or "特征描述" in name for name in spec.feature_names)


def test_detection_box_is_converted_to_polygon_in_segmentation_dataset(tmp_path):
    image_path = tmp_path / "source" / "case.jpg"
    label_path = tmp_path / "source" / "case.txt"
    image_path.parent.mkdir()
    cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
    label_path.write_text("1 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    case = CaseRecord("case", image_path, label_path, "train", {})

    seg_yaml, _ = build_task_datasets([case], tmp_path / "built")
    output_line = (seg_yaml.parent / "labels" / "train" / "case.txt").read_text().split()

    assert len(output_line) == 9
    assert output_line[0] == "1"


def test_strong_dataset_remaps_sparse_classes_to_contiguous_ids(tmp_path):
    image_path = tmp_path / "source" / "case.jpg"
    label_path = tmp_path / "source" / "case.txt"
    image_path.parent.mkdir()
    cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
    label_path.write_text(
        "2 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n"
        "3 0.4 0.2 0.6 0.2 0.6 0.4 0.4 0.4\n"
        "4 0.7 0.2 0.9 0.2 0.9 0.4 0.7 0.4\n",
        encoding="utf-8",
    )
    case = CaseRecord("case", image_path, label_path, "train", {})

    yaml_path = data.build_contiguous_strong_dataset([case], tmp_path / "built")

    output_ids = [int(line.split()[0]) for line in (yaml_path.parent / "labels" / "train" / "case.txt").read_text().splitlines()]
    assert output_ids == [0, 1, 2]
    assert yaml_path.read_text(encoding="utf-8").split("names:\n", maxsplit=1)[1] == (
        "  0: blister\n  1: necrosis\n  2: double_ring\n"
    )


def test_patient_metadata_matches_normalized_image_stem(tmp_path):
    patient_csv = tmp_path / "patients.csv"
    patient_csv.write_text(
        "image_name,case_id,结果评判文本,硬结平均径,测量间隔小时\n"
        "synthetic（1）.jpg,case-001,阴性,0,48\n",
        encoding="utf-8",
    )

    patient = load_patient_rows(patient_csv)["synthetic（1）"]

    assert patient["case_id"] == "case-001"
    assert patient["结果评判文本"] == "阴性"
    assert float(patient["硬结平均径"]) == 0.0
    assert float(patient["测量间隔小时"]) == 48.0
