# Dataset layout

To ensure anonymity, the specific dataset links will be made public after the paper is accepted.

After the download is complete, place the private files in the following layout:

```text
datasets/
├── ppd553_patient_info.csv
└── ppd553_seg/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
```

Labels use YOLO segmentation format with five classes:

| ID | Class |
|---:|---|
| 0 | bottle cap reference |
| 1 | red/swollen reaction area |
| 2 | blister |
| 3 | necrosis |
| 4 | double-ring reaction |

Patient metadata must match the schema described in the project README. Do not
commit images, labels, metadata, derived caches, or case identifiers to Git.
