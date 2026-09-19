# MediPPD

MediPPD is a task-routed multimodal pipeline for automated interpretation of
purified protein derivative (PPD) skin-test images. This repository contains
the code for the main method only: reaction-area segmentation and physical
measurement, strong-feature detection, frozen vision-language features,
task-routed prediction, spatial grounding, and final mask refinement.

## What the pipeline predicts

- A red/swollen reaction mask and its physical diameter.
- Blister, necrosis, and double-ring probabilities.
- A combined strong-reaction score.
- Spatial evidence maps for the predicted skin findings.

Physical calibration uses the bottle-cap reference in each image with a nominal
diameter of **30.0 mm**. The learned diameter correction is bounded so that the
explicit image-to-millimetre measurement chain remains visible.

## Method overview

The first stage trains two visual priors: a bottle-cap/reaction segmentation
model and a detector for the three strong skin findings. The pipeline then
builds global, reaction-region, and masked views and extracts frozen LLaVA-1.5
visual patch features. A task-routed network sends physical, case-level, and
grounding inputs through separate branches. A lightweight three-view fusion
head refines the reaction mask without replacing the YOLO prior.

## Repository layout

```text
configs/                 Main experiment configuration
core/                    Local Ultralytics-derived YOLO runtime
datasets/README.md       Private dataset layout and label definitions
medippd_gvlm/            MediPPD model, training, metrics, and pipeline
scripts/                 Public command-line entry points
tests/                   Data-free unit and integration tests
```

Generated checkpoints, caches, predictions, and reports are written under
`runs/` and `results/`; both directories are excluded from Git.

## Environment

The reference environment uses Python 3.9 and CUDA-enabled PyTorch. Install a
PyTorch build suitable for your CUDA driver if the pinned wheel in
`requirements.txt` is not appropriate for your machine.

```bash
conda create -n yolo_ppd python=3.9 -y
conda activate yolo_ppd
pip install -r requirements.txt
```

The main configuration uses the frozen visual tower from
`llava-hf/llava-1.5-7b-hf`. Obtain that checkpoint through the official
Hugging Face model page before running in offline mode. YOLO segmentation and
detection initialization weights are also required, but no `.pt` files are
stored in this repository.

## Dataset access and preparation

The medical dataset is not included. For research access, **contact the authors**
and comply with the applicable ethics, privacy, and data-use requirements.
After access is granted, follow [datasets/README.md](datasets/README.md) and set
`data.dataset_root` and `data.patient_csv` in
`configs/medippd_main.yaml`, or pass both paths on the command line.

Expected YOLO class IDs are:

| ID | Meaning |
|---:|---|
| 0 | bottle-cap reference |
| 1 | red/swollen reaction area |
| 2 | blister |
| 3 | necrosis |
| 4 | double-ring reaction |

The clinical encoder accepts only the documented demographic and observation
fields. Outcome labels, measured diameters, and free-text finding descriptions
are excluded from clinical inputs to prevent target leakage.

## Running the main experiment

Inspect all available options:

```bash
python scripts/run_main_experiment.py --help
```

Run the complete ordered pipeline:

```bash
python scripts/run_main_experiment.py \
  --dataset-root datasets/ppd553_seg \
  --patient-csv datasets/ppd553_patient_info.csv \
  --device cuda:0
```

Interrupted runs can validate and reuse completed artifacts:

```bash
python scripts/run_main_experiment.py \
  --dataset-root datasets/ppd553_seg \
  --patient-csv datasets/ppd553_patient_info.csv \
  --resume
```

Focused entry points are also available:

```bash
python scripts/prepare_data.py --help
python scripts/train_segmentation.py --help
python scripts/train_strong_features.py --help
```

The end-to-end phases are data preparation, segmentation-prior training,
strong-feature detector training, prediction export, frozen VLM caching,
task-routed model training, three-view mask fusion, evaluation, and report
generation. Output tables include the reaction-mask and physical-measurement
metrics, strong-feature metrics, runtime accounting, case-level predictions,
and detection diagnostics.

## Configuration

`configs/medippd_main.yaml` records the reference seed (`42`), cap diameter,
relative data/output paths, image sizes, training budgets, VLM checkpoint, and
task-routed optimization settings. Command-line paths override the public
repository-relative defaults. No server-specific absolute path is required.

## Tests

The test suite uses synthetic arrays and temporary files; it does not require
the private dataset or model weights.

```bash
python -m pytest -q
python -m compileall -q core medippd_gvlm scripts tests
```

## Privacy and intended use

Do not commit medical images, labels, patient metadata, case identifiers,
feature caches, model checkpoints, or predictions. This research code is not a
medical device and must not be used for clinical decision-making without
appropriate validation, governance, and regulatory review.

## Third-party code and licensing

The bundled `core/` runtime is derived from Ultralytics and retains AGPL-3.0
license headers. See [THIRD_PARTY.md](THIRD_PARTY.md) for provenance and license
details. No separate license is asserted here for the original MediPPD research
code.
