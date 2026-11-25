# SignaBridge — Stage 1: Dataset Feasibility

## Dataset
ASL-HG: American Sign Language Hand Gesture Image Dataset
DOI: 10.17632/j4y5w2c8w9.1
License: CC BY 4.0

## Environment (Fedora Linux)
See commands below.

## Usage
1. Download and extract dataset into `tools/datasets/asl_hg/`.
2. Download `hand_landmarker.task` into `tools/`.
3. Run `python inspect_dataset.py`.
4. Run `python extract_landmarks.py`.

## Expected outputs
- `tools/artifacts/dataset_report.json`
- `tools/artifacts/landmark_feasibility.json`