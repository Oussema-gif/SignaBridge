# SignaBridge — ML Pipeline Documentation

## Dataset
- **Name:** ASL-HG (American Sign Language Hand Gesture Image Dataset)
- **DOI:** 10.17632/j4y5w2c8w9.1
- **License:** CC BY 4.0
- **Classes used:** A, B, L, V, Y (from the 36 available)
- **Split Strategy:** Signer-independent (P1-P6 Train, P7-P8 Val, P9-P10 Test).

## Feature Extraction (Python & Java)
- **Input:** 21 MediaPipe hand landmarks (x, y, z).
- **Normalization:** Translate wrist (index 0) to origin, scale by distance to middle MCP (index 9).
- **Output:** 63-D float32 vector.
- **Parity:** Verified via `parity_vectors.json` (tolerance 1e-5).

## Model Architecture
- **Type:** Multi-Layer Perceptron (MLP)
- **Layers:** 63 -> 64 (ReLU) -> 32 (ReLU) -> 5 (Softmax)
- **Parameters:** 6,341
- **Training:** Adam, Label Smoothing (0.05), Early Stopping (Patience 10), Dropout (0.1).
- **Export:** TensorFlow Lite (Float32, 27 KB).

## Evaluation Results (Signer-Independent)
- **Test Accuracy:** 1.0000 (Macro F1: 1.0000)
- **Target Retention @ Threshold 0.9488:** 93.10%
- **Proxy-Unknown Rejection @ 0.9488:** 91.13%
- **Proxy-Unknown Rejection @ 0.80 (Master Prompt Default):** 53.87%
*(Note: The 0.80 default was rejected based on the empirical data).*

## Reproducibility
To reproduce the ML pipeline from scratch:
```bash
source tools/.venv/bin/activate
python tools/inspect_dataset.py
python tools/prepare_dataset.py
python tools/extract_landmarks_batch.py
python tools/train.py
python tools/evaluate.py
python tools/export_tflite.py

**`BRANDING.md`** (Brand Documentation)
```markdown
# SignaBridge Brand Guidelines

## Concept
A bridge between communication and understanding. The logo combines an open hand (communication) with a structural arc (a bridge).

## Color Palette
- **Primary Navy:** `#14284B`
- **Accent Teal:** `#1F8A8A`
- **Light Background:** `#EEF4F6`
- **Text:** `#14284B`

## Typography
- **Font:** Roboto (System default for Android, modern and accessible).

## States
- **Recognized:** Teal (`#1F8A8A`)
- **Uncertain:** Grey (`#8A8A8A`)
- **No Hand:** Dim White (`#B3FFFFFF`)