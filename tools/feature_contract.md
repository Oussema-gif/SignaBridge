# SignaBridge — Feature Extraction Contract (v1)

This document is the single source of truth for the 63-D feature vector that
flows from MediaPipe landmarks into both the Python training pipeline and the
Android TFLite inference pipeline. Any change here requires re-running the
Python/Java parity test before a release.

## Input
- 21 MediaPipe HandLandmarker landmarks.
- Each landmark `i` has normalized coordinates `(x_i, y_i, z_i)`:
  - `x_i`, `y_i` ∈ [0, 1] relative to image width/height.
  - `z_i` is depth relative to the wrist, same scale as `x_i`.
- Landmark indices:
  - 0 = wrist
  - 1–4 = thumb (CMC, MCP, IP, TIP)
  - 5–8 = index finger (MCP, PIP, DIP, TIP)
  - 9–12 = middle finger (MCP, PIP, DIP, TIP)
  - 13–16 = ring finger
  - 17–20 = pinky finger

## Output
- `float32[63]`.
- Layout: `[f_0x, f_0y, f_0z, f_1x, f_1y, f_1z, …, f_20x, f_20y, f_20z]`.

## Normalization (exact, in this order)
1. Reject if any of the 63 values is NaN or ±Inf.
2. `p_i = L_i − L_0` for all i (wrist at origin).
3. `d = ‖L_9 − L_0‖₂` (Euclidean distance, wrist → middle MCP).
4. Reject if `d < 1e-6`.
5. `f_i = p_i / d`.

## Invariance properties
- **Translation invariant**: yes, by construction (step 2).
- **Scale invariant**: yes, by construction (step 5).
- **Rotation invariant**: NO. Hand rotation relative to the camera is
  preserved deliberately; the classifier is trained with the same convention.
- **Handedness invariant**: NOT YET. V1 preserves the raw handedness as
  returned by MediaPipe. Handedness canonicalization (mirroring left hands)
  is deferred to a later module and will require experimental validation
  before it is enabled in either Python or Java.

## Rejection behaviour
- Python: `FeatureExtractionError` with a descriptive message.
- Java: `FeatureExtractionException` with the equivalent message.
- The Android recognition pipeline treats a rejected frame as `NO_HAND`.

## Non-goals
- We do not normalize for camera focal length.
- We do not normalize for perspective distortion.
- We do not attempt 3D pose alignment.

These omissions are intentional and will be documented as limitations in the
final ML README.