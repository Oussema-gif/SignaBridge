#!/usr/bin/env python3
"""
SignaBridge — canonical 63-D feature extractor.

This module is the authoritative reference for the numerical preprocessing
contract that the Android FeatureExtractor.java MUST reproduce.

Contract:
    Input : 21 MediaPipe hand landmarks, each (x, y, z) in normalized image
            coordinates (x,y in [0,1], z relative to wrist depth).
    Output: float32 vector of length 63.
    Rules : translate wrist (index 0) to origin; scale by Euclidean distance
            from wrist to middle-finger MCP (index 9); reject degenerate
            inputs (NaN, Inf, distance < 1e-6); preserve landmark order.

Any change to these rules requires re-running the Python/Java parity test.
"""

from __future__ import annotations

import numpy as np

WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63
MIN_SCALE = 1e-6


class FeatureExtractionError(ValueError):
    """Raised when a landmark set cannot be converted to a valid feature vector."""


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Convert a (21, 3) array of hand landmarks into a (63,) float32 feature vector.

    Parameters
    ----------
    landmarks : np.ndarray
        Array of shape (21, 3) with dtype convertible to float32.
        Row 0 is the wrist; row 9 is the middle-finger MCP.

    Returns
    -------
    np.ndarray
        Shape (63,), dtype float32.

    Raises
    ------
    FeatureExtractionError
        If shape is wrong, any value is non-finite, or the wrist→middle-MCP
        distance is degenerate (< MIN_SCALE).
    """
    pts = np.asarray(landmarks, dtype=np.float32)
    if pts.shape != (NUM_LANDMARKS, 3):
        raise FeatureExtractionError(
            f"Expected landmarks of shape ({NUM_LANDMARKS}, 3), got {pts.shape}"
        )
    if not np.all(np.isfinite(pts)):
        raise FeatureExtractionError("Landmarks contain NaN or Inf values.")

    wrist = pts[WRIST_INDEX]
    translated = pts - wrist  # (21, 3)

    middle_mcp = translated[MIDDLE_MCP_INDEX]
    scale = float(np.linalg.norm(middle_mcp))
    if scale < MIN_SCALE:
        raise FeatureExtractionError(
            f"Degenerate hand scale ({scale:.3e} < {MIN_SCALE:.0e}). "
            "Wrist and middle-finger MCP coincide."
        )

    normalized = translated / scale
    features = normalized.reshape(-1).astype(np.float32)

    assert features.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(features))
    return features


def extract_features_batch(landmarks_batch: np.ndarray) -> np.ndarray:
    """
    Vectorized version for pre-extracted landmark arrays.

    Parameters
    ----------
    landmarks_batch : np.ndarray
        Shape (N, 21, 3).

    Returns
    -------
    np.ndarray
        Shape (N, 63), dtype float32.

    Raises
    ------
    FeatureExtractionError
        If any sample is invalid. The offending index is included in the message.
    """
    batch = np.asarray(landmarks_batch, dtype=np.float32)
    if batch.ndim != 3 or batch.shape[1:] != (NUM_LANDMARKS, 3):
        raise FeatureExtractionError(
            f"Expected batch of shape (N, {NUM_LANDMARKS}, 3), got {batch.shape}"
        )
    n = batch.shape[0]
    features = np.empty((n, FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        try:
            features[i] = extract_features(batch[i])
        except FeatureExtractionError as exc:
            raise FeatureExtractionError(f"Sample {i}: {exc}") from exc
    return features


if __name__ == "__main__":
    # Minimal self-test with a canonical right-hand landmark set.
    # A straight-open hand: wrist at origin, middle MCP 0.1 in +y.
    sample = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    sample[MIDDLE_MCP_INDEX] = [0.0, 0.1, 0.0]
    f = extract_features(sample)
    assert f.shape == (63,), f.shape
    assert np.all(np.isfinite(f))
    assert abs(f[3 * MIDDLE_MCP_INDEX + 1] - 1.0) < 1e-6, "middle MCP y should be 1.0"
    print("feature_extractor self-test: PASS (63-D, middle-MCP y = 1.0)")